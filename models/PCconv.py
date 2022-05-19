from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn
import util.util as util
from util.Selfpatch import Selfpatch

from .sdff import SDFF
from .ca import RAL


# SE MODEL
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


# SK MODEL
class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.InstanceNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Convnorm(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', activ='leaky'):
        super().__init__()
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)

        if sample == 'down-3':
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1)
        if activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out[0])
        return out


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=conv_bias,
                                      multi_channel=True, return_mask=True)
        elif sample == 'same-7':
            self.conv = PartialConv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=conv_bias,
                                      multi_channel=True, return_mask=True)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=conv_bias,
                                      multi_channel=True, return_mask=True)
        else:
            self.conv = PartialConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=conv_bias,
                                      multi_channel=True, return_mask=True)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, 'activation'):
                out[0] = self.activation(out[0])
        return out


class ConvDown(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False, layers=1, activ=True):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nfmult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ == False and layers == 1:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True)
                    ]

            else:

                sequence += [
                    nn.Conv2d(in_c, out_c,
                              kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True)
                ]

            if activ == False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(nf_mult * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel,
                              stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear', align_corners=True)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BASE(nn.Module):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        sk = SKConv(inner_nc, 3, 32, 16)
        model = [sk]
        gus = util.gussin(1.5).cuda()
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fuse = ConvDown(512 * 2, 512, 1, 1)
        self.ral = RAL(kernel_size=3, stride=1, rate=2, softmax_scale=10.)

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)

        out_res = out_32
        out_32 = self.ral(out_32, out_32)

        b, c, h, w = out_32.size()  # torch.Size([1, 512, 32, 32])
        gus = self.gus.float()  # torch.Size([1024, 1, 32, 32])
        gus_out = out_32[0].expand(h * w, c, h, w)  # torch.Size([1024, 512, 32, 32])
        gus_out = gus * gus_out  # torch.Size([1024, 512, 32, 32])
        gus_out = torch.sum(gus_out, -1)  # torch.Size([1024, 512, 32])
        gus_out = torch.sum(gus_out, -1)  # torch.Size([1024, 512])
        gus_out = gus_out.contiguous().view(b, c, h, w)  # torch.Size([1, 512, 32, 32])
        csa2_in = torch.sigmoid(out_32)  # torch.Size([1, 512, 32, 32])
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))  # torch.Size([1, 512, 34, 34])
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))  # torch.Size([1, 512, 34, 34])
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1)
        # torch.Size([1024, 512, 3, 3])  torch.Size([1024, 512, 3, 3])  torch.Size([1024, 512, 1, 1])
        csa2_conv = csa2_conv.expand_as(csa2_f)  # torch.Size([1024, 512, 3, 3])
        csa_a = csa2_conv * csa2_f  # torch.Size([1024, 512, 3, 3])
        csa_a = torch.mean(csa_a, 1)  # torch.Size([1024, 3, 3])
        a_c, a_h, a_w = csa_a.size()  # torch.Size([1024, 3, 3])
        csa_a = csa_a.contiguous().view(a_c, -1)  # torch.Size([1024, 9])
        csa_a = F.softmax(csa_a, dim=1)  # torch.Size([1024, 9])
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)  # torch.Size([1024, 1, 3, 3])
        out = csa_a * csa2_fff  # torch.Size([1024, 512, 3, 3])
        out = torch.sum(out, -1)  # torch.Size([1024, 512, 3])
        out = torch.sum(out, -1)  # torch.Size([1024, 512])
        out_csa = out.contiguous().view(b, c, h, w)  # torch.Size([1, 512, 32, 32])
        out_32 = torch.cat([gus_out, out_csa], 1)  # torch.Size([1, 1024, 32, 32])
        out_32 = self.down(out_32)  # torch.Size([1, 512, 32, 32])

        out_32 = torch.cat([out_32, out_res], 1)
        out_32 = self.fuse(out_32)
        return out_32


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, inputt, mask_in=None):
        input = inputt[0]
        mask_in = inputt[1].float().cuda()
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            out = []
            out.append(output)
            out.append(self.update_mask)
            return out
        else:
            return output


class PCconv(nn.Module):
    def __init__(self):
        super(PCconv, self).__init__()
        self.down_128 = ConvDown(64, 128, 4, 2, padding=1, layers=2)
        self.down_64 = ConvDown(128, 256, 4, 2, padding=1)
        self.down_32 = ConvDown(256, 256, 1, 1)
        self.down_16 = ConvDown(512, 512, 4, 2, padding=1, activ=False)
        self.down_8 = ConvDown(512, 512, 4, 2, padding=1, layers=2, activ=False)
        self.down_4 = ConvDown(512, 512, 4, 2, padding=1, layers=3, activ=False)
        self.down = ConvDown(768, 256, 1, 1)
        self.fuse = ConvDown(512, 512, 1, 1)
        self.up = ConvUp(512, 256, 1, 1)
        self.up_128 = ConvUp(512, 64, 1, 1)
        self.up_64 = ConvUp(512, 128, 1, 1)
        self.up_32 = ConvUp(512, 256, 1, 1)
        self.base = BASE(512)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(256, 256, innorm=True)]
            seuqence_5 += [PCBActiv(256, 256, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(256, 256, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.detail_structure_fusion = SDFF(256, 256)

    def forward(self, input, mask):
        mask = util.cal_feat_mask(mask, 3, 1)
        # input[2]:256 32 32
        b, c, h, w = input[2].size()
        mask_1 = torch.add(torch.neg(mask.float()), 1)
        mask_1 = mask_1.expand(b, c, h, w)

        x_1 = self.activation(input[0])
        x_2 = self.activation(input[1])
        x_3 = self.activation(input[2])
        x_4 = self.activation(input[3])
        x_5 = self.activation(input[4])
        x_6 = self.activation(input[5])
        # Change the shape of each layer and intergrate low-level/high-level features
        x_1 = self.down_128(x_1)
        x_2 = self.down_64(x_2)
        x_3 = self.down_32(x_3)
        x_4 = self.up(x_4, (32, 32))
        x_5 = self.up(x_5, (32, 32))
        x_6 = self.up(x_6, (32, 32))

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_DE = torch.cat([x_1, x_2, x_3], 1)
        x_ST = torch.cat([x_4, x_5, x_6], 1)

        x_ST = self.down(x_ST)
        x_DE = self.down(x_DE)
        x_ST = [x_ST, mask_1]
        x_DE = [x_DE, mask_1]

        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_DE)
        x_DE_5 = self.cov_5(x_DE)
        x_DE_7 = self.cov_7(x_DE)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        x_DE_fi = self.down(x_DE_fuse)

        # Multi Scale PConv fill the Structure
        x_ST_3 = self.cov_3(x_ST)
        x_ST_5 = self.cov_5(x_ST)
        x_ST_7 = self.cov_7(x_ST)
        x_ST_fuse = torch.cat([x_ST_3[0], x_ST_5[0], x_ST_7[0]], 1)
        x_ST_fi = self.down(x_ST_fuse)

        # SDFF module
        x_cat = self.detail_structure_fusion(x_ST_fi, x_DE_fi)
        x_cat_fuse = self.fuse(x_cat)

        # Feature equalizations
        x_final = self.base(x_cat_fuse)

        # Add back to the input
        x_ST = x_final
        x_DE = x_final
        x_1 = self.up_128(x_DE, (128, 128)) + input[0]
        x_2 = self.up_64(x_DE, (64, 64)) + input[1]
        x_3 = self.up_32(x_DE, (32, 32)) + input[2]
        x_4 = self.down_16(x_ST) + input[3]
        x_5 = self.down_8(x_ST) + input[4]
        x_6 = self.down_4(x_ST) + input[5]

        out = [x_1, x_2, x_3, x_4, x_5, x_6]
        loss = [x_ST_fi, x_DE_fi]
        out_final = [out, loss]
        return out_final
