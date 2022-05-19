import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_patches(x, kernel_size=3, stride=1):
    if kernel_size != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()


class RAL(nn.Module):
    '''Region affinity learning.'''

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super(RAL, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale

    def forward(self, background, foreground):

        # accelerated calculation
        foreground = F.interpolate(foreground, scale_factor=1. / self.rate, mode='bilinear', align_corners=True)

        foreground_size, background_size = list(foreground.size()), list(background.size())

        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(background, kernel_size=background_kernel_size,
                                             stride=self.stride * self.rate)
        background_patches = background_patches.view(background_size[0], -1,
                                                     background_size[1], background_kernel_size, background_kernel_size)
        background_patches_list = torch.split(background_patches, 1, dim=0)

        foreground_list = torch.split(foreground, 1, dim=0)
        foreground_patches = extract_patches(foreground, kernel_size=self.kernel_size, stride=self.stride)
        foreground_patches = foreground_patches.view(foreground_size[0], -1,
                                                     foreground_size[1], self.kernel_size, self.kernel_size)
        foreground_patches_list = torch.split(foreground_patches, 1, dim=0)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1
        escape_NaN = torch.FloatTensor([1e-4])
        if torch.cuda.is_available():
            escape_NaN = escape_NaN.cuda()

        for foreground_item, foreground_patches_item, background_patches_item in zip(
                foreground_list, foreground_patches_list, background_patches_list
        ):
            foreground_patches_item = foreground_patches_item[0]
            foreground_patches_item_normed = foreground_patches_item / torch.max(
                torch.sqrt((foreground_patches_item * foreground_patches_item).sum([1, 2, 3], keepdim=True)),
                escape_NaN)

            score_map = F.conv2d(foreground_item, foreground_patches_item_normed, stride=1, padding=padding)
            score_map = score_map.view(1, foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                                       foreground_size[2], foreground_size[3])
            attention_map = F.softmax(score_map * self.softmax_scale, dim=1)
            attention_map = attention_map.clamp(min=1e-8)

            background_patches_item = background_patches_item[0]
            output_item = F.conv_transpose2d(attention_map, background_patches_item, stride=self.rate, padding=1) / 4.
            output_list.append(output_item)

        output = torch.cat(output_list, dim=0)
        output = output.view(background_size)
        return output