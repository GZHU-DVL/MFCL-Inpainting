import torch
import torch.nn as nn


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


class SDFF(nn.Module):
    # Soft-gating Dual Feature Fusion.

    def __init__(self, in_channels, out_channels):
        super(SDFF, self).__init__()

        self.structure_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SELayer(out_channels),
            nn.Sigmoid()
        )
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SELayer(out_channels),
            nn.Sigmoid()
        )

        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.detail_gamma = nn.Parameter(torch.zeros(1))

        self.detail_beta = nn.Parameter(torch.zeros(1))

    def forward(self, structure_feature, detail_feature):
        sd_cat = torch.cat((structure_feature, detail_feature), dim=1)

        map_detail = self.structure_branch(sd_cat)
        map_structure = self.detail_branch(sd_cat)

        detail_feature_branch = detail_feature + self.detail_beta * (structure_feature * (self.detail_gamma * (map_detail * detail_feature)))
        structure_feature_branch = structure_feature + self.structure_gamma * (map_structure * detail_feature)

        return torch.cat((structure_feature_branch, detail_feature_branch), dim=1)




