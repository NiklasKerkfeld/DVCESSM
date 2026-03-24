from typing import List

import torch
from torch import nn

from models.Layer import ConvLayer, AttentionBlock, ResidualBlock


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_size, out_channels):
        super().__init__()
        self.up_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels + skip_size, out_channels, kernel_size=3, padding=1)

    def forward(self, u, c):
        u = self.up_transpose(u)
        d = torch.cat([u, c], dim=1)
        u = self.conv(d)

        return u


class SegmentationUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 img_width: int,
                 img_height: int,
                 feature_maps: List[int] = [32, 64, 128, 256, 512, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = img_width
        self.height = img_height

        self.conv1 = ConvLayer(self.in_channels, feature_maps[0], kernel_size=3, padding=1)  # size
        self.down1 = ResidualBlock(feature_maps[0], feature_maps[1])    # size // 2
        self.down2 = ResidualBlock(feature_maps[1], feature_maps[2])    # size // 4
        self.down3 = ResidualBlock(feature_maps[2], feature_maps[3])    # size // 8
        self.down4 = ResidualBlock(feature_maps[3], feature_maps[4])    # size // 16
        self.down5 = ResidualBlock(feature_maps[4], feature_maps[5])    # size // 32

        self.h3_att = AttentionBlock(feature_maps[3], self.width // 8, self.height // 8)
        self.h4_att = AttentionBlock(feature_maps[4], self.width // 16, self.height // 16)
        self.u_att = AttentionBlock(feature_maps[5], self.width // 32, self.height // 32)

        self.up1 = UpBlock(feature_maps[5], feature_maps[4], feature_maps[4])
        self.up2 = UpBlock(feature_maps[4], feature_maps[3], feature_maps[3])
        self.up3 = UpBlock(feature_maps[3], feature_maps[2], feature_maps[2])
        self.up4 = UpBlock(feature_maps[2], feature_maps[1], feature_maps[1])
        self.up5 = UpBlock(feature_maps[1], feature_maps[0], feature_maps[0])

        self.final_conv = ConvLayer(feature_maps[0], self.out_channels, kernel_size=1, padding=1,
                                    activation=nn.Identity())

    def forward(self, x: torch.Tensor):
        # encoder
        x = self.conv1(x)
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        u = self.down5(h4)

        # decoder
        u = self.u_att(u)
        h4 = self.h4_att(h4)
        u = self.up1(u, h4)

        h3 = self.h3_att(h3)
        u = self.up2(u, h3)

        u = self.up3(u, h2)
        u = self.up4(u, h1)
        u = self.up5(u, x)

        return self.final_conv(u)


if __name__ == '__main__':
    model = SegmentationUNet(4, 4, 128, 128)

    image = torch.randn(4, 4, 128, 128)
    pred = model(image)
    print(pred.shape)
