"""Segmentation model."""
from typing import Optional, List

import torch
from torch import nn

from .layer import AttentionBlock


class ConvLayer(nn.Module):
    """
    Convolution Layer with Batch Normalization and activation.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 activation: nn.Module = nn.ReLU()):
        """
        Convolution Layer with Batch Normalization and activation.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: kernel size.
            padding: padding size (default: 3).
            stride: stride size (default: 1).
            activation: activation function (default: ReLU).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                              stride=stride)

        self.norm = nn.BatchNorm2d(out_channels)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual Block Layer with 3 Convolution Layer and Max Pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        """
        Residual Block Layer with 3 Convolution Layer and Max Pooling.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            pool: whether to use pooling or not.
        """
        super().__init__()
        self.pool = pool

        self.layer1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        h = self.layer1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.layer2(h)
        h = self.bn2(h)
        h = self.relu2(h)

        x = h + self.layer3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        if self.pool:
            x = self.max_pool(x)

        return x


class UpBlock(nn.Module):
    """
    Up Convolution block for Decoder.
    """

    def __init__(self, in_channels: int, skip_size: int, out_channels: int):
        """

        Args:
            in_channels: number of input channels.
            skip_size: number of channels of skip connection.
            out_channels: number of output channels.
        """
        super().__init__()
        self.up_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels + skip_size, out_channels, kernel_size=3, padding=1)

    def forward(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        u = self.up_transpose(u)
        d = torch.cat([u, c], dim=1)
        u = self.conv(d)

        return u


class SegmentationUNet(nn.Module):
    """
    U-Net model for semantic segmentation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 img_width: int,
                 img_height: int,
                 feature_maps: Optional[List[int]] = None,
                 attention: bool = False) -> None:
        """

        Args:
            in_channels: number of input channels.
            out_channels: number of segmentation classes.
            img_width: width of the input image.
            img_height: height of the input image.
            feature_maps: feature maps of each Block (default [32, 64, 128, 256, 512, 512]).
            attention: use attention mechanism.
        """

        super().__init__()
        if feature_maps is None:
            feature_maps = [32, 64, 128, 256, 512, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = img_width
        self.height = img_height

        self.conv1 = ConvLayer(self.in_channels, feature_maps[0], kernel_size=3, padding=1)  # size
        self.down1 = ResidualBlock(feature_maps[0], feature_maps[1])  # size // 2
        self.down2 = ResidualBlock(feature_maps[1], feature_maps[2])  # size // 4
        self.down3 = ResidualBlock(feature_maps[2], feature_maps[3])  # size // 8
        self.down4 = ResidualBlock(feature_maps[3], feature_maps[4])  # size // 16
        self.down5 = ResidualBlock(feature_maps[4], feature_maps[5])  # size // 32

        if attention:
            self.h3_att: nn.Module = AttentionBlock(feature_maps[3],
                                                    self.width // 8,
                                                    self.height // 8)
            self.h4_att: nn.Module = AttentionBlock(feature_maps[4],
                                                    self.width // 16,
                                                    self.height // 16)
            self.u_att: nn.Module = AttentionBlock(feature_maps[5],
                                                   self.width // 32,
                                                   self.height // 32)
        else:
            self.h3_att = nn.Identity()
            self.h4_att = nn.Identity()
            self.u_att = nn.Identity()

        self.up1 = UpBlock(feature_maps[5], feature_maps[4], feature_maps[4])
        self.up2 = UpBlock(feature_maps[4], feature_maps[3], feature_maps[3])
        self.up3 = UpBlock(feature_maps[3], feature_maps[2], feature_maps[2])
        self.up4 = UpBlock(feature_maps[2], feature_maps[1], feature_maps[1])
        self.up5 = UpBlock(feature_maps[1], feature_maps[0], feature_maps[0])

        # self.final_conv = ConvLayer(feature_maps[0], self.out_channels, kernel_size=1, padding=0,
        #               activation=nn.Identity())
        self.final_conv = nn.Sequential(
            ConvLayer(feature_maps[0], feature_maps[0], kernel_size=3, padding=1),
            ConvLayer(feature_maps[0], self.out_channels, kernel_size=1, padding=0,
                      activation=nn.Identity())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
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
