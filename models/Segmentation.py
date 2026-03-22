import torch
from torch import nn

from models.Layer import ConvLayer


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        x = self.conv3(x)

        return self.pool(x + h)


class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        return x + h


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
    def __init__(self, in_channels: int, out_channels: int, img_width: int, img_height: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = img_width
        self.height = img_height

        self.conv1 = ConvLayer(self.in_channels, 16, kernel_size=3, padding=1)  # size // 2
        self.down1 = DownBlock(16, 32)  # size // 4
        self.down2 = DownBlock(32, 64)  # size // 8
        self.down3 = DownBlock(64, 128)  # size // 16
        self.down4 = DownBlock(128, 256)  # # size // 32

        self.middel = DownBlock(256, 512)
        self.middel_att = AttentionBlock(512, self.width // 32, self.height // 32)

        self.up1 = UpBlock(512, 256, 256)
        self.up1_att = AttentionBlock(256, self.width // 16, self.height // 16)

        self.up2 = UpBlock(256, 128, 128)
        self.up2_att = AttentionBlock(128, self.width // 8, self.height // 8)

        self.up3 = UpBlock(128, 64, 64)
        self.up3_att = AttentionBlock(64, self.width // 4, self.height // 4)

        self.up4 = UpBlock(64, 32, 32)
        self.up5 = UpBlock(32, 16, 16)

        self.final_conv = ConvLayer(16, self.out_channels, kernel_size=3, padding=1,
                                    activation=nn.Identity())

    def forward(self, x: torch.Tensor):
        # first convolution
        x = self.conv1(x)

        # encoder
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)

        u = self.middel(h4)
        u = self.middel_att(u)

        # decoder
        u = self.up1(u, h4)
        u = self.up1_att(u)

        u = self.up2(u, h3)
        u = self.up2_att(u)

        u = self.up3(u, h2)
        u = self.up4(u, h1)
        u = self.up5(u, x)

        return self.final_conv(u)


if __name__ == '__main__':
    model = SegmentationUNet(4, 4, 128, 128)

    image = torch.randn(4, 4, 128, 128)
    pred = model(image)
    print(pred.shape)
