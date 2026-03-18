from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool: bool = True):
        super(ResidualBlock, self).__init__()
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

    def forward(self, x):
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
    def __init__(self, in_channels, skip_size, out_channels, scale_factor: float = 2):
        super(UpBlock, self).__init__()
        self.scale_factor = scale_factor

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_size, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, u, c):
        d = torch.cat([u, c], dim=1)
        u = self.conv(d)

        if self.scale_factor != 1.0:
            u = self.upsample(u)

        return u


class Generator(nn.Module):
    def __init__(self, channels: int, num_classes: int = 10, feature_maps: Sequence[int] = [64, 128, 256, 512]):
        super(Generator, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.feature_maps = feature_maps

        self.down1 = ResidualBlock(self.channels + self.num_classes, self.feature_maps[0])
        self.down2 = ResidualBlock(self.feature_maps[0], self.feature_maps[1])
        self.down3 = ResidualBlock(self.feature_maps[1], self.feature_maps[2])
        self.down4 = ResidualBlock(self.feature_maps[2], self.feature_maps[3])

        self.middel = ResidualBlock(self.feature_maps[3], self.feature_maps[3], pool=False)

        self.up1 = UpBlock(self.feature_maps[3], self.feature_maps[3], self.feature_maps[3])
        self.up2 = UpBlock(self.feature_maps[3], self.feature_maps[2], self.feature_maps[2])
        self.up3 = UpBlock(self.feature_maps[2], self.feature_maps[1], self.feature_maps[1])
        self.up4 = UpBlock(self.feature_maps[1], self.feature_maps[0], self.feature_maps[0])

        self.final = nn.Conv2d(self.feature_maps[0] + self.channels + self.num_classes, self.channels, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = F.one_hot(y, num_classes=self.num_classes).permute(0, 3, 1, 2)

        x = torch.cat((image, y), dim=1)

        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        h5 = self.middel(h4)

        u = self.up1(h5, h4)
        u = self.up2(u, h3)
        u = self.up3(u, h2)
        u = self.up4(u, h1)

        x = torch.cat([u, x], dim=1)

        return self.final(x)




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(4, num_classes=10).to(device)

    image = torch.randn(2, 4, 128, 128).to(device)
    target = torch.randint(1, 5, (2, 128, 128)).to(device)

    pred = model(image, target)
    print(pred.shape)
