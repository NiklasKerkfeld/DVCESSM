import os

from dotenv import load_dotenv

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

load_dotenv()

N_CLASSES = int(os.getenv('N_CLASSES'))
IN_CHANNELS = int(os.getenv('IN_CHANNELS'))
IN_HEIGHT = int(os.getenv('IN_HEIGHT'))
IN_WIDTH = int(os.getenv('IN_WIDTH'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
ITERATIONS = int(os.getenv('ITERATIONS'))
ALPHA = float(os.getenv('ALPHA'))
BETA = float(os.getenv('BETA'))


class DiscriminatorLoss(nn.Module):
    def __init__(self, label_smoothing: bool = False):
        super(DiscriminatorLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        if self.label_smoothing:
            target = 0.9 * target + 0.1 * (1 - target)
        return self.loss_fn(prediction, target.float())


class ChangeRegularizationLoss(nn.Module):
    def __init__(self, eps=1e-2):
        super(ChangeRegularizationLoss, self).__init__()
        self.eps = eps

    def _func(self, x: torch.Tensor):
        eps = torch.FloatTensor(self.eps).to(x.device)
        return torch.log(torch.abs(x) + eps) - torch.log(eps)
        # return torch.abs(x)

    def forward(self, changes):
        regularization = torch.mean(self._func(changes))

        return regularization


class ClassifierLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifierLayer, self).__init__()
        self.layer1 = spectral_norm(nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = spectral_norm(nn.Conv2d(out_channels // 4, out_channels, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer3 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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

        return self.pool(x)

class GeneratorLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeneratorLayer, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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

        return self.pool(x)


class BottleneckWithPrior(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.prior_mlp = nn.Linear(10, input_channels)
        self.bottleneck = GeneratorLayer(2 * input_channels, out_channels)

    def forward(self, x, prior):
        prior_mapped = self.prior_mlp(prior)  # (Batch, input_channels)
        prior_spatial = prior_mapped.view(x.size(0), x.size(1), 1, 1)
        prior_spatial = prior_spatial.expand(-1, -1, x.size(2), x.size(3))

        combined = torch.cat([x, prior_spatial], dim=1)

        return self.bottleneck(combined)


class Generator(nn.Module):
    def __init__(self, channels):
        super(Generator, self).__init__()
        # Encoder
        self.down_conv1 = GeneratorLayer(channels, 64)
        self.down_conv2 = GeneratorLayer(64, 128)
        self.down_conv3 = GeneratorLayer(128, 256)

        # Bottleneck
        self.bottleneck = BottleneckWithPrior(256, 512)

        # Decoder
        self.up_transpose1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up_conv1 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)

        self.up_transpose2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up_conv2 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)

        self.up_transpose3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up_conv3 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)

        self.up_transpose4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.out_conv = nn.Conv2d(32 + channels, channels, kernel_size=1)

    def forward(self, x, targets):
        # Encoder
        c1 = self.down_conv1(x)
        c2 = self.down_conv2(c1)
        c3 = self.down_conv3(c2)

        # Bottleneck
        b = self.bottleneck(c3, targets)

        # Decoder Stage 1
        u1 = self.up_transpose1(b)  # (512, H/4, W/4)
        d1 = torch.cat([u1, c3], dim=1)  # (512 + 256)
        u1 = self.up_conv1(d1)  # (256, H/4, W/4)

        # Decoder Stage 2
        u2 = self.up_transpose2(u1)  # (256, H/2, W/2)
        d2 = torch.cat([u2, c2], dim=1)  # (256 + 128)
        u2 = self.up_conv2(d2)  # (128, H/2, W/2)

        # Decoder Stage 3
        u3 = self.up_transpose3(u2)  # (128, H, W)
        d3 = torch.cat([u3, c1], dim=1)  # (128 + 64)
        u3 = self.up_conv3(d3)  # (64, H, W)

        # Decoder Stage 4
        u4 = self.up_transpose4(u3)
        d4 = torch.cat([u4, x], dim=1)
        out = self.out_conv(d4)

        return F.tanh(out)


class Classifier(nn.Module):
    def __init__(self, num_classes:int=N_CLASSES):
        super(Classifier, self).__init__()
        # Convolutional Layers
        self.conv1 = ClassifierLayer(IN_CHANNELS, 32)
        self.conv2 = ClassifierLayer(32, 64)
        self.conv3 = ClassifierLayer(64, 128)

        # Classification head
        self.fc1 = nn.Linear(128 * (IN_WIDTH // 8) * (IN_HEIGHT // 8), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == '__main__':
    generator = Generator(channels=1)

    input = torch.randn(4, 1, 28, 28)
    prior = torch.randn(4, 1)

    change = generator(input, prior)
    print(change.shape)
