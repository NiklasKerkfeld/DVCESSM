import torch
from torch import nn
import torch.nn.functional as F

from Parameter import IN_CHANNELS, N_CLASSES, IN_WIDTH, IN_HEIGHT


class ClassifierLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifierLayer, self).__init__()
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


class Classifier(nn.Module):
    def __init__(self, num_classes: int = N_CLASSES):
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
