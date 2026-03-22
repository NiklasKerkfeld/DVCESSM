from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 padding=1,
                 stride: int = 1,
                 activation=nn.ReLU()):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                              stride=stride)

        self.norm = nn.BatchNorm2d(out_channels)

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x



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


class LinearLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU()):
        super(LinearLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation


    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, width: int, height: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, width * height, channels))

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        b, c, h, w = x.shape

        h_norm = self.norm(x)
        h_flat = h_norm.view(b, c, h * w).transpose(1, 2)
        h_flat = h_flat + self.pos_embedding

        attn_output, _ = self.mha(h_flat, h_flat, h_flat)

        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)

        return x + attn_output