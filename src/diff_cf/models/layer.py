"""Layers used for models."""

import torch
from torch import nn


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


class AttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention Block.
    """
    def __init__(self, channels: int, width: int, height: int, num_heads: int = 8):
        """
        Multi-Head Self-Attention Block.

        Args:
            channels: number of channels.
            width: width of the image.
            height: height of the image.
            num_heads: number of heads (default: 8).
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        b, c, h, w = x.shape

        h_norm = self.norm(x)
        h_flat = h_norm.view(b, c, h * w).transpose(1, 2)
        h_flat = h_flat + self.pos_embedding

        attn_output, _ = self.mha(h_flat, h_flat, h_flat)

        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)

        return x + attn_output


class DiffusionConvLayer(nn.Module):
    """
    Convolutional layer for diffusion model.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 activation: nn.Module = nn.SiLU(),
                 apply_norm: bool = True):
        """
        Convolutional layer for diffusion model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of Convolutional kernel (default: 3).
            padding: Padding (default: 1).
            stride: Stride (default: 1).
            activation: Activation function (default: SiLU).
            apply_norm: use GroupNorm (default: True).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)

        num_groups = 32 if out_channels >= 32 else 1
        self.norm = nn.GroupNorm(num_groups, out_channels) if apply_norm else nn.Identity()

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class DownBlock(nn.Module):
    """
    Decoder Block for Diffusion model.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        """
        Decoder Block for Diffusion model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_emb_dim: number of dimensions of time embedding.
        """
        super().__init__()
        self.conv1 = DiffusionConvLayer(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DiffusionConvLayer(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = DiffusionConvLayer(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip: nn.Module = nn.Identity()
        if in_channels != out_channels:
            self.skip = DiffusionConvLayer(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)


    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv2(h)
        h = self.conv3(h)

        x = self.skip(x)

        return x + h


class UpBlock(nn.Module):
    """
    Decoder block of Diffusion model.
    """
    def __init__(self,
                 in_channels: int,
                 skip_size: int,
                 out_channels: int,
                 time_emb_dim: int,
                 scale_factor: float = 2):
        """
        Decoder block of Diffusion model.

        Args:
            in_channels: number of input channels
            skip_size: number of channels of skip connection input.
            out_channels: number of output channels.
            time_emb_dim: number of dimensions of time embedding.
            scale_factor: factor to upscale feature map (default: 2).
        """
        super().__init__()
        self.scale_factor = scale_factor

        self.conv = nn.Sequential(
            DiffusionConvLayer(in_channels + skip_size, out_channels, kernel_size=3, padding=1),
            DiffusionConvLayer(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)


    def forward(self, u: torch.Tensor, c: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        t_proj = self.time_mlp(t_emb)[:, :, None, None]

        d = torch.cat([u, c], dim=1)
        u = self.conv(d)
        u += t_proj

        if self.scale_factor != 1.0:
            u = self.upsample(u)

        return u
