"""Layers used for models."""
import math
from typing import Optional

import torch
from torch import nn


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

        if self.channels % self.num_heads != 0:
            raise ValueError('channels should be dividable by num_heads')

        self.norm = nn.GroupNorm(num_groups=math.gcd(32, self.channels),
                                 num_channels=channels)

        self.pos_embedding = nn.Parameter(torch.empty(1, width * height, channels))

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

        # init parameter
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.zeros_(self.mha.out_proj.weight)
        if self.mha.out_proj.bias is not None:
            nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        b, c, h, w = x.shape

        h_norm = self.norm(x)
        h_flat = h_norm.view(b, c, h * w).transpose(1, 2)
        h_flat = h_flat + self.pos_embedding

        attn_output, _ = self.mha(h_flat, h_flat, h_flat, need_weights=False)

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
                 activation: Optional[nn.Module] = None,
                 apply_norm: bool = True,
                 zero_initialization: bool = False):
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
            zero_initialization: initializes layer weights with zeros.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)



        self.norm = nn.GroupNorm(math.gcd(32, out_channels),
                                 out_channels) if apply_norm else nn.Identity()

        self.activation = activation if activation is not None else nn.SiLU()

        if zero_initialization:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class TimeConditionedResBlock(nn.Module):
    """
    Residual Block with time condition.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 dropout: float = 0.05):
        """
        Residual Block with time condition.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_emb_dim: number of dimensions of time embedding.
            dropout: dropout rate (default: 0.05).
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(
            math.gcd(32, in_channels), in_channels
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.norm2 = nn.GroupNorm(
            math.gcd(32, out_channels), out_channels
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_channels)
        )

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """Model forward."""
        h = self.conv1(self.activation(self.norm1(x)))

        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h

class UpBlock(nn.Module):
    """
    Decoder block of Diffusion model.
    """
    def __init__(self,
                 in_channels: int,
                 skip_size: int,
                 out_channels: int,
                 time_emb_dim: int,
                 scale_factor: float = 2,
                 dropout: float = 0.05):
        """
        Decoder block of Diffusion model.

        Args:
            in_channels: number of input channels
            skip_size: number of channels of skip connection input.
            out_channels: number of output channels.
            time_emb_dim: number of dimensions of time embedding.
            scale_factor: factor to upscale feature map (default: 2).
            dropout: dropout rate (default: 0.05).
        """
        super().__init__()
        self.scale_factor = scale_factor

        self.block = TimeConditionedResBlock(
            in_channels + skip_size,
            out_channels,
            time_emb_dim,
            dropout=dropout,
        )

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)


    def forward(self, u: torch.Tensor, c: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        u = self.block(torch.cat([u, c], dim=1), t_emb)

        if self.scale_factor != 1.0:
            u = self.upsample(u)

        return u
