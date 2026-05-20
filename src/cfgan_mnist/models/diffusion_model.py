"""Diffusion model class."""

import math
from typing import Optional, List

import torch
from torch import nn

from cfgan_mnist.models.layer import AttentionBlock


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                              stride=stride)

        num_groups = 8 if out_channels >= 8 else 1
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
        self.pool = DiffusionConvLayer(in_channels, in_channels, kernel_size=2, padding=0, stride=2)

        self.conv1 = DiffusionConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = DiffusionConvLayer(in_channels // 4, in_channels // 4, kernel_size=3,
                                        padding=1)
        self.conv3 = DiffusionConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)

        self.conv4: nn.Module = nn.Identity()
        if in_channels != out_channels:
            self.conv4 = DiffusionConvLayer(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, in_channels // 4)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        time_proj = self.time_mlp(t_emb)[:, :, None, None]

        x = self.pool(x)

        h = self.conv1(x)
        h = h + time_proj

        h = self.conv2(h)
        h = self.conv3(h)

        x = self.conv4(x)

        return x + h


class MiddleBlock(nn.Module):
    """
    Middle block between Encoder and Decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        """
        Middle block between Encoder and Decoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_emb_dim: number of dimensions of time embedding.
        """
        super().__init__()
        self.conv1 = DiffusionConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = DiffusionConvLayer(in_channels // 4, in_channels // 4, kernel_size=3,
                                        padding=1)
        self.conv3 = DiffusionConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, in_channels // 4)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        time_proj = self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv1(x)
        h = h + time_proj
        h = self.conv2(h)
        h = self.conv3(h)

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


class DiffusionModel(nn.Module):
    """
    Diffusion model.
    """
    emb: torch.Tensor

    def __init__(self,
                 channels: int,
                 width: int,
                 height: int,
                 time_dim: int,
                 nums_features: Optional[List[int]]=None) -> None:
        """

        Args:
            channels: number of input channels.
            width: width of the input image.
            height: height of the input image.
            time_dim: size of time embedding
            nums_features: feature maps of each Block (default [32, 64, 128, 256, 512, 512]).
        """
        super().__init__()
        if nums_features is None:
            nums_features = [64, 128, 256, 512, 1024]
        self.channels = channels
        self.time_dim = time_dim
        self.width = width
        self.height = height

        self.nums_features = nums_features

        half_dim = self.time_dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU()
        )

        self.conv1 = DiffusionConvLayer(channels, self.nums_features[0], kernel_size=3,
                                        padding=1)  # size // 2
        self.down1 = DownBlock(self.nums_features[0], self.nums_features[1],
                               self.time_dim)  # size // 4
        self.down2 = DownBlock(self.nums_features[1], self.nums_features[2],
                               self.time_dim)  # size // 8
        self.down3 = DownBlock(self.nums_features[2], self.nums_features[3],
                               self.time_dim)  # size // 16
        self.down4 = DownBlock(self.nums_features[3], self.nums_features[4],
                               self.time_dim)  # # size // 32

        self.middel = MiddleBlock(self.nums_features[4], self.nums_features[4], self.time_dim)
        self.middel_att = AttentionBlock(self.nums_features[4], self.width // 16, self.height // 16)

        self.up1 = UpBlock(self.nums_features[4], self.nums_features[4], self.nums_features[3],
                           self.time_dim)
        self.up1_att = AttentionBlock(self.nums_features[3], self.width // 8, self.height // 8)

        self.up2 = UpBlock(self.nums_features[3], self.nums_features[3], self.nums_features[2],
                           self.time_dim)
        self.up2_att = AttentionBlock(self.nums_features[2], self.width // 4, self.height // 4)

        self.up3 = UpBlock(self.nums_features[2], self.nums_features[2], self.nums_features[1],
                           self.time_dim)
        self.up3_att = AttentionBlock(self.nums_features[1], self.width // 2, self.height // 2)

        self.up4 = UpBlock(self.nums_features[1], self.nums_features[1], self.nums_features[0],
                           self.time_dim)
        self.up5 = UpBlock(self.nums_features[0], self.nums_features[0], self.nums_features[0],
                           self.time_dim, scale_factor=1.0)

        self.final_conv = DiffusionConvLayer(self.nums_features[0], channels, kernel_size=3,
                                             padding=1,
                                             activation=nn.Identity(), apply_norm=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Return timestep embeddings.
        Args:
            timesteps: tensor of timesteps.

        Returns:
            tensor of timestep embeddings.
        """
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # first convolution
        h1 = self.conv1(x)  # (64, h, w)

        # encoder
        h2 = self.down1(h1, t_emb)  # (128, h//2, w//2)
        h3 = self.down2(h2, t_emb)  # (256, h//4, w//4)
        h4 = self.down3(h3, t_emb)  # (512, h//8, w//8)
        h5 = self.down4(h4, t_emb)  # (1024, h//16, w//16)

        u = self.middel(h5, t_emb)  # (1024, h//16, w//16)
        u = self.middel_att(u)

        # decoder
        u = self.up1(u, h5, t_emb)  # (512, h//8, w//8)
        u = self.up1_att(u)

        u = self.up2(u, h4, t_emb)  # (256, h//4, w//4)
        u = self.up2_att(u)

        u = self.up3(u, h3, t_emb)  # (128, h//2, w//2)
        u = self.up4(u, h2, t_emb)  # (64, h, w)
        u = self.up5(u, h1, t_emb)  # (64, h, w)

        return self.final_conv(u)
