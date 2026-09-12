import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F

from .layer import AttentionBlock, DiffusionConvLayer, DownBlock, UpBlock


class DiffusionModel(nn.Module):
    """
    Unified Diffusion model supporting both unconditional generation 
    and target-prior segmentation mask conditioning.
    """
    emb: torch.Tensor

    def __init__(self,
                 channels: int,
                 width: int,
                 height: int,
                 time_dim: int,
                 nums_features: Optional[List[int]] = None,
                 use_prior: bool = False,
                 num_classes: int = 2) -> None:
        """
        Args:
            channels: number of input channels.
            width: width of the input image.
            height: height of the input image.
            time_dim: size of time embedding.
            nums_features: feature maps of each Residual Block (default [64, 128, 256, 512, 1024]).
            use_prior: If True, enables segmentation mask conditioning.
            num_classes: Number of classes for the segmentation mask (ignored if use_prior=False).
        """
        super().__init__()
        if nums_features is None:
            nums_features = [64, 128, 256, 512, 1024]
        self.channels = channels
        self.time_dim = time_dim
        self.width = width
        self.height = height
        self.use_prior = use_prior
        self.num_prior_channels = num_classes if use_prior else 0
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

        # --- FIRST CONVOLUTION ---
        self.conv1 = DiffusionConvLayer(channels + self.num_prior_channels, self.nums_features[0],
                                        kernel_size=3, padding=1)

        # --- STAGE 1: Downsample & Expand (64 -> 128) ---
        self.down1 = DiffusionConvLayer(self.nums_features[0], self.nums_features[0], kernel_size=3,
                                        padding=1, stride=2)
        self.block1a = DownBlock(self.nums_features[0] + self.num_prior_channels,
                                 self.nums_features[1], self.time_dim)

        self.block1b = DownBlock(self.nums_features[1], self.nums_features[1], self.time_dim)

        # --- STAGE 2: Downsample & Expand (128 -> 256) ---
        self.down2 = DiffusionConvLayer(self.nums_features[1], self.nums_features[1], kernel_size=3,
                                        padding=1, stride=2)
        self.block2a = DownBlock(self.nums_features[1] + self.num_prior_channels,
                                 self.nums_features[2], self.time_dim)
        self.block2b = DownBlock(self.nums_features[2], self.nums_features[2], self.time_dim)

        # --- STAGE 3: Downsample & Expand (256 -> 512) ---
        self.down3 = DiffusionConvLayer(self.nums_features[2], self.nums_features[2], kernel_size=3,
                                        padding=1, stride=2)
        self.block3a = DownBlock(self.nums_features[2] + self.num_prior_channels,
                                 self.nums_features[3], self.time_dim)
        self.block3b = DownBlock(self.nums_features[3], self.nums_features[3], self.time_dim)

        # --- STAGE 4: Downsample & Expand (512 -> 1024) ---
        self.down4 = DiffusionConvLayer(self.nums_features[3], self.nums_features[3], kernel_size=3,
                                        padding=1, stride=2)
        self.block4a = DownBlock(self.nums_features[3] + self.num_prior_channels,
                                 self.nums_features[4], self.time_dim)
        self.block4b = DownBlock(self.nums_features[4], self.nums_features[4], self.time_dim)

        # --- MIDDLE STAGE ---
        self.middel = DownBlock(self.nums_features[4], self.nums_features[4], self.time_dim)
        self.middel_att = AttentionBlock(self.nums_features[4], self.width // 16, self.height // 16)

        # --- DECODER STAGES ---
        self.up1 = UpBlock(self.nums_features[4] + self.num_prior_channels, self.nums_features[4],
                           self.nums_features[3], self.time_dim)
        self.up1_att = AttentionBlock(self.nums_features[3], self.width // 8, self.height // 8)

        self.up2 = UpBlock(self.nums_features[3] + self.num_prior_channels, self.nums_features[3],
                           self.nums_features[2], self.time_dim)
        self.up2_att = AttentionBlock(self.nums_features[2], self.width // 4, self.height // 4)

        self.up3 = UpBlock(self.nums_features[2] + self.num_prior_channels, self.nums_features[2],
                           self.nums_features[1], self.time_dim)
        self.up4 = UpBlock(self.nums_features[1] + self.num_prior_channels, self.nums_features[1],
                           self.nums_features[0], self.time_dim)
        self.up5 = UpBlock(self.nums_features[0] + self.num_prior_channels, self.nums_features[0],
                           self.nums_features[0], self.time_dim, scale_factor=1.0)

        self.final_conv = DiffusionConvLayer(self.nums_features[0], channels, kernel_size=3,
                                             padding=1, activation=nn.Identity(), apply_norm=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function."""
        if self.use_prior and target is None:
            raise ValueError("Target mask must be provided when use_prior=True")

        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Optional: Prepare target prior mask
        if self.use_prior:
            t_h1 = F.one_hot(target, num_classes=self.num_prior_channels).permute(0, 3, 1,
                                                                                  2).float()
            x = torch.concat((t_h1, x), dim=1)

        # First convolution
        h1 = self.conv1(x)  # (64, h, w)

        # --- ENCODER ---
        h2 = self.down1(h1)
        if self.use_prior:
            t_h2 = torch.max_pool2d(t_h1, kernel_size=2, stride=2)
            h2 = torch.concat((t_h2, h2), dim=1)
        h2 = self.block1a(h2, t_emb)
        h2 = self.block1b(h2, t_emb)

        h3 = self.down2(h2)
        if self.use_prior:
            t_h3 = torch.max_pool2d(t_h2, kernel_size=2, stride=2)
            h3 = torch.concat((t_h3, h3), dim=1)
        h3 = self.block2a(h3, t_emb)
        h3 = self.block2b(h3, t_emb)

        h4 = self.down3(h3)
        if self.use_prior:
            t_h4 = torch.max_pool2d(t_h3, kernel_size=2, stride=2)
            h4 = torch.concat((t_h4, h4), dim=1)
        h4 = self.block3a(h4, t_emb)
        h4 = self.block3b(h4, t_emb)

        h5 = self.down4(h4)
        if self.use_prior:
            t_h5 = torch.max_pool2d(t_h4, kernel_size=2, stride=2)
            h5 = torch.concat((t_h5, h5), dim=1)
        h5 = self.block4a(h5, t_emb)
        h5 = self.block4b(h5, t_emb)

        # --- MIDDLE ---
        u = self.middel(h5, t_emb)
        u = self.middel_att(u)

        # --- DECODER ---
        if self.use_prior:
            u = torch.concat((u, t_h5), dim=1)
        u = self.up1(u, h5, t_emb)
        u = self.up1_att(u)

        if self.use_prior:
            u = torch.concat((u, t_h4), dim=1)
        u = self.up2(u, h4, t_emb)
        # Note: up2_att was declared but omitted in both of your original forward passes. 
        # Kept uncalled to preserve original structure.

        if self.use_prior:
            u = torch.concat((u, t_h3), dim=1)
        u = self.up3(u, h3, t_emb)

        if self.use_prior:
            u = torch.concat((u, t_h2), dim=1)
        u = self.up4(u, h2, t_emb)

        if self.use_prior:
            u = torch.concat((u, t_h1), dim=1)
        u = self.up5(u, h1, t_emb)

        return self.final_conv(u)