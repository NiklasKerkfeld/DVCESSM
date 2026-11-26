"""Diffusion model."""
import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F

from .layer import AttentionBlock, DiffusionConvLayer, TimeConditionedResBlock, UpBlock


class DiffusionModel(nn.Module):
    """
    Unified Diffusion model supporting both unconditional generation 
    and target-prior segmentation mask conditioning.
    """
    emb: torch.Tensor
    channels: int
    width: int
    height: int

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
        # --- check parameter values ---
        if time_dim < 4 or time_dim % 2 != 0:
            raise ValueError("time_dim must be even and at least 4")

        if nums_features is not None and len(nums_features) != 5:
            raise ValueError("nums_features must contain exactly five channel sizes")

        if nums_features is not None and any(c <= 0 for c in nums_features):
            raise ValueError("All feature sizes must be positive")

        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("width and height must be positive multiples of 16")

        if channels <= 0:
            raise ValueError("channels must be positive")

        if use_prior and num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

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

        # --- TIME EMBEDDING ---
        half_dim = self.time_dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU()
        )

        # --- PRIOR EMBEDDING ---
        if self.use_prior:
            self.prior_mlp = nn.Conv2d(2 * self.num_prior_channels, self.num_prior_channels,
                                       kernel_size=1, padding=0)

        # --- FIRST CONVOLUTION ---
        self.conv1 = DiffusionConvLayer(channels + self.num_prior_channels, self.nums_features[0],
                                        kernel_size=3, padding=1)

        # --- STAGE 1: Downsample & Expand (64 -> 128) ---
        self.down1 = DiffusionConvLayer(self.nums_features[0], self.nums_features[0],
                                        kernel_size=3, padding=1, stride=2)
        self.block1a = TimeConditionedResBlock(self.nums_features[0] + self.num_prior_channels,
                                               self.nums_features[1], self.time_dim)

        self.block1b = TimeConditionedResBlock(self.nums_features[1],
                                               self.nums_features[1],
                                               self.time_dim)

        # --- STAGE 2: Downsample & Expand (128 -> 256) ---
        self.down2 = DiffusionConvLayer(self.nums_features[1], self.nums_features[1],
                                        kernel_size=3, padding=1, stride=2)
        self.block2a = TimeConditionedResBlock(self.nums_features[1] + self.num_prior_channels,
                                               self.nums_features[2],
                                               self.time_dim)
        self.block2b = TimeConditionedResBlock(self.nums_features[2],
                                               self.nums_features[2],
                                               self.time_dim)

        # --- STAGE 3: Downsample & Expand (256 -> 512) ---
        self.down3 = DiffusionConvLayer(self.nums_features[2], self.nums_features[2],
                                        kernel_size=3, padding=1, stride=2)
        self.block3a = TimeConditionedResBlock(self.nums_features[2] + self.num_prior_channels,
                                               self.nums_features[3],
                                               self.time_dim)
        self.block3b = TimeConditionedResBlock(self.nums_features[3],
                                               self.nums_features[3],
                                               self.time_dim)

        # --- STAGE 4: Downsample & Expand (512 -> 1024) ---
        self.down4 = DiffusionConvLayer(self.nums_features[3], self.nums_features[3],
                                        kernel_size=3, padding=1, stride=2)
        self.block4a = TimeConditionedResBlock(self.nums_features[3] + self.num_prior_channels,
                                               self.nums_features[4],
                                               self.time_dim)
        self.block4b = TimeConditionedResBlock(self.nums_features[4],
                                               self.nums_features[4],
                                               self.time_dim)

        # --- MIDDLE STAGE ---
        self.middle = TimeConditionedResBlock(self.nums_features[4],
                                              self.nums_features[4],
                                              self.time_dim)
        self.middle_att = AttentionBlock(self.nums_features[4],
                                         self.width // 16,
                                         self.height // 16)

        # --- DECODER STAGES ---
        self.up1 = UpBlock(self.nums_features[4] + self.num_prior_channels,
                           self.nums_features[4],
                           self.nums_features[3],
                           self.time_dim)
        self.up1_att = AttentionBlock(self.nums_features[3],
                                      self.width // 8,
                                      self.height // 8)

        self.up2 = UpBlock(self.nums_features[3] + self.num_prior_channels,
                           self.nums_features[3],
                           self.nums_features[2],
                           self.time_dim)
        self.up2_att = AttentionBlock(self.nums_features[2],
                                      self.width // 4,
                                      self.height // 4)

        self.up3 = UpBlock(self.nums_features[2] + self.num_prior_channels,
                           self.nums_features[2],
                           self.nums_features[1],
                           self.time_dim)
        self.up4 = UpBlock(self.nums_features[1] + self.num_prior_channels,
                           self.nums_features[1],
                           self.nums_features[0],
                           self.time_dim)
        self.up5 = UpBlock(self.nums_features[0] + self.num_prior_channels,
                           self.nums_features[0],
                           self.nums_features[0],
                           self.time_dim,
                           scale_factor=1.0)

        self.final_conv = DiffusionConvLayer(self.nums_features[0], channels,
                                             kernel_size=3,
                                             padding=1,
                                             activation=nn.Identity(),
                                             apply_norm=False,
                                             zero_initialization=True)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embedding for timesteps."""
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def get_prior(self, target: torch.Tensor, kernel: int):
        """
        Calculates prior form segmentation mask.
        Args:
            target: segmentation mask
            kernel: kernel size

        Returns:
            Contatinated Max pooling and average pooling of segmentation mask.
        """
        if kernel > 1:
            t_max = F.max_pool2d(target, kernel_size=kernel, stride=kernel)
            t_avg = F.avg_pool2d(target, kernel_size=kernel, stride=kernel)
            t = torch.cat((t_max, t_avg), dim=1)
        else:
            t = torch.cat((target, target), dim=1)

        return self.prior_mlp(t)


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
            hot_target = F.one_hot(
                target,     # type: ignore # pylint: disable=not-callable
                num_classes=self.num_prior_channels
            ).permute(0, 3, 1, 2).float()
            t_h1 = self.get_prior(hot_target, 1)
            x = torch.concat((t_h1, x), dim=1)

        # First convolution
        h1 = self.conv1(x)  # (64, h, w)

        # --- ENCODER ---
        h2 = self.down1(h1)
        if self.use_prior:
            t_h2 = self.get_prior(hot_target, 2)
            h2 = torch.concat((t_h2, h2), dim=1)
        h2 = self.block1a(h2, t_emb)
        h2 = self.block1b(h2, t_emb)

        h3 = self.down2(h2)
        if self.use_prior:
            t_h3 = self.get_prior(hot_target, 4)
            h3 = torch.concat((t_h3, h3), dim=1)
        h3 = self.block2a(h3, t_emb)
        h3 = self.block2b(h3, t_emb)

        h4 = self.down3(h3)
        if self.use_prior:
            t_h4 = self.get_prior(hot_target, 8)
            h4 = torch.concat((t_h4, h4), dim=1)
        h4 = self.block3a(h4, t_emb)
        h4 = self.block3b(h4, t_emb)

        h5 = self.down4(h4)
        if self.use_prior:
            t_h5 = self.get_prior(hot_target, 16)
            h5 = torch.concat((t_h5, h5), dim=1)
        h5 = self.block4a(h5, t_emb)
        h5 = self.block4b(h5, t_emb)

        # --- MIDDLE ---
        u = self.middle(h5, t_emb)
        u = self.middle_att(u)

        # --- DECODER ---
        if self.use_prior:
            u = torch.concat((u, t_h5), dim=1)
        u = self.up1(u, h5, t_emb)
        u = self.up1_att(u)

        if self.use_prior:
            u = torch.concat((u, t_h4), dim=1)
        u = self.up2(u, h4, t_emb)
        u = self.up2_att(u)

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
