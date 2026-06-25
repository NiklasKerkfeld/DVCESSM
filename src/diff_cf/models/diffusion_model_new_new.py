import math
from typing import Optional, List

import torch
from torch import nn

from .diffusion_model_new import DiffusionConvLayer, DownBlock, UpBlock
from .layer import AttentionBlock


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
        Diffusion model.

        Args:
            channels: number of input channels.
            width: width of the input image.
            height: height of the input image.
            time_dim: size of time embedding
            nums_features: feature maps of each Residual Block (default [64, 128, 256, 512, 1024]).
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
            nn.Linear(self.time_dim , self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim , self.time_dim),
            nn.SiLU()
        )

        self.conv1 = DiffusionConvLayer(channels, self.nums_features[0], kernel_size=3, padding=1)

        # --- STAGE 1: Downsample & Expand (64 -> 128) ---
        self.down1 = DiffusionConvLayer(self.nums_features[0], self.nums_features[1],
                                        kernel_size=3, padding=1, stride=2) # size // 2
        self.block1a = DownBlock(self.nums_features[1], self.nums_features[1], self.time_dim)
        self.block1b = DownBlock(self.nums_features[1], self.nums_features[1], self.time_dim)

        # --- STAGE 2: Downsample & Expand (128 -> 256) ---
        self.down2 = DiffusionConvLayer(self.nums_features[1], self.nums_features[2],
                                        kernel_size=3, padding=1, stride=2) # size // 4
        self.block2a = DownBlock(self.nums_features[2], self.nums_features[2], self.time_dim)
        self.block2b = DownBlock(self.nums_features[2], self.nums_features[2], self.time_dim)

        # --- STAGE 3: Downsample & Expand (256 -> 512) ---
        self.down3 = DiffusionConvLayer(self.nums_features[2], self.nums_features[3],
                                        kernel_size=3, padding=1, stride=2) # size // 8
        self.block3a = DownBlock(self.nums_features[3], self.nums_features[3], self.time_dim)
        self.block3b = DownBlock(self.nums_features[3], self.nums_features[3], self.time_dim)

        # --- STAGE 4: Downsample & Expand (512 -> 1024) ---
        self.down4 = DiffusionConvLayer(self.nums_features[3], self.nums_features[4],
                                        kernel_size=3, padding=1, stride=2) # size // 16
        self.block4a = DownBlock(self.nums_features[4], self.nums_features[4], self.time_dim)
        self.block4b = DownBlock(self.nums_features[4], self.nums_features[4], self.time_dim)

        # --- MIDDLE STAGE ---
        self.middel = DownBlock(self.nums_features[4], self.nums_features[4], self.time_dim)
        self.middel_att = AttentionBlock(self.nums_features[4], self.width // 16, self.height // 16)

        # --- DECODER STAGE (Unchanged, remains perfectly compatible) ---
        self.up1 = UpBlock(self.nums_features[4], self.nums_features[4], self.nums_features[3],
                           self.time_dim)
        self.up1_att = AttentionBlock(self.nums_features[3], self.width // 8, self.height // 8)

        self.up2 = UpBlock(self.nums_features[3], self.nums_features[3], self.nums_features[2],
                           self.time_dim)
        self.up2_att = AttentionBlock(self.nums_features[2], self.width // 4, self.height // 4)

        self.up3 = UpBlock(self.nums_features[2], self.nums_features[2], self.nums_features[1],
                           self.time_dim)
        self.up4 = UpBlock(self.nums_features[1], self.nums_features[1], self.nums_features[0],
                           self.time_dim)
        self.up5 = UpBlock(self.nums_features[0], self.nums_features[0], self.nums_features[0],
                           self.time_dim, scale_factor=1.0)

        self.final_conv = DiffusionConvLayer(self.nums_features[0], channels, kernel_size=3,
                                             padding=1, activation=nn.Identity(), apply_norm=False)

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
        h1 = self.conv1(x)              # (64, h, w)

        # encoder
        h2 = self.down1(h1)             # (128, h//2, w//2) - Expanded here now!
        h2 = self.block1a(h2, t_emb)
        h2 = self.block1b(h2, t_emb)

        h3 = self.down2(h2)             # (256, h//4, w//4) - Expanded here now!
        h3 = self.block2a(h3, t_emb)
        h3 = self.block2b(h3, t_emb)

        h4 = self.down3(h3)             # (512, h//8, w//8) - Expanded here now!
        h4 = self.block3a(h4, t_emb)
        h4 = self.block3b(h4, t_emb)

        h5 = self.down4(h4)             # (1024, h//16, w//16) - Expanded here now!
        h5 = self.block4a(h5, t_emb)
        h5 = self.block4b(h5, t_emb)

        u = self.middel(h5, t_emb)      # (1024, h//16, w//16)
        u = self.middel_att(u)

        # decoder
        u = self.up1(u, h5, t_emb)      # (512, h//8, w//8)
        u = self.up1_att(u)

        u = self.up2(u, h4, t_emb)      # (256, h//4, w//4)

        u = self.up3(u, h3, t_emb)      # (128, h//2, w//2)
        u = self.up4(u, h2, t_emb)      # (64, h, w)
        u = self.up5(u, h1, t_emb)      # (64, h, w)

        return self.final_conv(u)