import math

import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride:int=1, activation=nn.SiLU(), apply_norm:bool=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        num_groups = 8 if out_channels >= 8 else 1
        self.norm = nn.GroupNorm(num_groups, out_channels) if apply_norm else nn.Identity()

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, in_channels // 4)
        self.conv2 = ConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = ConvLayer(out_channels, out_channels, kernel_size=2, padding=0, stride=2)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_proj
        h = self.conv2(torch.nn.functional.silu(h))
        x = self.conv3(x)

        return self.pool(x + h)


class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, in_channels // 4)
        self.conv2 = ConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_proj
        h = self.conv2(torch.nn.functional.silu(h))
        x = self.conv3(x)

        return x + h


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_size, out_channels, time_emb_dim):
        super().__init__()
        self.up_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels + skip_size, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, in_channels + skip_size)

    def forward(self, u, c, t_emb):
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        
        u = self.up_transpose(u)
        d = torch.cat([u, c], dim=1)
        d += t_proj
        u = self.conv(d)
        
        return u


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
    

class DiffusionModel(nn.Module):
    def __init__(self, channels: int, width: int, height: int, time_dim: int):
        super().__init__()
        self.in_channels = channels
        self.time_dim = time_dim
        self.width = width
        self.height = height

        half_dim = self.time_dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim , self.time_dim ),
            nn.SiLU(),
            nn.Linear(self.time_dim , self.time_dim),
            nn.SiLU()
        )
        
        self.conv1 = ConvLayer(channels, 16, kernel_size=3, padding=1)    # size // 2
        self.down1 = DownBlock(16, 32, self.time_dim)           # size // 4
        self.down2 = DownBlock(32, 64, self.time_dim)           # size // 8
        self.down3 = DownBlock(64, 128, self.time_dim)          # size // 16
        self.down4 = DownBlock(128, 256, self.time_dim)         # # size // 32

        self.middel = DownBlock(256, 512, self.time_dim)
        self.middel_att = AttentionBlock(512, self.width // 32, self.height // 32)

        self.up1 = UpBlock(512, 256,256, self.time_dim)
        self.up1_att = AttentionBlock(256, self.width // 16, self.height // 16)

        self.up2 = UpBlock(256, 128, 128, self.time_dim)
        self.up2_att = AttentionBlock(128, self.width // 8, self.height // 8)

        self.up3 = UpBlock(128, 64, 64, self.time_dim)
        self.up3_att = AttentionBlock(64, self.width // 4, self.height // 4)

        self.up4 = UpBlock(64, 32, 32, self.time_dim)
        self.up5 = UpBlock(32, 16, 16, self.time_dim)

        self.final_conv = ConvLayer(16, channels, kernel_size=3, padding=1,
                                    activation=nn.Identity(), apply_norm=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor):
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # first convolution
        x = self.conv1(x)

        # encoder
        h1 = self.down1(x, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)
        h4 = self.down4(h3, t_emb)
        
        u = self.middel(h4, t_emb)
        u = self.middel_att(u)

        # decoder
        u = self.up1(u, h4, t_emb)
        print(f"{u.shape=}")
        u = self.up1_att(u)

        u = self.up2(u, h3, t_emb)
        u = self.up2_att(u)

        u = self.up3(u, h2, t_emb)
        u = self.up4(u, h1, t_emb)
        u = self.up5(u, x, t_emb)
        
        return self.final_conv(u)

if __name__ == '__main__':
    model = DiffusionModel(1, 32, 32,256)

    image = torch.randn(4, 1, 32, 32)
    timesteps = torch.randint(1, 1000, (4,))
    pred = model(image, timesteps)
    print(pred.shape)
