import math

import torch
from prettytable import PrettyTable
from torch import nn

from models.Layer import AttentionBlock


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=1, stride:int=1, activation=nn.SiLU(), apply_norm:bool=True):
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
        self.pool = ConvLayer(in_channels, in_channels, kernel_size=2, padding=0, stride=2)

        self.conv1 = ConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)

        self.conv4 = nn.Identity()
        if in_channels != out_channels:
            self.conv4 = ConvLayer(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, in_channels // 4)


    def forward(self, x, t_emb):
        time_proj = self.time_mlp(t_emb)[:, :, None, None]

        x = self.pool(x)

        h = self.conv1(x)
        h = h + time_proj

        h = self.conv2(h)
        h = self.conv3(h)

        x = self.conv4(x)

        return x + h


class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(in_channels // 4, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, in_channels // 4)

    def forward(self, x, t_emb):
        time_proj = self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv1(x)
        h = h + time_proj
        h = self.conv2(h)
        h = self.conv3(h)

        return x + h


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_size, out_channels, time_emb_dim, scale_factor:float=2):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv = nn.Sequential(
            ConvLayer(in_channels + skip_size, out_channels, kernel_size=3, padding=1),
            ConvLayer(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)


    def forward(self, u, c, t_emb):
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        
        d = torch.cat([u, c], dim=1)
        u = self.conv(d)
        u += t_proj

        if self.scale_factor != 1.0:
            u = self.upsample(u)

        return u
    

class DiffusionModel(nn.Module):
    def __init__(self, channels: int, width: int, height: int, time_dim: int):
        super().__init__()
        self.channels = channels
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

        self.nums_features = [64, 128, 256, 512, 1024]

        self.conv1 = ConvLayer(channels, self.nums_features[0], kernel_size=3, padding=1)           # size // 2
        self.down1 = DownBlock(self.nums_features[0], self.nums_features[1], self.time_dim)         # size // 4
        self.down2 = DownBlock(self.nums_features[1], self.nums_features[2], self.time_dim)         # size // 8
        self.down3 = DownBlock(self.nums_features[2], self.nums_features[3], self.time_dim)         # size // 16
        self.down4 = DownBlock(self.nums_features[3], self.nums_features[4], self.time_dim)         # # size // 32

        self.middel = MiddleBlock(self.nums_features[4], self.nums_features[4], self.time_dim)
        self.middel_att = AttentionBlock(self.nums_features[4], self.width // 16, self.height // 16)

        self.up1 = UpBlock(self.nums_features[4], self.nums_features[4], self.nums_features[3], self.time_dim)
        self.up1_att = AttentionBlock(self.nums_features[3], self.width // 8, self.height // 8)

        self.up2 = UpBlock(self.nums_features[3], self.nums_features[3], self.nums_features[2], self.time_dim)
        self.up2_att = AttentionBlock(self.nums_features[2], self.width // 4, self.height // 4)

        self.up3 = UpBlock(self.nums_features[2], self.nums_features[2], self.nums_features[1], self.time_dim)
        self.up3_att = AttentionBlock(self.nums_features[1], self.width // 2, self.height // 2)

        self.up4 = UpBlock(self.nums_features[1], self.nums_features[1], self.nums_features[0], self.time_dim)
        self.up5 = UpBlock(self.nums_features[0], self.nums_features[0], self.nums_features[0], self.time_dim, scale_factor=1.0)

        self.final_conv = ConvLayer(self.nums_features[0], channels, kernel_size=3, padding=1,
                                    activation=nn.Identity(), apply_norm=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor):
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # first convolution
        h1 = self.conv1(x)              # (64, h, w)

        # encoder
        h2 = self.down1(h1, t_emb)      # (128, h//2, w//2)
        h3 = self.down2(h2, t_emb)      # (256, h//4, w//4)
        h4 = self.down3(h3, t_emb)      # (512, h//8, w//8)
        h5 = self.down4(h4, t_emb)      # (1024, h//16, w//16)
        
        u = self.middel(h5, t_emb)      # (1024, h//16, w//16)
        u = self.middel_att(u)

        # decoder
        u = self.up1(u, h5, t_emb)      # (512, h//8, w//8)
        u = self.up1_att(u)

        u = self.up2(u, h4, t_emb)      # (256, h//4, w//4)
        u = self.up2_att(u)

        u = self.up3(u, h3, t_emb)      # (128, h//2, w//2)
        u = self.up4(u, h2, t_emb)      # (64, h, w)
        u = self.up5(u, h1, t_emb)      # (64, h, w)

        return self.final_conv(u)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params




if __name__ == '__main__':
    batch_size = 1
    channels = 4
    width = 128
    height = 128
    time_dim = 256

    model = DiffusionModel(channels, width, height, time_dim)
    print(model)
    count_parameters(model)

    image = torch.randn(batch_size, channels, width, height)
    timesteps = torch.randint(1, 1000, (batch_size,))
    pred = model(image, timesteps)
    print(pred.shape)
