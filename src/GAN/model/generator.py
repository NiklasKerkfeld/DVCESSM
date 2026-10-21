import torch
from torch import nn
import torch.nn.functional as F

def suitable_group_count(channels: int, maximum: int = 8) -> int:
    """Find a GroupNorm group count that divides channels."""
    groups = min(maximum, channels)

    while channels % groups != 0:
        groups -= 1

    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        groups = suitable_group_count(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PriorModel(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims

        self.mlp = nn.Sequential(
            nn.Linear(in_dims, in_dims * 2),
            nn.ReLU(),
            nn.Linear(in_dims * 2, in_dims * 4),
            nn.ReLU(),
            nn.Linear(in_dims * 4, out_dims * 2),
            nn.ReLU()
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adoption = self.mlp(embedding)
        return adoption[:, :self.out_dims, None, None], adoption[:, self.out_dims:, None, None]


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, prior_dim: int):
        super().__init__()

        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.convolution = ConvBlock(out_channels, out_channels)
        self.prior_mlp = PriorModel(prior_dim, out_channels)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        scale, shift = self.prior_mlp(prior)
        x = x * scale + shift
        return self.convolution(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()

        self.channel_projection = nn.Conv2d(
            in_channels,
            skip_channels,
            kernel_size=1,
        )

        self.convolution = ConvBlock(
            skip_channels * 2,
            skip_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        # Using the skip size also supports images with odd dimensions.
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.channel_projection(x)
        x = torch.cat([x, skip], dim=1)

        return self.convolution(x)


class SpatialAttentionBlock(nn.Module):
    """
    Multi-head self-attention over the spatial positions.

    Input:
        [batch, channels, height, width]

    Attention representation:
        [batch, height * width, channels]
    """

    def __init__(
        self,
        channels: int,
        number_of_heads: int,
        mlp_ratio: int = 2,
    ):
        super().__init__()

        if channels % number_of_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"number_of_heads ({number_of_heads})."
            )

        # Depthwise convolution supplies local positional information.
        self.position = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )

        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=number_of_heads,
            batch_first=True,
        )

        hidden_channels = channels * mlp_ratio

        self.mlp_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        x = x + self.position(x)

        sequence = x.flatten(2).transpose(1, 2)

        normalized = self.attention_norm(sequence)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        sequence = sequence + attention_output
        sequence = sequence + self.mlp(self.mlp_norm(sequence))

        return sequence.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )



class Generator(nn.Module):
    def __init__(self, in_channels: int, level: int=6):
        super(Generator, self).__init__()
        self.conv1 = ConvBlock(in_channels * 2, out_channels=64)

        self.down1 = DownBlock(64, 128, level)
        self.down2 = DownBlock(128, 256, level)

        self.down3 = DownBlock(256, 512, level)
        self.down_attention_3 = SpatialAttentionBlock(512, number_of_heads=8)

        self.down4 = DownBlock(512, 1024, level)
        self.down_attention_4 = SpatialAttentionBlock(1024, number_of_heads=8)

        self.up1 = UpBlock(1024, 512)
        self.up_attention_1 = SpatialAttentionBlock(512, number_of_heads=8)

        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.conv2 = ConvBlock(64 + in_channels, in_channels)
        self.powers = nn.Parameter(torch.arange(1, level + 1), requires_grad=False)


    def forward(self, image: torch.Tensor, prior: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        embedding = scalar ** self.powers

        x = torch.cat([image, prior], dim=1)
        h1 = self.conv1(x)
        h2 = self.down1(h1, embedding)
        h3 = self.down2(h2, embedding)
        h4 = self.down3(h3, embedding)
        h4 = self.down_attention_3(h4)

        h5 = self.down4(h4, embedding)
        h5 = self.down_attention_4(h5)

        up1 = self.up1(h5, h4)
        up1 = self.up_attention_1(up1)
        up2 = self.up2(up1, h3)
        up3 = self.up3(up2, h2)
        up4 = self.up4(up3, h1)

        x = torch.cat([up4, prior], dim=1)
        x = self.conv2(x)

        x = F.normalize(x, p=2.0, dim=(1, 2, 3))

        return x * scalar[:, :, None, None]


if __name__ == "__main__":
    in_channels = 3
    generator = Generator(in_channels)

    image = torch.randn(2, in_channels, 64, 64)
    prior = torch.randn(2, in_channels, 64, 64)
    scalar = torch.rand(2, 1) * 100
    print(scalar)

    adoption = generator(image, prior, scalar)
    print(adoption.shape)
    print(torch.norm(adoption, p=2.0, dim=(1, 2, 3)))


