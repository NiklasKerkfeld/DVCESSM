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


class GradientPriorInjection(nn.Module):
    """
    Projects the target-model gradient to the number of feature channels
    and adds it to a feature map.
    """

    def __init__(
        self,
        gradient_channels: int,
        feature_channels: int,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            gradient_channels,
            feature_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        features: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        resized_gradient = F.interpolate(
            gradient,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return features + self.projection(resized_gradient)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.convolution = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
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
    """
    U-Net generator for counterfactual image adoption.

    Arguments:
        image:
            Input image with shape [B, C, H, W].

        gradient_prior:
            Gradient from the target model with shape [B, C, H, W].

    Returns:
        Adoption with shape [B, C, H, W].
    """

    def __init__(
        self,
        image_channels: int,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        attention_heads: int = 8,
        attention_layers: int = 2,
        max_adoption: float | None = None,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("At least two U-Net channel levels are required.")

        self.max_adoption = max_adoption

        # Initial image features.
        self.input_block = ConvBlock(
            image_channels,
            channels[0],
        )

        # Gradient injection at each encoder level, including bottleneck.
        self.encoder_gradient_injections = nn.ModuleList([
            GradientPriorInjection(image_channels, feature_channels)
            for feature_channels in channels
        ])

        self.down_blocks = nn.ModuleList([
            DownBlock(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)
        ])

        self.attention_blocks = nn.Sequential(*[
            SpatialAttentionBlock(
                channels=channels[-1],
                number_of_heads=attention_heads,
            )
            for _ in range(attention_layers)
        ])

        reversed_skip_channels = list(reversed(channels[:-1]))

        self.up_blocks = nn.ModuleList()
        self.decoder_gradient_injections = nn.ModuleList()

        current_channels = channels[-1]

        for skip_channels in reversed_skip_channels:
            self.up_blocks.append(
                UpBlock(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                )
            )
            self.decoder_gradient_injections.append(
                GradientPriorInjection(
                    gradient_channels=image_channels,
                    feature_channels=skip_channels,
                )
            )
            current_channels = skip_channels

        self.output = nn.Conv2d(
            channels[0],
            image_channels,
            kernel_size=1,
        )

        self.apply(self._initialize_weights)

        # Exact zero adoption at initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_gradient(
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize independently for each item in the batch.
        rms = gradient.square().mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).sqrt()

        return gradient / rms.clamp_min(1e-6)

    def forward(
        self,
        image: torch.Tensor,
        gradient_prior: torch.Tensor,
    ) -> torch.Tensor:
        if image.shape != gradient_prior.shape:
            raise ValueError(
                "image and gradient_prior must have the same shape, "
                f"but received {image.shape} and {gradient_prior.shape}."
            )

        # Usually second-order gradients through the target model are
        # unnecessary, so treat the gradient as a fixed prior.
        gradient_prior = gradient_prior.detach()
        gradient_prior = self._normalize_gradient(gradient_prior)

        features = self.input_block(image)
        features = self.encoder_gradient_injections[0](
            features,
            gradient_prior,
        )

        skip_connections = []

        for index, down_block in enumerate(self.down_blocks):
            skip_connections.append(features)

            features = down_block(features)
            features = self.encoder_gradient_injections[index + 1](
                features,
                gradient_prior,
            )

        features = self.attention_blocks(features)

        for up_block, injection, skip in zip(
            self.up_blocks,
            self.decoder_gradient_injections,
            reversed(skip_connections),
        ):
            features = up_block(features, skip)
            features = injection(features, gradient_prior)

        adoption = self.output(features)

        if self.max_adoption is not None:
            adoption = self.max_adoption * torch.tanh(adoption)

        return adoption

if __name__ == "__main__":
    generator = Generator(image_channels=3)

    image = torch.randn(1, 3, 32, 32)
    prior = torch.randn(1, 3, 32, 32)

    adopt = generator(image, prior)
    print(adopt.shape)
    print(adopt.min(), adopt.max())
