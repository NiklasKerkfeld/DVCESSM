import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Adversarial(nn.Module):
    def __init__(
        self,
        image_channels: int,
        channels: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()

        layers = []
        in_channels = image_channels

        for out_channels in channels:
            layers.extend([
                spectral_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = spectral_norm(nn.Linear(channels[-1], 1))

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = getattr(module, "weight_orig", module.weight)
            nn.init.normal_(weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        features = self.pool(features)
        features = features.flatten(start_dim=1)
        return self.classifier(features)


if __name__ == "__main__":
    adversarial = Adversarial(image_channels=3)

    image = torch.randn(16, 3, 64, 64)
    pred = adversarial(image)
    print(pred.shape)