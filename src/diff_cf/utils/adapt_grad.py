"""Implementation of AdaptGrad from https://arxiv.org/html/2410.07711v2"""
import torch
from scipy.special import erfinv    # pylint: disable=no-name-in-module
from torch import nn


class AdaptGrad(nn.Module):
    """
    Implementation of AdaptGrad from https://arxiv.org/html/2410.07711v2
    """
    z_min: torch.Tensor
    z_max: torch.Tensor
    denominator: torch.Tensor

    def __init__(
            self,
            N: int,
            c: float,
            z_min: torch.Tensor,
            z_max: torch.Tensor,
    ):
        """
        Implementation of AdaptGrad from https://arxiv.org/html/2410.07711v2.

        Args:
            N: number of samples created
            c: confidence threshold
            data_min: minimum data value
            data_max: maximum data value
            data_means: mean data value
            data_stds: standard deviation data value
        """
        super().__init__()
        self.N = N     # pylint: disable=invalid-name
        self.c = c
        self.register_buffer('z_min', z_min[None])
        self.register_buffer('z_max', z_max[None])
        self.register_buffer('denominator', torch.sqrt(torch.tensor(2.0)) * erfinv(self.c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of forward function.

        Args:
            x: input tensor for model (B, C, H, W).

        Returns:
            input tensor: Noised tensor (B * N, C, H, W).
        """
        if self.N <= 1:
            return x

        with torch.no_grad():
            sigma = torch.min(self.z_max - x, torch.abs(self.z_min - x)) / self.denominator
            sigma = torch.where((x <= self.z_min) | (x >= self.z_max), torch.zeros_like(sigma),
                                sigma)
            noise = torch.randn((self.N, *x.shape), requires_grad=False, device=x.device) * sigma

        return (x + noise).flatten(0, 1)
