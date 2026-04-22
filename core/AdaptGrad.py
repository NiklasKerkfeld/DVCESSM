import torch
from omegaconf import DictConfig
from scipy.special import erfinv
from torch import nn


class AdaptGrad(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.N = cfg.diffusion.AdaptGrad.N
        self.c = cfg.diffusion.AdaptGrad.c
        self.x_min = ((cfg.data.min - torch.tensor(cfg.data.dataset.means)) / torch.tensor(cfg.data.dataset.stds))[None, :, None, None]
        self.x_max = ((cfg.data.max - torch.tensor(cfg.data.dataset.means)) / torch.tensor(cfg.data.dataset.stds))[None, :, None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denominator = torch.sqrt(torch.tensor(2.0)) * erfinv(self.c)
        sigma = torch.min(self.x_max - x, torch.abs(self.x_min - x)) / denominator

        sigma = torch.where((x <= self.x_min) | (x >= self.x_max), torch.zeros_like(sigma), sigma)

        noise = torch.randn((self.N, *x.shape), requires_grad=False, device=x.device) * sigma

        return (x + noise).flatten(0, 1)
