from typing import Literal, Tuple

import torch
from torch import Tensor, nn


def cosine_beta_schedule(T: int, s: float = 0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)

    alphas_cumprod = torch.cos((((t / T) + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0.0001, 0.999)


class Engine(nn.Module):
    def __init__(self,
                 T: int,
                 beta_min: float,
                 beta_max: float,
                 device: torch.device,
                 beta_schedule: Literal["linear", 'cosine'] = 'linear'
                 ):
        super().__init__()
        self.T = T
        self.device = device

        beta = cosine_beta_schedule(T) if beta_schedule == 'cosine' else torch.linspace(beta_min, beta_max, T)
        alphas = 1.0 - beta
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # precalculate values
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', (1 - alphas) / torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_beta', torch.sqrt(beta))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        alphas_comprod = self._extract(self.alphas_cumprod, t, t.shape)
        return alphas_comprod / (1 - alphas_comprod)

    @staticmethod
    def _extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
        """Extracts values from a 1D constant tensor for a batch of indices t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def sample(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        t = torch.randint(0, self.T, (images.shape[0],), device=self.device)

        noised_images, noise = self.diffuse(images, t)
        return noised_images, noise, t

    def diffuse(self, images: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # generate random noise for diffusion
        noise = torch.randn_like(images, device=images.device)

        # get values in the shape (batch_size, 1, 1, 1)
        sqrt_alpha_hat = self._extract(self.sqrt_alphas_cumprod, t, images.shape)
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alphas_cumprod, t,
                                                 images.shape)

        # add noise based on t to image
        noised = sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * noise

        return noised, noise

    def reverse(self, images: Tensor, epsilon_pred: Tensor, t: Tensor) -> Tensor:
        coeff1 = self._extract(self.coeff1, t, images.shape)
        coeff2 = self._extract(self.coeff2, t, images.shape)

        mean = coeff1 * (images - coeff2 * epsilon_pred)

        return mean

    def image_approximation(self, x_t: Tensor, epsilon_pred: Tensor, t: Tensor) -> Tensor:
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_alpha_hat = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)

        return (x_t - sqrt_one_minus_alpha_hat * epsilon_pred) / sqrt_alpha_hat

    def get_sigma(self, t: Tensor, shape: torch.Size):
        return self._extract(self.sqrt_beta, t, shape)


if __name__ == '__main__':
    engine = Engine(T=1000, beta_min=0.1, beta_max=1.0, device=torch.device('cpu'))
    t = torch.tensor([0, 1, 10, 50, 100, 250, 500, 800, 900, 990, 998, 999])
    snr = engine.snr(t)
    for i in range(len(t)):
        print(f"{t[i]}: {snr[i]}")