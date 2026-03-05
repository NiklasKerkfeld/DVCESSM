from typing import Literal, Tuple

import torch
from torch import Tensor, nn

from Parameter import T, BETA_MIN, BETA_MAX
from utils import cosine_beta_schedule


class DiffusionEngine(nn.Module):
    def __init__(self,
                 device: torch.device,
                 beta_schedule: Literal["linear", 'cosine'] = 'linear'
                 ):
        super().__init__()
        self.T: int = T
        self.device = device

        beta = cosine_beta_schedule(T) if beta_schedule == 'cosine' else torch.linspace(BETA_MIN, BETA_MAX, T)
        alphas = 1.0 - beta
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # precalculate values
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', (1 - alphas) / torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_beta', torch.sqrt(beta))

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
        noise = torch.randn_like(images)

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


























