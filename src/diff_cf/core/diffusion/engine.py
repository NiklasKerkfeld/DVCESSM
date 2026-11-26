"""Engine for the Diffusion process."""

from typing import Tuple, Literal

import torch
from torch import nn


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:  # pylint: disable=invalid-name
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)

    alphas_cumprod = torch.cos((((t / T) + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0.0001, 0.999)


# pylint: disable=abstract-method
class Engine(nn.Module):
    """Engine for the Diffusion process."""
    beta: torch.Tensor
    alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    coeff1: torch.Tensor
    coeff2: torch.Tensor

    def __init__(self,
                 T: int = 1000,
                 beta_schedule: Literal['cosine', 'linear'] = 'cosine',
                 beta_min: float = 1e-4,
                 beta_max: float = 0.02,
                 device: torch.device = torch.device('cpu')) -> None:
        """
        Engine for the Diffusion process.

        Args:
            T: Total number of diffusion timesteps.
            beta_schedule: Schedule type ('cosine' or 'linear').
            beta_min: Minimum beta value for linear schedule.
            beta_max: Maximum beta value for linear schedule.
            device: Device to run on.
        """
        super().__init__()

        self.T = T  # pylint: disable=invalid-name
        self.device = device

        if beta_schedule == 'cosine':
            beta = cosine_beta_schedule(self.T)
        elif beta_schedule == 'linear':
            beta = torch.linspace(beta_min, beta_max, self.T)
        else:
            raise ValueError(
                f"Unsupported beta_schedule: '{beta_schedule}'. Expected 'cosine' or 'linear'.")

        alphas = 1.0 - beta
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1,
                                                    dtype=alphas_cumprod.dtype,
                                                    device=alphas_cumprod.device),
                                         alphas_cumprod[:-1]])
        posterior_variance = (beta * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        # precalculate values
        self.register_buffer("beta", beta)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=0.0))
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', (1 - alphas) / torch.sqrt(1 - alphas_cumprod))

        self.to(self.device)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates Signal to Noise Ratio for that time step.

        Args:
            t: timestep.
        """
        alphas_comprod = self._extract(self.alphas_cumprod, t, t.shape)
        return alphas_comprod / (1 - alphas_comprod)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extracts values from a 1D constant tensor for a batch of indices t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def sample(self, images: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an image at random timestep t.
        Args:
            images: image to be noised

        Returns:
            noised image: image with added noise
            noise: the added noise
            t: timestep
        """
        t = torch.randint(0, self.T, (images.shape[0],), device=images.device)

        noised_images, noise, snr = self.diffuse(images, t)
        return noised_images, noise, t, snr

    def diffuse(self, images: torch.Tensor, t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Diffuses in given image (batch) at timestep t.
        Args:
            images: batch of images to be noised (B, C, H, W).
            t: tensor with timesteps

        Returns:
            noised image: image with added noise
            noise: the added noise
        """
        # generate random noise for diffusion
        noise = torch.randn_like(images, device=images.device)

        # get values in the shape (batch_size, 1, 1, 1)
        sqrt_alpha_hat = self._extract(self.sqrt_alphas_cumprod, t, images.shape)
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alphas_cumprod, t,
                                                 images.shape)

        # add noise based on t to image
        noised = sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * noise

        # Calculate SNR directly from the extracted components before they are expanded spatially
        # We can extract them as shape (batch_size,) or just flatten/mean the ratio
        # But a cleaner way is calculating it per-sample:
        snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t])

        return noised, noise, snr

    def ddim_mean_variance(
            self,
            images: torch.Tensor,
            epsilon_pred: torch.Tensor,
            t: torch.Tensor,
            target_t: torch.Tensor,
            eta: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one generalized DDIM transition from t to target_t.

        target_t may contain -1, which represents the clean-image endpoint.
        eta=0 gives deterministic DDIM sampling.
        """
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be between 0 and 1")

        if torch.any(target_t >= t):
            raise ValueError("target_t must be smaller than t")

        alpha_t = self._extract(
            self.alphas_cumprod,
            t,
            images.shape,
        )

        target_clamped = target_t.clamp(min=0)
        alpha_target = self._extract(self.alphas_cumprod, target_clamped, images.shape)

        endpoint_mask = (target_t < 0).reshape(target_t.shape[0], *((1,) * (images.ndim - 1)))
        alpha_target = torch.where(endpoint_mask, torch.ones_like(alpha_target), alpha_target)

        x0_pred = self.image_approximation(images, epsilon_pred, t)

        variance = (eta ** 2 * (1.0 - alpha_target) /
                         (1.0 - alpha_t) * (1.0 - alpha_t / alpha_target)).clamp(min=0.0)

        epsilon_coefficient = torch.sqrt((1.0 - alpha_target - variance).clamp(min=0.0))

        result = torch.sqrt(alpha_target) * x0_pred + epsilon_coefficient * epsilon_pred

        return result, variance

    def ddim_step(
            self,
            images: torch.Tensor,
            epsilon_pred: torch.Tensor,
            t: torch.Tensor,
            target_t: torch.Tensor,
            eta: float = 0.0,
    ) -> torch.Tensor:
        """Samples one generalized DDIM transition."""
        mean, variance = self.ddim_mean_variance(
            images=images,
            epsilon_pred=epsilon_pred,
            t=t,
            target_t=target_t,
            eta=eta,
        )

        if eta == 0.0:
            return mean

        return mean + torch.sqrt(variance) * torch.randn_like(images)

    def image_approximation(self,
                            x_t: torch.Tensor,
                            epsilon_pred: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        """
        Directly approximates the x_0 image based on the noise.

        Args:
            x_t: noised image (B, C, H, W).
            epsilon_pred: predicted noise (B, C, H, W).
            t: timestep.

        Returns:
            denoised image: approximated image at t = 0.
        """
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_alpha_hat = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)

        return (x_t - sqrt_one_minus_alpha_hat * epsilon_pred) / sqrt_alpha_hat

    def epsilon_approximation(self,
                              x_t: torch.Tensor,
                              x_0: torch.Tensor,
                              t_tensor: torch.Tensor) -> torch.Tensor:
        """
        Used current x_t and predicted x_0 to approximate the noise on the image x_t.

        Args:
            x_t: noised image (B, C, H, W).
            x_0: approximation of x_0 (B, C, H, W).
            t_tensor: timestep.

        Returns:
            approximated epsilon (B, C, H, W).
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor,
                                                 x_t.shape)
        return (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar
