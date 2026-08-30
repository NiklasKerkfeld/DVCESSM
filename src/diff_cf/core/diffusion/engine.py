"""Engine for the Diffusion process."""

from typing import Tuple, Literal

import torch
from omegaconf import DictConfig
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
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    coeff1: torch.Tensor
    coeff2: torch.Tensor
    sqrt_beta: torch.Tensor

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

        # precalculate values
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', (1 - alphas) / torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_beta', torch.sqrt(beta))

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
        t = torch.randint(0, self.T, (images.shape[0],), device=self.device)

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

    def reverse(self,
                images: torch.Tensor,
                epsilon_pred: torch.Tensor,
                t: torch.Tensor,
                steps: int = 1) -> torch.Tensor:
        """
        Reversed the diffusion process.

        Args:
            images: noised image (batch) at timestep t (B, C, H, W).
            epsilon_pred: predicted noise (B, C, H, W).
            t: timestep.
            steps: number of steps to make

        Returns:
            denoised image: imaged reversed by {steps} steps.
        """
        if steps == 1:
            coeff1 = self._extract(self.coeff1, t, images.shape)
            coeff2 = self._extract(self.coeff2, t, images.shape)

            return coeff1 * (images - coeff2 * epsilon_pred)

        x0_pred = self.image_approximation(images, epsilon_pred, t)
        target_t = (t - steps).clamp(min=0)

        sqrt_alpha_hat = self._extract(self.sqrt_alphas_cumprod, target_t, images.shape)
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alphas_cumprod, target_t,
                                                 images.shape)

        return sqrt_alpha_hat * x0_pred + sqrt_one_minus_alpha_hat * epsilon_pred

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

    def epsilon_approximation(self, x_t: torch.Tensor,
                              x_0: torch.Tensor,
                              t_tensor: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor,
                                                 x_t.shape)
        return (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar

    def get_sigma(self, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """
        Getter function for sigma.

        Args:
            t: timestep.
            shape: shape of sigma.
        """
        return self._extract(self.sqrt_beta, t, shape)
