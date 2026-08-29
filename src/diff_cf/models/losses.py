"""Loss functions for Counterfactual generation."""

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class CounterfactualLoss(nn.Module):
    """
    Loss functions for Counterfactual generation.
    """
    def __init__(self,
                 target: torch.Tensor,
                 border_width: int = 2,
                 use_mask: bool = True,
                 reduction: Literal['mean', 'sum'] = 'mean'):
        """
        Loss functions for Counterfactual generation.

        Args:
            target: target segmentation map.
            border_width: width of ignored border between segments.
            use_mask: masked all correct segmented pixels.
            reduction: reduction method.
        """
        super().__init__()
        self.target = target
        self.border_width = border_width
        self.use_mask = use_mask
        self._set_boarders(target)
        self.reduction = torch.sum if reduction == 'sum' else torch.mean

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Loss functions for Counterfactual generation.

        Args:
            logits: output logits.

        Returns:
            loss: loss value.
        """
        loss = F.cross_entropy(logits, self.target, reduction='none', ignore_index=-1)

        if self.use_mask:
            mask = (logits.argmax(dim=1) != self.target).float()
            loss = loss * mask

        return self.reduction(loss)

    def _set_boarders(self, target: torch.Tensor) -> None:
        if self.border_width > 0:
            k_size = self.border_width + 1

            dilated = F.max_pool2d(target.float(), kernel_size=k_size, stride=1,
                                   padding=self.border_width // 2)

            eroded = -F.max_pool2d(-target.float(), kernel_size=k_size, stride=1,
                                   padding=self.border_width // 2)

            mask = dilated != eroded

            self.target[mask] = -1


def log_distance(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(approx_image - image)
    return torch.sum(torch.log(1 + diff))


def geman_mcclure_distance(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    # Geman-McClure Distance
    diff = approx_image - image
    gamma = 0.1  # Scale threshold: pixels changing more than gamma incur ~1.0 loss
    return torch.sum(diff ** 2 / (diff ** 2 + gamma ** 2))

def differentiable_soft_l0_count(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    # Differentiable Soft L0 Count
    diff = torch.abs(approx_image - image)
    epsilon = 0.05  # Ignore changes smaller than 5% pixel value
    tau = 20.0  # Steeps the transition into a smooth step function

    return torch.sum(torch.sigmoid(tau * (diff - epsilon)))

def charbonnier_loss(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    # Charbonnier Loss
    diff = approx_image - image
    eps = 1e-3  # Smoothing factor
    return torch.sum(torch.sqrt(diff ** 2 + eps ** 2))

