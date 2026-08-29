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




