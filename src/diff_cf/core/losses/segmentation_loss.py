"""Loss functions for Counterfactual generation."""
from dataclasses import dataclass
from typing import Literal, Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SegmentationLossConfig:
    """Static hyperparameter configuration for CounterfactualLoss."""
    border_width: int = 2
    use_mask: bool = True
    reduction: Literal['mean', 'sum'] = 'mean'


class SegmentationLoss(nn.Module):
    """
    Loss functions for Counterfactual generation.
    """

    def __init__(self,
                 target: torch.Tensor,
                 config: SegmentationLossConfig):
        """
        Loss functions for Counterfactual generation.

        Args:
            target: target segmentation map.
            config: CounterfactualLossConfig with the parameter border_width, use_mask and reduction.
        """
        super().__init__()
        self.target = target.clone()
        self.border_width = config.border_width
        self.use_mask = config.use_mask
        self.reduction: Callable = torch.sum if config.reduction == 'sum' else torch.mean

        self._set_boarders()

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

    def _set_boarders(self) -> None:
        if self.border_width > 0:
            k_size = self.border_width + 1

            dilated = F.max_pool2d(self.target.float(), kernel_size=k_size, stride=1,
                                   padding=self.border_width // 2)

            eroded = -F.max_pool2d(-self.target.float(), kernel_size=k_size, stride=1,
                                   padding=self.border_width // 2)

            mask = dilated != eroded

            self.target[mask] = -1


class SegmentationLossFactory:
    """Factory that creates preprocessed CounterfactualLoss instances on demand."""

    def __init__(self, config: SegmentationLossConfig = SegmentationLossConfig()):
        self.config = config

    def __call__(self, target: torch.Tensor) -> SegmentationLoss:
        return SegmentationLoss(target=target, config=self.config)



