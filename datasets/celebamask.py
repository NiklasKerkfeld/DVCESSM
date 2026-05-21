"""Dataset class for CelebAMask dataset."""


from typing import List, Union, Tuple

from torchvision.transforms import v2

import torch
from datasets.base import BaseDataset


class CelebAMaskDataset(BaseDataset):
    """Dataset class for CelebAMask dataset."""
    def __init__(self,
                 samples: List[str],
                 means: List[float],
                 stds: List[float],
                 prob_blur: float = 0.0,
                 kernel_size: int = 3,
                 sigma: Union[float, Tuple[float, float]] = (0.0, 3.0),
                 use_augmentations: bool = True):
        super().__init__(samples, prob_blur, kernel_size, sigma, use_augmentations, 255.0)
        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

        self._normalize_transform = v2.Normalize(mean=means, std=stds)