"""Dataset class for CelebAMask dataset."""


from typing import List, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import GaussianBlur, RandomApply


class CelebAMaskDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset class for CelebAMask dataset."""
    def __init__(self,
                 samples: List[str],
                 means: List[float],
                 stds: List[float],
                 prob_blur: float = 0.0,
                 kernel_size: int = 3,
                 sigma: Union[float, Tuple[float, float]] = (0.0, 3.0),
                 train: bool = True):
        super().__init__()
        self.files = samples

        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

        self.normalize = v2.Normalize(mean=means, std=stds)

        self.augmentations = RandomApply(torch.nn.ModuleList([
            GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        ]), p=prob_blur) if (prob_blur > 0 and train is True) else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index])
        image = torch.from_numpy(data['image']).float() / 255.0
        image = self.normalize(image)
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        if self.augmentations is not None:
            image = self.augmentations(image)

        return image, label

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Reverses the z-score normalization.

        Args:
            image: image to be reversed.

        Returns:
            image: normal rgb image.
        """
        means = self.means.view(-1, 1, 1).to(image.device)
        stds = self.stds.view(-1, 1, 1).to(image.device)
        return (image * stds) + means
