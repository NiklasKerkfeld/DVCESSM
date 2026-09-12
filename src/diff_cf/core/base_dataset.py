"""Function to handle dataset instances."""

import glob
from typing import Tuple, List, Optional, Sequence
from pathlib import Path

from torchvision.transforms import v2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,
                 root: str,
                 samples: List[str],
                 means: list[float],
                 stds: list[float],
                 min: float,
                 max: float,
                 augmentations: bool = False,
                 probablity_blur: float = 0.2,
                 blur_sigma: Sequence[float] = [0.2, 2.0],
                 blur_kernel: int = 3,
                 probablity_noise: float = 0.2,
                 noise_sigma: Sequence[float] = [0.2, 2]
                 ) -> None:
        self.root = root
        self.files = samples

        self.augmentations = augmentations
        self.blur_p = probablity_blur
        self.blur_sigma = [float(x) for x in blur_sigma]
        self.blur_kernel = blur_kernel
        self.noise_p = probablity_noise
        self.noise_sigma = [float(x) for x in noise_sigma]

        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)
        self.z_min = ((min - self.means) / self.stds).view(3, 1, 1)
        self.z_max = ((max - self.means) / self.stds).view(3, 1, 1)

        self._normalize_transform: Optional[v2.Transform] = None

    def get_filepath(self, index: int) -> str:
        """Returns the raw file path for a given index."""
        return self.files[index]

    def find_index(self, path_or_stem: str) -> Optional[int]:
        """Finds the dataset index for a given file path or stem. Returns None if not present."""
        target_stem = Path(path_or_stem).stem
        for idx, file_path in enumerate(self.files):
            if Path(file_path).stem == target_stem:
                return idx
        return None

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the raw file and returns (image, label) tensors."""
        data = np.load(self.files[index])
        image = torch.from_numpy(data["image"]).float()
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        if self._normalize_transform is None:
            sample_mean = torch.from_numpy(data["mean"]).float()
            sample_std = torch.from_numpy(data["std"]).float()
            image = v2.functional.normalize(image, mean=sample_mean, std=sample_std)
        else:
            image = self._normalize_transform(image)

        if self.augmentations:
            if torch.rand((1,)) > self.blur_p:
                image = v2.functional.gaussian_blur(image,
                                                    [self.blur_kernel, self.blur_kernel],
                                                    self.blur_sigma)
            if torch.rand((1,)) > self.noise_p:
                image = v2.functional.gaussian_noise(
                    image,
                    sigma=torch.empty(1).uniform_(self.noise_sigma[0], self.noise_sigma[1]).item(),
                    clip=False
                )

        return image, label

    def _stats(self, index: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if index is None:
            if self.means is None or self.stds is None:
                raise ValueError("This dataset uses dynamic per-file normalization.")
            return self.means, self.stds

        return self.means, self.stds

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._load(index)

    def denormalize(self, image: torch.Tensor, index: Optional[int] = None) -> torch.Tensor:
        means, stds = self._stats(index)
        means, stds = means.to(image.device), stds.to(image.device)

        if image.dim() == 4:
            m, s = means.view(1, -1, 1, 1), stds.view(1, -1, 1, 1)
        elif image.dim() == 3:
            m, s = means.view(-1, 1, 1), stds.view(-1, 1, 1)
        else:
            raise ValueError(f"Unexpected image tensor shape: {image.shape}.")

        return (image * s) + m


def get_splits(root: str) -> Tuple[BaseDataset, BaseDataset]:
    files = sorted(glob.glob(f"{root}/preprocessed/*/*.npz"))

    train_samples, test_samples = train_test_split(
        files,
        test_size=0.05,
        random_state=42
    )

    return train_samples, test_samples