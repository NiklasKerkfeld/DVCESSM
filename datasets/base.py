"""Function to handle dataset instances."""

import glob
from typing import Tuple, List, Optional
from pathlib import Path

import hydra
from torchvision.transforms import v2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,
                 samples: List[str],
                 cfg: DictConfig,
                 augmentations: bool = False,
                 label: Optional[str] = None,
                 use_positive: bool = True,
                 ) -> None:
        self.cfg = cfg
        self.augmentations = augmentations
        self.all_samples = samples

        self.label = label
        self.use_positive = use_positive
        self.files = self._filter_samples(self.all_samples)
        # Filter samples (overridden in subclasses like CelebAMaskDataset)
        self.files = self._filter_samples(samples)

        self.blur_p = cfg.segmentation.augmentations.blur.p
        self.blur_sigma = [cfg.segmentation.augmentations.blur.sigma_min,
                           cfg.segmentation.augmentations.blur.sigma_max]
        self.blur_kernel = cfg.segmentation.augmentations.blur.kernel
        self.noise_p = cfg.segmentation.augmentations.noise.p
        self.noise_sigma = [cfg.segmentation.augmentations.noise.sigma_min,
                            cfg.segmentation.augmentations.noise.sigma_max]

        self.means: Optional[torch.Tensor] = None
        self.stds: Optional[torch.Tensor] = None
        self._normalize_transform: Optional[v2.Transform] = None

    def _filter_samples(self, samples: List[str]) -> List[str]:
        """Override in subclasses to filter samples by attribute labels."""
        return samples

    def set_attribute(self, label: Optional[str], use_positive: bool = True) -> None:
        """Dynamically switch attribute filtering without rebuilding the dataset object."""
        self.label = label
        self.use_positive = use_positive
        self.files = self._filter_samples(self.all_samples)

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
                image = v2.functional.gaussian_blur(image, self.blur_kernel, self.blur_sigma)
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

        if self.means is None or self.stds is None:
            data = np.load(self.files[index])
            sample_mean = torch.from_numpy(data["mean"]).float()
            sample_std = torch.from_numpy(data["std"]).float()
            return sample_mean, sample_std

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


def get_datasets(cfg: DictConfig,
                 use_augmentations: bool = False,
                 label: Optional[str] = None,
                 use_positive: bool = True) -> Tuple[BaseDataset, BaseDataset]:
    files = sorted(glob.glob(f"{cfg.data.root}/preprocessed/*/*.npz"))

    train_samples, test_samples = train_test_split(
        files,
        test_size=0.05,
        random_state=42
    )

    train_dataset = instantiate(cfg.data.dataset,
                                samples=train_samples,
                                cfg=cfg,
                                augmentations=use_augmentations,
                                label=label,
                                use_positive=use_positive,
                                _recursive_=False)

    test_dataset = instantiate(cfg.data.dataset,
                               samples=test_samples,
                               cfg=cfg,
                               augmentations=False,
                               label=label,
                               use_positive=use_positive,
                               _recursive_=False)

    print(f"Created training dataset ({len(train_dataset)}) & test dataset ({len(test_dataset)}) "
          f"[label='{label}', positive={use_positive}]")

    return train_dataset, test_dataset