"""Function to handle dataset instances."""

import glob
from typing import Tuple, List, Optional

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
                 ) -> None:
        self.files = samples

        # parameter augmentations
        self.augmentations = augmentations
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
        """Override in subclass if stats must be fetched dynamically per file."""
        if index is None:
            if self.means is None or self.stds is None:
                raise ValueError(
                    "This dataset uses dynamic per-file normalization. "
                    "You must provide an explicit 'index' to look up metadata stats."
                )
            return self.means, self.stds

        if self.means is None or self.stds is None:
            data = np.load(self.files[index])
            sample_mean = torch.from_numpy(data["mean"]).float()
            sample_std = torch.from_numpy(data["std"]).float()
            return sample_mean, sample_std

        return self.means, self.stds

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self._load(index)

        return image, label

    def denormalize(self, image: torch.Tensor, index: Optional[int] = None) -> torch.Tensor:
        """
        Reverses normalization for both single images (C, H, W) and batches (B, C, H, W).
        If an index is provided, it checks for per-sample stats. Otherwise,
        it falls back to global stats.
        """
        # Assuming self._stats(index) returns tensors of shape (C,)
        means, stds = self._stats(index)

        # Send stats to the correct device
        means = means.to(image.device)
        stds = stds.to(image.device)

        # Dynamically match shapes for broadcasting
        if image.dim() == 4:  # Batch case: (B, C, H, W)
            m = means.view(1, -1, 1, 1)
            s = stds.view(1, -1, 1, 1)
        elif image.dim() == 3:  # Single image case: (C, H, W)
            m = means.view(-1, 1, 1)
            s = stds.view(-1, 1, 1)
        else:
            raise ValueError(
                f"Unexpected image tensor shape: {image.shape}. Expected 3 or 4 dimensions.")

        return (image * s) + m


def get_datasets(cfg: DictConfig,
                 use_augmentations: bool = False) -> Tuple[BaseDataset, BaseDataset]:
    """
    cfg is the full OmegaConf object from Hydra
    """
    files = glob.glob(f"{cfg.data.root}/preprocessed/*/*.npz")

    train_samples, test_samples = train_test_split(
        files,
        test_size=0.05,
        random_state=42
    )

    train_dataset = instantiate(cfg.data.dataset,
                                samples=train_samples,
                                cfg=cfg,
                                augmentations=use_augmentations,
                                _recursive_=False)

    test_dataset = instantiate(cfg.data.dataset,
                               samples=test_samples,
                               cfg=cfg,
                               augmentations=False,
                               _recursive_=False)

    return train_dataset, test_dataset


@hydra.main(version_base=None, config_path="./../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataset, test_dataset = get_datasets(cfg)
    print(len(train_dataset))
    print(len(test_dataset))


if __name__ == "__main__":
    main()

