"""Function to handle dataset instances."""

import glob
import os
from pathlib import Path
from typing import Tuple, List, Union, Optional

from torchvision.transforms import v2

import numpy as np

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,
                 samples: List[str],
                 prob_blur: float = 0.0,
                 kernel_size: int = 3,
                 sigma: Union[float, Tuple[float, float]] = (0.0, 3.0),
                 use_augmentations: bool = True,
                 scale_factor: float = 1.0) -> None:
        self.files = samples
        self.scale_factor = scale_factor

        self.means: Optional[torch.Tensor] = None
        self.stds: Optional[torch.Tensor] = None
        self._normalize_transform: Optional[v2.Transform] = None

        self.augmentations = v2.RandomApply(torch.nn.ModuleList([
            v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        ]), p=prob_blur) if (prob_blur > 0 and use_augmentations is True) else v2.Identity()

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the raw file and returns (image, label) tensors."""
        data = np.load(self.files[index])
        image = torch.from_numpy(data["image"]).float() / self.scale_factor
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        if self._normalize_transform is None:
            sample_mean = torch.from_numpy(data["mean"]).float()
            sample_std = torch.from_numpy(data["std"]).float()
            image = v2.functional.normalize(image, mean=sample_mean, std=sample_std)
        else:
            image = self._normalize_transform(image)

        image = self.augmentations(image)

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
        Reverses normalization. If an index is provided, it checks for
        per-sample stats. Otherwise, it falls back to global stats.
        """
        means, stds = self._stats(index)

        dims = [1] * (image.dim() - 1)
        m = means.view(-1, *dims).to(image.device)
        s = stds.view(-1, *dims).to(image.device)

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
                                use_augmentations=use_augmentations)

    test_dataset = instantiate(cfg.data.dataset,
                               samples=test_samples,
                               use_augmentations=False)

    return train_dataset, test_dataset


def get_ood_datasets(cfg: DictConfig, attribute: str, use_augmentations: bool = False) -> Tuple[
    BaseDataset, BaseDataset]:
    print(f"Testing attribute: {attribute}")
    print(f"Data root from config: {cfg.data.root}")

    with open(f"{cfg.data.root}/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt") as f:
        data = f.readlines()[1:]

    data = [x.replace("\n", "").split(" ") for x in data]

    attributes = data[0]
    index = attributes.index(attribute) + 2
    images = [x[0] for x in data[1:]]
    value = [x[index] for x in data[1:]]

    positives, negatives = [], []
    for filename, value in zip(images, value):
        name = f"{int(filename[:-4]):05d}"
        batch = int(name) // 2000
        out_dir = Path(f"./data/CelebAMask/preprocessed/{batch}")
        out_path = out_dir / f"{name}.npz"

        if not os.path.exists(out_path):
            continue

        if value == '1':
            positives.append(out_path)
        else:
            negatives.append(out_path)

    positive_dataset = instantiate(cfg.data.dataset,
                                   samples=positives,
                                   use_augmentations=use_augmentations)

    negative_dataset = instantiate(cfg.data.dataset,
                                   samples=negatives,
                                   use_augmentations=False)

    return negative_dataset, positive_dataset


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")

    attribute = "Young"

    negative_dataset, positive_dataset = get_ood_datasets(cfg, attribute, use_augmentations=False)

    print(f"Positive samples available: {len(positive_dataset)}")
    print(f"Negative samples available: {len(negative_dataset)}")

    fig, axes = plt.subplots(6, 5, figsize=(16, 25))
    fig.suptitle(attribute, fontsize=18, fontweight="bold")

    for i in range(min(15, len(positive_dataset))):
        ax = axes[i // 5, i % 5]
        image, label = positive_dataset[i]
        image = positive_dataset.denormalize(image)

        ax.imshow(image.permute(1, 2, 0))
        # ax.set_title(f"Positive #{i}")
        ax.axis('off')

    for i in range(min(15, len(negative_dataset))):
        ax = axes[3 + i // 5, i % 5]
        image, label = negative_dataset[i]
        image = negative_dataset.denormalize(image)

        ax.imshow(image.permute(1, 2, 0))
        # ax.set_title(f"Negative #{i}")
        ax.axis('off')

    fig.text(0.5, 0.95, "Positive Examples", ha="center", va="center", fontsize=14,
             fontweight="bold")
    fig.text(0.5, 0.48, "Negative Examples", ha="center", va="center", fontsize=14,
             fontweight="bold")

    # Adjust layout so titles and images don't overlap
    plt.tight_layout()
    plt.show()
