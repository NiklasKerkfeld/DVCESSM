"""Dataset class for CelebAMask dataset."""
import os
from pathlib import Path
from typing import List, Union, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.transforms import v2

import torch
from datasets.base import BaseDataset


class CelebAMaskDataset(BaseDataset):
    """Dataset class for CelebAMask dataset."""

    def __init__(self,
                 samples: List[str],
                 cfg: DictConfig,
                 augmentations: bool = False):
        super().__init__(samples, cfg, augmentations)
        self.means = torch.tensor(cfg.data.means)
        self.stds = torch.tensor(cfg.data.stds)

        self._normalize_transform = v2.Normalize(mean=self.means, std=self.stds)


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
                                   cfg=cfg,
                                   augmentations=use_augmentations,
                                   _recursive_=False)

    negative_dataset = instantiate(cfg.data.dataset,
                                   samples=negatives,
                                   cfg=cfg,
                                   augmentations=False,
                                   _recursive_=False)

    return positive_dataset, negative_dataset


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")

    attribute = "Male"

    positive_dataset, negative_dataset = get_ood_datasets(cfg, attribute, use_augmentations=False)

    print(f"Positive samples available: {len(positive_dataset)}")
    print(f"Negative samples available: {len(negative_dataset)}")

    fig, axes = plt.subplots(6, 5, figsize=(16, 25))
    fig.suptitle(attribute, fontsize=18, fontweight="bold")

    for i in range(min(15, len(positive_dataset))):
        ax = axes[i // 5, i % 5]
        image, label = positive_dataset[i]
        image = positive_dataset.denormalize(image)

        ax.imshow(image.permute(1, 2, 0) / 255.0)
        ax.axis('off')

    for i in range(min(15, len(negative_dataset))):
        ax = axes[3 + i // 5, i % 5]
        image, label = negative_dataset[i]
        image = negative_dataset.denormalize(image)

        ax.imshow(image.permute(1, 2, 0) / 255.0)
        ax.axis('off')

    fig.text(0.5, 0.95, "Positive Examples", ha="center", va="center", fontsize=14,
             fontweight="bold")
    fig.text(0.5, 0.48, "Negative Examples", ha="center", va="center", fontsize=14,
             fontweight="bold")

    # Adjust layout so titles and images don't overlap
    plt.tight_layout()
    plt.show()
