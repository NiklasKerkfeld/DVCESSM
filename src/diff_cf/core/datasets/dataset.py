"""Function to handle dataset instances."""

import glob
from typing import Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def get_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset]:
    """
    cfg is the full OmegaConf object from Hydra
    """
    files = glob.glob(f"{cfg.data.root}/preprocessed/*/*.npz")

    train_samples, test_samples = train_test_split(
        files,
        test_size=0.05,
        random_state=42
    )

    train_dataset = instantiate(cfg.data.dataset, samples=train_samples, train=True)
    test_dataset = instantiate(cfg.data.dataset, samples=test_samples, train=False)

    return train_dataset, test_dataset
