"""Dataset class for BraTS dataset."""

from typing import List

from omegaconf import DictConfig

from datasets.base import BaseDataset


class BraTSDataset(BaseDataset):
    """Dataset class for BraTS dataset."""

    def __init__(self,
                 samples: List[str],
                 cfg: DictConfig,
                 augmentations: bool = False):
        super().__init__(samples, cfg, augmentations)
