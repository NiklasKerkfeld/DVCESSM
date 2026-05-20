"""Dataset class for BraTS dataset."""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset class for BraTS dataset."""
    def __init__(self, samples: List[str]):
        super().__init__()
        self.files = samples

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index])
        image = torch.from_numpy(data["image"]).float()
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        return image, label
