import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, samples: List[str]):
        super().__init__()
        self.files = []
        for sample in samples:
            self.files.extend(glob.glob(f"{sample}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        image = torch.from_numpy(data["image"]).float()
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        return image, label

