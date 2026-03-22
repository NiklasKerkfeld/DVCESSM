import glob
from typing import Literal, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SuperDataset(Dataset):
    def __init__(self, samples: List[str]):
        super().__init__()
        self.files = []
        for sample in samples:
            self.files.extend(glob.glob(f"{sample}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        return data


class DiffusionDataset(SuperDataset):
    def __init__(self, samples: List[str]):
        super().__init__(samples)

    def __getitem__(self, index):
        data = np.load(self.files[index])

        return torch.from_numpy(data["image"]).float()


class SegmentationDataset(Dataset):
    def __init__(self, samples: List[str]):
        super().__init__(samples)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        image = torch.from_numpy(data["image"]).float()
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        return image, label


def get_datasets(task: Literal['Diffusion', 'Segmentation'], test_size:float=0.05, random_state:int=42):
    dataset = DiffusionDataset if task == 'Diffusion' else SegmentationDataset

    files = glob.glob(f"data/BraTS/V2024/preprocessed/*")
    train_samples, test_samples = train_test_split(files, test_size=test_size, random_state=random_state)


    return dataset(train_samples), dataset(test_samples)


if __name__ == "__main__":
    train_dataset, test_dataset = get_datasets('Diffusion')
    print(len(train_dataset))
    print(len(test_dataset))
