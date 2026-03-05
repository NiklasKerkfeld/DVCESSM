import torch
from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, images: torch.Tensor):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

