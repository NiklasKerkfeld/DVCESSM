import glob
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CelebAMaskDataset(Dataset):
    def __init__(self, samples: List[str], means: List[float], stds: List[float]):
        super().__init__()
        self.files = samples

        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

        self.normalize = v2.Normalize(mean=means, std=stds)
        self.denormalize = v2.Compose([
            v2.Normalize(mean=torch.zeros_like(self.means), std=1/self.stds),
            v2.Normalize(mean=-self.means, std=torch.ones_like(self.stds)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        image = torch.from_numpy(data['image']).float() # / 255.0
        image = self.normalize(image)
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        return image, label

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        return self.denormalize(image)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    files = glob.glob(f"data/CelebAMask/preprocessed/*")
    train_samples, test_samples = train_test_split(files, test_size=0.05, random_state=42)

    mean = [0.5170, 0.4166, 0.3633]
    std = [0.2652, 0.2379, 0.2308]
    dataset = CelebAMaskDataset(test_samples, mean, std)

    image = dataset[0]

    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    plt.imshow(dataset.denormalize(image).permute(1, 2, 0))
    plt.show()