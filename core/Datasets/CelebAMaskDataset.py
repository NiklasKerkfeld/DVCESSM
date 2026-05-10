import glob
from typing import List, Union, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import GaussianBlur, RandomApply


class CelebAMaskDataset(Dataset):
    def __init__(self,
                 samples: List[str],
                 means: List[float],
                 stds: List[float],
                 prob_blur: float = 0.0,
                 kernel_size: int = 3,
                 sigma: Union[float, Tuple[float, float]] = (0.0, 3.0)):
        super().__init__()
        self.files = samples

        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

        self.normalize = v2.Normalize(mean=means, std=stds)

        self.augmentations = RandomApply(torch.nn.ModuleList([
            GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        ]), p=prob_blur) if prob_blur > 0 else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        image = torch.from_numpy(data['image']).float() / 255.0
        image = self.normalize(image)
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        if self.augmentations is not None:
            image = self.augmentations(image)

        return image, label

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        means = self.means.view(-1, 1, 1).to(image.device)
        stds = self.stds.view(-1, 1, 1).to(image.device)
        return (image * stds) + means


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    files = glob.glob(f"data/CelebAMask/preprocessed/*/*.npz")
    train_samples, test_samples = train_test_split(files, test_size=0.05, random_state=42)

    mean = [0.5170, 0.4166, 0.3633]
    std = [0.2652, 0.2379, 0.2308]

    augmentations = GaussianBlur(15, sigma=(5.0))

    dataset = CelebAMaskDataset(test_samples, mean, std, augmentations)

    image, _ = dataset[0]

    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    plt.imshow(dataset.denormalize(image).permute(1, 2, 0))
    plt.show()

    image -= torch.amin(image, dim=(1, 2))[:, None, None]
    image /= torch.amax(image, dim=(1, 2))[:, None, None]

    plt.imshow(image.permute(1, 2, 0))
    plt.show()
