from typing import List, Tuple

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class StandardDataset(Dataset):
    def __init__(self, images: torch.Tensor, std: float = 0.0, p: float = 0.0, p_block: float = 0.0, max_block_height: int = None, max_block_width: int = None):
        super(StandardDataset, self).__init__()
        if images.dim() == 3:
            self.images = images[:, None]
        else:
            self.images = images
        self.std = std
        self.p = p
        self.p_block = p_block
        self.max_block_width = max_block_width if max_block_width is not None else self.images.shape[2]
        self.max_block_height = max_block_height if max_block_height is not None else self.images.shape[3]
        self.length, self.channel, self.width, self.height = self.images.shape

        assert self.max_block_width <= self.width, "max block width must be <= width of the images"
        assert self.max_block_height <= self.height, "max block height must be <= height of the images"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self._process(image)
        return image

    def _process(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clone()

        # add black block augmentation
        if torch.rand(1) < self.p_block:
            block_h = torch.randint(1, self.max_block_height, (1, )).item()
            block_w = torch.randint(1, self.max_block_width, (1, )).item()
            top = torch.randint(0, self.height - block_h, (1, )).item()
            left = torch.randint(0, self.width - block_w, (1, )).item()

            image[:, top:top + block_h, left:left + block_w] = 0

        # Gaussian noise
        if self.std != 0.0:
            image += torch.randn_like(image) * self.std
            image = torch.clamp(image, 0, 1)

        # salt and paper noise
        if self.p != 0.0:
            noise = torch.randint_like(image, 1000) < (1000 * self.p)
            prob = torch.randint_like(image, 2)

            image[noise] = prob[noise]

        # return normalized result
        return (image - 0.5) / 0.5

class DiscriminatorDataset(StandardDataset):
    def __init__(self, images: torch.Tensor, limit: int = 3, std: float = 0.0, p: float = 0.0, p_block: float = 0.0, max_block_height: int = None, max_block_width: int = None):
        super(DiscriminatorDataset, self).__init__(images, std, p, p_block, max_block_height, max_block_width)
        self.limit = limit
        self.fake_images = []
        self.add_count = 0

    def __getitem__(self, idx):
        if idx <= (len(self.images) * min(self.limit, self.add_count)):
            image = self.images[idx % len(self.images)]
            target = 0
        else:
            image = self.fake_images[
                idx - (len(self.images) * min(self.limit, self.add_count))]
            target = 1

        image = self._process(image)

        return image, target

    def __len__(self):
        return (2 * min(self.limit, self.add_count)) * len(self.real_images)

    def add_fakes(self, new_fakes: List[torch.Tensor]) -> None:
        assert len(new_fakes) == len(
            self.real_images), f"{len(new_fakes)} != {len(self.real_images)}"

        if self.add_count >= self.limit:
            remove_idx = numpy.random.choice(np.arange(len(self.fake_images)),
                                             len(self.real_images))
            for idx in sorted(remove_idx, reverse=True):
                del self.fake_images[idx]

        self.fake_images.extend(new_fakes)
        self.add_count += 1


class ClassificatorDataset(StandardDataset):
    def __init__(self, images: torch.Tensor, targets: torch.Tensor, std: float = 0.0, p: float = 0.0, p_block: float = 0.0, max_block_height: int = None, max_block_width: int = None):
        super(ClassificatorDataset, self).__init__(images, std, p, p_block, max_block_height, max_block_width)
        self.targets = targets

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        image = self._process(image)
        return image, target


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([train_dataset[i][0] for i in range(len(train_dataset))])
    targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    print(images.shape, targets.shape)

    dataset = ClassificatorDataset(images, targets,
                                   std=0.1,
                                   p=0.1,
                                   p_block=1.0,
                                   max_block_height=10,
                                   max_block_width=10)

    for i in range(10):
        image, label = dataset[i]
        plt.imshow((image[0] * 0.5 + 0.5).clip(0.0, 1.0), cmap='gray')
        plt.title(f"label: {str(label.item())}")
        plt.show()


