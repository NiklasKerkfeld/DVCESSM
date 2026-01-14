from typing import List, Tuple

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class DiscriminatorDataset(Dataset):
    def __init__(self, real_images: List[torch.Tensor], limit: int = 3, std: float = 0.0, p: float = 0.0):
        super(DiscriminatorDataset, self).__init__()
        self.limit = limit
        self.real_images = real_images
        self.fake_images = []
        self.add_count = 0
        self.std = std
        self.p = p

    def __getitem__(self, idx):
        if idx < (len(self.real_images) * min(self.limit, self.add_count)):
            image = self.real_images[idx % len(self.real_images)]
            target = 0
        else:
            image = self.fake_images[
                idx - (len(self.real_images) * min(self.limit, self.add_count))]
            target = 1

        image, target = self._process(image, target)

        return image, target

    def __len__(self):
        return (2 * min(self.limit, self.add_count)) * len(self.real_images)

    def _process(self, image: torch.Tensor, target: int) -> Tuple[torch.Tensor, int]:
        # Gaussian noise
        if self.std != 0.0:
            image += torch.randn_like(image) * self.std

        # salt and paper noise
        if self.p != 0.0:
            noise = torch.randint_like(image, 1000) < (1000 * self.p)
            prob = torch.randint_like(image, 2)

            image[noise] = prob[noise]

        # return normalized result
        return (image - 0.5) / 0.5, target

    def add_fakes(self, fake_images):
        assert len(fake_images) == len(
            self.real_images), f"{len(fake_images)} != {len(self.real_images)}"

        if self.add_count >= self.limit:
            remove_idx = numpy.random.choice(np.arange(len(self.fake_images)),
                                             len(self.real_images))
            for idx in sorted(remove_idx, reverse=True):
                del self.fake_images[idx]

        self.fake_images.extend(fake_images)
        self.add_count += 1


class ClassificatorDataset(Dataset):
    def __init__(self, images: torch.Tensor, targets: torch.Tensor, std: float = 0.0, p: float = 0.0, p_block: float = 0.0, max_block_height: int = None, max_block_width: int = None):
        super(ClassificatorDataset, self).__init__()
        if images.dim() == 3:
            self.images = images[:, None]
        else:
            self.images = images

        self.targets = targets
        self.std = std
        self.p = p
        self.p_block = p_block
        self.block_width = max_block_width if max_block_width is not None else self.images.shape[2]
        self.block_height = max_block_height if max_block_height is not None else self.images.shape[3]
        self.length, self.channel, self.width, self.height = self.images.shape

        assert self.block_width <= self.width, "max block width must be <= width of the images"
        assert self.block_height <= self.height, "max block height must be <= height of the images"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        image, target = self._process(image, target)
        return image, target

    def _process(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gaussian noise
        if self.std != 0.0:
            image += torch.randn_like(image) * self.std

        # salt and paper noise
        if self.p != 0.0:
            noise = torch.randint_like(image, 1000) < (1000 * self.p)
            prob = torch.randint_like(image, 2)

            image[noise] = prob[noise]

        # add black block augmentation
        if torch.rand(1) < self.p_block:
            block_h = torch.randint(0, self.height, (1, ))
            block_w = torch.randint(0, self.width, (1, ))
            top = torch.randint(0, self.height - block_h, (1, ))
            left = torch.randint(0, self.width - block_w, (1, ))

            image = image.clone()
            image[:, top:top + block_h, left:left + block_w] = 0

        # return normalized result
        return (image - 0.5) / 0.5, target


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([train_dataset[i][0] for i in range(len(train_dataset))])
    targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    print(images.shape, targets.shape)

    dataset = ClassificatorDataset(images, targets, std=0.1, p=0.1, p_block=1.0)
    image, label = dataset[0]

    plt.imshow(image, cmap='gray')
    plt.show()


