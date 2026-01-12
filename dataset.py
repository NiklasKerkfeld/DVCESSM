from typing import List

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class DiscriminatorDataset(Dataset):
    def __init__(self, real_images: List[torch.Tensor], limit: int = 3):
        super(DiscriminatorDataset, self).__init__()
        self.limit = limit
        self.real_images = real_images
        self.fake_images = []
        self.add_count = 0

    def __getitem__(self, idx):
        if idx < (len(self.real_images) * self.limit):
            return self.real_images[idx % len(self.real_images)], 0
        else:
            return self.fake_images[idx - (len(self.real_images) * min(self.limit, self.add_count))], 1

    def __len__(self):
        return (2 *  self.limit) * len(self.real_images)

    def add_fakes(self, fake_images):
        assert len(fake_images) == len(self.real_images), f"{len(fake_images)} != {len(self.real_images)}"

        if self.add_count >= self.limit:
            remove_idx = numpy.random.choice(np.arange(len(self.fake_images)), len(self.real_images))
            for idx in sorted(remove_idx, reverse=True):
                del self.fake_images[idx]

        self.fake_images.extend(fake_images)
        self.add_count += 1

   
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = [train_dataset[i][0] for i in range(len(train_dataset))]

    dataset = DiscriminatorDataset(images)
    dataset.add_fakes(fake_images=images)
    dataset.add_fakes(fake_images=images)

    print(len(dataset))
    image, label = dataset[0]
    print(image.shape, label)

    image, label = dataset[180000]
    print(image.shape, label)

    image, label = dataset[240000]
    print(image.shape, label)
