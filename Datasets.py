import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from Parameter import IN_CHANNELS, IN_HEIGHT, IN_WIDTH, BETA_MIN, BETA_MAX, T


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
            image = torch.clamp(image, -1, 1)

        # salt and paper noise
        if self.p != 0.0:
            noise = torch.randint_like(image, 1000) < (1000 * self.p)
            prob = torch.randint_like(image, 2)

            image[noise] = prob[noise]

        # return normalized result
        return image

class DiscriminatorDataset(StandardDataset):
    def __init__(self, images: torch.Tensor, limit: int = 3, std: float = 0.0, p: float = 0.0, p_block: float = 0.0, max_block_height: int = None, max_block_width: int = None):
        super(DiscriminatorDataset, self).__init__(images, std, p, p_block, max_block_height, max_block_width)
        self.limit = limit
        self.fake_images = torch.zeros((limit * len(self.images), IN_CHANNELS, IN_HEIGHT, IN_WIDTH), dtype=torch.float)
        self.add_count = 0

    def __getitem__(self, idx):
        if idx <= (len(self.images) * min(self.limit, self.add_count)):
            image = self.images[idx % len(self.images)]
            target = 0.0
        else:
            image = self.fake_images[
                idx - (len(self.images) * min(self.limit, self.add_count))]
            target = 1.0

        image = self._process(image)

        return image, target

    def __len__(self):
        return (2 * self.prop) * len(self.images)

    @property
    def prop(self):
        return min(self.limit, self.add_count)

    def add_fakes(self, new_fakes: torch.Tensor) -> None:
        assert len(new_fakes) == len(
            self.images), f"{len(new_fakes)} != {len(self.images)}"

        if self.add_count >= self.limit:
            indices = numpy.random.choice(np.arange(len(self.fake_images)),
                                             len(self.images))
        else:
            indices = numpy.arange(self.length) + (self.add_count * self.prop)

        self.fake_images[indices] = new_fakes
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


class DiffusionDataset(Dataset):
    def __init__(self, images: torch.Tensor):
        self.images = images

        self.beta = torch.linspace(BETA_MIN, BETA_MAX, T)
        self.alphas = 1.0 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        t = torch.randint(0, T, (1,))[0]
        noise = torch.randn(image.size())

        noised = torch.sqrt(self.alphas_cumprod[t]) * image + torch.sqrt(
            1 - self.alphas_cumprod[t]) * noise
        return noised, noise, t



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([train_dataset[i][0][:, None] for i in range(len(train_dataset))])
    targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    print(images.shape, targets.shape)

    dataset = DiscriminatorDataset(images)
    dataset.add_fakes(images)

    print(len(dataset))
    image, label = dataset[0]
    print(image.shape)
    print(label)

    image, label = dataset[120000]
    print(image.shape)
    print(label)

    dataset.add_fakes(images)

    print(len(dataset))
    image, label = dataset[0]
    print(image.shape)
    print(label)

    image, label = dataset[180000]
    print(image.shape)
    print(label)


