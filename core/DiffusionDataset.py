import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        image, _ = self.raw_dataset[index]

        return image


class BraTSDiffusionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "data/BraTS/V2024/preprocessed"
        self.images = list(glob.glob(f"{self.path}/*/*.npz"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = np.load(self.images[index])

        return torch.from_numpy(data["image"]).float()


def get_MNIST_dataset(train: bool = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, 0.0),
        transforms.Normalize(0.5, 0.5)
    ])

    raw_dataset = MNIST(root='./data', train=train, download=True, transform=transform)

    return MNISTDataset(raw_dataset)


def get_BraTS_dataset():
    return BraTSDiffusionDataset()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    channel_names = ['T1', 'T1ce', 'T2', 'FLAIR']

    dataset = BraTSDiffusionDataset()
    image, mask, has_mask = dataset[10]

    print(image.shape)
    print(mask.shape)

    # 2. Initialize the figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('MRI Multi-Channel Analysis', fontsize=16)

    # 3. Plot the 4 MRI Channels
    for i in range(4):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].set_title(channel_names[i])
        axes[i].axis('off')

    # 4. Plot the Mask (using a discrete colormap for clarity)
    # 'jet' or 'viridis' work well for masks, but 'tab10' is great for categorical labels
    im_mask = axes[4].imshow(mask[0], cmap='gray')
    axes[4].set_title('Segmentation Mask')
    axes[4].axis('off')

    # Add a colorbar for the mask to show label values
    # plt.colorbar(im_mask, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
