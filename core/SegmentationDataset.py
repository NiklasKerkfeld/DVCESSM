import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "data/BraTS/V2024/preprocessed"
        self.images = list(glob.glob(f"{self.path}/*/*.npz"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = np.load(self.images[index])
        image = torch.from_numpy(data["image"]).float()
        label = torch.from_numpy(data["label"]).long().squeeze(0)

        return image, label


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    channel_names = ['T1', 'T1ce', 'T2', 'FLAIR']

    dataset = SegmentationDataset()
    print(len(dataset))
    image, mask = dataset[85]

    print(image.shape)
    print(mask.shape)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('MRI Multi-Channel Analysis', fontsize=16)

    for i in range(4):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].set_title(channel_names[i])
        axes[i].axis('off')

    im_mask = axes[4].imshow(mask, cmap='gray')
    axes[4].set_title('Segmentation Mask')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()