import torch
from torch import Tensor
from torch.utils.data import Dataset


class ClassificatorDataset(Dataset):
    def __init__(self,
                 raw_dataset,
                 image_width: int,
                 image_height: int,
                 std: float = 0.0,
                 p: float = 0.0,
                 p_block: float = 0.0,
                 max_block_width: int = 0,
                 max_block_height: int = 0):
        self.raw_dataset = raw_dataset
        self.width = image_width
        self.height = image_height
        self.std = std
        self.p = p
        self.p_block = p_block
        self.max_block_width = max_block_width
        self.max_block_height = max_block_height

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        img, label = self.raw_dataset[idx]
        # Apply your noise logic to img here
        img = self._process(img)

        return img, label


    def _process(self, image: Tensor) -> Tensor:
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
