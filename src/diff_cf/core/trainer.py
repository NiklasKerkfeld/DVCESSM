"""Abstract class for all Trainer."""
from typing import Optional, Literal

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .base_dataset import BaseDataset


class BaseTrainer:
    """
    Abstract class for all Trainer.
    """
    def __init__(self,
                 dataset: BaseDataset,
                 device: torch.device,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 image_type: Literal['mri', 'rgb'] = 'rgb',
                 name: str = 'SuperTrainer',
                 checkpoint_path: str = './checkpoints',
                 log_dir: str = '.logs/'
                 ) -> None:
        """
        Args:
            dataset: Dataset for training
            device: device to run on.
            name: name of the process.
            log_dir: Path to logging folder.
        """
        self.dataset = dataset
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
        self.log_dir = log_dir
        self.checkpoint = checkpoint_path
        self.image_type = image_type

        self.epoch = 0

        self.train_dataloader = DataLoader(dataset=dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers)

        self.writer = SummaryWriter(f'{self.log_dir}/{self.name}')

    def save_checkpoint(self) -> None:
        """Saves current state in checkpoint."""
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads given checkpoint.

        Args:
            checkpoint_path: path to checkpoint.
        """
        raise NotImplementedError

    def _visualize_image(self, image: torch.Tensor, index: Optional[int] = None) -> torch.Tensor:
        """
        Converts the images so they can be displayed.

        Args:
            images: images of shape (C, H, W)

        Returns:

        """
        if self.image_type == 'mri':
            image = self.dataset.denormalize(image, index)
            return make_grid(image[:, None], nrow=2, normalize=True)

        return self.dataset.denormalize(image, index)

    def _log_images(self, images: torch.Tensor | list[torch.Tensor], label: str):
        grid = make_grid(images, nrow=4, normalize=True)
        self.writer.add_image(label, grid, global_step=self.epoch)
