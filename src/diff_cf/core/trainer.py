"""Abstract class for all Trainer."""
from typing import Tuple, List, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class BaseTrainer:
    """
    Abstract class for all Trainer.
    """
    def __init__(self,
                 dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
                 device: torch.device,
                 cfg: DictConfig,
                 name: str = 'SuperTrainer',
                 log_dir: str = '.logs/'
                 ) -> None:
        """
        Args:
            dataset: Dataset for training
            device: device to run on.
            cfg: Configuration dictionary.
            name: name of the process.
            log_dir: Path to logging folder.
        """
        self.dataset = dataset
        self.device = device
        self.name = name
        self.log_dir = log_dir
        self.checkpoint = "./checkpoints"
        self.image_type = cfg.data.image_type

        self.epoch = 0

        self.train_dataloader = DataLoader(dataset=dataset,
                                           batch_size=cfg.data.batch_size,
                                           shuffle=True,
                                           num_workers=cfg.data.num_workers)

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

    def _log_images(self, images: torch.Tensor | List[torch.Tensor], label: str):
        grid = make_grid(images, nrow=4, normalize=True)
        self.writer.add_image(label, grid, global_step=self.epoch)
