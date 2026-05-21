"""Abstract class for all Trainer."""
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class SuperTrainer:
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

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads given checkpoint.

        Args:
            checkpoint_path: path to checkpoint.
        """

    def _visualize_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_type == 'mri':
            return image[:, 0, None]

        return self.dataset.denormalize(image)
