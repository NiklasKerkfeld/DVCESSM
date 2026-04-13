from typing import Literal

import torch
from torch.utils.tensorboard import SummaryWriter


class SuperTrainer:
    def __init__(self,
                 device: torch.device,
                 name: str = 'Trainer',
                 log_dir: str = './runs',
                 checkpoint: str = "./checkpoints",
                 image_type: Literal['mri', 'rgb'] = 'mri'
                 ):
        self.device = device
        self.name = name
        self.log_dir = log_dir
        self.checkpoint = checkpoint
        self.image_type = image_type

        self.epoch = 0

        self.writer = SummaryWriter(f'{self.log_dir}/{self.name}')

    def save_checkpoint(self):
        pass

    def load_checkpoint(self, checkpoint_path: str):
        pass

    def _visualize_image(self, image):
        if self.image_type == 'mri':
            return image[:, 0, None]
        else:
            return image
