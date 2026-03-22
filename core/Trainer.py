import torch
from torch.utils.tensorboard import SummaryWriter


class SuperTrainer:
    def __init__(self,
                 device: torch.device,
                 name: str = 'Trainer',
                 log_dir: str = './runs',
                 checkpoint: str = "./checkpoints"
                 ):
        self.device = device
        self.name = name
        self.log_dir = log_dir
        self.checkpoint = checkpoint

        self.epoch = 0

        self.writer = SummaryWriter(f'{self.log_dir}/{self.name}')

    def save_checkpoint(self):
        pass

    def load_checkpoint(self, checkpoint_path: str):
        pass

