import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class SuperTrainer:
    def __init__(self,
                 dataset: Dataset,
                 device: torch.device,
                 cfg: DictConfig,
                 name: str = 'SuperTrainer',
                 log_dir: str = '.logs/'
                 ):
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

    def save_checkpoint(self):
        pass

    def load_checkpoint(self, checkpoint_path: str):
        pass

    def _visualize_image(self, image):
        if self.image_type == 'mri':
            return image[:, 0, None]
        else:
            return (image * self.std) + self.mean
