import os
from omegaconf import DictConfig

import torch
from monai.losses import DiceLoss
from torch import nn, Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from core.Trainer import SuperTrainer


class Trainer(SuperTrainer):
    def __init__(self,
                 train_dataset: Dataset,
                 valid_dataset: Dataset,
                 model: nn.Module,
                 device: torch.device,
                 cfg: DictConfig,
                 name: str = 'SegmentationTrainer',
                 log_dir: str = '.logs/'
                 ):
        super().__init__(train_dataset, device, cfg, name, log_dir)
        self.model = model
        self.lr = cfg.segmentation.lr
        self.num_classes = model.out_channels

        self.loss_fn = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)

        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=cfg.data.batch_size,
                                           shuffle=False,
                                           num_workers=cfg.data.num_workers)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, f'{self.checkpoint}/{self.name}.pt')

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    def train(self, epochs: int):
        images = self._log_examples(self.valid_dataloader)

        self.model.train()

        for self.epoch in range(1, epochs + 1):
            self.train_one_epoch(self.train_dataloader, self.optimizer)
            self.evaluate(self.valid_dataloader, self.device)

            self.log_samples(images)

            self.save_checkpoint()

    def _log_examples(self, valid_dataloader: DataLoader) -> Tensor:
        images = torch.stack([valid_dataloader.dataset[i][0] for i in range(5)])
        targets = torch.stack([valid_dataloader.dataset[i][1] for i in range(5)])

        if self.image_type == 'mri':
            image_grid = make_grid(images.flatten(0, 1)[:, None], nrow=4, normalize=True)
        else:
            image_grid = make_grid(images, nrow=5, normalize=True)

        target_grid = make_grid(targets.unsqueeze(1).float(), nrow=5, normalize=True)

        self.writer.add_image('images', image_grid)
        self.writer.add_image('targets', target_grid)

        return images

    def train_one_epoch(self, dataloader: DataLoader, optimizer):
        self.model.train()
        running_loss = 0

        bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {self.epoch}")
        for images, mask in bar:
            images, mask = images.to(self.device), mask.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, mask.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            bar.set_postfix(loss=loss.detach().item())

        self.writer.add_scalar('loss', (running_loss / len(dataloader)), global_step=self.epoch)

    @torch.no_grad()
    def evaluate(self, dataloader, device):
        self.model.eval()
        # Initialize a confusion matrix: [Target, Prediction]
        conf_matrix = torch.zeros((self.num_classes, self.num_classes), device=device)

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                # Mask out ignore_index if you have one (e.g., 255)
                mask = (labels >= 0) & (labels < self.num_classes)

                # Efficiently update confusion matrix
                conf_matrix += torch.bincount(
                    self.num_classes * labels[mask].view(-1) + preds[mask].view(-1),
                    minlength=self.num_classes ** 2
                ).reshape(self.num_classes, self.num_classes)

        # Calculate Metrics
        tp = conf_matrix.diag()
        fp = conf_matrix.sum(dim=0) - tp
        fn = conf_matrix.sum(dim=1) - tp

        # Avoid division by zero
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)

        for i in range(self.num_classes):
            self.writer.add_scalar(f'class_{i}/iou', iou[i], global_step=self.epoch)
            self.writer.add_scalar(f'class_{i}/precision', precision[i], global_step=self.epoch)
            self.writer.add_scalar(f'class_{i}/recall', recall[i], global_step=self.epoch)
            self.writer.add_scalar(f'class_{i}/f1', f1[i], global_step=self.epoch)

    @torch.no_grad()
    def log_samples(self, images: torch.Tensor):
        self.model.eval()

        out = self.model(images.to(self.device))
        out = torch.argmax(out, dim=1)
        out = out.unsqueeze(1).float()

        self.writer.add_image(
            'predictions',
            make_grid(out, nrow=5, normalize=True),
            global_step=self.epoch)
