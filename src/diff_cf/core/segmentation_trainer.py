"""Trainer for the Segmentation model."""

import os
from typing import Optional, Tuple

from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from ..utils.utils import visualize_seg
from .trainer import BaseTrainer


class SegmentationTrainer(BaseTrainer):
    """
    Trainer for the Segmentation model.
    """

    def __init__(self,
                 train_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
                 valid_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
                 model: nn.Module,
                 device: torch.device,
                 cfg: DictConfig,
                 name: str = 'SegmentationTrainer',
                 log_dir: str = '.logs/'
                 ) -> None:
        """
        Args:
            train_dataset: Dataset with training data.
            valid_dataset: Dataset with validation data.
            model: Model to be trained.
            device: device to run on.
            cfg: Configuration dictionary.
            name: name of the process.
            log_dir: Path to logging folder.
        """
        super().__init__(train_dataset, device, cfg, name, log_dir)
        self.model = model
        self.lr = cfg.segmentation.lr
        self.num_classes = cfg.data.n_classes
        self.checkpoint = cfg.segmentation.checkpoint_path

        # parameter for robust training
        self.robust = cfg.segmentation.robust.use
        self.epsilon = cfg.segmentation.robust.epsilon
        self.steps = cfg.segmentation.robust.steps
        self.alpha = cfg.segmentation.robust.alpha
        self.alpha = 2.5 * (self.epsilon / self.steps) if self.alpha is None else self.alpha
        self.start_epoch = cfg.segmentation.robust.start_epoch
        self.warm_up = cfg.segmentation.robust.warm_up

        self.loss_fn = CrossEntropyLoss()

        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=cfg.data.batch_size,
                                           shuffle=False,
                                           num_workers=cfg.data.num_workers)

    def save_checkpoint(self) -> None:
        """Saves current state in checkpoint."""
        os.makedirs(self.checkpoint, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, f'{self.checkpoint}/{self.name}.pt')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads given checkpoint.

        Args:
            checkpoint_path: path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    def train(self, epochs: int) -> None:
        """
        Trains model for given epochs.

        Args:
            epochs: number of trainings epochs.

        Returns:
            None
        """
        images = self._log_examples(self.valid_dataloader)

        self.model.train()

        for self.epoch in range(self.epoch + 1, epochs + 1):
            self.train_one_epoch()
            self.evaluate()

            self.log_prediction(images)

            self.save_checkpoint()

    def train_one_epoch(self) -> None:
        """Trains one epoch."""
        self.model.train()
        running_loss = 0

        progress_bar = tqdm(self.train_dataloader, total=len(self.train_dataloader),
                            desc=f"Epoch {self.epoch}")
        for images, mask in progress_bar:
            images, mask = images.to(self.device), mask.to(self.device)

            images = self._projected_gradient_descent(images, mask)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, mask)  # .unsqueeze(1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            progress_bar.set_postfix(loss=loss.detach().item())

        self.writer.add_scalar('loss', (running_loss / len(self.train_dataloader)),
                               global_step=self.epoch)

    def _projected_gradient_descent(self,
                                    images: torch.Tensor,
                                    mask: torch.Tensor,
                                    eps: Optional[float] = None) -> torch.Tensor:
        """
        Adapts image with projected gradient descent.

        Args:
            images: batch on images (B, C, H, W).
            mask: batch of target masks (B, H, W).
            eps: Epsilon (normally None as self.epsilon is used)

        Returns:
            Adapted images.
        """
        # if training is not robust skip
        if not self.robust:
            return images

        # if training in before start of robust skip
        if self.epoch <= self.start_epoch and eps is None:
            return images

        # set eps in warmup phase
        if self.epoch <= self.start_epoch + self.warm_up and eps is None:
            eps = self.epsilon * ((self.epoch - self.start_epoch) / self.warm_up)

        # set eps after warmup phase
        elif eps is None:
            eps = self.epsilon

        # calc adversarial example perturbation
        random_noise = torch.randn_like(images, device=images.device) * (eps / 2)
        adv_perturbation = random_noise.requires_grad_(True)

        self.model.eval()

        for _ in range(self.steps):
            adv_example = images + adv_perturbation
            outputs = self.model(adv_example)
            loss = self.loss_fn(outputs, mask)
            grad = \
                torch.autograd.grad(loss, adv_perturbation, retain_graph=False, create_graph=False)[
                    0]

            grad_norms = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1) + 1e-10
            grad = grad / grad_norms.view(-1, 1, 1, 1)  # Normalize gradient

            with torch.no_grad():
                adv_perturbation += self.alpha * grad

                pert_norm = torch.norm(adv_perturbation.view(adv_perturbation.shape[0], -1), p=2,
                                       dim=1)
                scaling_factor = torch.clamp(eps / (pert_norm + 1e-10), max=1.0)
                adv_perturbation *= scaling_factor.view(-1, 1, 1, 1)

            adv_perturbation.requires_grad_(True)

        self.model.train()

        return (images + adv_perturbation).detach()

    @torch.no_grad()
    def evaluate(self) -> None:
        """
        Evaluates model and logs results on tensorboard.
        """
        self.model.eval()
        conf_matrix = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        with torch.no_grad():
            for images, labels in tqdm(self.valid_dataloader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)

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

    def _log_examples(self,
                      valid_dataloader: DataLoader[
                          Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Logs examples to tensorboard.
        Args:
            valid_dataloader: validation dataloader for examples.

        Returns:
            batch of example images (B, C, H, W).
        """
        images = torch.stack([valid_dataloader.dataset[i][0] for i in range(5)])
        targets = torch.stack([valid_dataloader.dataset[i][1] for i in range(5)])
        adv_perturbation = self._projected_gradient_descent(images.to(self.device),
                                                            targets.to(self.device),
                                                            eps=self.epsilon)

        if self.image_type == 'mri':
            image_grid = make_grid(images.flatten(0, 1)[:, None], nrow=4, normalize=True)
            adv_grid = make_grid(adv_perturbation.flatten(0, 1)[:, None], nrow=4, normalize=True)
        else:
            image_grid = make_grid(images, nrow=5, normalize=True)
            adv_grid = make_grid(adv_perturbation, nrow=5, normalize=True)

        target_grid = make_grid([visualize_seg(t) for t in targets.unsqueeze(1)], nrow=5, normalize=True)

        self.writer.add_image('images', image_grid)
        self.writer.add_image('adversarial', adv_grid)
        self.writer.add_image('targets', target_grid)

        return images

    @torch.no_grad()
    def log_prediction(self, images: torch.Tensor) -> None:
        """
        Logs current predictions of examples to tensorboard.

        Args:
            images: Batch of examples (B, C, H, W).

        Returns:
            None
        """
        self.model.eval()

        out = self.model(images.to(self.device))
        pred = torch.argmax(out, dim=1).unsqueeze(1)

        self.writer.add_image(
            'predictions',
            make_grid([visualize_seg(p) for p in pred], nrow=5, normalize=True),
            global_step=self.epoch)
