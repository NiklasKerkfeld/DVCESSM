# pylint: disable=duplicate-code
"""Trainer for the Diffusion model."""

import copy
import math
import os
from typing import Optional, Literal

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

from ..base_dataset import BaseDataset
from .engine import Engine
from ..trainer import BaseTrainer
from ...models.diffusion_model import DiffusionModel
from ...utils.utils import visualize_seg


class Trainer(BaseTrainer):
    """Trainer for the Diffusion model."""
    model: DiffusionModel

    def __init__(self,
                 dataset: BaseDataset,
                 model: DiffusionModel,
                 engine: Engine,
                 device: torch.device,
                 epochs: int = 500,
                 use_target: bool = False,
                 warm_up: int = 500,
                 lr: float = 1e-4,
                 batch_size: int = 64,
                 accumulation_steps: int = 1,
                 num_workers: int = 4,
                 gamma: float = 5.0,
                 ema_decay: float = 0.999,
                 image_type: Literal['mri', 'rgb'] = 'rgb',
                 name: str = 'DiffusionTrainer',
                 checkpoint_path: str = './checkpoints/Diffusion',
                 log_dir: str = '.logs/'
                 ):
        super().__init__(dataset=dataset,
                         model=model,
                         device=device,
                         epochs=epochs,
                         lr=lr,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         image_type=image_type,
                         name=name,
                         checkpoint_path=checkpoint_path,
                         log_dir=log_dir)
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1")

        self.engine = engine
        self.gamma = gamma
        self.accumulation_steps = accumulation_steps
        self.use_target = use_target

        # --- EMA Setup ---
        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.to(self.device)
        self.ema_model.eval()

        self.model_params = list(self.model.parameters())
        self.ema_params = list(self.ema_model.parameters())
        self.compiled_model = torch.compile(self.model)

        self.engine.to(self.device)
        self.step = 0

        self.num_warmup_steps = warm_up
        updates_per_epoch = math.ceil(len(self.train_dataloader) / self.accumulation_steps)
        total_training_steps = self.epochs * updates_per_epoch

        def lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, total_training_steps - self.num_warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    @torch.no_grad()
    def update_ema(self) -> None:
        """Step the EMA weights."""
        current_decay = min(self.ema_decay,(1 + self.step) / (10 + self.step))

        # Equivalent to:
        # ema = current_decay * ema + (1 - current_decay) * model
        torch._foreach_lerp_(self.ema_params, self.model_params, 1.0 - current_decay)

        # Copy BatchNorm statistics and other buffers.
        model_buffers = dict(self.model.named_buffers())
        for name, ema_buffer in self.ema_model.named_buffers():
            ema_buffer.copy_(model_buffers[name])

    def save_checkpoint(self) -> None:
        """Saves current state in checkpoint."""
        os.makedirs(self.checkpoint, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'step' in checkpoint:
            self.step = checkpoint['step']

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    def train(self) -> None:
        """Trains model."""
        num_batches = len(self.train_dataloader)
        if num_batches == 0:
            raise ValueError(
                "Training DataLoader is empty; reduce batch_size or disable drop_last."
            )
        if len(self.dataset) < 4:
            raise ValueError("At least four samples are required for visualization.")

        samples = torch.stack([self.dataset[i][0] for i in range(4)])
        sample_targets = torch.stack([self.dataset[i][1] for i in range(4)])

        self._log_images([self._visualize_image(s, i) for i, s in enumerate(samples)],
                         "samples/images")
        self._log_images([visualize_seg(st) for st in sample_targets], "samples/targets")

        self.compiled_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for self.epoch in range(self.epoch + 1, self.epochs + 1):
            losses = torch.zeros((len(self.train_dataloader),))
            epoch_t_loss = torch.zeros(self.engine.T, device=self.device)
            epoch_t_counts = torch.zeros(self.engine.T, device=self.device)
            epoch_grad_norms = []

            progress_bar = tqdm(self.train_dataloader,
                                total=len(self.train_dataloader),
                                desc=f"Epoch {self.epoch}")

            for i, (x, y) in enumerate(progress_bar):
                x = x.to(self.device, non_blocking=True)
                x_t, noise, t, snr = self.engine.sample(x)

                if self.use_target:
                    y = y.to(self.device, non_blocking=True)
                    pred = self.compiled_model(x_t, t, y)
                else:
                    pred = self.compiled_model(x_t, t)

                # calculate snr weighted loss
                per_sample_loss = (pred - noise).pow(2).mean(dim=list(range(1, pred.ndim)))
                snr_weight = torch.minimum(snr, torch.full_like(snr, self.gamma)) / snr
                raw_loss = (per_sample_loss * snr_weight).mean()

                # scale loss by size of group
                group_start = (i // self.accumulation_steps) * self.accumulation_steps
                group_size = min(self.accumulation_steps, num_batches - group_start)
                loss = raw_loss / group_size

                # backwards
                loss.backward()

                # Track unreduced loss per timestep
                with torch.no_grad():
                    epoch_t_loss.index_add_(0, t, per_sample_loss)
                    epoch_t_counts.index_add_(0, t, torch.ones_like(t, dtype=torch.float,
                                                                    device=self.device))

                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == num_batches:
                    self.step += 1

                    # tracking grad norm
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                         max_norm=float('inf'))
                    epoch_grad_norms.append(grad_norm.item())

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.update_ema()

                loss_value = raw_loss.detach().item()
                progress_bar.set_postfix({"loss": loss_value})
                losses[i] = loss_value

            # visualization
            num_bins = 10
            bin_size = self.engine.T // num_bins
            for b in range(num_bins):
                start = b * bin_size
                end = (b + 1) * bin_size if b < num_bins - 1 else self.engine.T

                # Filter only for timesteps that were actually hit during this epoch
                visited_mask = epoch_t_counts[start:end] > 0
                if visited_mask.any():
                    # Calculate mean loss for this specific interval bin
                    bin_loss = (epoch_t_loss[start:end][visited_mask] / epoch_t_counts[start:end][
                        visited_mask]).mean()
                    self.writer.add_scalar(f"Loss/loss_by_t/T_{start}_to_{end}", bin_loss.item(),
                                           self.epoch)

            self.writer.add_scalar("Loss/loss", torch.mean(losses), self.epoch)

            self.writer.add_scalar("Train/learning_rate",
                                   self.optimizer.param_groups[0]['lr'],
                                   self.epoch)
            self.writer.add_scalar("Train/gradient_norm",
                                   sum(epoch_grad_norms) / len(epoch_grad_norms),
                                   self.epoch)

            self._visualize_one_step(samples, targets=sample_targets)
            self._visualize_sampling(targets=sample_targets)
            self.writer.flush()

            self.save_checkpoint()

        self.writer.close()
        print("Training complete and logs flushed.")

    @torch.inference_mode()
    def generate_image(self,
                       batch_size: int = 1,
                       use_ema: bool = True,
                       targets: Optional[torch.Tensor] = None,
                       show_progress: bool = False,
                       step: int = 1,
                       eta: float = 0.0) -> torch.Tensor:
        """
        Generates images.

        eta=0 gives deterministic DDIM sampling.

        Args:
            batch_size: number of images.
            targets: tensor with targets if model was trained with prior targets
            use_ema: use ema model
            show_progress: shows results every 50 steps
            step: number of steps taken at once in reverse diffusion process
            eta:

        Returns:
            images: generated images (B, C, H, W).
        """
        # check if parameter are valid
        if step < 1:
            raise ValueError("step must be at least 1")

        if self.use_target:
            assert targets is not None, "No targets provided!"
            targets = targets.to(self.device)

        # make sure model is on device and in eval mode
        model = self.ema_model if use_ema else self.model
        model.eval()
        model.to(self.device)

        # init time steps.
        time_steps = list(range(self.engine.T - 1, -1, -step))
        if time_steps[-1] != 0:
            time_steps.append(0)
        target_steps = time_steps[1:] + [-1]
        transitions = list(zip(time_steps, target_steps))

        # init progress bar if show_progress == True
        progress_steps = transitions
        if show_progress:
            progress_steps = tqdm(transitions,
                                  desc="Generating images",
                                  total=len(transitions),
                                  leave=False)

        # init noise image
        x = torch.randn((batch_size, model.channels, model.height, model.width)).to(self.device)

        frames = []
        for t, target_t in progress_steps:  #
            # init t_tensor and target_t_tensor for ddim_step
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            target_t_tensor = torch.full_like(t_tensor, target_t)

            if self.use_target:
                epsilon_pred = model(x, t_tensor, targets)
            else:
                epsilon_pred = model(x, t_tensor)

            x = self.engine.ddim_step(
                images=x,
                epsilon_pred=epsilon_pred,
                t=t_tensor,
                target_t=target_t_tensor,
                eta=eta,
            ).clamp(self.dataset.z_min.to(self.device), self.dataset.z_max.to(self.device))

        return x.detach().cpu().clamp(self.dataset.z_min, self.dataset.z_max)

    @torch.no_grad()
    def _visualize_sampling(self,
                            use_ema: bool = True,
                            targets: Optional[torch.Tensor] = None) -> None:
        self.model.eval()

        samples = self.generate_image(batch_size=4, use_ema=use_ema, targets=targets)
        samples_list = [self._visualize_image(s, i) for i, s in enumerate(samples)]
        grid = make_grid(samples_list,
                         nrow=4,
                         normalize=True,
                         value_range=(0, 255))
        self.writer.add_image("Visual_Check/samples", grid, self.epoch)

        self.model.train()

    @torch.no_grad()
    def _visualize_one_step(self,
                            images: torch.Tensor,
                            use_ema: bool = True,
                            targets: Optional[torch.Tensor] = None) -> None:
        """
        fixed_images: A batch of clean images (x0) normalized to [-1, 1].
        """
        model = self.ema_model if use_ema else self.model
        model.eval()
        images = images.to(self.device)

        if self.use_target:
            if targets is None: raise ValueError("No targets provided!")
            targets = targets.to(self.device)

        check_steps = [
            int(self.engine.T * 0.1),
            int(self.engine.T * 0.3),
            int(self.engine.T * 0.5),
            int(self.engine.T * 0.8),
            int(self.engine.T * 0.9),
            self.engine.T - 1,
        ]

        for t in check_steps:
            t_tensor = torch.full((images.shape[0],), t, device=self.device)
            x_t, _, _ = self.engine.diffuse(images, t_tensor)

            if self.use_target:
                epsilon_pred = model(x_t, t_tensor, targets)
            else:
                epsilon_pred = model(x_t, t_tensor)

            x_0_recon = self.engine.image_approximation(x_t, epsilon_pred, t_tensor)

            self._log_one_step(t, x_0_recon, x_t)

        self.model.train()

    def _log_one_step(self,
                      t: int,
                      x_0: torch.Tensor,
                      x_t: torch.Tensor) -> None:

        x_t_list = [self._visualize_image(x, i) for i, x in enumerate(x_t)]
        x_0_list = [self._visualize_image(x, i) for i, x in enumerate(x_0)]
        self._log_images(x_t_list + x_0_list, f"Visual_Check/T_{t}")
