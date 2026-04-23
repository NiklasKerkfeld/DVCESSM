import copy
import os
from typing import Optional
from omegaconf import DictConfig

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from core.Diffusion.Engine import Engine
from core.Trainer import SuperTrainer
from models.DiffusionModel import DiffusionModel
from utils import to_vis


class Trainer(SuperTrainer):
    def __init__(self,
                 dataset: Dataset,
                 model: DiffusionModel,
                 engine: Engine,
                 device: torch.device,
                 cfg: DictConfig,
                 name: str = 'DiffusionTrainer',
                 log_dir: str = '.logs/'
                 ):
        super().__init__(dataset, device, cfg, name, log_dir)
        self.model = model
        self.engine = engine
        self.lr = cfg.diffusion.lr
        self.accumulation_steps = cfg.data.accumulation_steps
        self.checkpoint = cfg.diffusion.checkpoint_path

        # --- EMA Setup ---
        self.ema_decay = cfg.diffusion.get("ema_decay", 0.999)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.to(self.device)
        self.ema_model.eval()  # EMA model is never in train mode
        self.step = 0
        # -----------------

        self.model.to(self.device)
        self.engine.to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def update_ema(self):
        """Step the EMA weights."""
        self.step += 1
        current_decay = min(self.ema_decay, (1 + self.step) / (10 + self.step))

        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(current_decay).add_(param.data, alpha=1 - current_decay)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, f'{self.checkpoint}/{self.name}.pt')

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    def train(self, epochs: int):
        samples = torch.stack([self.dataset[i][0] for i in range(4)])

        self.model.train()

        for self.epoch in range(self.epoch + 1, epochs + 1):
            losses = torch.zeros((len(self.train_dataloader),))
            self.optimizer.zero_grad()

            bar = tqdm(self.train_dataloader, total=len(self.train_dataloader ), desc=f"Epoch {self.epoch}")
            for i, (x, _) in enumerate(bar):
                x = x.to(self.device)
                x_t, noise, t = self.engine.sample(x)

                pred = self.model(x_t, t)
                loss = self.loss_fn(pred, noise)
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.update_ema()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            # visualization
            self.writer.add_scalar("loss", torch.mean(losses), self.epoch)
            self._visualize_one_step(samples)
            self._visualize_sampling()
            self.writer.flush()

            self.save_checkpoint()

        self.writer.close()
        print("Training complete and logs flushed.")

    @torch.no_grad()
    def generate_image(self, batch_size: Optional[int] = 1, use_ema: bool = True):
        model = self.ema_model if use_ema else self.model
        model.eval()

        images = torch.randn(batch_size, model.channels, model.width, model.height).to(self.device)
        model.to(self.device)

        for t in reversed(range(self.engine.T)):
            t_tensor = torch.full((images.shape[0],), t, device=self.device)
            epsilon_pred = model(images, t_tensor)

            # calculate mean of image at t-1
            mean = self.engine.reverse(images, epsilon_pred, t_tensor)

            if t > 0:
                noise = torch.randn_like(images)
                sigma = self.engine.get_sigma(t_tensor, images.shape)
                images = mean + sigma * noise
            else:
                images = mean

        return images.detach()

    @torch.no_grad()
    def _visualize_sampling(self, use_ema: bool = True):
        self.model.eval()

        samples = self.generate_image(batch_size=4, use_ema=use_ema)
        grid = make_grid(self._visualize_image(samples), nrow=4)
        self.writer.add_image(f"Visual_Check/samples", grid, self.epoch)

        self.model.train()

    @torch.no_grad()
    def _visualize_one_step(self, images: torch.Tensor, use_ema: bool = True):
        """
        fixed_images: A batch of clean images (x0) normalized to [-1, 1].
        """
        model = self.ema_model if use_ema else self.model
        model.eval()
        images = images.to(self.device)

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
            x_t, _ = self.engine.diffuse(images, t_tensor)

            epsilon_pred = model(x_t, t_tensor)

            x_0_recon = self.engine.image_approximation(x_t, epsilon_pred, t_tensor)

            self._log_one_step(self.epoch, t, x_0_recon, x_t)

        self.model.train()

    def _log_one_step(self, epoch: int, t: int, x_0_recon: Tensor, x_t: Tensor):
        comparison = torch.cat([to_vis(x_t), to_vis(x_0_recon)], dim=0)
        grid = make_grid(self._visualize_image(comparison), nrow=4)
        self.writer.add_image(f"Visual_Check/T_{t}", grid, epoch)