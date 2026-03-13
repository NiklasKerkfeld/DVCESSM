import os
from typing import Optional

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from core.DiffusionEngine import DiffusionEngine
from models.DiffusionModel import DiffusionModel
from utils import to_vis


class DiffusionTrainer:
    def __init__(self,
                 model: DiffusionModel,
                 engine: DiffusionEngine,
                 lr: float,
                 device: torch.device,
                 name='DiffusionModel',
                 ):
        self.model = model
        self.engine = engine
        self.lr = lr
        self.device = device

        self.epoch = 1

        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.writer = SummaryWriter(f'runs/Diffusion/{name}')

    def save_checkpoint(self):
        os.makedirs("./checkpoints", exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, './checkpoints/diffusion_training_checkpoint.pt')

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())


    def save_model(self):
        # save model
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/DiffusionModel.pth")

    def train(self, dataloader: DataLoader, epochs: int):
        samples = torch.stack([dataloader.dataset[i] for i in torch.randint(0, len(dataloader.dataset), (4,))])

        self.model.to(self.device)
        self.model.train()

        self.engine.to(self.device)

        for self.epoch in range(self.epoch, epochs + 1):
            losses = torch.zeros((len(dataloader),))
            bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {self.epoch}")
            for i, x in enumerate(bar):
                x = x.to(self.device)
                x_t, noise, t = self.engine.sample(x)

                self.optimizer.zero_grad()
                pred = self.model(x_t, t)
                loss = self.loss_fn(pred, noise)
                loss.backward()
                self.optimizer.step()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            # visualization
            self.writer.add_scalar("loss", torch.mean(losses), self.epoch)
            self._visualize_one_step(samples)
            self._visualize_sampling()
            self.writer.flush()

            self.save_checkpoint()

        self.save_model()
        self.writer.close()
        print("Training complete and logs flushed.")

    @torch.no_grad()
    def _visualize_sampling(self):
        self.model.eval()

        samples = self.generate_image(batch_size=4)
        grid = make_grid(samples[:, 0, None], nrow=4)
        self.writer.add_image(f"Visual_Check/samples", grid, self.epoch)

        self.model.train()

    @torch.no_grad()
    def _visualize_one_step(self, images: torch.Tensor):
        """
        fixed_images: A batch of clean images (x0) normalized to [-1, 1].
        """
        self.model.eval()
        images = images.to(self.device)

        check_steps = [
            int(self.engine.T * 0.1),
            int(self.engine.T * 0.5),
            int(self.engine.T * 0.8),
            self.engine.T - 1,
        ]

        for t in check_steps:
            t_tensor = torch.full((images.shape[0],), t, device=self.device)
            x_t, _ = self.engine.diffuse(images, t_tensor)

            epsilon_pred = self.model(x_t, t_tensor)

            x_0_recon = self.engine.image_approximation(x_t, epsilon_pred, t_tensor)

            self._log_one_step(self.epoch, t, x_0_recon, x_t)

        self.model.train()

    def _log_one_step(self, epoch: int, t: int, x_0_recon: Tensor, x_t: Tensor):
        comparison = torch.cat([to_vis(x_t), to_vis(x_0_recon)], dim=0)
        grid = make_grid(comparison[:, 0, None], nrow=4)
        self.writer.add_image(f"Visual_Check/T_{t}", grid, epoch)

    @torch.no_grad()
    def generate_image(self, batch_size: Optional[int] = 1):
        images = torch.randn(batch_size, self.model.channels, self.model.width, self.model.height)
        self.model.eval()

        images = images.to(self.device)
        self.model.to(self.device)

        for t in reversed(range(self.engine.T)):
            t_tensor = torch.full((images.shape[0],), t, device=self.device)
            epsilon_pred = self.model(images, t_tensor)

            # calculate mean of image at t-1
            mean = self.engine.reverse(images, epsilon_pred, t_tensor)
            mean = torch.clamp(mean, -1.0, 1.0)

            if t > 0:
                noise = torch.randn_like(images)
                sigma = self.engine.get_sigma(t_tensor, images.shape)
                images = mean + sigma * noise
            else:
                images = mean

        return images.detach()