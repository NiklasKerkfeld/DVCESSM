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
                 device: torch.device,
                 name='DiffusionModel',
                 ):
        self.model = model
        self.engine = engine
        self.device = device

        self.loss_fn = nn.MSELoss()

        self.writer = SummaryWriter(f'runs/Diffusion/{name}')

    def train(self, dataloader: DataLoader, epochs: int, lr: float):
        optimizer = Adam(self.model.parameters(), lr=lr)
        samples = torch.stack([dataloader.dataset[i] for i in torch.randint(0, len(dataloader.dataset), (5,))])

        self.model.to(self.device)
        self.model.train()

        self.engine.to(self.device)

        for epoch in range(1, epochs + 1):
            losses = torch.zeros((len(dataloader),))
            bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")
            for i, x in enumerate(bar):
                x = x.to(self.device)
                x_t, noise, t = self.engine.sample(x)

                optimizer.zero_grad()
                pred = self.model(x_t, t)
                loss = self.loss_fn(pred, noise)
                loss.backward()
                optimizer.step()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            # visualization
            self.writer.add_scalar("loss", torch.mean(losses), epoch)
            self._visualize_one_step(samples, epoch)
            self._visualize_sampling(epoch)

        # save model
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/DiffusionModel.pth")

    @torch.no_grad()
    def _visualize_sampling(self, epoch: int):
        self.model.eval()

        samples = self.generate_image(batch_size=4)
        grid = make_grid(samples, nrow=4)
        self.writer.add_image(f"Visual_Check/samples", grid, epoch)

        self.model.train()

    @torch.no_grad()
    def _visualize_one_step(self, images: torch.Tensor, epoch: int):
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

            self._log_one_step(epoch, t, x_0_recon, x_t)

        self.model.train()

    def _log_one_step(self, epoch: int, t: int, x_0_recon: Tensor, x_t: Tensor):
        comparison = torch.cat([to_vis(x_t[:4]), to_vis(x_0_recon[:4])], dim=0)
        grid = make_grid(comparison, nrow=4)
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