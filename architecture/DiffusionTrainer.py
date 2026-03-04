from typing import Optional, Literal

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from architecture.DiffusionModel import DiffusionModel
from Datasets import DiffusionDataset
from Parameter import T, BETA_MIN, BETA_MAX, BATCH_SIZE
from utils import to_vis, cosine_beta_schedule


class DiffusionTrainer:
    def __init__(self,
                 dims: int,
                 device: torch.device,
                 name='DiffusionModel',
                 beta_schedule: Literal["linear", 'cosine'] = 'linear'
                 ):
        self.diffusion_model = DiffusionModel(dims)
        self.device = device

        self.T = T
        self.beta = cosine_beta_schedule(self.T) if beta_schedule == 'cosine' else torch.linspace(BETA_MIN, BETA_MAX, self.T)
        self.alphas = 1.0 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.loss_fn = nn.MSELoss()

        self.writer = SummaryWriter(f'runs/Diffusion/{name}')

    def load_model(self):
        self.diffusion_model = torch.load("./models/DiffusionModel.pt", weights_only=False, map_location=self.device)

    def train_diffusion(self, images: torch.Tensor, epochs: int, lr: float):
        dataset = DiffusionDataset(images)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        self.diffusion_model.to(self.device)
        self.diffusion_model.train()

        for epoch in range(1, epochs + 1):
            losses = torch.zeros((len(dataloader),))
            bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")
            for i, (x_t, noises, t) in enumerate(bar):
                x_t = x_t.to(self.device)
                noises = noises.to(self.device)
                t = t.to(device=self.device, dtype=torch.long)

                optimizer.zero_grad()
                pred = self.diffusion_model(x_t, t)
                loss = self.loss_fn(pred, noises)
                loss.backward()
                optimizer.step()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            # visualization
            self.writer.add_scalar("loss", torch.mean(losses), epoch)
            self._visualize_one_step(images[:5], epoch)
            self._visualize_sampling(epoch)

        # save model
        torch.save(self.diffusion_model, "./models/DiffusionModel.pt")

    @torch.no_grad()
    def _visualize_sampling(self, epoch: int):
        self.diffusion_model.eval()

        samples = self.generate_image(batch_size=4)
        grid = make_grid(samples, nrow=4)
        self.writer.add_image(f"Visual_Check/samples", grid, epoch)

        self.diffusion_model.train()

    @torch.no_grad()
    def _visualize_one_step(self, fixed_images: torch.Tensor, epoch: int):
        """
        fixed_images: A batch of clean images (x0) normalized to [-1, 1].
        """
        self.diffusion_model.eval()

        # Define the timesteps we want to inspect (e.g., start, middle, end)
        # Using 4 stages to see the progression
        check_steps = [
            int(self.T * 0.1),  # Early noise (t=20)
            int(self.T * 0.5),  # Medium noise (t=100)
            int(self.T * 0.8),  # Heavy noise (t=160)
            self.T - 1,  # Almost pure noise (t=199)
        ]

        for t in check_steps:
            images = fixed_images.to(self.device)
            noise = torch.randn_like(images)
            x_t = torch.sqrt(self.alphas_cumprod[t]) * images + torch.sqrt(1 - self.alphas_cumprod[t]) * noise

            epsilon_pred = self.diffusion_model(x_t, torch.tensor([t], device=self.device))

            x_0_recon = self._image_approximation(x_t, epsilon_pred, t)

            comparison = torch.cat([to_vis(x_t[:4]), to_vis(x_0_recon[:4])], dim=0)
            grid = make_grid(comparison, nrow=4)
            self.writer.add_image(f"Visual_Check/T_{t}", grid, epoch)

            if t == check_steps[-1]:
                self.writer.add_scalar("min_predicted_noise", epsilon_pred.min(), epoch)
                self.writer.add_scalar("max_predicted_noise", epsilon_pred.max(), epoch)
                self.writer.add_scalar("mean_predicted_noise", epsilon_pred.mean(), epoch)

        self.diffusion_model.train()

    @torch.no_grad()
    def generate_image(self, image: Optional[torch.Tensor] = None, T: Optional[int] = None, batch_size: Optional[int] = 1):
        image = torch.randn(batch_size, 1, 32, 32) if image is None else image
        T = self.T if T is None else T

        self.diffusion_model.eval()

        image = image.to(self.device)
        self.diffusion_model.to(self.device)

        for t in reversed(range(T)):
            epsilon_pred = self.diffusion_model(image, t)

            # calculate mean of image at t-1
            mean = self._reverse(image, epsilon_pred, t)

            if t > 0:
                noise = torch.randn_like(image)
                sigma = torch.sqrt(self.beta[t])
                image = mean + sigma * noise
            else:
                image = mean  # Final step is just the clean estimate

        return to_vis(image).detach()

    def _image_approximation(self, x_t: Tensor, epsilon_pred: Tensor, t: int):
        return (x_t - torch.sqrt(1 - self.alphas_cumprod[t]) * epsilon_pred) / torch.sqrt(
            self.alphas_cumprod[t])
