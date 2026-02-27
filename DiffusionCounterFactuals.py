from typing import Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from architecture.DiffusionModel import DiffusionModel
from Datasets import DiffusionDataset
from Parameter import T, BETA_MIN, BETA_MAX, BATCH_SIZE


class CounterFactualDiffusion:
    def __init__(self,
                 classifier: nn.Module,
                 dims: int,
                 device: torch.device,
                 name='DiffusionModel'):
        self.diffusion_model = DiffusionModel(dims)
        self.classifier = classifier
        self.device = device

        self.T = T
        self.beta = torch.linspace(BETA_MIN, BETA_MAX, self.T)
        self.alphas = 1.0 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.loss_fn = nn.MSELoss()

        self.writer = SummaryWriter(f'runs/Diffusion/{name}')

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
                t = t.to(self.device)

                optimizer.zero_grad()
                pred = self.diffusion_model(x_t, t)
                loss = self.loss_fn(pred, noises)
                loss.backward()
                optimizer.step()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            self.writer.add_scalar("loss", torch.mean(losses), epoch)
            self.visual_check(images[:5], epoch)

        torch.save(self.diffusion_model, "./models/DiffusionModel.pt")

    def visual_check(self, fixed_images: torch.Tensor, epoch: int):
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
            self.T
        ]

        with torch.no_grad():
            for t_val in check_steps:
                images = fixed_images.to(self.device)
                batch_size = images.shape[0]
                t_tensor = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)

                noise = torch.randn_like(images)
                a_bar = self.alphas_cumprod[t_val]
                x_t = torch.sqrt(a_bar) * images + torch.sqrt(1 - a_bar) * noise

                pred_noise = self.diffusion_model(x_t, t_tensor)

                x_0_recon = (x_t - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)

                def to_vis(x):
                    return torch.clamp((x + 1) / 2, 0, 1)

                comparison = torch.cat([to_vis(x_t[:4]), to_vis(x_0_recon[:4])], dim=0)
                grid = make_grid(comparison, nrow=4)
                self.writer.add_image(f"Visual_Check/T_{t_val}", grid, epoch)

                if t_val == check_steps[-1]:
                    self.writer.add_scalar("min_predicted_noise", pred_noise.min(), epoch)
                    self.writer.add_scalar("max_predicted_noise", pred_noise.max(), epoch)
                    self.writer.add_scalar("mean_predicted_noise", pred_noise.mean(), epoch)

        self.diffusion_model.train()

    def load_model(self):
        self.diffusion_model = torch.load("./models/DiffusionModel.pt", weights_only=False)

    def generate_image(self, image: Optional[torch.Tensor] = None, T: Optional[int] = None):
        image = torch.randn(1, 1, 32, 32) if image is None else image
        T = self.T if T is None else T

        self.diffusion_model.eval()

        image = image.to(self.device)
        self.diffusion_model.to(self.device)

        for t in reversed(range(T)):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            beta_t = self.beta[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            with torch.no_grad():
                epsilon_pred = self.diffusion_model(image, t_tensor)
                print(f"t: {t} | Image Mean: {image.mean():.4f}, Image Std: {image.std():.4f} | Noise Range: [{epsilon_pred.min():.2f}, {epsilon_pred.max():.2f}]")

            pred_x0 = (image - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)

            if t > 0:
                alpha_bar_t_prev = self.alphas_cumprod[t - 1]
                coeff_x0 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * pred_x0 + coeff_xt * image

                sigma = torch.sqrt(beta_t)
                image = mu + sigma * torch.randn_like(image)
            else:
                image = pred_x0.clamp(-1.0, 1.0)  # Final step is just the clean estimate

        return image.detach().cpu().numpy()

    def generate_counterfactual(self, images: torch.Tensor, targets: torch.Tensor):
        batch_size = images.size(0)

        self.diffusion_model.eval()
        self.classifier.eval()

        images = images.to(self.device)
        self.classifier.to(self.device)
        self.diffusion_model.to(self.device)

        x = self._ddim_inversion(images)
        s = 2.0

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred = self.diffusion_model(x, t_tensor)

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier(x_in)
                loss = torch.mean(logits[:, targets])
                grad = torch.autograd.grad(loss, x_in)[0]

            combined_noise = noise_pred - torch.sqrt(1 - self.alphas_cumprod[t]) * s * grad

            x = self._step_backwards(x, combined_noise, t)

        x /= 0.5
        x += 0.5
        return x.clip(0, 1)

    def _ddim_inversion(self, image: torch.Tensor):
        x = image.clone()
        for t in range(self.T - 1):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
            noise_pred = self.diffusion_model(x, t_tensor)

            alpha_now = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t + 1]

            x0_out = (x - torch.sqrt(1 - alpha_now) * noise_pred) / torch.sqrt(alpha_now)
            x = torch.sqrt(alpha_next) * x0_out + torch.sqrt(1 - alpha_next) * noise_pred

        return x

    def _step_backwards(self, x: torch.Tensor, combined_noise: torch.Tensor, t: int):
        alpha_now = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod_prev[t]

        estimated_x0 = (x - torch.sqrt(1 - alpha_now) * combined_noise) / torch.sqrt(alpha_now)
        estimated_x0 = torch.clamp(estimated_x0, -1, 1)
        x_prev = torch.sqrt(alpha_prev) * estimated_x0 + torch.sqrt(1 - alpha_prev) * combined_noise
        return x_prev
