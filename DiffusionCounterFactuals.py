from typing import Optional

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from architecture.DiffusionModel import DiffusionModel
from Datasets import DiffusionDataset
from Parameter import T, BETA_MIN, BETA_MAX, BATCH_SIZE, WEIGHT_CLASSIFIER, WEIGHT_DISTANCE
from utils import to_vis


class CounterFactualDiffusion:
    def __init__(self,
                 classifier: nn.Module,
                 dims: int,
                 device: torch.device,
                 name='DiffusionModel',
                 distance_fn: nn.Module = torch.nn.L1Loss,
                 weight_classifier: float= WEIGHT_CLASSIFIER,
                 weight_distance: float = WEIGHT_DISTANCE):
        self.weight_classifier = weight_classifier
        self.weight_distance = weight_distance
        self.diffusion_model = DiffusionModel(dims)
        self.classifier = classifier
        self.device = device
        self.distance_fn = distance_fn

        self.T = T
        self.beta = torch.linspace(BETA_MIN, BETA_MAX, self.T)
        self.alphas = 1.0 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

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

    @torch.no_grad()
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
        ]

        for t in check_steps:
            images = fixed_images.to(self.device)
            noise = torch.randn_like(images)
            x_t = torch.sqrt(self.alphas_cumprod[t]) * images + torch.sqrt(1 - self.alphas_cumprod[t]) * noise

            epsilon_pred = self.diffusion_model(x_t, t)

            x_0_recon = self._image_approximation(x_t, epsilon_pred, t)

            comparison = torch.cat([to_vis(x_t[:4]), to_vis(x_0_recon[:4])], dim=0)
            grid = make_grid(comparison, nrow=4)
            self.writer.add_image(f"Visual_Check/T_{t}", grid, epoch)

            if t == check_steps[-1]:
                self.writer.add_scalar("min_predicted_noise", epsilon_pred.min(), epoch)
                self.writer.add_scalar("max_predicted_noise", epsilon_pred.max(), epoch)
                self.writer.add_scalar("mean_predicted_noise", epsilon_pred.mean(), epoch)

        samples = self.generate_image(batch_size=4)
        grid = make_grid(samples, nrow=4)
        self.writer.add_image(f"Visual_Check/samples", grid, epoch)

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

    @torch.no_grad()
    def generate_counterfactual(self, images: torch.Tensor, targets: torch.Tensor):
        self.diffusion_model.eval()
        self.classifier.eval()

        images = images.to(self.device)
        self.classifier.to(self.device)
        self.diffusion_model.to(self.device)

        x = self._diffuse(images, T // 2)

        for t in reversed(range(self.T)):
            epsilon_pred = self.diffusion_model(x, t)
            mean = self._reverse(x, epsilon_pred, t)

            with torch.enable_grad():
                g_update = self.calc_g_update(x, images, targets, epsilon_pred, t)

            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)

            sigma = torch.sqrt(self.beta[t])
            mean += sigma * mu_norm * g_update

            if t > 0:
                noise = torch.randn_like(images)
                x = mean + sigma * noise
            else:
                x = mean

        return to_vis(x).detach()

    def calc_g_update(self, x: Tensor, images: Tensor, targets: Tensor, epsilon_pred: Tensor, t: int) -> Tensor:
        x = x.detach().requires_grad_(True)
        approx_images = self._image_approximation(x, epsilon_pred, t)

        # classifier
        logits = self.classifier(approx_images)
        log_probs = torch.log_softmax(logits, dim=1)
        target_log_prob = log_probs[torch.arange(x.shape[0], device=x.device), targets].sum()
        grad_c = torch.autograd.grad(target_log_prob, x, retain_graph=True)[0]
        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)
        # distance
        distance = torch.norm(images - approx_images, p=1)
        grad_d = torch.autograd.grad(distance, x)[0]
        grad_d /= (torch.norm(grad_d.view(grad_d.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)
        # g_update
        return self.weight_classifier * grad_c - self.weight_distance * grad_d

    def _diffuse(self, images: Tensor, t):
        noise = torch.randn_like(images)

        noised = torch.sqrt(self.alphas_cumprod[t]) * images + torch.sqrt(
            1 - self.alphas_cumprod[t]) * noise
        return noised

    def _reverse(self, images: Tensor, epsilon_pred: Tensor, t: int):
        coeff1 = 1 / torch.sqrt(self.alphas[t])
        coeff2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
        mean = coeff1 * (images - coeff2 * epsilon_pred)

        return mean

    def _image_approximation(self, x_t: Tensor, epsilon_pred: Tensor, t: int):
        return (x_t - torch.sqrt(1 - self.alphas_cumprod[t]) * epsilon_pred) / torch.sqrt(self.alphas_cumprod[t])
