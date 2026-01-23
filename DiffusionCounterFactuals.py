import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
            for i, (images, noises, t) in enumerate(bar):
                images = images.to(self.device)
                noises = noises.to(self.device)
                t = t.to(self.device)

                optimizer.zero_grad()
                pred = self.diffusion_model(images, t)
                loss = self.loss_fn(pred, noises)
                loss.backward()
                optimizer.step()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            self.writer.add_scalar("loss", torch.mean(losses), epoch)

        torch.save(self.diffusion_model, "./models/DiffusionModel.pt")

    def load_model(self):
        self.diffusion_model = torch.load("./models/DiffusionModel.pt", weights_only=False)

    def generate_counterfactual(self, images: torch.Tensor, targets: torch.Tensor):
        batch_size = images.size(0)

        self.diffusion_model.eval()
        self.classifier.eval()

        images = images.to(self.device)
        self.classifier.to(self.device)
        self.diffusion_model.to(self.device)

        x = self._ddim_inversion(images)
        s = 0.1

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred = self.diffusion_model(x, t_tensor)

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier(x_in)
                loss = torch.mean(logits[:, targets])
                grad = torch.autograd.grad(loss, x_in)[0]

            combined_noise = noise_pred - torch.sqrt(1 - self.alphas_cumprod[t]) * s * t * grad

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
        x_prev = torch.sqrt(alpha_prev) * estimated_x0 + torch.sqrt(1 - alpha_prev) * combined_noise
        return x_prev
