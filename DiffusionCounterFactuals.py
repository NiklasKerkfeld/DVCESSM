import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architecture.DiffusionModel import DiffusionModel
from Datasets import DiffusionDataset
from Parameter import T, BETA_MIN, BETA_MAX


class CounterFactualDiffusion:
    def __init__(self,
                 classifier: nn.Module,
                 images: torch.Tensor,
                 device: torch.device,
                 lr: float,
                 name='DiffusionModel'):
        self.diffusion_model = DiffusionModel(images.shape[1])
        self.classifier = classifier
        self.device = device

        self.T = T
        self.beta = torch.linspace(BETA_MIN, BETA_MAX, self.T)
        self.alphas = 1.0 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        dataset = DiffusionDataset(images)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        self.writer = SummaryWriter(f'runs/Diffusion/{name}')

    def train_diffusion(self, epochs: int):
        self.diffusion_model.to(self.device)
        self.diffusion_model.train()

        for epoch in range(1, epochs + 1):
            losses = torch.zeros((len(self.dataloader),))
            bar = tqdm(self.dataloader, total=len(self.dataloader), desc=f"Epoch {epoch}")
            for i, (images, noises, t) in enumerate(bar):
                images = images.to(self.device)
                noises = noises.to(self.device)
                t = t.to(self.device)

                self.optimizer.zero_grad()
                pred = self.diffusion_model(images, t)
                loss = self.loss_fn(pred, noises)
                loss.backward()
                self.optimizer.step()

                bar.set_postfix({"loss": loss.detach().item()})
                losses[i] = loss.detach().item()

            self.writer.add_scalar("loss", torch.mean(losses), epoch)

        torch.save(self.diffusion_model, "./models/DiffusionModel.pt")

    def load_model(self):
        self.diffusion_model = torch.load("./models/DiffusionModel.pt", weights_only=False)

    def generate_counterfactual(self, image: torch.Tensor, target_class: int):
        self.diffusion_model.eval()
        self.classifier.eval()

        image = image.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.diffusion_model = self.diffusion_model.to(self.device)

        x = self._ddim_inversion(image)
        s = 50.0

        for t in reversed(range(self.T)):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred = self.diffusion_model(x, t_tensor)

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier(x_in)
                loss = logits[0, target_class]
                grad = torch.autograd.grad(loss, x_in)[0]

            combined_noise = noise_pred - torch.sqrt(1 - self.alphas_cumprod[t]) * s * grad

            x = self._step_backwards(x, combined_noise, t)

        return x

    def _ddim_inversion(self, image: torch.Tensor):
        x = image.clone()
        for t in range(self.T - 1):
            noise_pred = self.diffusion_model(x, torch.tensor([t]))

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

if __name__ == '__main__':
    from torchvision.transforms import transforms
    from torchvision.datasets import MNIST

    from architecture.Classifier import Classifier
    from matplotlib import pyplot as plt

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(0.5, 0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([x[0] for x in train_dataset], dim=0).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(10)
    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    trainer = CounterFactualDiffusion(classifier, images[:1024], device, lr=1e-3)
    trainer.load_model()

    plt.imshow(images[1024][0], cmap="gray")
    plt.show()

    cf = trainer.generate_counterfactual(images[1024][None], target_class=6).detach().numpy()

    print(f"{cf.shape=}")
    plt.imshow(cf[0, 0], cmap="gray")
    plt.show()