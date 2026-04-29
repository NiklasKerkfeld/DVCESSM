import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from core.AdaptGrad import AdaptGrad
from core.Diffusion.Engine import Engine
from torch.nn.functional import cross_entropy
from torchvision.transforms.functional import gaussian_blur


class CounterfactualGenerator:
    def __init__(self,
                 segmentation: nn.Module,
                 diffusion_model: nn.Module,
                 engine: Engine,
                 device: torch.device,
                 cfg: DictConfig):
        super().__init__()
        self.segmentation_model = segmentation
        self.diffusion_model = diffusion_model
        self.engine = engine
        self.device = device
        self.cfg = cfg
        self.diffusion_checkpoint = cfg.diffusion.checkpoint_path
        self.diffusion_name = cfg.diffusion.name
        self.segmentation_checkpoint = cfg.segmentation.checkpoint_path
        self.segmentation_name = cfg.segmentation.name
        self.weight_classifier = cfg.diffusion.weight_classifier
        self.weight_distance = cfg.diffusion.weight_distance
        self.adaptgrad = AdaptGrad(cfg)

        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=f".{cfg.system.log_dir}/Generation")

        self.load_checkpoint(f"{self.diffusion_checkpoint}/{self.diffusion_name}.pt",
                             f"{self.segmentation_checkpoint}/{self.segmentation_name}.pt")
        self.segmentation_model.eval()
        self.diffusion_model.eval()

    def load_checkpoint(self, diffusion_checkpoint: str, segmentation_checkpoint: str):
        checkpoint = torch.load(diffusion_checkpoint, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])

        checkpoint = torch.load(segmentation_checkpoint, map_location=self.device)
        self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def generate(self, images: Tensor, targets: Tensor, start_fraction: float):
        prediction_org = self.segmentation_model(images).detach().cpu().argmax(dim=1)
        counterfactuals = self._generate_counterfactual(images, targets, start_fraction)
        adaptions = counterfactuals - images
        prediction_cf = self.segmentation_model(counterfactuals).detach().cpu().argmax(dim=1)

        self.writer.close()  # Close writer after generation
        return counterfactuals.cpu(), adaptions.cpu(), prediction_org.cpu(), prediction_cf.cpu()

    @torch.no_grad()
    def _generate_counterfactual(self, image: torch.Tensor, targets: torch.Tensor,
                                 start_fraction: float):
        self.diffusion_model.eval()
        self.segmentation_model.eval()

        image = image.to(self.device)
        self.segmentation_model.to(self.device)
        self.diffusion_model.to(self.device)
        self.adaptgrad.to(self.device)

        start_t = int(self.engine.T * start_fraction)
        t_tensor = torch.full((image.shape[0],), start_t, device=self.device)
        x, _ = self.engine.diffuse(image, t_tensor)

        for t in tqdm(reversed(range(start_t)), total=start_t, desc="generating counterfactual"):
            t_tensor = torch.full((image.shape[0],), t, device=self.device)
            epsilon_pred = self.diffusion_model(x, t_tensor).detach()

            with torch.enable_grad():
                g_update, approx_image = self._calc_g_update(x, image, targets, epsilon_pred,
                                                             t_tensor)

            mean = self.engine.reverse(x, epsilon_pred, t_tensor)
            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
            sigma = self.engine.get_sigma(t_tensor, image.shape)

            # Adaptive Parameterization
            mean = mean + (sigma ** 2 * mu_norm * g_update)

            # --- TensorBoard Logging ---
            if t % 50 == 0:
                self.writer.add_image('Process/1_Noisy_xt', vutils.make_grid(x, normalize=True), t)
                self.writer.add_image('Process/2_Approx_x0',
                                      vutils.make_grid(approx_image, normalize=True), t)
                self.writer.add_scalar('Gradients/mu_norm', mu_norm.mean().item(), t)
                self.writer.add_scalar('Gradients/sigma_sq', (sigma ** 2).mean().item(), t)

            if t > 0:
                noise = torch.randn_like(image)
                x = mean + sigma * noise
            else:
                x = mean

        return x.detach()

    def _calc_g_update(self, x: Tensor, image: Tensor, targets: Tensor, epsilon_pred: Tensor,
                       t: Tensor):
        x = x.detach().requires_grad_(True)
        # Equation 8: Estimate clean image from noisy latent
        approx_image = self.engine.image_approximation(x, epsilon_pred, t)

        # Classification Loss
        approx_image_batch = self.adaptgrad(approx_image)
        logits = self.segmentation_model(approx_image_batch)
        loss_c = cross_entropy(logits, targets.repeat((self.adaptgrad.N, 1, 1)))
        grad_c = torch.autograd.grad(-loss_c, x, retain_graph=True)[0]

        # Force spatial coherence (Gradient Smoothing)
        grad_c = gaussian_blur(grad_c, kernel_size=7, sigma=1.5)

        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        # Distance Loss
        distance = torch.norm(image - approx_image, p=1)
        grad_d = torch.autograd.grad(distance, x)[0]
        grad_d /= (torch.norm(grad_d.view(grad_d.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        g_update = self.weight_classifier * grad_c - self.weight_distance * grad_d
        return g_update, approx_image