import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from tqdm import tqdm

from core.AdaptGrad import AdaptGrad
from core.Diffusion.Engine import Engine
from models.Losses import masked_cross_entropy


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
        self.diffusion_checkpoint = cfg.diffusion.checkpoint_path
        self.diffusion_name = cfg.diffusion.name
        self.segmentation_checkpoint = cfg.segmentation.checkpoint_path
        self.segmentation_name = cfg.segmentation.name
        self.weight_classifier = cfg.segmentation.weight_classifier
        self.weight_distance = cfg.segmentation.weight_distance
        self.adaptgrad = AdaptGrad(cfg)

        self.load_checkpoint(f"{self.diffusion_checkpoint}/{self.diffusion_name}.pt",
                             f"{self.segmentation_checkpoint}/{self.segmentation_name}.pt")
        self.segmentation_model.eval()
        self.diffusion_model.eval()

    def load_checkpoint(self, diffusion_checkpoint: str, segmentation_checkpoint: str):
        print(f"{diffusion_checkpoint=}, {segmentation_checkpoint=}")
        checkpoint = torch.load(diffusion_checkpoint, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict']) #

        checkpoint = torch.load(segmentation_checkpoint, map_location=self.device)
        self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def generate(self, images: Tensor, targets: Tensor, start_fraction: float):
        prediction_org = self.segmentation_model(images).detach().cpu().argmax(dim=1)

        counterfactuals = self._generate_counterfactual(images, targets, start_fraction)
        adaptions = counterfactuals - images

        prediction_cf = self.segmentation_model(counterfactuals).detach().cpu().argmax(dim=1)

        return counterfactuals.cpu(), adaptions.cpu(), prediction_org.cpu(), prediction_cf.cpu()

    @torch.no_grad()
    def _generate_counterfactual(self, images: torch.Tensor, targets: torch.Tensor,
                                 start_fraction: float):
        self.diffusion_model.eval()
        self.segmentation_model.eval()

        images = images.to(self.device)
        self.segmentation_model.to(self.device)
        self.diffusion_model.to(self.device)

        start_t = int(self.engine.T * start_fraction)
        t_tensor = torch.full((images.shape[0],), start_t, device=self.device)
        x, _ = self.engine.diffuse(images, t_tensor)

        for t in tqdm(reversed(range(start_t)), total=start_t, desc="generating counterfactual"):
            t_tensor = torch.full((images.shape[0],), t, device=self.device)
            epsilon_pred = self.diffusion_model(x, t_tensor)
            mean = self.engine.reverse(x, epsilon_pred, t_tensor)

            with torch.enable_grad():
                g_update = self._calc_g_update(x, images, targets, epsilon_pred, t_tensor)

            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)

            sigma = self.engine.get_sigma(t_tensor, images.shape)
            mean = mean + (sigma * mu_norm * g_update)

            if t > 0:
                noise = torch.randn_like(images)
                x = mean + sigma * noise
            else:
                x = mean

        return x.detach()

    def _calc_g_update(self, x: Tensor, images: Tensor, targets: Tensor, epsilon_pred: Tensor,
                       t: Tensor) -> Tensor:
        x = x.detach().requires_grad_(True)
        approx_images = self.engine.image_approximation(x, epsilon_pred, t)

        # model
        approx_images = self.adaptgrad(approx_images)
        logits = self.segmentation_model(approx_images)
        loss_c = masked_cross_entropy(logits, targets.repeat((self.adaptgrad.N, 1, 1)))
        grad_c = torch.autograd.grad(-loss_c, x, retain_graph=True)[0]
        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)
        # distance
        distance = torch.norm(images - approx_images, p=1)
        grad_d = torch.autograd.grad(distance, x)[0]
        grad_d /= (torch.norm(grad_d.view(grad_d.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)
        # g_update
        return self.weight_classifier * grad_c - self.weight_distance * grad_d
