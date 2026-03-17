import torch
from torch import nn, Tensor
from tqdm import tqdm
import torch.nn.functional as F

from core.DiffusionEngine import DiffusionEngine


def masked_cross_entropy(logits, targets):
    # Get current predictions: [B, H, W]
    current_preds = logits.argmax(dim=1)

    # Create a mask where predictions are WRONG (1 if wrong, 0 if correct)
    mask = (current_preds != targets).float()

    # Calculate per-pixel cross entropy
    loss = F.cross_entropy(logits, targets, reduction='none')

    # Only keep loss for wrong pixels
    return (loss * mask).sum()


def capped_cross_entropy(logits, targets, threshold=0.75):
    # Calculate probabilities
    probs = torch.softmax(logits, dim=1)
    # Get probability of the target class at each pixel
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Mask pixels that are already 'confident enough'
    # Loss is only calculated for pixels where target_prob < threshold
    mask = (target_probs < threshold).float()

    loss = F.cross_entropy(logits, targets, reduction='none')
    return (loss * mask).sum()


class CounterfactualGenerator:
    def __init__(self,
                 model: nn.Module,
                 diffusion_model: nn.Module,
                 engine: DiffusionEngine,
                 device: torch.device,
                 weight_classifier: float,
                 weight_distance: float):
        super().__init__()
        self.model = model
        self.diffusion_model = diffusion_model
        self.engine = engine
        self.device = device
        self.weight_classifier = weight_classifier
        self.weight_distance = weight_distance

    @torch.no_grad()
    def generate_counterfactual(self, images: torch.Tensor, targets: torch.Tensor, start_fraction: float):
        self.diffusion_model.eval()
        self.model.eval()

        images = images.to(self.device)
        self.model.to(self.device)
        self.diffusion_model.to(self.device)

        start_t = int(self.engine.T * start_fraction)
        t_tensor = torch.full((images.shape[0],), start_t, device=self.device)
        x, _ = self.engine.diffuse(images, t_tensor)

        for t in tqdm(reversed(range(start_t)), total=start_t, desc="generating counterfactual"):
            t_tensor = torch.full((images.shape[0],), t, device=self.device)
            epsilon_pred = self.diffusion_model(x, t_tensor)
            mean = self.engine.reverse(x, epsilon_pred, t_tensor)

            with torch.enable_grad():
                g_update = self.calc_g_update(x, images, targets, epsilon_pred, t_tensor)

            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)

            sigma = self.engine.get_sigma(t_tensor, images.shape)
            mean = mean + (sigma * mu_norm * g_update)

            if t > 0:
                noise = torch.randn_like(images)
                x = mean + sigma * noise
            else:
                x = mean

        return x.detach()

    def calc_g_update(self, x: Tensor, images: Tensor, targets: Tensor, epsilon_pred: Tensor,
                      t: Tensor) -> Tensor:
        x = x.detach().requires_grad_(True)
        approx_images = self.engine.image_approximation(x, epsilon_pred, t)

        # model
        logits = self.model(approx_images)
        loss_c = masked_cross_entropy(logits, targets)
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

