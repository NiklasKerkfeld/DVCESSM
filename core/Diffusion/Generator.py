from typing import Tuple, Dict

import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import nn, Tensor
from tqdm import tqdm

from core.BaseGenerator import BaseGenerator
from core.Diffusion.Engine import Engine
from models.Losses import CounterfactualLoss
from uitls.utils import visualize_adaption, visualize_seg, calc_iou


class CounterfactualGenerator(BaseGenerator):
    def __init__(self,
                 segmentation: nn.Module,
                 diffusion_model: nn.Module,
                 engine: Engine,
                 device: torch.device,
                 cfg: DictConfig,
                 name='counterfactual'):
        super().__init__(segmentation, device, cfg, name)
        self.diffusion_model = diffusion_model
        self.engine = engine

        self.weight_segmentation = cfg.generation.weight_segmentation
        self.weight_distance = cfg.generation.weight_distance
        self.start_fraction = cfg.generation.start_fraction

        self.diffusion_model.to(self.device).eval()

    @torch.no_grad()
    def _generate_counterfactual(self,
                                 image: torch.Tensor,
                                 target: torch.Tensor):
        loss_fn = CounterfactualLoss(target.repeat((self.adaptgrad.N, 1, 1)),
                                     use_mask=self.use_mask,
                                     border_width=self.border_width)

        start_t = int(self.engine.T * self.start_fraction)
        t_tensor = torch.full((image.shape[0],), start_t, device=self.device)
        x, _ = self.engine.diffuse(image, t_tensor)

        for t in tqdm(reversed(range(start_t)), total=start_t, desc="generating counterfactual"):
            t_tensor = torch.full((image.shape[0],), t, device=self.device)
            epsilon_pred = self.diffusion_model(x, t_tensor).detach()
            mean = self.engine.reverse(x, epsilon_pred, t_tensor)

            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
            sigma = self.engine.get_sigma(t_tensor, image.shape)

            with torch.enable_grad():
                g_update, logs_dict = self._calc_g_update(
                    x,
                    image,
                    loss_fn,
                    epsilon_pred,
                    t_tensor)

            if t > 0:
                mean = mean - (sigma ** 2 * mu_norm * g_update)
                x = mean + sigma * torch.randn_like(image)
            else:
                x = mean

            # --- TensorBoard Logging ---
            if t % 50 == 0:
                self.log(x, target, mean, mu_norm, g_update, logs_dict, start_t, t)

        return x.detach()

    def _calc_g_update(self, x: Tensor, image: Tensor, loss_fn: CounterfactualLoss,
                       epsilon_pred: Tensor,
                       t: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        x = x.detach().requires_grad_(True)
        approx_image = self.engine.image_approximation(x, epsilon_pred, t)

        # Segmentation Loss
        approx_image_batch = self.adaptgrad(approx_image)
        logits = self.segmentation_model(approx_image_batch)

        # Loss and Gradients
        loss_c = loss_fn(logits)
        grad_c = torch.autograd.grad(loss_c, x, retain_graph=True)[0]

        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        # Distance Loss
        distance = torch.norm(image - approx_image, p=2)
        grad_d = torch.autograd.grad(distance, x)[0]
        grad_d /= (torch.norm(grad_d.view(grad_d.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        g_update = self.weight_segmentation * grad_c + self.weight_distance * grad_d

        return g_update.detach(), {
            'approx_image': approx_image.detach(),
            'grad_c': grad_c.detach(),
            'grad_d': grad_d.detach(),
            'logits': logits.detach()
        }

    def log(self, x: Tensor, target: Tensor, mean: Tensor, mu_norm: Tensor, g_update: Tensor,
            results: Dict[str, Tensor], start_t: int, t: int):
        log_step = start_t - t

        # log scalars
        prediction = torch.argmax(results['logits'][0], dim=0)
        iou = calc_iou(prediction, target, num_classes=self.num_classes)
        self.writer.add_scalar('Metrics/IoU', iou, log_step)
        self.writer.add_scalar('Gradients/mean', mean.mean().item(), log_step)
        self.writer.add_scalar('Gradients/mu_norm', mu_norm.mean().item(), log_step)
        self.writer.add_scalar('Gradients/g_update', g_update.abs().mean().item(), log_step)
        self.writer.add_scalar('Gradients/grad_c', results['grad_c'].abs().mean().item(), log_step)
        self.writer.add_scalar('Gradients/grad_d', results['grad_d'].abs().mean().item(), log_step)

        self.writer.add_image('Process/x_t', vutils.make_grid(x, normalize=True),
                              log_step)
        self.writer.add_image('Process/approx_x_0',
                              vutils.make_grid(results['approx_image'], normalize=True), log_step)

        prediction = visualize_seg(prediction.detach().cpu()).permute(2, 0, 1).float()
        print(f"{prediction.shape=}")
        g_visual = visualize_adaption(g_update.sum(dim=1, keepdim=True))[0, 0]
        grad_c_visual = visualize_adaption(results['grad_c'].sum(dim=1, keepdim=True))[0, 0]
        grad_d_visual = visualize_adaption(results['grad_d'].sum(dim=1, keepdim=True))[0, 0]

        self.writer.add_image('Process/g_update',
                              vutils.make_grid(g_visual, normalize=True), log_step)
        self.writer.add_image('Process/grad_c',
                              vutils.make_grid(grad_c_visual, normalize=True), log_step)
        self.writer.add_image('Process/grad_d',
                              vutils.make_grid(grad_d_visual, normalize=True), log_step)
        self.writer.add_image('Process/prediction', prediction, log_step)
