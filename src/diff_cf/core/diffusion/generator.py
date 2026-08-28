"""Diffusion based Counterfactual Generator."""

from typing import Tuple, Dict, Optional

import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import nn
import torchvision.transforms.functional as F
from tqdm import tqdm

from ..base_generator import BaseGenerator
from .engine import Engine
from ...models.losses import CounterfactualLoss, geman_mcclure_distance
from ...utils.cone_projection import cone_projection
from ...utils.utils import visualize_adaption, visualize_seg
from ...utils.metrics import calc_iou, calc_f1_score


class CounterfactualGenerator(BaseGenerator):
    """Diffusion based Counterfactual Generator."""

    def __init__(self,
                 segmentation: nn.Module,
                 diffusion_model: nn.Module,
                 engine: Engine,
                 device: torch.device,
                 cfg: DictConfig,
                 robust_segmentation: Optional[nn.Module] = None,
                 log: bool = False,
                 name: str = 'counterfactual') -> None:
        """
        Diffusion based Counterfactual Generator.

        Args:
            segmentation: segmentation model.
            diffusion_model: diffusion model.
            engine: engine for the diffusion process.
            device: device to run on.
            cfg: Configuration dictionary.
            name: name of the process.
        """

        super().__init__(segmentation, device, cfg, log, name)
        self.diffusion_model = diffusion_model
        self.engine = engine

        self.weight_segmentation = cfg.generation.weight_segmentation
        self.weight_distance = cfg.generation.weight_distance
        self.start_fraction = cfg.generation.start_fraction
        self.alpha = cfg.generation.alpha
        self.use_smoothing = cfg.generation.GradSmoothing.enable
        self.max_sigma = cfg.generation.GradSmoothing.max_sigma
        self.step_size = cfg.generation.approximation_step_size
        if not self.step_size:
            self.step_size = int("inf")

        self.z_min = ((cfg.data.min - torch.tensor(cfg.data.means)) /
                      torch.tensor(cfg.data.stds)).view(3, 1, 1).to(self.device)
        self.z_max = ((cfg.data.max - torch.tensor(cfg.data.means)) /
                      torch.tensor(cfg.data.stds)).view(3, 1, 1).to(self.device)

        self.diffusion_model.to(self.device).eval()

        self.robust_segmentation = robust_segmentation
        if self.robust_segmentation is not None:
            self.robust_segmentation.to(self.device).eval()

    @torch.no_grad()
    def _generate_counterfactual(self,
                                 image: torch.Tensor,
                                 target: torch.Tensor) -> torch.Tensor:
        loss_fn = CounterfactualLoss(target.repeat((self.adaptgrad.N, 1, 1)),
                                     use_mask=self.use_mask,
                                     border_width=self.border_width)

        start_t = int(self.engine.T * self.start_fraction)
        t_tensor = torch.full((image.shape[0],), start_t, device=self.device)
        x, _, _ = self.engine.diffuse(image, t_tensor)

        for t in tqdm(reversed(range(start_t)), total=start_t, desc="generating counterfactual"):
            t_tensor = torch.full((image.shape[0],), t, device=self.device)
            epsilon_pred = self._pred_epsilon(x, t_tensor)
            mean = self.engine.reverse(x, epsilon_pred, t_tensor)

            with torch.enable_grad():
                g_update, logs_dict = self._calc_g_update(
                    x,
                    image,
                    loss_fn,
                    epsilon_pred,
                    t_tensor)

            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)

            if t > 0:
                sigma = self.engine.get_sigma(t_tensor, image.shape)
                mean = mean - (sigma ** 2 * mu_norm * g_update)
                x = mean + sigma * torch.randn_like(image)
            else:
                x = mean

            # clamp image in image domain
            x = x.clamp(self.z_min, self.z_max)

            # --- TensorBoard Logging ---
            if t % 50 == 0:
                self.log(x, target, mean, mu_norm, g_update, logs_dict, start_t, t)

        return x.detach()

    def _pred_epsilon(self, x: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        t_val = t_tensor[0].item()

        # Return standard 1-step prediction at t=0 or when t is smaller than the step size
        if t_val == 0 or t_val <= self.step_size:
            return self.diffusion_model(x, t_tensor).detach()

        # Build sub-timesteps from current t down to 0 with fixed stride
        time_steps = list(range(int(t_val), 0, -self.step_size))
        if time_steps[-1] != 0:
            time_steps.append(0)

        curr_x = x.clone()
        for i in range(len(time_steps) - 1):
            sub_t = time_steps[i]
            next_t = time_steps[i + 1]
            delta_t = sub_t - next_t

            sub_t_tensor = torch.full((x.shape[0],), sub_t, device=x.device, dtype=torch.long)
            eps = self.diffusion_model(curr_x, sub_t_tensor).detach()
            curr_x = self.engine.reverse(curr_x, eps, sub_t_tensor, steps=delta_t)

        x_0_approx = curr_x
        return self.engine.epsilon_approximation(x, x_0_approx, t_tensor)

    def _calc_g_update(self,
                       x: torch.Tensor,
                       image: torch.Tensor,
                       loss_fn: CounterfactualLoss,
                       epsilon_pred: torch.Tensor,
                       t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = x.detach().requires_grad_(True)
        approx_image = self.engine.image_approximation(x, epsilon_pred, t)

        # Segmentation Loss
        approx_image_batch = self.adaptgrad(approx_image)
        logits = self.segmentation_model(approx_image_batch)

        # Loss and Gradients
        loss_seg = loss_fn(logits)
        grad_c = torch.autograd.grad(loss_seg, x, retain_graph=True)[0]

        if self.robust_segmentation is not None:
            logits_rob = self.robust_segmentation(approx_image_batch)
            loss_rob = loss_fn(logits_rob)
            grad_rob = torch.autograd.grad(loss_rob, x, retain_graph=True)[0]

            grad_c = cone_projection(grad_c, grad_rob, self.alpha)

        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)
        if self.use_smoothing:
            grad_c = self._apply_threshold(grad_c)

        # Distance Loss
        distance = geman_mcclure_distance(image,
                                          approx_image)  # torch.norm(image - approx_image, p=2)
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

    def _apply_threshold(self, grad: torch.Tensor):
        abs_grad = torch.abs(grad)
        threshold = torch.quantile(abs_grad, 0.9)
        mask = abs_grad > threshold
        grad[mask] = 0
        return grad


    def _apply_dynamic_smoothing(self, grad: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.engine.sqrt_alphas_cumprod[t[0]].item()

        # Scale max_sigma inversely with signal magnitude
        sigma = self.max_sigma * (1.0 - sqrt_alpha_bar)

        if sigma < 0.2:
            return grad

        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        return F.gaussian_blur(grad, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def log(self, x: torch.Tensor,
            target: torch.Tensor,
            mean: torch.Tensor,
            mu_norm: torch.Tensor,
            g_update: torch.Tensor,
            results: Dict[str, torch.Tensor],
            start_t: int,
            t: int) -> None:
        """
        Logs process to tensorboard.

        Args:
            x: current image.
            target: target segmentation map.
            g_update: g_update.
            results: dict with approx_image, grad_c, grad_d and logits
            start_t: starting point of diffusion
            t: current timestep.

        Returns:
            None
        """
        log_step = start_t - t

        # log scalars
        prediction = torch.argmax(results['logits'][0], dim=0)
        iou = calc_iou(prediction, target, num_classes=self.num_classes)
        f1, precision, recall = calc_f1_score(prediction, target, num_classes=self.num_classes)
        self.writer.add_scalar('Metrics/IoU', iou, log_step)
        self.writer.add_scalar('Metrics/f1', f1, log_step)
        self.writer.add_scalar('Metrics/precision', precision, log_step)
        self.writer.add_scalar('Metrics/recall', recall, log_step)

        self.writer.add_scalar('Gradients/mean', mean.mean().item(), log_step)
        self.writer.add_scalar('Gradients/mu_norm', mu_norm.mean().item(), log_step)
        self.writer.add_scalar('Gradients/g_update', g_update.abs().mean().item(), log_step)
        self.writer.add_scalar('Gradients/grad_c', results['grad_c'].abs().mean().item(), log_step)
        self.writer.add_scalar('Gradients/grad_d', results['grad_d'].abs().mean().item(), log_step)

        self.writer.add_image('Process/x_t', vutils.make_grid(x, normalize=True),
                              log_step)
        self.writer.add_image('Process/approx_x_0',
                              vutils.make_grid(results['approx_image'], normalize=True), log_step)

        prediction = visualize_seg(prediction.detach().cpu()).float()
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
