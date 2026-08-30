"""Diffusion based Counterfactual Generator."""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
import torchvision.utils as vutils
from torch import nn
import torchvision.transforms.functional as F
from tqdm import tqdm

from ..losses import BaseDistanceLoss, SegmentationLoss, SegmentationLossConfig
from ..base_generator import BaseGenerator
from .engine import Engine
from ...utils.adapt_grad import AdaptGrad
from ...utils.utils import visualize_adaption


@dataclass(frozen=True)
class GradSmoothingConfig:
    """Configuration for dynamic gradient smoothing."""
    spacial_gradient_smoothing: bool = False
    temporal_gradient_smoothing: bool = False
    filter_gradients: bool = False
    percentile: float = 0.9
    alpha: float = 0.99
    max_sigma: float = 2.5


class CounterfactualGenerator(BaseGenerator):
    """Diffusion based Counterfactual Generator."""

    def __init__(self,
                 segmentation_model: nn.Module,
                 diffusion_model: nn.Module,
                 engine: Engine,
                 segmentation_loss_config: SegmentationLossConfig,
                 distance_loss: BaseDistanceLoss,
                 z_bounds: Tuple[torch.Tensor, torch.Tensor],
                 weight_segmentation: float = 0.15,
                 weight_distance: float = 0.15,
                 start_fraction: float = 0.5,
                 approximation_step_size: Optional[int] = None,
                 robust_model: Optional[nn.Module] = None,
                 adapt_grad: Optional[AdaptGrad] = None,
                 smoothing_cfg: GradSmoothingConfig = GradSmoothingConfig(),
                 n_classes: int = 2,
                 cone_projection_alpha: Optional[float] = None,
                 log_dir: Optional[str] = None,
                 log_to_tensorboard: bool = False,
                 device: torch.device = torch.device('cpu'),
                 name: str = 'DiffusionGenerator') -> None:
        """
        Diffusion based Counterfactual Generator.

        Args:
            segmentation_model: segmentation model.
            segmentation_loss_config: configuration for the SegmentationLoss.
            distance_loss: loss function for messaging the distance to original image.
            robust_model: robust segmentation model to improve segmentation gradients using cone projection.
            adapt_grad: Instance of AdaptGrad to improve segmentation gradients.
            cone_projection_alpha: alpha angle in radians for cone projection.
            smoothing_cfg: configuration of gradient smoothing.
            n_classes: number of classes.
            log_dir: logging directory.
            log_to_tensorboard: activated tensorboard logging.
            device: device to run on.
            name: name of the process.
        """

        super().__init__(segmentation_model, segmentation_loss_config, distance_loss, z_bounds,
                         robust_model, adapt_grad, cone_projection_alpha, n_classes,
                         log_dir, log_to_tensorboard, device, name)
        self.diffusion_model = diffusion_model.to(self.device).eval()
        self.engine = engine.to(self.device)

        self.weight_segmentation = weight_segmentation
        self.weight_distance = weight_distance
        self.start_fraction = start_fraction
        self.step_size = approximation_step_size if approximation_step_size else self.engine.T

        # smoothing
        self.smoothing = smoothing_cfg
        self.smoothing_step = 0
        self.grad_c: Optional[torch.Tensor] = None
        self.grad_d: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _generate_counterfactual(self,
                                 image: torch.Tensor,
                                 target: torch.Tensor) -> torch.Tensor:
        # reset gradients
        self.reset_state()

        loss_fn = self.segmentation_loss_factory(target=target.repeat((self.adaptgrad.N, 1, 1)))

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
                    t_tensor
                )

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
            if self.log_to_tensorboard and t % 50 == 0:
                self.log(
                    target=target,
                    logits=logs_dict['logits'],
                    step=start_t - t,
                    x=x.detach().cpu(),
                    mean=mean.detach().cpu(),
                    mu_norm=mu_norm.detach().cpu(),
                    g_update=g_update.detach().cpu(),
                    grad_c=logs_dict['grad_c'],
                    grad_d=logs_dict['grad_d'],
                    approx_image=logs_dict['approx_image'],
                    start_t=start_t,
                )

        return x.detach()

    def reset_state(self):
        """Call this at the start of a new diffusion trajectory."""
        self.grad_c = None
        self.grad_d = None

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
                       loss_fn: SegmentationLoss,
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

        if self.robust_model is not None:
            logits_rob = self.robust_model(approx_image_batch)
            loss_rob = loss_fn(logits_rob)
            grad_rob = torch.autograd.grad(loss_rob, x, retain_graph=True)[0]

            grad_c = self._cone_projection(grad_c, grad_rob)

        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        # Distance Loss
        distance = self.distance_loss(image, approx_image)

        grad_d = torch.autograd.grad(distance, x)[0]
        grad_d /= (torch.norm(grad_d.view(grad_d.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        if self.smoothing.spacial_gradient_smoothing:
            grad_c = self._apply_dynamic_smoothing(grad_c, t)
            grad_d = self._apply_dynamic_smoothing(grad_d, t)

        if self.smoothing.temporal_gradient_smoothing:
            self.smoothing_step += 1
            # init gradients in the frist round
            if self.grad_c is None:
                self.grad_c = torch.zeros_like(grad_c)
            if self.grad_d is None:
                self.grad_d = torch.zeros_like(grad_d)

            # rolling average
            self.grad_c = (1 - self.smoothing.alpha) * self.grad_c + self.smoothing.alpha * grad_c
            self.grad_d = (1 - self.smoothing.alpha) * self.grad_d + self.smoothing.alpha * grad_d

            bias_correction = 1.0 - (1.0 - self.smoothing.alpha) ** self.smoothing_step

            grad_c = self.grad_c / bias_correction
            grad_d = self.grad_d / bias_correction

        # only change very strong gradients ignore changes of grad_d there
        if self.smoothing.filter_gradients:
            mask = self._get_mask(grad_c)
            grad_c = torch.where(mask, torch.zeros_like(grad_c), grad_c)
            grad_d = torch.where(~mask, torch.zeros_like(grad_d), grad_d)

        g_update = self.weight_segmentation * grad_c + self.weight_distance * grad_d

        return g_update.detach(), {
            'approx_image': approx_image.detach(),
            'grad_c': grad_c.detach(),
            'grad_d': grad_d.detach(),
            'logits': logits.detach()
        }

    @staticmethod
    def _threshold_grad(grad: torch.Tensor, percentile: float = 0.9):
        abs_grad = torch.abs(grad)
        threshold = torch.quantile(abs_grad, percentile)
        mask = abs_grad < threshold
        grad[mask] = 0
        return grad

    def _get_mask(self, grad: torch.Tensor):
        abs_grad = torch.abs(grad)
        threshold = torch.quantile(abs_grad, self.smoothing.percentile)
        return abs_grad < threshold

    def _apply_dynamic_smoothing(self, grad: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Smoothes the gradient over the image dimension using a gaussian filter.

        The sigma value of the Gaus filter is adopted to the step, so that the gradient is smoother
        in early stages of the diffusion process.
        Args:
            grad: Tensor with the gradients.
            t: time step.

        Returns:
            torch.Tensor with the smoothed gradients.
        """
        sqrt_alpha_bar = self.engine.sqrt_alphas_cumprod[t[0]].item()

        # Scale max_sigma inversely with signal magnitude
        sigma = self.smoothing.max_sigma * (1.0 - sqrt_alpha_bar)

        if sigma < 0.2:
            return grad

        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        return F.gaussian_blur(grad, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def log(self,
            target: torch.Tensor,
            logits: torch.Tensor,
            step: int,
            x: Optional[torch.Tensor] = None,
            mean: Optional[torch.Tensor] = None,
            mu_norm: Optional[torch.Tensor] = None,
            g_update: Optional[torch.Tensor] = None,
            grad_c: Optional[torch.Tensor] = None,
            grad_d: Optional[torch.Tensor] = None,
            approx_image: Optional[torch.Tensor] = None,
            **kwargs
            ):
        """Logs diffusion process state to TensorBoard."""

        super().log(
            target=target,
            logits=logits,
            step=step,
        )

        # Subclass-specific metrics
        if mean is not None:
            self.writer.add_scalar('Gradients/mean', mean.mean().item(), step)

        if mu_norm is not None:
            self.writer.add_scalar('Gradients/mu_norm', mu_norm.mean().item(), step)

        if g_update is not None:
            self.writer.add_scalar('Gradients/g_update', g_update.abs().mean().item(), step)

        if grad_c is not None:
            self.writer.add_scalar('Gradients/grad_c', grad_c.abs().mean().item(),
                                   step)

        if grad_d is not None:
            self.writer.add_scalar('Gradients/grad_d', grad_d.abs().mean().item(),
                                   step)

        if approx_image is not None:
            self.writer.add_image('Process/approx_x_0',
                                  vutils.make_grid(approx_image, normalize=True),
                                  step)

        if x is not None:
            self.writer.add_image('Process/x_t', vutils.make_grid(x, normalize=True),
                                  step)

        if g_update is not None and grad_c is not None and grad_d is not None:
            g_visual = visualize_adaption(g_update.sum(dim=1, keepdim=True))[0, 0]
            grad_c_visual = visualize_adaption(grad_c.sum(dim=1, keepdim=True))[0, 0]
            grad_d_visual = visualize_adaption(grad_d.sum(dim=1, keepdim=True))[0, 0]

            self.writer.add_image('Process/g_update',
                                  vutils.make_grid(g_visual, normalize=True), step)
            self.writer.add_image('Process/grad_c',
                                  vutils.make_grid(grad_c_visual, normalize=True), step)
            self.writer.add_image('Process/grad_d',
                                  vutils.make_grid(grad_d_visual, normalize=True), step)
