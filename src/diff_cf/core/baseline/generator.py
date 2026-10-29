"""Baseline Generator class."""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from ..base_generator import BaseGenerator, GradSmoothingConfig
from ..losses import BaseDistanceLoss, SegmentationLossConfig
from ...utils.adapt_grad import AdaptGrad

class BaselineGenerator(BaseGenerator):
    """
    Baseline Generator.
    Optimizes a parameter map added to the input image to satisfy a target prediction.
    """
    alpha: float
    beta: float

    def __init__(self,
                 segmentation_model: nn.Module,
                 segmentation_loss_config: SegmentationLossConfig,
                 distance_loss: BaseDistanceLoss,
                 z_bounds: Tuple[torch.Tensor, torch.Tensor],
                 robust_model: Optional[nn.Module] = None,
                 adapt_grad: Optional[AdaptGrad] = None,
                 lr: float = 1e-3,
                 steps: int = 100,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 cone_projection_alpha: Optional[float] = None,
                 smoothing_cfg: GradSmoothingConfig = GradSmoothingConfig(),
                 n_classes: int = 2,
                 log_dir: Optional[str] = None,
                 log_to_tensorboard: bool = False,
                 device: torch.device = torch.device('cpu'),
                 name: str = 'BaselineGenerator',
                 ):
        """
        Args:
            segmentation_model: segmentation model.
            segmentation_loss_config: Factory for the SegmentationLoss.
            distance_loss: loss function for messaging the distance to original image.
            robust_model: robust segmentation model to improve gradients using cone projection.
            adapt_grad: Instance of AdaptGrad to improve segmentation gradients.
            lr: learning rate for AdamW to update adaption parameters.
            steps: number of steps.
            alpha: weight of adoption cost
            beta: weight of smoothness loss.
            cone_projection_alpha: alpha angle in radians for cone projection.
            smoothing_cfg: configuration of gradient smoothing.
            n_classes: number of classes.
            log_dir: logging directory.
            log_to_tensorboard: activated tensorboard logging.
            device: device to run on.
            name: name of the process.
        """
        super().__init__(segmentation_model, segmentation_loss_config, distance_loss, z_bounds,
                         robust_model, adapt_grad, cone_projection_alpha, smoothing_cfg, n_classes,
                         log_dir, log_to_tensorboard, device, name)
        self.lr = lr
        self.steps = steps
        self.alpha = alpha
        self.beta = beta

        self.filter = torch.tensor([[[-1 / 8, -1 / 8, -1 / 8],
                                     [-1 / 8, 1, -1 / 8],
                                     [-1 / 8, -1 / 8, -1 / 8]]])

    def _generate_counterfactual(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Runs the optimization loop.

        Args:
            image: Input image.
            target: Target segmentation map.

        Returns:
            counterfactual: Output image.
        """
        # calculate max and min values for parameter
        p_min = self.z_min - image
        p_max = self.z_max - image

        parameter = nn.Parameter(torch.zeros_like(image))
        optimizer = AdamW([parameter], lr=self.lr)

        loss_fn = self.segmentation_loss_factory(
            target=target.repeat((self.target_repetitions, 1, 1))
        )

        s_filter = self.filter.repeat(image.shape[1], 1, 1, 1).to(self.device)

        N = parameter.numel() // 3  # pylint: disable=invalid-name

        with torch.enable_grad():
            progress_bar = trange(self.steps, desc='Generating...')
            for t in progress_bar:
                optimizer.zero_grad()

                # calc counterfactual
                counterfactual = image + parameter

                # calculate regularization gradient
                cost = (1 / N) * self.distance_loss.loss(parameter)
                smoothness = torch.abs(F.conv2d(parameter, s_filter, groups=image.shape[1])).mean()
                loss_reg = self.alpha * cost + self.beta * smoothness
                grad_reg = torch.autograd.grad(loss_reg, parameter)[0]

                # create noised batch with AdaptGrad to reduce gradient noise
                image_batch = self.adaptgrad(counterfactual)

                # calc gradients with segmentation model
                output = self.segmentation_model(image_batch)
                loss_seg = loss_fn(output)
                grad = torch.autograd.grad(loss_seg, parameter,
                                           retain_graph=self.robust_model is not None)[0]

                # if set calculate gradients with robust model
                if self.robust_model:
                    loss_rob = loss_fn(self.robust_model(image_batch))
                    grad_rob = torch.autograd.grad(loss_rob, parameter)[0]

                    # use cone projection to combine gradients
                    grad = self._cone_projection(grad, grad_rob)

                if self.smoothing.filter_gradients:
                    mask = self._get_mask(grad_c).to(grad_c.dtype)
                    grad_c = mask * grad_c

                parameter.grad = grad + grad_reg
                optimizer.step()

                with torch.no_grad():
                    parameter.clip(p_min, p_max)

                loss = (loss_seg + loss_reg).detach().cpu().item()

                # --- TensorBoard Logging ---
                if self.log_to_tensorboard and (t % 25 == 0 or t == self.steps - 1):
                    self.log(target,
                             output,
                             t,
                             counterfactual,
                             loss,
                             cost.item(),
                             smoothness.item())

                progress_bar.set_description(f"loss: {loss:.4f}")

        return counterfactual.detach()

    def log(self,
            target: torch.Tensor,
            logits: torch.Tensor,
            step: int,
            counterfactual: Optional[torch.Tensor] = None,
            loss: Optional[float] = None,
            cost: Optional[float] = None,
            smoothness: Optional[float] = None,
            **kwargs):
        """Logs baseline metrics to TensorBoard."""
        super().log(
            target=target,
            logits=logits,
            step=step,
        )

        if loss is not None:
            self.writer.add_scalar('Metrics/loss', loss, step)

        if cost is not None:
            self.writer.add_scalar('Metrics/cost', cost, step)

        if smoothness is not None:
            self.writer.add_scalar('Metrics/smoothness', smoothness, step)

        if counterfactual is not None:
            self.writer.add_image('Process/counterfactual',
                                  vutils.make_grid(counterfactual, normalize=True), step)
