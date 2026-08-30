"""Baseline Generator class."""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from losses import BaseDistanceLoss, SegmentationLossConfig
from utils.adapt_grad import AdaptGrad
from ..base_generator import BaseGenerator
from ...utils.utils import visualize_seg
from ...utils.metrics import calc_iou


class BaselineGenerator(BaseGenerator):
    """
    Baseline Generator.
    Optimizes a parameter map added to the input image to satisfy a target prediction.
    """

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
            robust_model: robust segmentation model to improve segmentation gradients using cone projection.
            adapt_grad: Instance of AdaptGrad to improve segmentation gradients.
            lr: learning rate for AdamW to update adaption parameters.
            steps: number of steps.
            alpha: weight of adoption cost
            beta: weight of smoothness loss.
            cone_projection_alpha: alpha angle in radians for cone projection.
            n_classes: number of classes.
            log_dir: logging directory.
            log_to_tensorboard: activated tensorboard logging.
            device: device to run on.
            name: name of the process.
        """
        super().__init__(segmentation_model, segmentation_loss_config, distance_loss, z_bounds,
                         robust_model, adapt_grad, cone_projection_alpha, n_classes,
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

        loss_fn = self.segmentation_loss_factory(target=target.repeat((self.adaptgrad.N, 1, 1)))

        s_filter = self.filter.repeat(image.shape[1], 1, 1, 1).to(self.device)

        N = parameter.numel() // 3  # pylint: disable=invalid-name

        with torch.enable_grad():
            progress_bar = trange(self.steps, desc='Generating...')
            for t in progress_bar:
                optimizer.zero_grad()

                cost = (1 / N) * self.distance_loss.loss(parameter)

                smoothness = torch.abs(F.conv2d(parameter,
                                                s_filter,
                                                groups=image.shape[1])).mean()

                counterfactual = image + parameter

                image_batch = self.adaptgrad(counterfactual)
                output = self.segmentation_model(image_batch)
                loss = loss_fn(output)

                loss = loss + self.alpha * cost + self.beta * smoothness

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    parameter.clip(p_min, p_max)

                # --- TensorBoard Logging ---
                if t % 25 == 0 or t == self.steps - 1:
                    self.log(counterfactual,
                             target,
                             output,
                             loss.item(),
                             cost.item(),
                             smoothness.item(),
                             t)

                progress_bar.set_description(f"loss: {loss.detach().cpu().item():.4f}")

        return counterfactual.detach()

    def log(self,
            counterfactual: torch.Tensor,
            target: torch.Tensor,
            output: torch.Tensor,
            loss: float,
            cost: float,
            smoothness: float,
            t: int) -> None:
        """
        Logs process on tensorboard.

        Args:
            counterfactual: current counterfactual image.
            target: Target segmentation map.
            output: current output of the segmentation model.
            loss: current loss.
            cost: current cost of the adaption.
            smoothness: current smoothness loss
            t: current step of the process

        Returns:
            None
        """

        prediction = torch.argmax(output[0], dim=0)
        iou = calc_iou(prediction, target, num_classes=self.n_classes)

        self.writer.add_scalar('Metrics/loss', loss, t)
        self.writer.add_scalar('Metrics/cost', cost, t)
        self.writer.add_scalar('Metrics/smoothness', smoothness, t)
        self.writer.add_scalar('Metrics/IoU', iou, t)

        self.writer.add_image('Counterfactual',
                              vutils.make_grid(counterfactual, normalize=True), t)
        self.writer.add_image('Prediction',
                              visualize_seg(prediction.detach().cpu()).float(), t)
