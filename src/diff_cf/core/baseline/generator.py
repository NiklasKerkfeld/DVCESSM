"""Baseline Generator class."""

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from ..base_generator import BaseGenerator
from ...models.losses import CounterfactualLoss
from ...utils.utils import visualize_seg
from ...utils.metrics import calc_iou


class BaselineGenerator(BaseGenerator):
    """
    Baseline Generator.
    Optimizes a parameter map added to the input image to satisfy a target prediction.
    """
    def __init__(self,
                 segmentation_model: nn.Module,
                 device: torch.device,
                 cfg: DictConfig,
                 log: bool = False,
                 name: str = 'BaselineGenerator'
                 ):
        """
        Args:
            segmentation_model: Segmentation model used for prediction.
            name: Name for result directories.
            cfg: Configuration dictionary.
            device: Device to run on.
        """
        super().__init__(segmentation_model, device, cfg, log, name)
        self.lr = cfg.generation.lr
        self.steps = cfg.generation.steps
        self.alpha = cfg.generation.alpha
        self.beta = cfg.generation.beta

        self.filter = torch.tensor([[[-1 / 8, -1 / 8, -1 / 8],
                                     [-1/8, 1, -1/8],
                                     [-1/8, -1/8, -1/8]]])

    def _generate_counterfactual(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Runs the optimization loop.

        Args:
            image: Input image.
            target: Target segmentation map.

        Returns:
            counterfactual: Output image.
        """
        parameter = nn.Parameter(torch.zeros_like(image))
        optimizer = AdamW([parameter], lr=self.lr)

        loss_fn = CounterfactualLoss(target.repeat((self.adaptgrad.N, 1, 1)),
                                     use_mask=self.use_mask,
                                     border_width=self.border_width)

        s_filter = self.filter.repeat(image.shape[1], 1, 1, 1).to(self.device)

        N = parameter.numel() // 3  # pylint: disable=invalid-name

        with torch.enable_grad():
            progress_bar = trange(self.steps, desc='Generating...')
            for t in progress_bar:
                optimizer.zero_grad()

                counterfactual = image + parameter
                cost = (1 / N) * torch.norm(parameter, p=2)

                smoothness = torch.abs(F.conv2d(parameter,
                                                s_filter,
                                                groups=image.shape[1])).mean()

                image_batch = self.adaptgrad(counterfactual)
                output = self.segmentation_model(image_batch)
                loss = loss_fn(output)

                loss = loss + self.alpha * cost + self.beta * smoothness

                loss.backward()
                optimizer.step()

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
        iou = calc_iou(prediction, target, num_classes=self.num_classes)

        self.writer.add_scalar('Metrics/loss', loss, t)
        self.writer.add_scalar('Metrics/cost', cost, t)
        self.writer.add_scalar('Metrics/smoothness', smoothness, t)
        self.writer.add_scalar('Metrics/IoU', iou, t)

        self.writer.add_image('Counterfactual',
                              vutils.make_grid(counterfactual, normalize=True), t)
        self.writer.add_image('Prediction',
                              visualize_seg(prediction.detach().cpu()).float(), t)
