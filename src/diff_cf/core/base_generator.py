"""Abstract class for all generators."""

from typing import Tuple

import torch
import torchvision.utils as vutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ..utils.adapt_grad import AdaptGrad
from ..utils.utils import visualize_seg


class BaseGenerator:
    """
    Abstract class for all generators.
    """

    def __init__(self,
                 segmentation_model: nn.Module,
                 device: torch.device,
                 cfg: DictConfig,
                 log_to_tensorboard: bool = False,
                 name: str = 'BaseGenerator'):
        """

        Args:
            segmentation_model: segmentation model.
            device: device to run on.
            cfg: Configuration dictionary.
            name: name of the process.
        """
        super().__init__()
        self.segmentation_model = segmentation_model
        self.device = device
        self.log_to_tensorboard = log_to_tensorboard
        self.name = name

        self.n_classes = cfg.data.n_classes

        self.segmentation_loss_factory = instantiate(cfg.generation.SegmentationLoss)

        self.adaptgrad = AdaptGrad(cfg)

        self.segmentation_model.to(self.device).eval()

        # Initialize TensorBoard
        if self.log_to_tensorboard:
            print(f"logging to {cfg.system.log_dir}/Generation/{name}")
            self.writer = SummaryWriter(log_dir=f"{cfg.system.log_dir}/Generation/{name}")

    @torch.no_grad()
    def generate(self,
                 image: torch.Tensor,
                 target: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The common workflow for all generators.

        Args:
            image: Input image.
            target: Target segmentation map.

        Returns:
            counterfactual: counterfactual.
            adaptions: changes between original image and counterfactual.
            pred_org: predicted segmentation map on original image.
            pred_cf: predicted segmentation map on counterfactual.
        """
        if self.log_to_tensorboard:
            self.writer.add_image('Image', vutils.make_grid(image[0], normalize=True), 0)
            self.writer.add_image('Target',
                                  visualize_seg(target[0].detach().cpu()).float(),
                                  0)

        image, target = image.to(self.device), target.to(self.device)
        self.segmentation_model.to(self.device).eval()
        self.adaptgrad.to(self.device)

        pred_org = self.segmentation_model(image).detach().cpu().argmax(dim=1)
        if self.log_to_tensorboard:
            self.writer.add_image('Prediction', visualize_seg(pred_org[0].detach().cpu()).float(), 0)

        counterfactual = self._generate_counterfactual(image, target)
        adaptions = counterfactual - image
        pred_cf = self.segmentation_model(counterfactual).detach().cpu().argmax(dim=1)

        return counterfactual.cpu(), adaptions.cpu(), pred_org, pred_cf

    def _generate_counterfactual(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses.

        Args:
            image: Input image.
            target: Target segmentation map.

        Returns:
            counterfactual: counterfactual.
        """
        raise NotImplementedError
