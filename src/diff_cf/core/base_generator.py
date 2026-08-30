"""Abstract class for all generators."""
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .losses import SegmentationLossFactory, BaseDistanceLoss, SegmentationLossConfig
from ..utils.adapt_grad import AdaptGrad
from ..utils.utils import visualize_seg, scalar_product


class BaseGenerator:
    """
    Abstract class for all generators.
    """

    def __init__(self,
                 segmentation_model: nn.Module,
                 segmentation_loss_config: SegmentationLossConfig,
                 distance_loss: BaseDistanceLoss,
                 z_bounds: Tuple[torch.Tensor, torch.Tensor],
                 robust_model: Optional[nn.Module] = None,
                 adapt_grad: Optional[AdaptGrad] = None,
                 cone_projection_alpha: Optional[float] = None,
                 n_classes: int = 2,
                 log_dir: Optional[str] = None,
                 log_to_tensorboard: bool = False,
                 device: torch.device = torch.device('cpu'),
                 name: str = 'BaseGenerator'):
        """
        Args:
            segmentation_model: segmentation model.
            segmentation_loss_config: Factory for the SegmentationLoss.
            distance_loss: loss function for messaging the distance to original image.
            robust_model: robust segmentation model to improve segmentation gradients using cone projection.
            adapt_grad: Instance of AdaptGrad to improve segmentation gradients.
            cone_projection_alpha: alpha angle in radians for cone projection.
            n_classes: number of classes.
            log_dir: logging directory.
            log_to_tensorboard: activated tensorboard logging.
            device: device to run on.
            name: name of the process.
        """
        super().__init__()
        self.segmentation_model = segmentation_model.to(device).eval()
        self.segmentation_loss_factory = SegmentationLossFactory(segmentation_loss_config)
        self.distance_loss = distance_loss
        self.robust_model = robust_model.to(device).eval() if robust_model else None
        self.z_min, self.z_max = z_bounds[0].to(device), z_bounds[1].to(device)
        self.n_classes = n_classes
        self.log_dir = log_dir
        self.log_to_tensorboard = log_to_tensorboard
        self.device = device
        self.name = name

        self.alpha = cone_projection_alpha if cone_projection_alpha else None
        if self.robust_model is not None and self.alpha is None:
            raise ValueError(
                "When robust model is provided alpha value for cone projection must be set!")

        self.adaptgrad = adapt_grad if adapt_grad else torch.nn.Identity()

        # Initialize TensorBoard
        if self.log_to_tensorboard:
            if self.log_dir is None:
                raise ValueError("log_dir must be provided when log_to_tensorboard is True")
            print(f"logging to {self.log_dir}/Generation/{name}")
            print(f"logging to {self.log_dir}/Generation/{name}")
            self.writer = SummaryWriter(log_dir=f"{self.log_dir}/Generation/{name}")

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
            self.writer.add_image('Prediction', visualize_seg(pred_org[0].detach().cpu()).float(),
                                  0)

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

    def _cone_projection(self, grad_seg: torch.Tensor, grad_rob: torch.Tensor) -> torch.Tensor:
        """
        Does the cone projection from this paper https://arxiv.org/pdf/2210.11841 to ensure use full gradients.

        Args:
            grad_seg: gradient of the segmentation model.
            grad_rob: gradient of the robust model.

        Returns:
            new gradient for guiding the diffusion model.
        """
        if self.alpha is None:
            raise ValueError("alpha must be provided when cone projection is used!")

        vec_seg = grad_seg.flatten(1)
        vec_rob = grad_rob.flatten(1)
        angle = torch.acos(
            F.cosine_similarity(vec_seg, vec_rob, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))

        p_ortho = vec_rob - (scalar_product(vec_rob, vec_seg) / (
                scalar_product(vec_seg, vec_seg) + 1e-8)) * vec_seg
        p_ortho_norm = torch.norm(p_ortho, p=2, dim=-1, keepdim=True)
        vec_seg_norm = torch.norm(vec_seg, p=2, dim=-1, keepdim=True)

        u = math.sin(self.alpha) * (p_ortho / (p_ortho_norm + 1e-8)) + math.cos(self.alpha) * (
                vec_seg / (vec_seg_norm + 1e-8))

        projection = scalar_product(u, vec_rob) * u

        p_cone = torch.where((angle > self.alpha).unsqueeze(-1), projection, vec_rob)

        return p_cone.view_as(grad_seg)
