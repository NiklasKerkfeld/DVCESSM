import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from core.AdaptGrad import AdaptGrad
from uitls.utils import visualize_seg


class BaseGenerator(nn.Module):
    def __init__(self, segmentation_model: nn.Module, device: torch.device, cfg: DictConfig,
                 name: str = 'BaseGenerator'):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.device = device
        self.cfg = cfg
        self.name = name
        self.border_width = cfg.generation.border_width
        self.use_mask = cfg.generation.use_mask
        self.num_classes = cfg.data.n_classes

        self.adaptgrad = AdaptGrad(cfg)

        self.segmentation_model.to(self.device).eval()

        # Initialize TensorBoard
        print(f"logging to {cfg.system.log_dir}/Generation/{name}")
        self.writer = SummaryWriter(log_dir=f"{cfg.system.log_dir}/Generation/{name}")

    @torch.no_grad()
    def generate(self, image: torch.Tensor, target: torch.Tensor):
        """The common workflow for all generators."""
        self.writer.add_image('Image', vutils.make_grid(image[0], normalize=True), 0)
        self.writer.add_image('Target', visualize_seg(target.detach().cpu())[0].permute(2, 0, 1).float(), 0)

        image, target = image.to(self.device), target.to(self.device)
        self.segmentation_model.to(self.device).eval()
        self.adaptgrad.to(self.device)

        pred_org = self.segmentation_model(image).detach().cpu().argmax(dim=1)
        counterfactual = self._generate_counterfactual(image, target)
        adaptions = counterfactual - image
        pred_cf = self.segmentation_model(counterfactual).detach().cpu().argmax(dim=1)

        return counterfactual.cpu(), adaptions.cpu(), pred_org, pred_cf

    def _generate_counterfactual(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError