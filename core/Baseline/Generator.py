import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from tqdm import trange

from core.BaseGenerator import BaseGenerator
from models.Losses import CounterfactualLoss
from uitls.utils import calc_iou, visualize_seg


class BaselineGenerator(BaseGenerator):
    """
    Baseline Generator.
    Optimizes a parameter map added to the input image to satisfy a target prediction.
    """
    def __init__(self,
                 segmentation_model: nn.Module,
                 device: torch.device,
                 cfg: DictConfig,
                 name: str = 'BaselineGenerator'
                 ):
        """
        Args:
            segmentation_model: Segmentation model used for prediction.
            name: Name for result directories.
        """
        super().__init__(segmentation_model, device, cfg, name)
        self.lr = cfg.generation.lr
        self.steps = cfg.generation.steps

    def _generate_counterfactual(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Runs the optimization loop."""
        parameter = nn.Parameter(torch.zeros_like(image))
        optimizer = Adam([parameter], lr=self.lr)

        loss_fn = CounterfactualLoss(target.repeat((self.adaptgrad.N, 1, 1)),
                                     use_mask=self.use_mask,
                                     border_width=self.border_width)

        N = parameter.numel() // 3

        with torch.enable_grad():
            bar = trange(self.steps, desc='Generating...')
            for t in bar:
                optimizer.zero_grad()

                counterfactual = image + parameter
                cost = (1 / N) * torch.norm(parameter, p=2)

                image_batch = self.adaptgrad(counterfactual)
                output = self.segmentation_model(image_batch)
                loss = loss_fn(output)
                loss = loss + self.cfg.generation.alpha * cost

                loss.backward()
                optimizer.step()

                # --- TensorBoard Logging ---
                if t % 25 == 0 or t == self.steps - 1:
                    self.log(counterfactual, target, output, loss.item(), cost.item(), t)

                bar.set_description(f"loss: {loss.detach().cpu().item():.4f}")

        return counterfactual.detach()


    def log(self, counterfactual: torch.Tensor, target: torch.Tensor, output: torch.Tensor, loss: float, cost: float, t: int):
        prediction = torch.argmax(output[0], dim=0)
        iou = calc_iou(prediction, target, num_classes=self.num_classes)

        self.writer.add_scalar('Metrics/loss', loss, t)
        self.writer.add_scalar('Metrics/cost', cost, t)
        self.writer.add_scalar('Metrics/IoU', iou, t)

        self.writer.add_image('Counterfactual', vutils.make_grid(counterfactual, normalize=True), t)
        self.writer.add_image('Prediction', visualize_seg(prediction.detach().cpu()).permute(2, 0, 1).float(), t)

