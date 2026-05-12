import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from tqdm import trange

from core.AdaptGrad import AdaptGrad
from models.Losses import CounterfactualLoss


class BaselineGenerator:
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
        self.segmentation_model = segmentation_model
        self.device = device
        self.alpha = cfg.generation.alpha
        self.lr = cfg.generation.lr
        self.steps = cfg.generation.steps
        self.segmentation_checkpoint = cfg.segmentation.checkpoint_path
        self.segmentation_name = cfg.segmentation.name
        self.name = name

        self.adaptgrad = AdaptGrad(cfg)

        self.load_checkpoint(f"{self.segmentation_checkpoint}/{self.segmentation_name}.pt")
        self.segmentation_model.eval()

    def load_checkpoint(self, segmentation_checkpoint: str):
        checkpoint = torch.load(segmentation_checkpoint, map_location=self.device)
        self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def generate(self, image: torch.Tensor, target: torch.Tensor):
        """
        Generates a counterfactual for the given image and target.

        Args:
            image: Input image tensor (B, C, W, H).
            target: Target segmentation tensor (B, W, H).
        """
        image = image.to(self.device)
        target = target.to(self.device)

        self.segmentation_model.eval()

        self.segmentation_model.to(self.device)
        self.adaptgrad.to(self.device)

        prediction_org = self.segmentation_model(image).detach().cpu().argmax(dim=1)
        counterfactual = self._generate_counterfactual(image, target)
        adaptions = counterfactual - image
        prediction_cf = self.segmentation_model(counterfactual).detach().cpu().argmax(dim=1)

        return counterfactual.cpu(), adaptions.cpu(), prediction_org.cpu(), prediction_cf.cpu()

    def _generate_counterfactual(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Runs the optimization loop."""
        parameter = nn.Parameter(torch.zeros_like(image))
        optimizer = Adam([parameter], lr=self.lr)

        loss_fn = CounterfactualLoss(target.repeat((self.adaptgrad.N, 1, 1)))

        with torch.enable_grad():
            bar = trange(self.steps, desc='Generating...')
            for _ in bar:
                optimizer.zero_grad()

                counterfactual = image + parameter
                cost = torch.mean(torch.abs(parameter))

                image_batch = self.adaptgrad(counterfactual)
                output = self.segmentation_model(image_batch)
                loss = loss_fn(output)
                loss = loss + self.alpha * cost

                loss.backward()
                optimizer.step()

                bar.set_description(f"loss: {loss.detach().cpu().item():.4f}")

        return counterfactual.detach()

