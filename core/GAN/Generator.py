import torch
from torch import nn

from models.GAN import Generator, Adversarial


class CounterfactualGenerator:
    def __init__(self,
                 generator: Generator,
                 adversarial: Adversarial,
                 segmentation: nn.Module,
                 checkpoint: str,
                 device: torch.device,
                 ):
        super().__init__()
        self.generator = generator
        self.adversarial = adversarial
        self.segmentation = segmentation
        self.checkpoint = checkpoint
        self.device = device

        self.load_checkpoint(self.checkpoint)

        self.generator.eval()
        self.adversarial.eval()
        self.segmentation.eval()

    def load_checkpoint(self, checkpoint: str):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.adversarial.load_state_dict(checkpoint['adversarial_state_dict'])
        self.segmentation.load_state_dict(checkpoint['segmentation_state_dict'])

    @torch.no_grad()
    def generate(self, images: torch.Tensor, targets: torch.Tensor):
        prediction_org = self.segmentation(images).detach().cpu().argmax(dim=1)

        adaptions = self.generator(images, targets, prediction_org)
        counterfactuals = images + adaptions

        prediction_cf = self.segmentation(counterfactuals).detach().cpu().argmax(dim=1)

        return counterfactuals, adaptions, prediction_org, prediction_cf
