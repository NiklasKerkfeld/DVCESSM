from typing import Optional

import torch
from torch import nn

from GAN.model import Generator


class CounterfactualGAN(nn.Module):
    def __init__(self,
                 generator: Generator,
                 discriminator: nn.Module,
                 target_model: nn.Module,
                 use_target: bool = False,
                 use_prediction: bool = False,
                 max_adoption: float = 100.0):
        super().__init__()
        self.use_target = use_target
        self.use_prediction = use_prediction
        self.max_adoption = max_adoption

        self.generator = generator
        self.discriminator = discriminator
        self.target_model = target_model
        self.seg_loss = nn.CrossEntropyLoss()

        # freeze target model
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                scalar: torch.Tensor,
                ) -> torch.Tensor:
        x = x.detach().requires_grad_(True)

        with torch.enable_grad():
            pred = self.target_model(x)
            loss = self.seg_loss(pred, y)
            grad_prior = torch.autograd.grad(loss, [x])[0].detach()

        if self.use_target and self.use_prediction:
            pred = torch.argmax(pred, dim=1)
            return self.generator(x, grad_prior, scalar, y, pred)

        if self.use_target:
            return self.generator(x, grad_prior, scalar, y)

        if self.use_prediction:
            return self.generator(x, grad_prior, scalar, pred)

        return self.generator(x, grad_prior, scalar)
