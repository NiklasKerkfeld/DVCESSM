import torch
from torch import nn


class CounterfactualGAN(nn.Module):
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 target_model: nn.Module,
                 max_adoption: float = 100.0):
        super().__init__()

        self.max_adoption = max_adoption

        self.generator = generator
        self.discriminator = discriminator
        self.target_model = target_model
        self.seg_loss = nn.CrossEntropyLoss()

        # freeze target model
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)

        with torch.enable_grad():
            pred = self.target_model(x)
            loss = self.seg_loss(pred, y)
            grad_prior = torch.autograd.grad(loss, [x])[0].detach()

        return self.generator(x, grad_prior, scalar)
