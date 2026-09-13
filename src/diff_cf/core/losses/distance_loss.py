"""Loss functions for similarity loss between counterfactual and original image."""
import torch
from torch import nn


class BaseDistanceLoss(nn.Module):
    """Base class for all similarity loss functions."""
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
        """Forward pass for similarity loss between counterfactual and original image."""
        raise NotImplementedError

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Loss calculation based on difference between original and counterfactual image."""
        raise NotImplementedError


class NormDistance(BaseDistanceLoss):
    """Similarity loss based on norm."""
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(self, image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
        return self.loss(image - approx_image)

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.norm(diff, p=self.p)


class LogDistance(BaseDistanceLoss):
    """Similarity loss based logarithmic absolute difference."""
    def __init__(self, scale: float = 1.4427):
        super().__init__()
        self.scale = scale

    def forward(self, image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(approx_image - image)
        return self.loss(diff)

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.log(1 + diff) * self.scale)


class GemanMcClureDistance(BaseDistanceLoss):
    """Similarity loss based on GemanMcClure distance."""
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
        diff = approx_image - image
        return self.loss(diff)

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.sum(diff ** 2 / (diff ** 2 + self.gamma ** 2))


class DifferentiableSoftL0Count(BaseDistanceLoss):
    """Similarity loss based on differentiable SoftL0."""
    def __init__(self, epsilon: float = 0.05, tau: float = 20.0):
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau

    def forward(self, image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(approx_image - image)
        return self.loss(diff)

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.sigmoid(self.tau * (diff - self.epsilon)))


class CharbonnierLoss(BaseDistanceLoss):
    """Similarity loss based on Charbonnier distance."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
        diff = approx_image - image
        return self.loss(diff)

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.sqrt(diff ** 2 + self.eps ** 2))
