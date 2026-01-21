import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self, label_smoothing: bool = False):
        super(DiscriminatorLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        if self.label_smoothing:
            target = 0.9 * target + 0.1 * (1 - target)
        return self.loss_fn(prediction, target.float())


class ChangeRegularizationLossExperimental(nn.Module):
    def __init__(self, eps=1e-2):
        super(ChangeRegularizationLossExperimental, self).__init__()
        self.eps = eps

    def _func(self, x: torch.Tensor):
        eps = torch.FloatTensor([self.eps]).to(x.device)
        return torch.log(torch.abs(x) + eps) - torch.log(eps)
        # return torch.abs(x)

    def forward(self, changes):
        regularization = torch.mean(self._func(changes))

        return regularization


class ChangeRegularizationLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super(ChangeRegularizationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, changes):
        l1_reg = torch.mean(torch.abs(changes))
        l2_reg = torch.mean(changes ** 2)

        return self.alpha * l1_reg + self.beta * l2_reg
