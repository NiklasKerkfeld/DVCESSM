from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class CounterfactualLoss(nn.Module):
    def __init__(self,
                 target: torch.Tensor,
                 border_width: int = 2,
                 reduction: Literal['mean', 'sum'] = 'mean'):
        super(CounterfactualLoss, self).__init__()
        self.target = target
        self.border_width = border_width
        self._set_boarders(target)
        self.reduction = torch.sum if reduction == 'sum' else torch.mean

    def forward(self, logits: torch.Tensor):
        mask = (logits.argmax(dim=1) != self.target).float()

        loss = F.cross_entropy(logits, self.target, reduction='none', ignore_index=-1)

        return self.reduction(loss * mask)

    def _set_boarders(self, target: torch.Tensor):
        k_size = self.border_width + 1

        dilated = F.max_pool2d(target.float(), kernel_size=k_size, stride=1,
                               padding=self.border_width // 2)

        eroded = -F.max_pool2d(-target.float(), kernel_size=k_size, stride=1,
                               padding=self.border_width // 2)

        mask = (dilated != eroded)

        self.target[mask] = -1


class MultiTaskLoss(nn.Module):
    def __init__(self, n_losses: int):
        super().__init__()
        # Initialize with zeros so exp(-s) starts at 1.0
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: torch.Tensor):
        """
        losses: a tensor of shape (n_losses,)
        """
        # precision = 1 / (2 * sigma^2)
        precision = torch.exp(-self.log_vars)

        # Weighted loss + complexity penalty (log_var)
        weighted_losses = precision * losses + self.log_vars
        return weighted_losses.sum()


class DiscriminatorLoss(nn.Module):
    def __init__(self, label_smoothing: bool = False):
        super(DiscriminatorLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        if self.label_smoothing:
            target = 0.9 * target + 0.1 * (1 - target)
        return self.loss_fn(prediction, target.float())


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # This is your Alpha (class weights)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C, H, W] (Logits)
        targets: [N, H, W] (Class indices: 0, 1, 2...)
        """
        # Convert logits to probabilities
        log_p = F.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)

        # Gather the log-probabilities of the target classes
        # This is equivalent to standard Cross Entropy
        log_p_target = F.nll_loss(log_p, targets, weight=self.weight, reduction='none')

        # Get the probability of the target class (pt)
        # We need to gather p based on the target indices
        p_target = torch.gather(p, 1, targets.unsqueeze(1)).squeeze(1)

        # Calculate Focal Loss: (1 - pt)^gamma * CE
        loss = (1 - p_target) ** self.gamma * log_p_target

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.reduction = torch.sum if reduction == 'sum' else torch.mean

    def forward(self, logits, targets):
        mask = (logits.argmax(dim=1) != targets).float()
        loss = F.cross_entropy(logits, targets, reduction='none')

        return self.reduction(loss * mask)


def masked_cross_entropy(logits, targets):
    # Get current predictions: [B, H, W]
    current_preds = logits.argmax(dim=1)

    # Create a mask where predictions are WRONG (1 if wrong, 0 if correct)
    mask = (current_preds != targets).float()

    # Calculate per-pixel cross entropy
    loss = F.cross_entropy(logits, targets, reduction='none')

    # Only keep loss for wrong pixels
    return (loss * mask).sum()


def capped_cross_entropy(logits, targets, threshold=0.75):
    # Calculate probabilities
    probs = torch.softmax(logits, dim=1)
    # Get probability of the target class at each pixel
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Mask pixels that are already 'confident enough'
    # Loss is only calculated for pixels where target_prob < threshold
    mask = (target_probs < threshold).float()

    loss = F.cross_entropy(logits, targets, reduction='none')
    return (loss * mask).sum()


import hydra
from omegaconf import DictConfig, OmegaConf
from core.Datasets.dataset import get_datasets
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    _, dataset = get_datasets(cfg)

    print(f"dataset size: {len(dataset)}")

    target = torch.stack([dataset[1][1]], dim=0)

    loss_fn = CounterfactualLoss(target.repeat(16, 1, 1))

    logits = torch.randn(16, 19, 256, 256)
    print(loss_fn(logits))

    print(f"{loss_fn.target.min(), loss_fn.target.max()}")
    plt.imshow(loss_fn.target[0])
    plt.show()


if __name__ == '__main__':
    main()
