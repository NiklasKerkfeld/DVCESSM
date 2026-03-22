import torch
from torch import nn
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self, label_smoothing: bool = False):
        super(DiscriminatorLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        if self.label_smoothing:
            target = 0.9 * target + 0.1 * (1 - target)
        return self.loss_fn(prediction, target.float())

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