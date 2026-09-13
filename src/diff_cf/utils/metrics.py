"""Metrics for measuring segmentations"""

from typing import Tuple

import torch


def calc_iou(labels: torch.Tensor, preds: torch.Tensor, num_classes: int = 5) -> float:
    """
    Calculates the Mean Intersection over Union (mIoU) for multi-class segmentation.
    Args:
        labels: ground truth mask of shape (H, W)
        preds: predicted mask of shape (H, W)
        num_classes: number of segmentation classes.

    Returns:
        mIoU: Mean Intersection over Union (mIoU).
    """
    conf_matrix = torch.zeros((num_classes, num_classes), device=labels.device)

    conf_matrix += torch.bincount(
        num_classes * labels.view(-1) + preds.view(-1),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    # Calculate Metrics
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    # calc iou
    present_classes = (conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0)) > 0
    denominator = tp + fp + fn
    iou = tp[present_classes] / denominator[present_classes]

    return iou.mean().item()


def calc_f1_score(labels: torch.Tensor,
                  preds: torch.Tensor,
                  num_classes: int = 5,
                  eps: float= 1e-7) -> Tuple[float, float, float]:
    """Calculates Macro Precision, Recall, and F1-Score for multi-class segmentation.

    Args:
        labels: Ground truth mask of shape (H, W) or (B, H, W).
        preds: Predicted mask of shape (H, W) or (B, H, W).
        num_classes: Number of segmentation classes.

    Returns:
        Tuple of (macro_f1, macro_precision, macro_recall).
    """
    conf_matrix = torch.zeros((num_classes, num_classes), device=labels.device)

    # Flatten tensors
    labels_flat = labels.view(-1).long()
    preds_flat = preds.view(-1).long()

    conf_matrix += torch.bincount(
        num_classes * labels_flat + preds_flat, minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)

    # Base present classes ONLY on Ground Truth targets present in this sample
    gt_present = conf_matrix.sum(dim=1) > 0

    if not gt_present.any():
        return 0.0, 0.0, 0.0

    return (f1[gt_present].mean().item(),
            precision[gt_present].mean().item(),
            recall[gt_present].mean().item())
