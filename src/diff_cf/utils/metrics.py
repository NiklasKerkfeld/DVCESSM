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


def calc_f1_score(
    labels: torch.Tensor, preds: torch.Tensor, num_classes: int = 5
) -> Tuple[float, float, float]:
    """Calculates Mean Precision, Recall, and F1-Score for multi-class segmentation.

    Args:
        labels: ground truth mask of shape (H, W)
        preds: predicted mask of shape (H, W)
        num_classes: number of segmentation classes.

    Returns:
        Dict containing 'precision', 'recall', and 'f1' scores.
    """
    conf_matrix = torch.zeros((num_classes, num_classes), device=labels.device)

    conf_matrix += torch.bincount(
        num_classes * labels.view(-1) + preds.view(-1), minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    # Calculate True Positives, False Positives, False Negatives
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)

    # Filter to classes present in ground truth or predictions
    present_classes = (conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0)) > 0

    return (f1[present_classes].mean().item(),
            precision[present_classes].mean().item(),
            recall[present_classes].mean().item())