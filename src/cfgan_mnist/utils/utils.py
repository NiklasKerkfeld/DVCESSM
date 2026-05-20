"""Utility functions."""
from typing import Optional, Tuple

from prettytable import PrettyTable

from torch import nn

import torch
import torch.nn.functional as F


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


def to_vis(x: torch.Tensor) -> torch.Tensor:
    """Maps z-score image back to image."""
    return torch.clamp((x + 1) / 2, 0.0, 1.0)


def visualize_mri(image: torch.Tensor) -> torch.Tensor:
    """
    Converts mri images into rgb images.

    Args:
        image: image tensor of shape (B, S, H, W)
        image tensor of shape (B, S, 3, H, W)
    """
    img_min = torch.amin(image, (2, 3))[:, :, None, None]
    img_max = torch.amax(image, (2, 3))[:, :, None, None]
    image = (image - img_min) / (img_max - img_min)
    return image[:, :, None].repeat(1, 1, 3, 1, 1)


def visualize_counterfactuals(image: torch.Tensor, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Normalizes 'image' and any number of additional tensors (*args)
    based on the min/max of the first 'image'.
    """
    img_min = torch.amin(image, dim=(2, 3), keepdim=True)
    img_max = torch.amax(image, dim=(2, 3), keepdim=True)

    def transform(img: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min + 1e-8)
        return img.unsqueeze(2).repeat(1, 1, 3, 1, 1)

    return transform(image), *[transform(a) for a in args]


def visualize_seg(seg_map: torch.Tensor) -> torch.Tensor:
    """
    Visualizes segmentation maps.

    Args:
        seg_map: segmentation map.

    Returns:
        colors image of segmentation map.
    """
    palette = torch.tensor([
        [0., 0., 0.],  # Class 0
        [1., 0., 0.],  # Class 1
        [0., 1., 0.],  # Class 2
        [0., 0., 1.],  # Class 3
        [0., 1., 1.],  # Class 4
        [1., 0., 1.],  # Class 5
        [1., 1., 0.],  # Class 6
        [.5, 0., 0.],  # Class 7
        [0., .5, 0.],  # Class 8
        [0., 0., .5],  # Class 9
        [0., .5, .5],  # Class 10
        [.5, 0., .5],  # Class 11
        [.5, .5, 0.],  # Class 12
        [.25, 0., 0.],  # Class 13
        [0., .25, 0.],  # Class 14
        [0., 0., .25],  # Class 15
        [0., 1., .5],  # Class 16
        [1., 0., .5],  # Class 17
        [1., .5, 0.],  # Class 18
        [0., .5, 1.],  # Class 19
        [.25, 0., 1.],  # Class 20
        [.25, 1., 0.],  # Class 21
        [0., 1., .25],  # Class 22
        [1., 0., .25],  # Class 23
        [1., .25, 0.],  # Class 24
        [0., .25, 1.],  # Class 25
        [.25, 0., 1.],  # Class 26
        [.25, 1., 0.]  # Class 27
    ], dtype=torch.float32)

    return palette[seg_map]


def visualize_adaption(adaption: torch.Tensor,
                       maximas: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Visualizes adaption maps.

    Args:
        adaption: change in image (B, C, H, W)
        maximas: maximal change in image (C,)

    Returns:
        visualization
    """
    r = torch.relu(adaption)
    g = torch.zeros_like(adaption)
    b = torch.relu(-adaption)

    palate = torch.stack([r, g, b], dim=-1)

    if maximas is not None:
        palate /= maximas

    return palate.permute(0, 1, 4, 2, 3)


def gaussian_blur_2d(grad: torch.Tensor,
                     kernel_size: int = 3,
                     sigma: float = 1.0) -> torch.Tensor:
    """
    Applies a Gaussian blur to a 4D gradient tensor (B, C, H, W).

    Args:
        grad: gradient tensor (B, C, H, W)
        kernel_size: kernel size
        sigma: standard deviation
    """
    # Create a 1D Gaussian kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=grad.dtype,
                     device=grad.device)
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Expand to 2D
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.expand(grad.shape[1], 1, kernel_size, kernel_size)

    # Apply as a depthwise convolution to keep channels separate
    # pylint: disable=not-callable
    smoothed_grad = F.conv2d(grad, kernel_2d, groups=grad.shape[1], padding=kernel_size // 2)
    return smoothed_grad


def count_parameters(model: nn.Module) -> int:
    """
    Counts the parameters of a given model.

    Args:
        model: model for counting the parameters.

    Returns:
        None
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
