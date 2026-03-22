import torch


def random_target(input_tensor, n):
    offset = torch.randint(1, n - 1, input_tensor.shape, device=input_tensor.device)
    target = ((input_tensor - 1 + offset) % (n - 1)) + 1
    return target


def calculate_iou(preds, labels, num_classes, ignore_index=None):
    """
    Calculates the Mean Intersection over Union (mIoU) for multi-class segmentation.

    Args:
        preds (Tensor): Predicted mask of shape (H, W) or (N, H, W).
        labels (Tensor): Ground truth mask of same shape as preds.
        num_classes (int): Number of segmentation classes.
        ignore_index (int, optional): Index to ignore (e.g., background or void).

    Returns:
        torch.Tensor: A tensor containing the IoU for each class.
        float: The mean IoU across all valid classes.
    """
    iou_list = []

    # Flatten tensors to simplify calculation
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        # Create binary masks for the current class
        pred_inds = (preds == cls)
        target_inds = (labels == cls)

        # Intersection: pixels that are 'cls' in both maps
        intersection = (pred_inds & target_inds).long().sum().item()

        # Union: pixels that are 'cls' in either map
        union = (pred_inds | target_inds).long().sum().item()

        if union == 0:
            # If there is no ground truth or prediction for this class,
            # we usually skip it to avoid NaN.
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)

    # Filter out NaNs and calculate mean
    ious = torch.tensor(iou_list)
    valid_ious = ious[~torch.isnan(ious)]
    miou = valid_ious.mean().item() if len(valid_ious) > 0 else 1.0

    return ious, miou


def get_one_of_each(dataset):
    samples = {}

    # Iterate through the dataset
    for i in range(len(dataset)):
        image, label = dataset[i]
        # If we haven't found this digit yet, save it
        if label not in samples:
            samples[label] = image

        # Stop once we have all 10 digits (0-9)
        if len(samples) == 10:
            break

    # Sort by label to ensure they are in order 0-9
    return torch.cat([samples[i] for i in range(10)])[:, None]


def cosine_beta_schedule(T: int, s: float = 0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)

    # Calculate alpha_bar (cumprod)
    alphas_cumprod = torch.cos((((t / T) + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Derive betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clamp betas to prevent singularities (max value of 0.999 is standard)
    return torch.clip(betas, 0.0001, 0.9999)


def to_vis(x):
    return torch.clamp((x + 1) / 2, 0.0, 1.0)


def visualize_mri(image: torch.Tensor):
    """
    Converts mri images into rgb images
    :param image: image tensor of shape (B, S, H, W)
    :return: image tensor of shape (B, S, 3, H, W)
    """
    img_min = torch.amin(image, (2, 3))[:, :, None, None]
    img_max = torch.amax(image, (2, 3))[:, :, None, None]
    image = (image - img_min) / (img_max - img_min)
    return image[:, :, None].repeat(1, 1, 3, 1, 1)


def visualize_seg(seg_map: torch.Tensor):
    palette = torch.tensor([
        [0, 0, 0],  # Class 0
        [1, 0, 0],  # Class 1
        [0, 1, 0],  # Class 2
        [0, 0, 1],  # Class 3
        [0, 1, 1],  # Class 4
        [1, 0, 1],  # Class 5
        [1, 1, 0]  # Class 6
    ], dtype=torch.float32)

    return palette[seg_map]

def visualize_adaption(adaption: torch.Tensor):
    r = torch.relu(adaption)
    g = torch.zeros_like(adaption)
    b = torch.relu(-adaption)

    palate = torch.stack([r, g, b], dim=-1)
    return palate


if __name__ == '__main__':
    images = torch.linspace(-1, 1, 1000).reshape(1000, 1) @ torch.linspace(-1, 1, 1000).reshape(1, 1000)
    print(images.shape)
    visualize_adaption(images[None])