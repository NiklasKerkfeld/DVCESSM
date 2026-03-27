import torch


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

        pred_inds = (preds == cls)
        target_inds = (labels == cls)

        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()

        if union == 0:
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

    for i in range(len(dataset)):
        image, label = dataset[i]
        if label not in samples:
            samples[label] = image

        if len(samples) == 10:
            break

    return torch.cat([samples[i] for i in range(10)])[:, None]


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


def visualize_counterfactuals(image: torch.Tensor, *args: torch.Tensor):
    """
    Normalizes 'image' and any number of additional tensors (*args)
    based on the min/max of the first 'image'.
    """
    img_min = torch.amin(image, dim=(2, 3), keepdim=True)
    img_max = torch.amax(image, dim=(2, 3), keepdim=True)

    def transform(img):
        img = torch.clamp(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min + 1e-8)
        return img.unsqueeze(2).repeat(1, 1, 3, 1, 1)

    return transform(image), *[transform(a) for a in args]


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
    print(f"{palate.shape=}")
    return palate.permute(0, 1, 4, 2, 3)


if __name__ == '__main__':
    image = torch.randn((1, 4, 224, 224))
    cf1 = torch.randn((1, 4, 224, 224))
    cf2 = torch.randn((1, 4, 224, 224))
    cf3 = torch.randn((1, 4, 224, 224))
    image, cf1, cf2, cf3 = visualize_counterfactuals(image, cf1, cf2, cf3)