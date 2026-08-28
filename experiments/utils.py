import os

import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchvision.utils import make_grid

from datasets.brats import BraTSDataset
from datasets.celebamask import CelebAMaskDataset

from diff_cf.utils.utils import visualize_seg, visualize_counterfactuals, visualize_adaption
from utils.metrics import calc_iou, calc_f1_score


def visualize_rgb(dataset: CelebAMaskDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  cfg: DictConfig):
    image, target = dataset[n][0], dataset[n][1]

    print("Before denormalize:")
    print("Counterfactual:")
    print("min: ", counterfactual.min(), "max: ", counterfactual.max())
    print((counterfactual < 0).float().mean(), (counterfactual > 1).float().mean())

    print("Image:")
    print("min: ", image.min(), "max: ", image.max())
    print((image < 0).float().mean(), (image > 1).float().mean())

    image = dataset.denormalize(image) / 255.0
    counterfactual = dataset.denormalize(counterfactual[0]) / 255.0

    print("\nAfter denormalize:")
    print("Counterfactual:")
    print("min: ", counterfactual.min(), "max: ", counterfactual.max())
    print((counterfactual < 0).float().mean(), (counterfactual > 1).float().mean())

    print("\n Image:")
    print("min: ", image.min(), "max: ", image.max())
    print((image < 0).float().mean(), (image > 1).float().mean())

    iou_org = calc_iou(target, prediction_org, cfg.data.n_classes)
    f1_org, precision_org, recall_org = calc_f1_score(target, prediction_org, cfg.data.n_classes)

    iou_cf = calc_iou(target, prediction_cf, cfg.data.n_classes)
    f1_cf, precision_cf, recall_cf = calc_f1_score(target, prediction_cf, cfg.data.n_classes)

    target = visualize_seg(target)
    pred_original = visualize_seg(prediction_org[0])
    pred_counterfactual = visualize_seg(prediction_cf[0])

    x_min = ((cfg.data.min - torch.tensor(cfg.data.means)) / torch.tensor(
        cfg.data.stds))
    x_max = ((cfg.data.max - torch.tensor(cfg.data.means)) / torch.tensor(
        cfg.data.stds))

    maximas = torch.maximum(torch.abs(x_min), torch.abs(x_max))

    adaption_visual = visualize_adaption(adaption, maximas)

    os.makedirs(f"{cfg.system.figures_dir}/{run_name}", exist_ok=True)

    with open(f"{cfg.system.figures_dir}/{run_name}/config.txt", "a") as f:
        f.write(OmegaConf.to_yaml(cfg))

    with open(f"{cfg.system.figures_dir}/{run_name}/results.txt", "a") as f:
        f.write(f"Results:\n"
                f"\t Original:\n"
                f"\t\t IoU: \t {iou_org}\n"
                f"\t\t F1: \t {f1_org}\n"
                f"\t\t Precision: \t {precision_org}\n"
                f"\t\t Recall: \t {recall_org}\n"
                f"\t Counterfactual:\n"
                f"\t\t IoU: \t {iou_cf}\n"
                f"\t\t F1: \t {f1_cf}\n"
                f"\t\t Precision: \t {precision_cf}\n"
                f"\t\t Recall: \t {recall_cf}\n"
                f"\t Difference:\n"
                f"\t\t Recall: \t {(image-counterfactual).abs().sum()}\n"
                f"\t\t IoU: \t {iou_cf-iou_org}\n"
                f"\t\t F1: \t {f1_cf-f1_org}\n"
                f"\t\t Precision: \t {precision_cf-prediction_org}\n"
                f"\t\t Recall: \t {recall_cf-recall_org}\n"
                )

    torchvision.utils.save_image(target,
                                 f"{cfg.system.figures_dir}/{run_name}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_counterfactual.png")

    torchvision.utils.save_image(image,
                                 f"{cfg.system.figures_dir}/{run_name}/original.png")
    torchvision.utils.save_image(counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/counterfactual.png")

    for i, channel in enumerate(["red", "green", "blue"]):
        torchvision.utils.save_image(adaption_visual[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/adaption_{channel}.png")

    adaption_visual = visualize_adaption(adaption.sum(dim=1, keepdim=True),
                                         maximas=torch.max(maximas))

    stacked_image = make_grid([
        image,
        adaption_visual[0, 0],
        counterfactual,
        pred_original,
        pred_counterfactual,
        target
    ], nrow=3)
    torchvision.utils.save_image(stacked_image,
                                 f"{cfg.system.figures_dir}/{run_name}/comparison.png")


def visualize_mri(dataset: BraTSDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  cfg: DictConfig):
    image, target = dataset[n][0], dataset[n][1]

    target = visualize_seg(target)[0]
    pred_original = visualize_seg(prediction_org)[0]
    pred_counterfactual = visualize_seg(prediction_cf)[0]

    adaption = visualize_adaption(adaption)
    image, counterfactual = visualize_counterfactuals(image, counterfactual)

    os.makedirs(f"{cfg.system.figures_dir}/{run_name}", exist_ok=True)

    with open(f"{cfg.system.figures_dir}/{run_name}/config.txt", "a") as f:
        f.write(str(cfg))

    torchvision.utils.save_image(target, f"{cfg.system.figures_dir}/{run_name}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_counterfactual.png")

    for i, channel in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        torchvision.utils.save_image(image[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/original_{channel}.png")
        torchvision.utils.save_image(counterfactual[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/counterfactual_{channel}.png")
        torchvision.utils.save_image(adaption[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/adaption_{channel}.png")

    stacked_image = make_grid(torch.stack([
        image[0],
        adaption[0],
        counterfactual[0]
    ]).flatten(0, 1), nrow=4)
    torchvision.utils.save_image(stacked_image,
                                 f"{cfg.system.figures_dir}/{run_name}/comparison.png")


def parse_attributes(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Line 0: Total number of images (e.g., "30000")
    # Line 1: Attribute names
    attr_names = lines[1].strip().split()

    # Initialize the data structure to store results
    # Every attribute will have a dict with 'positive' and 'negative' lists
    dataset_split = {attr: {'positive': [], 'negative': []} for attr in attr_names}

    # Process image lines (from index 2 onwards)
    for line in lines[2:]:
        parts = line.strip().split()
        if not parts:
            continue  # Skip empty lines if any

        img_name = parts[0]
        # The rest of the elements are the -1 or 1 labels
        labels = [int(x) for x in parts[1:]]

        # Map each label to its corresponding attribute
        for attr, label in zip(attr_names, labels):
            if label == 1:
                dataset_split[attr]['positive'].append(img_name)
            elif label == -1:
                dataset_split[attr]['negative'].append(img_name)

    return dataset_split
