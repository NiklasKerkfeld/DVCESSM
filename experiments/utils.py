import csv
import os
from pathlib import Path

import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchvision.utils import make_grid

from datasets.brats import BraTSDataset
from datasets.celebamask import CelebAMaskDataset

from diff_cf.utils.utils import visualize_seg, visualize_counterfactuals, visualize_adaption
from diff_cf.utils.metrics import calc_iou, calc_f1_score


def visualize_rgb(dataset: CelebAMaskDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  time_needed: float,
                  cfg: DictConfig):

    # create output folder
    output_folder = Path(cfg.system.evaluation_dir)
    if cfg.generation.batch_name:
        output_folder /= Path(cfg.generation.batch_name)
    output_folder /= Path(run_name)
    output_folder.mkdir(parents=True, exist_ok=True)


    image, target = dataset[n][0], dataset[n][1]

    print("Before denormalize:")
    print("Counterfactual:")
    print("min: ", counterfactual.min(), "max: ", counterfactual.max())
    print((counterfactual < 0).float().mean(), (counterfactual > 1).float().mean())

    print("Image:")
    print("min: ", image.min(), "max: ", image.max())
    print((image < 0).float().mean(), (image > 1).float().mean())

    iou_org = calc_iou(target, prediction_org, cfg.data.n_classes)
    f1_org, precision_org, recall_org = calc_f1_score(target, prediction_org, cfg.data.n_classes)

    iou_cf = calc_iou(target, prediction_cf, cfg.data.n_classes)
    f1_cf, precision_cf, recall_cf = calc_f1_score(target, prediction_cf, cfg.data.n_classes)

    image = dataset.denormalize(image) / 255.0
    counterfactual = dataset.denormalize(counterfactual[0]) / 255.0

    print("\nAfter denormalize:")
    print("Counterfactual:")
    print("min: ", counterfactual.min(), "max: ", counterfactual.max())
    print((counterfactual < 0).float().mean(), (counterfactual > 1).float().mean())

    print("\n Image:")
    print("min: ", image.min(), "max: ", image.max())
    print((image < 0).float().mean(), (image > 1).float().mean())

    target = visualize_seg(target)
    pred_original = visualize_seg(prediction_org[0])
    pred_counterfactual = visualize_seg(prediction_cf[0])

    x_min = ((cfg.data.min - torch.tensor(cfg.data.means)) / torch.tensor(
        cfg.data.stds))
    x_max = ((cfg.data.max - torch.tensor(cfg.data.means)) / torch.tensor(
        cfg.data.stds))

    maximas = torch.maximum(torch.abs(x_min), torch.abs(x_max))

    adaption_visual = visualize_adaption(adaption, maximas)

    with open(output_folder / "config.txt", "a") as f:
        f.write(OmegaConf.to_yaml(cfg))

    log_results(
        output_folder,
        iou_org=iou_org,
        f1_org=f1_org,
        precision_org=precision_org,
        recall_org=recall_org,
        iou_cf=iou_cf,
        f1_cf=f1_cf,
        precision_cf=precision_cf,
        recall_cf=recall_cf,
        dist=(image-counterfactual).abs().mean().item(),
        time_needed=time_needed
    )

    torchvision.utils.save_image(target, output_folder / "target.png")
    torchvision.utils.save_image(pred_original, output_folder / "prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual, output_folder / "prediction_counterfactual.png")

    torchvision.utils.save_image(image, output_folder / "original.png")
    torchvision.utils.save_image(counterfactual, output_folder / "counterfactual.png")

    for i, channel in enumerate(["red", "green", "blue"]):
        torchvision.utils.save_image(adaption_visual[0, i], output_folder / f"adaption_{channel}.png")

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
    torchvision.utils.save_image(stacked_image, output_folder / "comparison.png")


def visualize_mri(dataset: BraTSDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  cfg: DictConfig):

    # create output folder
    output_folder = Path(cfg.system.evaluation_dir)
    if cfg.generation.batch_name:
        output_folder /= Path(cfg.generation.batch_name)
    output_folder /= Path(run_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    image, target = dataset[n][0], dataset[n][1]

    target = visualize_seg(target)[0]
    pred_original = visualize_seg(prediction_org)[0]
    pred_counterfactual = visualize_seg(prediction_cf)[0]

    adaption = visualize_adaption(adaption)
    image, counterfactual = visualize_counterfactuals(image, counterfactual)

    with open( output_folder / "config.txt", "a") as f:
        f.write(str(cfg))

    torchvision.utils.save_image(target, output_folder / "target.png")
    torchvision.utils.save_image(pred_original, output_folder / "prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual, output_folder / "prediction_counterfactual.png")

    for i, channel in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        torchvision.utils.save_image(image[0, i], output_folder / "original_{channel}.png")
        torchvision.utils.save_image(counterfactual[0, i], output_folder / "counterfactual_{channel}.png")
        torchvision.utils.save_image(adaption[0, i], output_folder / "adaption_{channel}.png")

    stacked_image = make_grid(torch.stack([
        image[0],
        adaption[0],
        counterfactual[0]
    ]).flatten(0, 1), nrow=4)
    torchvision.utils.save_image(stacked_image, output_folder / "comparison.png")

def log_results(output_folder: Path,
                iou_org: float,
                f1_org: float,
                precision_org: float,
                recall_org: float,
                iou_cf: float,
                f1_cf: float,
                precision_cf: float,
                recall_cf: float,
                dist: float,
                time_needed: float):
    # Define file path
    csv_path = output_folder / "results.csv"

    # Write headers if file doesn't exist yet
    file_exists = csv_path.exists()

    fields = [
        "Metric",
        "Original",
        "Counterfactual",
        "Difference",
    ]

    iou_diff = iou_cf - iou_org
    f1_diff = f1_cf - f1_org
    prec_diff = precision_cf - precision_org
    rec_diff = recall_cf - recall_org

    rows = [
        ["Distance", "-", "-", f"{dist:.4f}"],
        ["Time", "-", "-", f"{time_needed:.4f}"],
        ["IoU", f"{iou_org:.4f}", f"{iou_cf:.4f}", f"{iou_diff:.4f}"],
        ["F1", f"{f1_org:.4f}", f"{f1_cf:.4f}", f"{f1_diff:.4f}"],
        ["Precision", f"{precision_org:.4f}", f"{precision_cf:.4f}", f"{prec_diff:.4f}"],
        ["Recall", f"{recall_org:.4f}", f"{recall_cf:.4f}", f"{rec_diff:.4f}"],
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fields)
        writer.writerows(rows)


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
