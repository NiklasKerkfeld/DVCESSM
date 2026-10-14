import os
import time
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate

import hydra
import torch
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from diff_cf.core.base_dataset import get_splits
from diff_cf.core.baseline.generator import BaselineGenerator
from diff_cf.core.diffusion.generator import CounterfactualGenerator
from diff_cf.models.diffusion_model import DiffusionModel
from diff_cf.utils import calc_iou, calc_f1_score


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # print config
    print(OmegaConf.to_yaml(cfg))

    # set random seeds:
    if cfg.generation.seed is not None:
        torch.random.manual_seed(cfg.generation.seed)

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # create output folder
    output_folder = f"evaluation/{cfg.generation.batch_name}"
    os.makedirs(output_folder, exist_ok=True)

    # load test dataset
    _, test_data = get_splits(root=cfg.data.root)
    dataset = instantiate(cfg.data.dataset, samples=test_data)
    print(f"dataset size: {len(dataset)}")

    # init segmentation model and load weights
    segmentation_model = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
    )

    checkpoint = torch.load(
        f"{cfg.segmentation.checkpoint_path}/{cfg.generation.model}.pt",
        map_location=device
    )
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    # Losses
    segmentation_loss_config = instantiate(cfg.generation.SegmentationLoss)
    distance_loss = instantiate(cfg.generation.distance_loss)

    # AdaptGrad
    adapt_grad = instantiate(cfg.generation.AdaptGrad, z_min=dataset.z_min, z_max=dataset.z_max)

    if cfg.generation.type == 'Diffusion':
        # Smoothing
        smoothing_cfg = instantiate(cfg.generation.GradSmoothing)

        # Init robust model if given
        robust_model = None
        if cfg.generation.robust is not None:
            robust_model = BasicUNet(
                spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
                in_channels=cfg.data.image_channels,
                out_channels=cfg.data.n_classes,
            )
            checkpoint = torch.load(
                f"{cfg.segmentation.checkpoint_path}/{cfg.generation.robust}.pt",
                map_location=device
            )
            robust_model.load_state_dict(checkpoint['model_state_dict'])

        # init diffusion model and load weights
        diffusion_model = DiffusionModel(channels=cfg.data.image_channels,
                                         width=cfg.data.image_width,
                                         height=cfg.data.image_height,
                                         time_dim=cfg.diffusion.time_dim,
                                         nums_features=cfg.diffusion.nums_features,
                                         use_prior=cfg.diffusion.use_target,
                                         num_classes=cfg.data.n_classes)

        checkpoint = torch.load(
            f"{cfg.diffusion.checkpoint_path}/{cfg.diffusion.name}.pt",
            map_location=device
        )
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])

        # init diffusion engine
        engine = instantiate(cfg.engine, device=device)

        # init diffusion counterfactual generator
        generator = CounterfactualGenerator(
            segmentation_model=segmentation_model,
            diffusion_model=diffusion_model,
            engine=engine,
            segmentation_loss_config=segmentation_loss_config,
            distance_loss=distance_loss,
            z_bounds=(dataset.z_min, dataset.z_max),
            use_target_prior=cfg.diffusion.use_target,
            robust_model=robust_model,
            weight_segmentation=cfg.generation.weight_segmentation,
            weight_distance=cfg.generation.weight_distance,
            start_fraction=cfg.generation.start_fraction,
            approximation_step_size=cfg.generation.approximation_step_size,
            adapt_grad=adapt_grad,
            smoothing_cfg=smoothing_cfg,
            n_classes=cfg.data.n_classes,
            cone_projection_alpha=cfg.generation.cone_projection_alpha,
            log_to_tensorboard=False,
            device=device
        )

    else:
        # Alternative load Baseline Generator
        generator = BaselineGenerator(
            segmentation_model=segmentation_model,
            segmentation_loss_config=segmentation_loss_config,
            distance_loss=distance_loss,
            z_bounds=(dataset.z_min, dataset.z_max),
            adapt_grad=adapt_grad,
            lr=cfg.generation.lr,
            steps=cfg.generation.steps,
            alpha=cfg.generation.alpha,
            beta=cfg.generation.beta,
            cone_projection_alpha=cfg.generation.cone_projection_alpha,
            n_classes=cfg.data.n_classes,
            log_to_tensorboard=False,
            device=device
        )

    df = pd.read_csv("data/CelebAMask/CelebAMask-HQ/results.csv")

    results = []
    for i in trange(len(dataset)):
        instance = Path(dataset.get_filepath(i)).stem

        if df[df['file'] == int(instance)]['iou'].values[0] > 0.75:
            continue

        image = torch.stack([dataset[i][0]], dim=0)
        target = torch.stack([dataset[i][1]], dim=0)

        start_time = time.time()
        counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target)
        time_needed = time.time() - start_time

        iou_org = calc_iou(target, prediction_org, cfg.data.n_classes)
        f1_org, precision_org, recall_org = calc_f1_score(target, prediction_org,
                                                          cfg.data.n_classes)

        iou_cf = calc_iou(target, prediction_cf, cfg.data.n_classes)
        f1_cf, precision_cf, recall_cf = calc_f1_score(target, prediction_cf, cfg.data.n_classes)
        results.append({
            "instance": instance,
            "time": time_needed,
            "iou_org": iou_org,
            "f1_org": f1_org,
            "precision_org": precision_org,
            "recall_org": recall_org,
            "iou_cf": iou_cf,
            "f1_cf": f1_cf,
            "precision_cf": precision_cf,
            "recall_cf": recall_cf,
            "improvement_iou": iou_cf - iou_org,
            "improvement_f1": f1_cf - f1_org,
            "improvement_recall": recall_cf - recall_org,
            "improvement_precision": precision_cf - precision_org,
            "distance": (image-counterfactual).abs().mean().item(),
            "mean_squared": ((image - counterfactual) ** 2).mean().item()
        })

    df = pd.DataFrame(results)
    df.to_csv(f'{output_folder}/results.csv', index=False)
    with open(f"{output_folder}/config.txt", "a") as f:
        f.write(OmegaConf.to_yaml(cfg))

    metrics = {
        "IoU": "improvement_iou",
        "F1": "improvement_f1",
        "Recall": "improvement_recall",
        "Precision": "improvement_precision",
        "Distance": "distance",
        "Mean Squared": "mean_squared",
    }

    with open(f"{output_folder}/results.txt", "a") as f:
        for label, col in metrics.items():
            # Compute stats, round to 4 digits, and convert to a tuple
            stats = tuple(df[col].agg(["mean", "median", "std", "min", "max"]).round(4))
            f.write(f"{label}: {stats}\n")


if __name__ == '__main__':
    main()
