import os
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate

import torch
import hydra
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from baseline.generator import BaselineGenerator
from base_dataset import get_datasets
from diff_cf.utils.metrics import calc_iou, calc_f1_score
from diff_cf.utils.utils import visualize_seg, visualize_adaption
from diffusion.generator import CounterfactualGenerator
from diff_cf.models.diffusion_model_final import DiffusionModel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    if cfg.gan.name is not None:
        run_name = cfg.gan.name
    else:
        raise ValueError(f"Parameter gan.name not defined!")

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, test_dataset = get_datasets(cfg, cfg.segmentation.augmentations.enable)
    if cfg.segmentation.ood.enable:
        test_dataset.set_attribute(cfg.segmentation.ood.attribute,
                                   use_positive=not cfg.segmentation.ood.train_on_positive)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    segmentation = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
    )

    checkpoint = torch.load(
        f"{cfg.segmentation.checkpoint_path}/{cfg.generation.model}.pt",
        map_location=device
    )
    segmentation.load_state_dict(checkpoint['model_state_dict'])

    if cfg.generation.type == 'Diffusion':
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
        else:
            robust_model = None

        diffusion = DiffusionModel(channels=cfg.data.image_channels,
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
        diffusion.load_state_dict(checkpoint['model_state_dict'])

        engine = instantiate(cfg.engine)

        generator = CounterfactualGenerator(segmentation,
                                            diffusion,
                                            engine,
                                            device,
                                            cfg,
                                            robust_model,
                                            name=run_name)

    else:
        generator = BaselineGenerator(segmentation, device, cfg, name=run_name)
    
    results = []
    bar = tqdm(test_loader, total=len(test_loader))
    count = 0
    for i, (image, target) in enumerate(bar):
        image.to(device)
        target.to(device)
        if target.sum() == 0.0:
            continue

        counterfactual, adaption, prediction_org, prediction_cf = generator.generate(
            image, target)

        iou_org = calc_iou(prediction_org, target, num_classes=cfg.data.n_classes)
        f1_org, precision_org, recall_org = calc_f1_score(prediction_org, target,
                                                          num_classes=cfg.data.n_classes)

        iou_cf = calc_iou(prediction_cf, target, num_classes=cfg.data.n_classes)
        f1_cf, precision_cf, recall_cf = calc_f1_score(prediction_cf, target,
                                                       num_classes=cfg.data.n_classes)

        image = test_dataset.denormalize(image)
        counterfactual = test_dataset.denormalize(counterfactual)
        adaption = visualize_adaption(adaption)
        prediction_org = visualize_seg(prediction_org)
        prediction_cf = visualize_seg(prediction_cf)
        target = visualize_seg(target)
        
        file = Path(test_dataset.get_filepath(i)).stem

        os.makedirs(f"./{cfg.system.evaluation_dir}/evaluation/{file}", exist_ok=True)
        save_image(image[0], f"./{cfg.system.evaluation_dir}/evaluation/{file}/image.png")
        save_image(prediction_org[0].permute(2, 0, 1),
                   f"./{cfg.system.evaluation_dir}/evaluation/{file}/prediction_org.png")
        save_image(adaption[0],
                   f"./{cfg.system.evaluation_dir}/evaluation/{file}/adaption.png")
        save_image(counterfactual[0], f"./{cfg.system.evaluation_dir}/evaluation/{file}/counterfactual.png")
        save_image(prediction_cf[0].permute(2, 0, 1),
                   f"./{cfg.system.evaluation_dir}/evaluation/{file}/prediction_cf.png")
        save_image(target[0].permute(2, 0, 1), f"./{cfg.system.evaluation_dir}/evaluation/{file}/target.png")

        with open(f"./{cfg.system.evaluation_dir}/evaluation/{file}/results.txt", "w") as f:
            # Writing individual class IoUs
            f.write("Original Segmentation:\n")
            f.write(f"\tiou: {iou_org}")
            f.write(f"\tf1: {f1_org}")
            f.write(f"\tprecision: {precision_org}")
            f.write(f"\trecall: {recall_org}")

            f.write("Counterfactual Segmentation:\n")
            f.write(f"\tiou: {iou_cf}")
            f.write(f"\tf1: {f1_cf}")
            f.write(f"\tprecision: {precision_cf}")
            f.write(f"\trecall: {recall_cf}")
        
        results.append({'file': file, 
                        'iou_org': iou_org, 'f1_org': f1_org, 'precision_org': precision_org, 'recall_org': recall_org,
                        'iou_cf': iou_cf, 'f1_cf': f1_cf, 'precision_cf': precision_cf, 'recall_cf': recall_cf})
            
        bar.set_description(f"{count} samples")
        
    pd.DataFrame(results, 
                 columns=['file', 'iou_org', 'f1_org', 'precision_org', 'recall_org', 'iou_cf', 'f1_cf', 'precision_cf', 'recall_cf']).to_csv(
        f"{cfg.system.evaluation_dir}/evaluation/results.csv", index=False)


if __name__ == '__main__':
    main()
