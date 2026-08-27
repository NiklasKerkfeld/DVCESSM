import glob
from pathlib import Path
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datasets.base import get_datasets, BaseDataset
from src.diff_cf.core.segmentation_trainer import SegmentationTrainer
from monai.networks.nets import BasicUNet

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    log_dir = f'{cfg.system.log_dir}/Segmentation'
    if cfg.segmentation.name is not None:
        run_name = cfg.segmentation.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_SegmentationTraining"

    print(f"{run_name=}")

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_dataset, test_dataset = get_datasets(cfg, cfg.segmentation.augmentations.enable)
    if cfg.segmentation.ood.enable:
        train_dataset.set_attribute(cfg.segmentation.ood.attribute, use_positive=cfg.segmentation.ood.train_on_positive)
        test_dataset.set_attribute(cfg.segmentation.ood.attribute, use_positive=not cfg.segmentation.ood.train_on_positive)

    print(f"{len(train_dataset)} train samples")
    print(f"{len(test_dataset)} test samples")

    print(train_dataset[0][0].shape)

    model = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
    ).to(device)

    trainer = SegmentationTrainer(
        train_dataset,
        test_dataset,
        model,
        device,
        cfg,
        name=run_name,
        log_dir=log_dir
    )

    checkpoint = f'{cfg.segmentation.checkpoint_path}/{run_name}.pt'
    if Path(checkpoint).is_file():
        trainer.load_checkpoint(checkpoint)
        print(f"continue training form {checkpoint}...")

    trainer.train(cfg.segmentation.epochs)


if __name__ == '__main__':
    main()
