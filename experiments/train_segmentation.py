import glob
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.diff_cf.core.base_dataset import get_splits
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

    train_data, test_data= get_splits(
        root=cfg.data.root,
    )

    label = cfg.segmentation.ood.attribute if cfg.segmentation.ood.enable else None
    train_on_positive = cfg.segmentation.ood.train_on_positive if cfg.segmentation.ood.enable else True

    train_dataset = instantiate(cfg.data.dataset,
                                samples=train_data,
                                label=label,
                                use_positive=train_on_positive)

    test_dataset = instantiate(cfg.data.dataset,
                                samples=test_data,
                                label=label,
                                use_positive=not train_on_positive)


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
        n_classes=cfg.data.n_classes,
        device=device,
        lr=cfg.segmentation.lr,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_robust=cfg.segmentation.robust.enable,
        robust_start=cfg.segmentation.robust.start_epoch,
        robust_warm_up=cfg.segmentation.robust.warm_up,
        robust_epsilon=cfg.segmentation.robust.epsilon,
        robust_steps=cfg.segmentation.robust.steps,
        robust_alpha=cfg.segmentation.robust.alpha,
        image_type=cfg.data.image_type,
        name=run_name,
        log_dir=log_dir
    )

    checkpoint = f'{cfg.segmentation.checkpoint_path}/{run_name}.pt'
    if Path(checkpoint).is_file():
        trainer.load_checkpoint(checkpoint)
        print(f"continue training form {checkpoint}...")

    trainer.train()


if __name__ == '__main__':
    main()
