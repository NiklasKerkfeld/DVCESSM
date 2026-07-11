import glob
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datasets.base import get_datasets
from datasets.celebamask import get_ood_datasets
from src.diff_cf.core.segmentation_trainer import SegmentationTrainer
from src.diff_cf.models.segmentation import SegmentationUNet

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

    if cfg.segmentation.ood is None:
        train_dataset, test_dataset = get_datasets(cfg, cfg.segmentation.use_augmentations)
    else:
        train_dataset, test_dataset = get_ood_datasets(cfg,
                                                       cfg.segmentation.ood,
                                                       cfg.segmentation.use_augmentations)
    print(f"{len(train_dataset)} train samples")
    print(f"{len(test_dataset)} test samples")

    print(train_dataset[0][0].shape)

    model = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
        attention=cfg.segmentation.attention,
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
