import glob

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from core.SegmentationDataset import SegmentationDataset
from core.SementationTrainer import SegmentationTrainer
from models.Segmentation import SegmentationUNet

torch.manual_seed(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Logging & Run Number
    nr = len(glob.glob('runs/Segmentation/*'))
    print(f"Start {nr}. Segmentation training ...")

    # print the resolved config to your terminal for sanity
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = SegmentationDataset()
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False
    )

    model = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    ).to(device)

    trainer = SegmentationTrainer(
        model=model,
        device=device,
        name=f"{nr}_{cfg.segmentation.name_prefix}"
    )

    trainer.train(train_loader, test_loader, epochs=cfg.segmentation.epochs, lr=cfg.segmentation.lr)


if __name__ == '__main__':
    main()
