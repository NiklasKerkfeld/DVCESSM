import glob

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from core.GANTrainer import GANTrainer
from core.SegmentationDataset import SegmentationDataset
from models.Classifier import Classifier
from models.GAN import Generator
from models.Segmentation import SegmentationUNet

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Logging & Run Number
    if cfg.training.name is not None:
        run_name = cfg.training.name
    else:
        # If no run_id, it's a NEW experiment. Generate a fresh ID.
        nr = len(glob.glob('runs/Diffusion/*'))
        run_name = f"{nr}_DiffusionTraining"

    # print the resolved config to your terminal for sanity
    print(OmegaConf.to_yaml(cfg))

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    full_dataset = SegmentationDataset()
    train_dataset, test_dataset, _ = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    generator = Generator(cfg.data.image_channels, cfg.data.n_classes)
    adversarial = Classifier(cfg.data.image_channels, cfg.data.n_classes, cfg.data.image_width, cfg.data.image_height)
    segmentation = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    )
    segmentation.load_state_dict(torch.load(cfg.inference.segmentation_path, map_location=device))

    trainer = GANTrainer(
        generator,
        adversarial,
        segmentation,
        device=device,
        name=run_name,
        lr_generator=cfg.gan.lr_generator,
        lr_adversarial=cfg.gan.lr_adversarial,
        lambda_adv=cfg.gan.lambda_adv,
        lambda_seg=cfg.gan.lambda_seg,
        lambda_adpt=cfg.gan.lambda_adpt
    )

    trainer.train(
        train_loader,
        test_loader,
        epochs=cfg.training.epochs,
    )


if __name__ == '__main__':
    main()
