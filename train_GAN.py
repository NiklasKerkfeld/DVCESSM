import glob

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from core.Datasets import get_datasets
from core.GAN.Trainer import Trainer
from models.GAN import Generator, Adversarial
from models.Segmentation import SegmentationUNet

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    log_dir = f'{cfg.system.log_dir}/GAN'
    if cfg.gan.name is not None:
        run_name = cfg.gan.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_GANTraining"

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_dataset, test_dataset = get_datasets('Segmentation')

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
    adversarial = Adversarial(cfg.data.image_channels, 1, cfg.data.image_width, cfg.data.image_height)
    segmentation = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
        attention=cfg.segmentation.attention
    )

    checkpoint = torch.load(f'{cfg.segmentation.checkpoint_path}/{cfg.gan.segmentation_model}.pt', map_location=device)
    segmentation.load_state_dict(checkpoint['model_state_dict'])

    trainer = Trainer(
        generator,
        adversarial,
        segmentation,
        lr_generator=cfg.gan.lr_generator,
        lr_adversarial=cfg.gan.lr_adversarial,
        lambda_adv=cfg.gan.lambda_adv,
        lambda_seg=cfg.gan.lambda_seg,
        lambda_rec=cfg.gan.lambda_rec,
        lambda_per=cfg.gan.lambda_per,
        free_rec=cfg.gan.free_rec,
        free_per=cfg.gan.free_per,
        device=device,
        name=run_name,
        log_dir=log_dir,
        checkpoint=cfg.gan.checkpoint_path
    )

    trainer.train(
        train_loader,
        test_loader,
        epochs=cfg.gan.epochs,
    )


if __name__ == '__main__':
    main()
