import hydra
import torch
from hydra.utils import instantiate
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from GAN.Trainer import CounterfactualGANTrainer
from base_dataset import get_splits
from GAN.CounterfactualGAN import CounterfactualGAN
from GAN.model import Adversarial
from GAN.model import Generator


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = get_splits(
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

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg.data.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.data.num_workers,
                                  pin_memory=device.type == "cuda",
                                  persistent_workers=cfg.num_workers > 1,
                                  drop_last=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=cfg.data.num_workers,
                                 pin_memory=device.type == "cuda",
                                 persistent_workers=cfg.data.num_workers > 1,
                                 drop_last=False)

    # init segmentation model and load weights
    target_model = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=3,
        out_channels=cfg.data.n_classes,
    )

    checkpoint = torch.load(
        f"{cfg.segmentation.checkpoint_path}/{cfg.generation.model}.pt",
        map_location=device
    )
    target_model.load_state_dict(checkpoint['model_state_dict'])

    generator = Generator(image_channels=3)
    adversarial = Adversarial(image_channels=3)
    gan = CounterfactualGAN(generator, adversarial, target_model)

    trainer = CounterfactualGANTrainer(gan=gan)

    trainer.train(train_dataloader, test_dataloader, epochs=3)


if __name__ == '__main__':
    main()
