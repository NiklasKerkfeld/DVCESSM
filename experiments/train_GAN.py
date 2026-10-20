import glob

import hydra
import torch
from hydra.utils import instantiate
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf

from GAN.Trainer import CounterfactualGANTrainer
from diff_cf.core.base_dataset import get_splits
from GAN.CounterfactualGAN import CounterfactualGAN
from GAN.model import Adversarial
from GAN.model import Generator


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    log_dir = f'{cfg.system.log_dir}/GAN/'
    if cfg.diffusion.name is not None:
        run_name = cfg.diffusion.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_GAN"

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

    generator = Generator(in_channels=3, level=6)
    adversarial = Adversarial(image_channels=3)
    gan = CounterfactualGAN(generator, adversarial, target_model)

    trainer = CounterfactualGANTrainer(gan=gan,
                                       train_dataset=train_dataset,
                                       test_dataset=test_dataset,
                                       batch_size=cfg.data.batch_size,
                                       num_workers=cfg.data.num_workers,
                                       device=device,
                                       log_dir=log_dir,
                                       name=run_name)

    trainer.train(epochs=10)


if __name__ == '__main__':
    main()
