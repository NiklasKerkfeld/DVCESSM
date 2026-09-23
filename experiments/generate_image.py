import os

from hydra.utils import instantiate

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

from diff_cf.core.base_dataset import get_splits
from diff_cf.models.diffusion_model import DiffusionModel
from diff_cf.core.diffusion.trainer import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_data, _ = get_splits(root=cfg.data.root)

    dataset = instantiate(cfg.data.dataset, samples=train_data)

    model = DiffusionModel(channels=cfg.data.image_channels,
                           width=cfg.data.image_width,
                           height=cfg.data.image_height,
                           time_dim=cfg.diffusion.time_dim,
                           nums_features=cfg.diffusion.nums_features,
                           use_prior=cfg.diffusion.use_target,
                           num_classes=cfg.data.n_classes)
    engine = instantiate(cfg.engine)

    trainer = Trainer(
        dataset,
        model,
        engine,
        device
    )

    checkpoint = f'{cfg.diffusion.checkpoint_path}/{cfg.generation.model}.pt'
    trainer.load_checkpoint(checkpoint)
    print(f"loaded model from {checkpoint}.")

    # set randomness
    if cfg.generation.seed:
        torch.manual_seed(cfg.generation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.generation.seed)
    else:
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed_all()

    image = trainer.generate_image(show_progress=True, step=1, eta=1.0)
    print(f"{image.min()}, ", f"{image.max()}, ", f"{image.mean()}, ", f"{image.std()}")
    image = trainer._visualize_image(image)
    print(f"{image.min()}, ", f"{image.max()}, ", f"{image.mean()}, ", f"{image.std()}")

    os.makedirs(f"{cfg.system.figures_dir}", exist_ok=True)
    save_image(image / 255.0, f"{cfg.system.figures_dir}/generated_image.png")


if __name__ == '__main__':
    main()