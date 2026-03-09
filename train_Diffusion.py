import glob
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from core.DiffusionDataset import get_MNIST_dataset, get_BraTS_dataset
from models.DiffusionModel import DiffusionModel
from core.DiffusionEngine import DiffusionEngine
from core.DiffusionTrainer import DiffusionTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Logging & Run Number
    nr = len(glob.glob('runs/Diffusion/*'))
    print(f"Start {nr}. Diffusion ...")

    # print the resolved config to your terminal for sanity
    print(OmegaConf.to_yaml(cfg))

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if cfg.data.name == 'MNIST':
        dataset = get_MNIST_dataset()
    elif cfg.data.name == 'BraTS':
        dataset = get_BraTS_dataset()
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")


    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    model = DiffusionModel(cfg.model.dims, cfg.data.image_width, cfg.data.image_height, cfg.model.time_dim)
    engine = DiffusionEngine(**cfg.engine, device=device)

    trainer = DiffusionTrainer(
        model,
        engine,
        device,
        name=f"{nr}_{cfg.training.name_prefix}"
    )

    trainer.train(
        train_loader,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr
    )


if __name__ == '__main__':
    main()
