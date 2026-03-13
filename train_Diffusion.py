import glob
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from core.DiffusionDataset import get_MNIST_dataset, get_BraTS_dataset
from models.DiffusionModel import DiffusionModel
from core.DiffusionEngine import DiffusionEngine
from core.DiffusionTrainer import DiffusionTrainer


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

    model = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height, cfg.model.time_dim)
    engine = DiffusionEngine(**cfg.engine, device=device)

    trainer = DiffusionTrainer(
        model,
        engine,
        cfg.training.lr,
        device,
        name=run_name
    )

    filepath = Path(cfg.training.checkpoint)
    if filepath.is_file():
        trainer.load_checkpoint(cfg.training.checkpoint)
        print(f"continue training form {cfg.training.checkpoint}...")

    trainer.train(
        train_loader,
        epochs=cfg.training.epochs,
    )


if __name__ == '__main__':
    main()
