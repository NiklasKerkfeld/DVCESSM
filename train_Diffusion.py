import glob
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from core.Datasets import get_datasets
from models.DiffusionModel import DiffusionModel
from core.Diffusion.Engine import Engine
from core.Diffusion.Trainer import Trainer


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    log_dir = f'{cfg.system.log_dir}/Diffusion/*'
    if cfg.diffusion.name is not None:
        run_name = cfg.diffusion.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_DiffusionTraining"

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    dataset, _ = get_datasets('Diffusion', cfg.data.root)

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    model = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height, cfg.model.time_dim)
    engine = Engine(**cfg.diffusion.engine, device=device)

    trainer = Trainer(
        model,
        engine,
        cfg.diffusion.lr,
        device,
        name=run_name,
        log_dir=log_dir,
        checkpoint=cfg.diffusion.checkpoint,
    )

    checkpoint = f'{cfg.diffusion.checkpoint_path}/{run_name}.pt'
    if Path(checkpoint).is_file():
        trainer.load_checkpoint(checkpoint)
        print(f"continue training form {checkpoint}...")

    trainer.train(
        train_loader,
        epochs=cfg.diffusion.epochs,
    )


if __name__ == '__main__':
    main()
