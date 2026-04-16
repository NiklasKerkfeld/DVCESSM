import glob
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from core.Datasets.dataset import get_datasets
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
    log_dir = f'{cfg.system.log_dir}/Diffusion/'
    if cfg.diffusion.name is not None:
        run_name = cfg.diffusion.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_DiffusionTraining"

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    dataset, _ = get_datasets(cfg)

    model = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height,
                           cfg.diffusion.time_dim)
    engine = Engine(**cfg.engine, device=device)

    trainer = Trainer(
        dataset,
        model,
        engine,
        device,
        cfg,
        name=run_name,
        log_dir=log_dir
    )

    checkpoint = f'{cfg.diffusion.checkpoint_path}/{run_name}.pt'
    if Path(checkpoint).is_file():
        trainer.load_checkpoint(checkpoint)
        print(f"continue training form {checkpoint}...")

    trainer.train(cfg.diffusion.epochs)


if __name__ == '__main__':
    main()
