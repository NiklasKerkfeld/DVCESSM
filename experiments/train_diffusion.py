import glob
from pathlib import Path

from hydra.utils import instantiate

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.diff_cf.core.base_dataset import get_splits
from src.diff_cf.models.diffusion_model import DiffusionModel
from src.diff_cf.core.diffusion.trainer import Trainer

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
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

    train_data, _ = get_splits(root=cfg.data.root)

    dataset = instantiate(cfg.data.dataset, samples=train_data)

    model = DiffusionModel(channels=cfg.data.image_channels,
                           width=cfg.data.image_width,
                           height=cfg.data.image_height,
                           time_dim=cfg.diffusion.time_dim,
                           nums_features=cfg.diffusion.nums_features,
                           use_prior=cfg.diffusion.use_target,
                           num_classes=cfg.data.n_classes)

    engine = instantiate(cfg.engine, device=device)

    trainer = Trainer(
        dataset,
        model,
        engine,
        device,
        use_target=cfg.diffusion.use_target,
        epochs=cfg.diffusion.epochs,
        warm_up=cfg.diffusion.warm_up,
        lr=cfg.diffusion.lr,
        batch_size=cfg.data.batch_size,
        accumulation_steps=cfg.data.accumulation_steps,
        num_workers=cfg.data.num_workers,
        gamma=cfg.diffusion.gamma,
        ema_decay=cfg.diffusion.ema_decay,
        image_type=cfg.data.image_type,
        name=run_name,
        checkpoint_path=cfg.diffusion.checkpoint_path,
        log_dir=log_dir
    )

    checkpoint = f'{cfg.diffusion.checkpoint_path}/{run_name}.pt'
    if Path(checkpoint).is_file():
        trainer.load_checkpoint(checkpoint)
        print(f"continue training form {checkpoint}...")

    trainer.train()


if __name__ == '__main__':
    main()
