import glob
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from core.DiffusionDataset import DiffusionDataset
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

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, 0.0),
        transforms.Normalize(0.5, 0.5)
    ])

    raw_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    dataset = DiffusionDataset(raw_dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    model = DiffusionModel(cfg.model.dims, cfg.model.time_dim)
    engine = DiffusionEngine(**cfg.engine, device=device)

    trainer = DiffusionTrainer(
        model,
        engine,
        device,
        name=f"{nr}_{cfg.training.name_prefix}"
    )

    trainer.train_diffusion(
        train_loader,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr
    )


if __name__ == '__main__':
    main()
