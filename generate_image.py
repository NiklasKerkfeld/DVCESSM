import os
import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import make_grid, save_image

from core.Diffusion.Engine import Engine
from core.Diffusion.Trainer import Trainer
from models.DiffusionModel import DiffusionModel
from utils import visualize_mri


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    diffusion = DiffusionModel(cfg.model.dims, cfg.data.image_width, cfg.data.image_height,
                               cfg.model.time_dim)
    checkpoint = torch.load(f"{cfg.diffusion.checkpoint_path}/{cfg.diffusion.name}.pth", map_location=device)
    diffusion.load_state_dict(checkpoint['model_state_dict'])

    engine = Engine(**cfg.diffusion.engine, device=device)

    diffusion_model = Trainer(diffusion, engine, 0.0, device)

    gen_image = diffusion_model.generate_image()
    gen_image = visualize_mri(gen_image)[0].cpu() # (B, S, 3, H, W)

    gen_image = make_grid(gen_image, nrow=4)

    os.makedirs(f"{cfg.system.figures_dir}", exist_ok=True)
    save_image(gen_image, f"{cfg.system.figures_dir}/generated_image.png")


if __name__ == '__main__':
    main()