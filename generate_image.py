import os
import hydra
import torch
from omegaconf import DictConfig
from matplotlib import pyplot as plt

from core.DiffusionEngine import DiffusionEngine
from core.DiffusionTrainer import DiffusionTrainer
from models.DiffusionModel import DiffusionModel
from utils import to_vis


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = DiffusionModel(cfg.model.dims, cfg.data.image_width, cfg.data.image_height, cfg.model.time_dim)
    model.load_state_dict(torch.load(cfg.inference.diffusion_path, map_location=device))

    engine = DiffusionEngine(**cfg.engine, device=device)

    diffusion_model = DiffusionTrainer(model, engine, device)

    gen_image = diffusion_model.generate_image()[0, 0].cpu()

    plt.imshow(to_vis(gen_image).numpy(), cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.title("Generated Image")

    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f"./figures/generated_image.png")
    plt.close()


if __name__ == '__main__':
    main()