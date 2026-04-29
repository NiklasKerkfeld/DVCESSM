import os
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image
from tqdm import tqdm

from core.Datasets.dataset import get_datasets
from core.Diffusion.Engine import Engine
from models.DiffusionModel_new import DiffusionModel

@torch.no_grad()
def reconstruct_image(model, engine, image: torch.Tensor, T: int, device: torch.device = torch.device("cpu")):
    model.eval()

    image = image[None].to(device)
    model.to(device)

    for t in tqdm(reversed(range(T)), total=T):
        t_tensor = torch.full((image.shape[0],), t, device=device)
        epsilon_pred = model(image, t_tensor)

        # calculate mean of image at t-1
        mean = engine.reverse(image, epsilon_pred, t_tensor)

        if t > 0:
            noise = torch.randn_like(image)
            sigma = engine.get_sigma(t_tensor, image.shape)
            image = mean + sigma * noise
        else:
            image = mean

    return image.detach()


@torch.no_grad()
def generate_image(model, engine, batch_size: Optional[int] = 1, device: torch.device = torch.device("cpu")):
    images = torch.randn(batch_size, model.channels, model.width, model.height)
    model.eval()

    images = images.to(device)
    model.to(device)

    for t in tqdm(reversed(range(engine.T)), total=engine.T):
        t_tensor = torch.full((images.shape[0],), t, device=device)
        epsilon_pred = model(images, t_tensor)

        # calculate mean of image at t-1
        mean = engine.reverse(images, epsilon_pred, t_tensor)

        if t > 0:
            noise = torch.randn_like(images)
            sigma = engine.get_sigma(t_tensor, images.shape)
            images = mean + sigma * noise
        else:
            images = mean

    return images.detach()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    diffusion = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height,
                               cfg.diffusion.time_dim)
    checkpoint = torch.load(f"{cfg.diffusion.checkpoint_path}/{cfg.diffusion.name}.pt", map_location=device)
    diffusion.load_state_dict(checkpoint['model_state_dict']) #

    _, dataset = get_datasets(cfg)

    engine = Engine(cfg, device=device)

    if True:
        T = 500
        image, _ = dataset[0]
        image, _ = engine.diffuse(image, torch.tensor([T]))
        import matplotlib.pyplot as plt
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        gen_image = reconstruct_image(diffusion, engine, image, T, device=device)

    else:
        gen_image = generate_image(diffusion, engine, batch_size=1, device=device)

    gen_image = dataset.denormalize(gen_image)

    os.makedirs(f"{cfg.system.figures_dir}", exist_ok=True)
    save_image(gen_image, f"{cfg.system.figures_dir}/generated_image.png")


if __name__ == '__main__':
    main()