import torch
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import nn

from core.DiffusionDataset import get_MNIST_dataset, get_BraTS_dataset
from models.DiffusionModel import DiffusionModel
from core.DiffusionEngine import DiffusionEngine

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def evaluate_T(model, engine, images: torch.Tensor, device):
    loss_fn = nn.MSELoss()
    model.eval()

    images = images.to(device)
    model.to(device)
    engine.to(device)

    loss = torch.zeros((engine.T,))

    for t in range(engine.T):
        t_tensor = torch.full((images.shape[0],), t, device=device)
        noised, noise = engine.diffuse(images, t_tensor)

        epsilon_pred = model(noised, t_tensor)
        loss[t] = loss_fn(epsilon_pred, noise)

    plt.plot(loss)
    plt.xlabel("T")
    plt.ylabel("loss")
    plt.savefig("figures/loss.png")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if cfg.data.name == 'MNIST':
        dataset = get_MNIST_dataset()
    elif cfg.data.name == 'BraTS':
        dataset = get_BraTS_dataset()
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    samples = torch.cat([dataset[i][None] for i in torch.randint(0, len(dataset), (64,))], dim=0)

    model = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height,
                           cfg.model.time_dim)
    checkpoint = torch.load(cfg.training.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    engine = DiffusionEngine(**cfg.engine, device=device)

    evaluate_T(model, engine, samples, device)


if __name__ == '__main__':
    main()
