import hydra
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import trange

from core.CounterfactualGenerator import CounterfactualGenerator
from core.DiffusionEngine import DiffusionEngine
from models.Classifier import Classifier
from models.DiffusionModel import DiffusionModel
from utils import get_one_of_each, to_vis


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # 2. Data Loading (Using config for paths)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, 0.0),
        transforms.Normalize(0.5, 0.5)
    ])
    train_dataset = MNIST(root=cfg.data.root, train=True, download=True, transform=transform)
    samples = get_one_of_each(train_dataset).to(device)

    classifier = Classifier(
        cfg.classifier.dims,
        cfg.classifier.num_classes,
        cfg.classifier.image_width,
        cfg.classifier.image_height
    )
    classifier.load_state_dict(torch.load(cfg.inference.classifier_path, map_location=device))

    model = DiffusionModel(cfg.model.dims, cfg.data.image_width, cfg.data.image_height, cfg.model.time_dim)
    checkpoint = torch.load(cfg.inference.diffusion_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    engine = DiffusionEngine(**cfg.engine, device=device)

    cfd = CounterfactualGenerator(classifier, model, engine, device,
                                  weight_classifier=cfg.inference.weight_classifier,
                                  weight_distance=cfg.inference.weight_distance)

    counterfactuals = torch.zeros(
        len(samples),
        cfg.data.n_classes,
        cfg.data.image_width,
        cfg.data.image_height
    )
    predictions = torch.zeros(
        len(samples),
        cfg.data.n_classes,
        dtype=torch.long)  # New tensor


    for i in trange(cfg.data.n_classes):
        targets = torch.full((len(samples), 1), i, dtype=torch.long)
        cf = cfd.generate_counterfactual(
            samples,
            targets=targets,
            start_fraction=cfg.inference.start_fraction
        ).detach().cpu()
        pred = classifier(cf).argmax(dim=1)
        predictions[:, i] = pred.cpu()
        counterfactuals[:, i] = cf.squeeze(1)

    plot_results(counterfactuals, samples, predictions)


def plot_results(counterfactuals: Tensor, samples: Tensor, predictions: Tensor):
    num_samples = counterfactuals.shape[0]
    num_targets = counterfactuals.shape[1]

    counterfactuals = to_vis(counterfactuals)
    counterfactuals[torch.arange(num_samples), torch.arange(num_targets)] = 0.0
    cf_flat = counterfactuals.reshape(-1, 1, samples.shape[-2], samples.shape[-1])
    cf_grid = torchvision.utils.make_grid(cf_flat, nrow=num_samples, padding=2)

    orig_grid = torchvision.utils.make_grid(to_vis(samples), nrow=1, padding=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12),
                                   gridspec_kw={'width_ratios': [1, 10]})

    # --- Samples ---
    ax1.imshow(orig_grid.permute(1, 2, 0), cmap="gray")
    ax1.set_title("Original", fontsize=16, pad=20)
    ax1.set_xticks([])
    ax1.set_yticks(np.arange(10) * 34 + 18)
    ax1.set_yticklabels(np.arange(num_samples))

    # --- Main Plot (Swapped Axes) ---
    ax2.imshow(cf_grid.permute(1, 2, 0), cmap="gray")
    ax2.set_title("Counterfactual Generations", fontsize=16, pad=20)

    # X-axis
    ax2.set_xlabel('Target Class', fontsize=12)
    ax2.set_xticks(np.arange(num_targets) * 34 + 18)
    ax2.set_xticklabels(np.arange(num_targets))

    # Y-axis
    ax2.set_ylabel('Original Class', fontsize=12)
    ax2.set_yticks(np.arange(num_samples) * 34 + 18)
    ax2.set_yticklabels(np.arange(num_samples))

    # Cell size in pixels (Padding=2, MNIST=32 after padding in your transform)
    cell_size = 32 + 2
    offset = 4  # Small offset from the top-left corner

    for row in range(num_samples):
        for col in range(num_targets):
            if row == col:
                continue

            pred_val = predictions[row, col].item()
            target_val = col

            # Color: Green if successful CF, Red if failed
            color = "lime" if pred_val == target_val else "red"

            # Calculate pixel position
            x_pos = col * cell_size + offset
            y_pos = row * cell_size + offset + 8  # +8 to adjust for text baseline

            ax2.text(x_pos, y_pos, str(pred_val), color=color,
                     fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.tight_layout()
    plt.savefig("./figures/counterfactuals.png")
    plt.close()


if __name__ == '__main__':
    main()
