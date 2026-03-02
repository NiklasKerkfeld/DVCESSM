import os

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from tqdm import trange

from DiffusionCounterFactuals import CounterFactualDiffusion
from Parameter import N_CLASSES, IN_WIDTH, IN_HEIGHT
from architecture.Classifier import Classifier
from matplotlib import pyplot as plt

from utils import get_one_of_each, to_vis


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(0.5, 0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    samples = get_one_of_each(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(10)
    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    trainer = CounterFactualDiffusion(classifier, 1, device)
    trainer.load_model()

    counterfactuals = torch.zeros(len(samples), N_CLASSES, IN_WIDTH, IN_HEIGHT)
    for i in trange(N_CLASSES):
        targets = torch.full((len(samples), 1), i, dtype=torch.long)
        cf = trainer.generate_counterfactual(samples, targets=targets).detach().cpu()
        pred = classifier(cf).argmax(dim=1)

        print(f"{i} {pred}")
        counterfactuals[:, i] = cf.squeeze(1)

    counterfactuals[torch.arange(10), torch.arange(10)] = 0.0
    cf_flat = counterfactuals.reshape(-1, 1, IN_WIDTH, IN_HEIGHT)
    cf_grid = torchvision.utils.make_grid(cf_flat, nrow=10, padding=2)

    orig_grid = torchvision.utils.make_grid(to_vis(samples), nrow=1, padding=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12),
                                   gridspec_kw={'width_ratios': [1, 10]})

    # --- Samples ---
    ax1.imshow(orig_grid.permute(1, 2, 0), cmap="gray")
    ax1.set_title("Original", fontsize=16, pad=20)
    ax1.set_xticks([])
    ax1.set_yticks(np.arange(10) * 34 + 18)
    ax1.set_yticklabels(np.arange(10))

    # --- Main Plot (Swapped Axes) ---
    ax2.imshow(cf_grid.permute(1, 2, 0), cmap="gray")
    ax2.set_title("Counterfactual Generations", fontsize=16, pad=20)

    # X-axis
    ax2.set_xlabel('Target Class', fontsize=12)
    ax2.set_xticks(np.arange(10) * 34 + 18)
    ax2.set_xticklabels(np.arange(10))

    # Y-axis
    ax2.set_ylabel('Original Class', fontsize=12)
    ax2.set_yticks(np.arange(10) * 34 + 18)
    ax2.set_yticklabels(np.arange(10))

    plt.tight_layout()
    plt.savefig("./figures/counterfactuals.png")
    plt.close()


if __name__ == '__main__':
    main()
