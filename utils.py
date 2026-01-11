import os

import torch
from matplotlib import pyplot as plt

import torch.nn.functional as F

from models import Generator

def process(images, std=0.0, p=0.0):
    # Gaussian noise
    if std != 0.0:
        images += torch.randn_like(images) * std

    # salt and paper noise
    if p != 0.0:
        noise = torch.randint_like(images, 1000) < (1000 * p)
        prob = torch.randint_like(images, 2)

        images[noise] = prob[noise]

    # return normalized result
    return (images - 0.5) / 0.5


def random_target(input_tensor, n):
    offset = torch.randint(1, n - 1, input_tensor.shape, device=input_tensor.device)
    target = ((input_tensor - 1 + offset) % (n - 1)) + 1
    return target


def get_one_of_each(dataset):
    samples = {}

    # Iterate through the dataset
    for i in range(len(dataset)):
        image, label = dataset[i]
        # If we haven't found this digit yet, save it
        if label not in samples:
            samples[label] = image

        # Stop once we have all 10 digits (0-9)
        if len(samples) == 10:
            break

    # Sort by label to ensure they are in order 0-9
    return torch.cat([samples[i] for i in range(10)])[:, None]


def plot_samples(generator: Generator, samples: torch.Tensor, device, title:str =""):
    generator = generator.to(device)
    samples = samples.to(device)

    images = [[x for x in samples]]
    for y in range(10):
        targets = torch.full((len(samples),), y)
        changes = generator(process(samples), F.one_hot(targets, num_classes=10).float())
        new_images = samples + changes
        images.append([x for x in new_images])

    fig, axes = plt.subplots(11, 10, figsize=(10, 11))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for row_idx, row_data in enumerate(images):
        for col_idx, img_tensor in enumerate(row_data):
            ax = axes[row_idx, col_idx]

            img = img_tensor.detach().cpu().squeeze().numpy()

            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            if col_idx == 0:
                if row_idx == 0:
                    ax.set_title("Original", loc='left', fontsize=10)
                else:
                    ax.set_title(f"Target {row_idx - 1}", loc='left', fontsize=10)

    plt.tight_layout()
    plt.suptitle(title)

    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f"figures/{title}.png")
    plt.close()
