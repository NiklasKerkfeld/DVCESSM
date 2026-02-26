import os

import torch
from matplotlib import pyplot as plt

from DiffusionCounterFactuals import CounterFactualDiffusion
from architecture.Classifier import Classifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(10)
    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    diffusion_model = CounterFactualDiffusion(classifier, 1, device, name=f"DiffusionModel")
    gen_image = diffusion_model.generate_image()[0, 0]

    print(gen_image.max())
    print(gen_image.min())

    # gen_image = ((gen_image + 1.0) / 2.0).clip(0.0, 1.0)

    plt.imshow(gen_image, cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.title("Generated Image")

    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/generated_image.png")
    plt.close()


if __name__ == '__main__':
    main()