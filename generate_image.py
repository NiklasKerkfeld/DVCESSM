import os

import torch
from matplotlib import pyplot as plt

from torchvision.transforms import transforms
from torchvision.datasets import MNIST

from DiffusionCounterFactuals import CounterFactualDiffusion
from Parameter import BETA_MIN, BETA_MAX, T
from architecture.Classifier import Classifier


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(0.5, 0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([x[0] for x in train_dataset], dim=0).unsqueeze(1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(10)
    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    diffusion_model = CounterFactualDiffusion(classifier, 1, device, name=f"DiffusionModel")
    diffusion_model.load_model()

    image = images[0][None]
    print(image.shape)
    t = 1

    beta = torch.linspace(BETA_MIN, BETA_MAX, T)
    alphas = 1.0 - beta
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noised = torch.sqrt(alphas_cumprod[t]) * image + torch.sqrt(
        1 - alphas_cumprod[t]) * torch.randn(image.size())

    gen_image = diffusion_model.generate_image()[0, 0].cpu().numpy()

    print(gen_image.max())
    print(gen_image.min())

    # gen_image = ((gen_image + 1.0) / 2.0).clip(0.0, 1.0)

    plt.imshow(gen_image, cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.title("Generated Image")

    os.makedirs("./results", exist_ok=True)
    # plt.savefig(f"./results/generated_image.png")
    # plt.close()
    plt.show()


if __name__ == '__main__':
    main()