import glob

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from DiffusionCounterFactuals import CounterFactualDiffusion
from architecture.Classifier import Classifier


def main():
    nr = len(glob.glob('runs/Diffusion/'))
    print(f"Start {nr}. Diffusion ...")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(0.5, 0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([x[0] for x in train_dataset], dim=0).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(10)
    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    trainer = CounterFactualDiffusion(classifier, 1, device, lr=1e-3, name=f"{nr}_DiffusionModel")
    trainer.train_diffusion(images, epochs=15)


if __name__ == '__main__':
    main()