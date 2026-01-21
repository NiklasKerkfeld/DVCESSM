import glob
import os

import torch

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from CounterFactualGenerator import CounterFactualGenerator
from architecture.Classifier import Classifier, train_classifier
from Parameter import N_CLASSES, EPOCHS


def main():
    nr = len(glob.glob('runs/GAN/'))
    print(f"Start {nr}. CounterFactualGenerator...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(N_CLASSES)

    if os.path.exists("./models/classifier.pt"):
        classifier.load_state_dict(torch.load("./models/classifier.pt"))
    else:
        train_classifier(classifier, train_dataset, device, num_epochs=EPOCHS)

    cfg = CounterFactualGenerator(classifier, train_dataset, device=device, name=f"{nr}_CounterFactualGenerator")
    cfg.train()

if __name__ == '__main__':
    main()
