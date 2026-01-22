import glob

import torch
from torch.utils.data import Subset

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from CounterFactualGenerator import CounterFactualGenerator
from architecture.Classifier import Classifier
from Parameter import N_CLASSES


def main():
    nr = len(glob.glob('runs/GAN/'))
    print(f"Start {nr}. CounterFactualGenerator...")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(mean=0.5, std=0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset = Subset(train_dataset, torch.arange(1024))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(N_CLASSES)

    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    cfg = CounterFactualGenerator(classifier, train_dataset, device=device, name=f"{nr}_CounterFactualGenerator")
    cfg.train()

if __name__ == '__main__':
    main()
