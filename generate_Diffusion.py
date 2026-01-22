import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from tqdm import trange

from DiffusionCounterFactuals import CounterFactualDiffusion
from Parameter import N_CLASSES, IN_WIDTH, IN_HEIGHT
from architecture.Classifier import Classifier
from matplotlib import pyplot as plt

from utils import get_one_of_each


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(0.5, 0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = torch.concatenate([x[0] for x in train_dataset], dim=0).unsqueeze(1)

    samples = get_one_of_each(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(10)
    classifier.load_state_dict(torch.load("./models/classifier.pt"))

    trainer = CounterFactualDiffusion(classifier, images, device, lr=1e-3)
    trainer.load_model()

    counterfactuals = torch.zeros(N_CLASSES, len(samples), IN_WIDTH, IN_HEIGHT)
    for i in trange(N_CLASSES):
        cf = trainer.generate_counterfactual(samples, target_class=i).detach().cpu()
        print(f"{cf.shape=}")
        counterfactuals[i] = cf.squeeze(1)

    counterfactuals[torch.arange(10), torch.arange(10)] = 0.0
    img_grid = torchvision.utils.make_grid(counterfactuals.reshape(-1, 1, IN_WIDTH, IN_HEIGHT), nrow=10)

    plt.imshow(img_grid.permute(1, 2, 0), cmap="gray")
    plt.savefig("./figures/counterfactuals.png")
    plt.close()


if __name__ == '__main__':
    main()
