import glob
import os
from dotenv import load_dotenv

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from CounterFactualGenerator import CounterFactualGenerator
from dataset import ClassificatorDataset
from models import Classifier

load_dotenv()

N_CLASSES = int(os.getenv('N_CLASSES'))
IN_CHANNELS = int(os.getenv('IN_CHANNELS'))
IN_HEIGHT = int(os.getenv('IN_HEIGHT'))
IN_WIDTH = int(os.getenv('IN_WIDTH'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
ITERATIONS = int(os.getenv('ITERATIONS'))
ALPHA = float(os.getenv('ALPHA'))
BETA = float(os.getenv('BETA'))

STD_CLS = float(os.getenv('STD_CLS'))
P_CLS = float(os.getenv('P_CLS'))
B_CLS = float(os.getenv('B_CLS'))
B_H_MAX_CLS = int(os.getenv('B_H_MAX_CLS'))
B_W_MAX_CLS = int(os.getenv('B_W_MAX_CLS'))


def train_classifier(classifier, dataset, device, batch_size: int = BATCH_SIZE, num_epochs=5):
    images = torch.concatenate([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    dataset = ClassificatorDataset(images, targets,
                                   std=STD_CLS,
                                   p=P_CLS,
                                   p_block=B_CLS,
                                   max_block_width=B_H_MAX_CLS,
                                   max_block_height=B_W_MAX_CLS)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    classifier.to(device)
    classifier.train()  # Set model to training mode

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, "training", total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    os.makedirs("./models", exist_ok=True)
    torch.save(classifier.state_dict(), "./models/classifier.pt")


def main():
    nr = len(glob.glob('runs/*'))
    print(f"Start {nr}. CounterFactualGenerator...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset = Subset(train_dataset, torch.arange(1024))

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
