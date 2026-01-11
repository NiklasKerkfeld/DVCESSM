from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.nn.functional as F

from models import Classifier, Generator, DiscriminatorLoss
from utils import random_target, get_one_of_each, plot_samples, process

N_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10
ITERATIONS = 100

def train_generator(generator: Generator, classifier: Classifier, discriminator: Classifier,
                    train_loader: DataLoader, device: torch.device):

    generator.to(device)
    discriminator.to(device)
    classifier.to(device)

    for param in generator.parameters():
        param.requires_grad = True

    for param in classifier.parameters():
        param.requires_grad = False

    for param in discriminator.parameters():
        param.requires_grad = False

    criterion_classifier = nn.CrossEntropyLoss()
    criterion_discriminator = DiscriminatorLoss()

    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    bar = tqdm(train_loader, "training", total=len(train_loader))
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        targets = random_target(labels, N_CLASSES)

        images = process(images)

        optimizer.zero_grad()
        changes = generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
        counterfactuals = images + changes

        pred_class = classifier(counterfactuals)
        pred_disc = discriminator(counterfactuals)[:, 0]

        regularization = torch.mean(torch.abs(changes))
        loss_class = criterion_classifier(pred_class, targets)
        loss_disc = criterion_discriminator(pred_disc, torch.zeros_like(labels))
        loss = loss_class + loss_disc + regularization

        loss.backward()
        optimizer.step()

        pred = torch.argmax(pred_class, dim=1)
        like_label = (pred == labels).float().sum()
        like_target = (pred == targets).float().sum()
        label_acc = like_label / len(labels)
        target_acc = like_target / len(labels)

        bar.set_description(f"training loss: {loss.item():.4f} "
                            f"(class: {loss_class.item():.4f} "
                            f"disc: {loss_disc.item():.4f} "
                            f"regularization: {regularization.item():.4f} "
                            f"label acc {label_acc.item():.4f} "
                            f"target acc {target_acc.item():.4f})")

def train_discriminator(generator: Generator,
                        discriminator: Classifier,
                        train_loader, device):
    for param in generator.parameters():
        param.requires_grad = False

    for param in discriminator.parameters():
        param.requires_grad = True

    generator.to(device)
    discriminator.to(device)

    criterion_discriminator = DiscriminatorLoss()

    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # generate faces
    list_images = []
    list_labels = []
    bar = tqdm(train_loader, "generating", total=len(train_loader))
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        targets = random_target(labels, N_CLASSES)

        with torch.no_grad():
            change = generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
            list_images.append(images + change)
            list_labels.append(torch.ones_like(labels))
            list_images.append(images)
            list_labels.append(torch.zeros_like(labels))

    images = torch.cat(list_images, dim=0)
    labels = torch.cat(list_labels, dim=0)

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    bar = tqdm(dataloader, "training", total=len(dataloader))
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)

        images = process(images, std=0.1, p=0.1)

        pred = discriminator(images)[:, 0]

        optimizer.zero_grad()
        loss = criterion_discriminator(pred, labels)
        loss.backward()
        optimizer.step()

        pred = (pred > 0.5).int()
        correct = (pred == labels).float().sum()
        accuracy = correct / len(labels)

        bar.set_description(f"training loss: {loss.item():.4f} "
                            f"accuracy {accuracy.item():.4f})")


def train_classifier(classifier, train_loader, device, num_epochs=5):
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
            # Move data to the same device as the model (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # add noise and normalize
            images = process(images, std=0.1, p=0.1)

            # 1. Forward pass
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            # 2. Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute new gradients
            optimizer.step()  # Update weights

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    # train_dataset = Subset(train_dataset, range(10240))

    samples = get_one_of_each(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier((1, 32, 32), 10)
    train_classifier(classifier, train_loader, device, num_epochs=EPOCHS)

    discriminator = Classifier((1, 32, 32), 1)
    generator = Generator(1)

    for i in range(1, ITERATIONS + 1):
        print(f"Iteration {i}:")
        train_generator(generator, classifier, discriminator, train_loader, device)
        plot_samples(generator, samples, title=f"Iteration {i}")
        train_discriminator(generator, discriminator, train_loader, device)



if __name__ == '__main__':
    main()
