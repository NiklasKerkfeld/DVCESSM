import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from Datasets import ClassificatorDataset
from Parameter import BATCH_SIZE, STD_CLS, P_CLS, B_CLS, B_H_MAX_CLS, B_W_MAX_CLS, N_CLASSES, EPOCHS
from architecture.Classifier import Classifier


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
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    os.makedirs("./models", exist_ok=True)
    torch.save(classifier.state_dict(), "./models/classifier.pt")


def evaluate(model, dataset, device, batch_size=64):
    images = torch.concatenate([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    dataset = ClassificatorDataset(images, targets)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.eval()

    test_loss = 0
    all_preds = []
    all_labels = []

    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)

        with torch.no_grad():
            output = model(data)

        test_loss += F.cross_entropy(output, labels, reduction='sum').item()

        pred = F.softmax(output, dim=1).argmax(dim=1)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())


    test_loss /= len(dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    correct = (all_preds == all_labels).sum()
    accuracy = correct / len(dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    print(f'\nTest Results:')
    print(f'\tAverage loss: {test_loss:.4f}')
    print(f'\tAccuracy:     {accuracy:.4f}')
    print(f'\tPrecision:    {precision:.4f}')
    print(f'\tRecall:       {recall:.4f}')
    print(f'\tF1-Score:     {f1:.4f}')

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds))


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(2, 0.0),
                                    transforms.Normalize(0.5, 0.5)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    classifier = Classifier(num_classes=N_CLASSES)
    train_classifier(classifier, train_dataset, device, num_epochs=EPOCHS)
    # classifier.load_state_dict(torch.load("./models/classifier.pt"))
    evaluate(classifier, test_dataset, device)


if __name__ == '__main__':
    main()
