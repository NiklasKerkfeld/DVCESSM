import os
import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from core.ClassifierDataset import ClassificatorDataset
from models.Classifier import Classifier


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix(loss=loss.item(), acc=f"{100 * correct / total:.1f}%")

    return running_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, 0.0),
        transforms.Normalize(0.5, 0.5)
    ])

    raw_train = MNIST(root=cfg.data.root, train=True, download=True, transform=transform)
    raw_test = MNIST(root=cfg.data.root, train=False, download=True, transform=transform)

    # Wrap in your custom ClassificatorDataset for noise/blocking
    train_dataset = ClassificatorDataset(
        raw_train,
        cfg.data.image_width,
        cfg.data.image_height,
        **cfg.classifier.augmentation
    )
    test_dataset = ClassificatorDataset(
        raw_test,
        cfg.data.image_width,
        cfg.data.image_height
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    model = Classifier(
        in_channels=cfg.data.image_channels,
        num_classes=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.classifier.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.classifier.epochs + 1):
        print(f'Epoch {epoch}/{cfg.classifier.epochs}')
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {loss:.4f} | Train Acc: {acc:.2f}%")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), cfg.inference.classifier_path)

    evaluate(model, test_loader, device)


if __name__ == '__main__':
    main()
