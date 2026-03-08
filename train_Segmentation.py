import os
import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.SegmentationDataset import SegmentationDataset
from models.Segmentation import SegmentationUNet

torch.manual_seed(42)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, mask in pbar:
        images, mask = images.to(device), mask.to(device)

        outputs = model(images)
        loss = criterion(outputs, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    # Initialize a confusion matrix: [Target, Prediction]
    conf_matrix = torch.zeros((num_classes, num_classes), device=device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Mask out ignore_index if you have one (e.g., 255)
            mask = (labels >= 0) & (labels < num_classes)

            # Efficiently update confusion matrix
            conf_matrix += torch.bincount(
                num_classes * labels[mask].view(-1) + preds[mask].view(-1),
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # Calculate Metrics
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    # Avoid division by zero
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    print_metrics(num_classes, precision, recall, f1, iou)


def print_metrics(num_classes, prec, rec, f1, iou):
    print(f"\n{'Class':<10} | {'IoU':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 55)
    for i in range(num_classes):
        print(f"{i:<10} | {iou[i]:.4f} | {prec[i]:.4f} | {rec[i]:.4f} | {f1[i]:.4f}")

    print("-" * 55)
    print(
        f"{'Mean':<10} | {iou.mean():.4f} | {prec.mean():.4f} | {rec.mean():.4f} | {f1.mean():.4f}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = SegmentationDataset()
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    model = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.segmentation.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.segmentation.epochs + 1):
        print(f'Epoch {epoch}/{cfg.segmentation.epochs}')
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {loss:.4f} | Train Acc: {acc:.2f}%")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), cfg.inference.segmentation_path)

    evaluate(model, test_loader, device, cfg.data.num_classes)


if __name__ == '__main__':
    main()
