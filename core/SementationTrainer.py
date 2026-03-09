import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm


class SegmentationTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 name='SegmentationModel',
                 ):
        self.model = model
        self.device = device
        self.name = name
        self.epoch = 0
        self.num_classes = model.out_channels

        self.loss_fn = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(f'runs/Segmentation/{name}')

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, lr: float):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        images = torch.stack([valid_dataloader.dataset[i][0] for i in range(5)])
        targets = torch.stack([valid_dataloader.dataset[i][1] for i in range(5)])
        self.writer.add_image('images', make_grid(images, nrow=5, normalize=True))
        self.writer.add_image('targets', make_grid(targets.unsqueeze(1).float(), nrow=5, normalize=True))

        self.model.to(self.device)
        self.model.train()

        for self.epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_dataloader, optimizer)
            self.writer.add_scalar('loss', loss, global_step=self.epoch)

            # evaluation
            iou, precision, recall, f1 = self.evaluate(valid_dataloader, self.device)
            for i in range(self.num_classes):
                self.writer.add_scalar(f'class_{i}/iou', iou[i], global_step=self.epoch)
                self.writer.add_scalar(f'class_{i}/precision', precision[i], global_step=self.epoch)
                self.writer.add_scalar(f'class_{i}/recall', recall[i], global_step=self.epoch)
                self.writer.add_scalar(f'class_{i}/f1', f1[i], global_step=self.epoch)

            # sample logging
            self.log_samples(images)

        # save model
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), f"checkpoints/{self.name}.pth")

    def train_one_epoch(self, dataloader: DataLoader, optimizer):
        self.model.train()
        running_loss = 0

        bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {self.epoch}")
        for images, mask in bar:
            images, mask = images.to(self.device), mask.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            bar.set_postfix(loss=loss.detach().item())

        return (running_loss / len(dataloader))


    @torch.no_grad()
    def evaluate(self, dataloader, device):
        self.model.eval()
        # Initialize a confusion matrix: [Target, Prediction]
        conf_matrix = torch.zeros((self.num_classes, self.num_classes), device=device)

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                # Mask out ignore_index if you have one (e.g., 255)
                mask = (labels >= 0) & (labels < self.num_classes)

                # Efficiently update confusion matrix
                conf_matrix += torch.bincount(
                    self.num_classes * labels[mask].view(-1) + preds[mask].view(-1),
                    minlength=self.num_classes ** 2
                ).reshape(self.num_classes, self.num_classes)

        # Calculate Metrics
        tp = conf_matrix.diag()
        fp = conf_matrix.sum(dim=0) - tp
        fn = conf_matrix.sum(dim=1) - tp

        # Avoid division by zero
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)

        return iou, precision, recall, f1

    @torch.no_grad()
    def log_samples(self, images: torch.Tensor):
        self.model.eval()

        out = self.model(images.to(self.device))
        out = torch.argmax(out, dim=1)

        self.writer.add_image(
            'predictions',
            make_grid(out, nrow=5, normalize=True),
            global_step=self.epoch)
