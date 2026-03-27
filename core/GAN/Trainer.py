import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchvision.utils import make_grid
from tqdm import tqdm

from core.Trainer import SuperTrainer
from models.GAN import Generator, Adversarial
from utils import visualize_mri, visualize_seg, visualize_adaption


class Trainer(SuperTrainer):
    def __init__(self,
                 generator: Generator,
                 adversarial: Adversarial,
                 segmentation: nn.Module,
                 lr_generator: float = 0.001,
                 lr_adversarial: float = 0.001,
                 lambda_adv: float = 1.0,
                 lambda_seg: float = 1.0,
                 lambda_rec: float = 1.0,
                 lambda_per: float = 1.0,
                 free_rec: float = 0.0,
                 free_per: float = 0.0,
                 device: torch.device = torch.device('cpu'),
                 name: str = 'GANModel',
                 log_dir: str = 'runs/GAN',
                 checkpoint: str = "./checkpoints/GAN"
                 ):
        super().__init__(device, name, log_dir, checkpoint)
        self.generator = generator
        self.adversarial = adversarial
        self.segmentation = segmentation

        self.lr_generator = lr_generator
        self.lr_adversarial = lr_adversarial

        self.lambda_adv = lambda_adv
        self.lambda_seg = lambda_seg
        self.lambda_adpt = lambda_rec
        self.lambda_per = lambda_per
        self.free_rec = free_rec
        self.free_per = free_per

        self.dis_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = nn.CrossEntropyLoss()
        self.rec_loss_fn = nn.MSELoss()

        self.generator.to(self.device)
        self.adversarial.to(self.device)
        self.segmentation.to(self.device)

        for param in self.segmentation.parameters():
            param.requires_grad = False

        self.generator_optimizer = AdamW(self.generator.parameters(), lr=self.lr_generator)
        self.adversarial_optimizer = AdamW(self.adversarial.parameters(), lr=self.lr_adversarial)

        self.losses = {}

        self.metrics = MetricCollection({
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall()
        }).to(self.device)

        self.reset_loss()

    def reset_loss(self):
        self.metrics.reset()

        self.losses = {
            "generation": {
                "loss": [],
                "perturbation": [],
                "reconstruction": [],
                "adversarial": [],
                "segmentation": []
            },
            "adversarial": {
                "loss": [],
                "accuracy": [],
                "f1": [],
                "precision": [],
                "recall": []
            },
            "validation": {
                "loss": [],
                "perturbation": [],
                "reconstruction": [],
                "adversarial": [],
                "segmentation": []
            }
        }

    def save_checkpoint(self):
        os.makedirs(self.checkpoint, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'adversarial_state_dict': self.adversarial.state_dict(),
            'adversarial_optimizer_state_dict': self.adversarial_optimizer.state_dict(),
            'segmentation_state_dict': self.segmentation.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, f'{self.checkpoint}/{self.name}.pt')

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

        self.adversarial.load_state_dict(checkpoint['adversarial_state_dict'])
        self.adversarial_optimizer.load_state_dict(checkpoint['adversarial_optimizer_state_dict'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int):
        for self.epoch in range(1, epochs + 1):
            self.reset_loss()

            self.train_epoch(train_dataloader)
            self.evaluation(valid_dataloader)

            self.log_losses()

            self.save_checkpoint()

    def train_epoch(self, train_dataloader: DataLoader):
        self.segmentation.eval()

        for image, target in tqdm(train_dataloader, total=len(train_dataloader),
                                  desc=f"Epoch {self.epoch}"):
            image, target = image.to(self.device), target.to(self.device)

            counterfactual = self.train_generator(image, target)
            self.train_adversarial(image, target, counterfactual)

    def train_generator(self,
                        image: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        self.generator.train()
        self.adversarial.eval()
        self.segmentation.eval()
        self.generator.zero_grad()

        with torch.no_grad():
            prediction_org = self.segmentation(image).argmax(1).long()

        perturbation_cf = self.generator(image, target, prediction_org)
        counterfactual = image + perturbation_cf
        perturbation_rec = self.generator(counterfactual, prediction_org, target)
        reconstruction = counterfactual + perturbation_rec
        rec_loss = self.rec_loss_fn(reconstruction, image)
        rec_loss = F.relu(rec_loss - self.free_rec)

        per_loss = torch.mean(torch.abs(perturbation_cf)) + torch.mean(torch.abs(perturbation_rec))
        per_loss = F.relu(per_loss - self.free_per)

        adv_pred = self.adversarial(counterfactual, target)
        adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(image.shape[0], dtype=torch.long,
                                                          device=self.device))

        prediction_cf = self.segmentation(counterfactual)
        seg_loss = self.seg_loss_fn(prediction_cf, target)

        loss = (self.lambda_adv * adv_loss +
                self.lambda_seg * seg_loss +
                self.lambda_adpt * rec_loss +
                self.lambda_per * per_loss)

        loss.backward()
        self.generator_optimizer.step()

        self.losses["generation"]["loss"].append(loss.detach().cpu().item())
        self.losses["generation"]["perturbation"].append(per_loss.detach().cpu().item())
        self.losses["generation"]["reconstruction"].append(rec_loss.detach().cpu().item())
        self.losses["generation"]["adversarial"].append(adv_loss.detach().cpu().item())
        self.losses["generation"]["segmentation"].append(seg_loss.detach().cpu().item())

        return counterfactual

    def train_adversarial(self,
                          image: torch.Tensor,
                          target: torch.Tensor,
                          counterfactual: torch.Tensor):
        self.generator.eval()
        self.adversarial.train()
        self.segmentation.eval()
        self.adversarial.zero_grad()

        real_pred = self.adversarial(image, target)
        real_labels = torch.zeros(image.size(0), dtype=torch.long, device=self.device)
        d_loss_real = self.dis_loss_fn(real_pred, real_labels)

        fake_pred = self.adversarial(counterfactual.detach(), target)
        fake_labels = torch.ones(image.size(0), dtype=torch.long, device=self.device)
        d_loss_fake = self.dis_loss_fn(fake_pred, fake_labels)

        loss = d_loss_real + d_loss_fake
        loss.backward()
        self.adversarial_optimizer.step()

        # --- Metrics Logging ---
        with torch.no_grad():
            all_preds = torch.cat([real_pred, fake_pred], dim=0)
            all_labels = torch.cat([real_labels, fake_labels], dim=0)

            batch_metrics = self.metrics(torch.argmax(all_preds, dim=1), all_labels)

        # Store results
        self.losses["adversarial"]["loss"].append(loss.detach().cpu().item())
        for key, value in batch_metrics.items():
            self.losses["adversarial"][key].append(value.item())

    @torch.no_grad()
    def evaluation(self, valid_dataloader: DataLoader):
        self.generator.eval()
        self.adversarial.eval()
        self.segmentation.eval()

        for image, target in tqdm(valid_dataloader, desc='validation'):
            image, target = image.to(self.device), target.to(self.device)

            with torch.no_grad():
                prediction = self.segmentation(image).argmax(1).long()

            perturbation_cf = self.generator(image, target, prediction)
            counterfactual = image + perturbation_cf
            perturbation_rec = self.generator(counterfactual, prediction, target)
            reconstruction = counterfactual + perturbation_rec
            rec_loss = self.rec_loss_fn(reconstruction, image)

            per_loss = torch.mean(torch.abs(perturbation_cf)) + torch.mean(torch.abs(perturbation_rec))

            adv_pred = self.adversarial(counterfactual, target)
            adv_loss = self.dis_loss_fn(adv_pred,
                                        torch.zeros(counterfactual.shape[0], dtype=torch.long,
                                                    device=self.device))

            prediction_cf = self.segmentation(counterfactual)
            seg_loss = self.seg_loss_fn(prediction_cf, target)

            loss = (self.lambda_adv * adv_loss +
                    self.lambda_seg * seg_loss +
                    self.lambda_adpt * rec_loss +
                    self.lambda_per * per_loss)

            # Store results
            self.losses["validation"]["loss"].append(loss.item())
            self.losses["validation"]["perturbation"].append(per_loss.item())
            self.losses["validation"]["reconstruction"].append(rec_loss.item())
            self.losses["validation"]["adversarial"].append(adv_loss.item())
            self.losses["validation"]["segmentation"].append(seg_loss.item())

        # Log averages to Tensorboard
        self.save_visual_explanation(image.cpu(),
                                     perturbation_cf.cpu(),
                                     counterfactual.cpu(),
                                     reconstruction.cpu(),
                                     target.cpu(),
                                     prediction.cpu(),
                                     prediction_cf.argmax(1).cpu())

    @torch.no_grad()
    def save_visual_explanation(self, img, perturbation, cf, reconstruction, target_mask, seg_org,
                                seg_cf):
        img = visualize_mri(img)
        cf = visualize_mri(cf)
        reconstruction = visualize_mri(reconstruction)
        perturbation = visualize_adaption(perturbation)
        orig_map = visualize_seg(seg_org).permute(0, 3, 1, 2)
        cf_map = visualize_seg(seg_cf).permute(0, 3, 1, 2)
        target_map = visualize_seg(target_mask).permute(0, 3, 1, 2)

        for i, seq in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            visual = torch.stack([img[:, i], perturbation[:, i], cf[:, i], reconstruction[:, i]],
                                 dim=1).flatten(0, 1)
            self.writer.add_image(f'image/{seq}',
                                  make_grid(visual, nrow=4, padding=2),
                                  global_step=self.epoch)

        visual = torch.stack([orig_map, cf_map, target_map], dim=1).flatten(0, 1)
        self.writer.add_image('image/maps',
                              make_grid(visual, nrow=3, padding=2),
                              global_step=self.epoch)

    def log_losses(self):
        self.writer.add_scalar("generation/loss",
                               torch.mean(torch.tensor(self.losses["generation"]["loss"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/perturbation",
                               torch.mean(
                                   torch.tensor(self.losses["generation"]["perturbation"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/reconstruction",
                               torch.mean(
                                   torch.tensor(self.losses["generation"]["reconstruction"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/adversarial",
                               torch.mean(
                                   torch.tensor(self.losses["generation"]["adversarial"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/segmentation",
                               torch.mean(
                                   torch.tensor(self.losses["generation"]["segmentation"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial/loss",
                               torch.mean(torch.tensor(self.losses["adversarial"]["loss"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial/f1",
                               torch.mean(torch.tensor(self.losses["adversarial"]["f1"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial/accuracy",
                               torch.mean(torch.tensor(self.losses["adversarial"]["accuracy"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial/precision",
                               torch.mean(torch.tensor(self.losses["adversarial"]["precision"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial/recall",
                               torch.mean(torch.tensor(self.losses["adversarial"]["recall"])),
                               global_step=self.epoch)

        avg_loss = sum(self.losses["validation"]["loss"]) / len(self.losses["validation"]["loss"])
        self.writer.add_scalar("validation/loss", avg_loss, global_step=self.epoch)
        self.writer.add_scalar("validation/perturbation",
                               sum(self.losses["validation"]["perturbation"]) / len(
                                   self.losses["validation"]["perturbation"]),
                               global_step=self.epoch)
        self.writer.add_scalar("validation/reconstruction",
                               sum(self.losses["validation"]["reconstruction"]) / len(
                                   self.losses["validation"]["reconstruction"]),
                               global_step=self.epoch)
        self.writer.add_scalar("validation/adversarial",
                               sum(self.losses["validation"]["adversarial"]) / len(
                                   self.losses["validation"]["adversarial"]),
                               global_step=self.epoch)
        self.writer.add_scalar("validation/segmentation",
                               sum(self.losses["validation"]["segmentation"]) / len(
                                   self.losses["validation"]["segmentation"]),
                               global_step=self.epoch)
