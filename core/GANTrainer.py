import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from models.GAN import Generator, Adversarial
from utils import visualize_mri, visualize_seg


class GANTrainer:
    def __init__(self,
                 generator: Generator,
                 adversarial: Adversarial,
                 segmentation: nn.Module,
                 device: torch.device,
                 name='GANModel',
                 lr_generator: float = 0.001,
                 lr_adversarial: float = 0.001,
                 lambda_adv: float = 1.0,
                 lambda_seg: float = 1.0,
                 lambda_adpt: float = 1.0,
                 lambda_per: float = 1.0
                 ):
        self.generator = generator
        self.adversarial = adversarial
        self.segmentation = segmentation
        self.device = device
        self.name = name
        self.lr_generator = lr_generator
        self.lr_adversarial = lr_adversarial
        self.lambda_adv = lambda_adv
        self.lambda_seg = lambda_seg
        self.lambda_adpt = lambda_adpt
        self.lambda_per = lambda_per

        self.epoch = 0

        self.dis_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = nn.CrossEntropyLoss()
        self.rec_loss_fn = nn.MSELoss()

        self.losses = {
            "generation loss": {
                "loss": [],
                "adoption": [],
                "adversarial": [],
                "segmentation": []
            },
            "adversarial loss": []
        }

        self.writer = SummaryWriter(f'runs/GAN/{name}')

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int):
        self.generator.to(self.device)
        self.adversarial.to(self.device)
        self.segmentation.to(self.device)

        generator_optimizer = AdamW(self.generator.parameters(), lr=self.lr_generator)
        adversarial_optimizer = AdamW(self.adversarial.parameters(), lr=self.lr_adversarial)

        for self.epoch in range(1, epochs + 1):
            self.train_epoch(train_dataloader, generator_optimizer, adversarial_optimizer)
            self.evaluation(valid_dataloader)

            # save model
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(self.generator.state_dict(), f"checkpoints/{self.name}_generator.pth")
            torch.save(self.adversarial.state_dict(), f"checkpoints/{self.name}_adversarial.pth")

    def train_epoch(self, train_dataloader: DataLoader, generator_optimizer,
                    adversarial_optimizer):
        self.generator.train()
        self.adversarial.train()
        self.segmentation.eval()

        self.losses = {
            "generation loss": {
                "loss": [],
                "adoption": [],
                "reconstruction": [],
                "adversarial": [],
                "segmentation": []
            },
            "adversarial loss": []
        }
        for image, target in tqdm(train_dataloader, total=len(train_dataloader),
                                  desc='train generator'):
            image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                prediction = self.segmentation(image).argmax(1).long()

            # train generator
            counterfactual = self.train_generator(image, target, prediction, generator_optimizer)
            self.train_adversarial(image, target, counterfactual, adversarial_optimizer)

        self.writer.add_scalar("generation/loss",
                               torch.mean(torch.tensor(self.losses["generation loss"]["loss"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/adoption",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["adoption"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/reconstruction",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["reconstruction"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/adversarial",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["adversarial"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation/segmentation",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["segmentation"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial/loss",
                               torch.mean(torch.tensor(self.losses["adversarial loss"])),
                               global_step=self.epoch)

    def train_generator(self,
                        image: torch.Tensor,
                        target: torch.Tensor,
                        prediction: torch.Tensor,
                        optimizer: AdamW) -> torch.Tensor:
        self.generator.zero_grad()

        adoption_cf = self.generator(image, target, prediction)
        counterfactual = image + adoption_cf
        adoption_rec = self.generator(counterfactual, prediction, target)
        reconstruction = counterfactual + adoption_rec
        rec_loss = self.rec_loss_fn(reconstruction, image)

        per_loss = torch.mean(torch.abs(adoption_cf)) + torch.mean(torch.abs(adoption_rec))

        adv_pred = self.adversarial(counterfactual, target)
        adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(image.shape[0], dtype=torch.long,
                                                          device=self.device))

        seg_pred = self.segmentation(counterfactual)
        seg_loss = self.seg_loss_fn(seg_pred, target)

        loss = (self.lambda_adpt * rec_loss +
                self.lambda_adv * adv_loss +
                self.lambda_seg * seg_loss +
                self.lambda_per * per_loss)
        loss.backward()
        optimizer.step()

        self.losses["generation loss"]["loss"].append(loss.detach().cpu().item())
        self.losses["generation loss"]["adoption"].append(per_loss.detach().cpu().item())
        self.losses["generation loss"]["reconstruction"].append(rec_loss.detach().cpu().item())
        self.losses["generation loss"]["adversarial"].append(adv_loss.detach().cpu().item())
        self.losses["generation loss"]["segmentation"].append(seg_loss.detach().cpu().item())

        return counterfactual

    def train_adversarial(self, image: torch.Tensor, target: torch.Tensor,
                          counterfactual: torch.Tensor, optimizer):
        self.adversarial.train()

        real_pred = self.adversarial(image, target)
        d_loss_real = self.dis_loss_fn(real_pred, torch.zeros(image.size(0), dtype=torch.long,
                                                              device=self.device))

        fake_pred = self.adversarial(counterfactual.detach(), target)
        d_loss_fake = self.dis_loss_fn(fake_pred, torch.ones(image.size(0), dtype=torch.long,
                                                             device=self.device))

        loss = d_loss_real + d_loss_fake

        loss.backward()
        optimizer.step()

        self.adversarial.zero_grad()

        self.losses["adversarial loss"].append(loss.detach().cpu().item())

    @torch.no_grad()
    def evaluation(self, valid_dataloader: DataLoader):
        self.generator.eval()
        self.adversarial.eval()
        self.segmentation.eval()

        val_losses = {
            "loss": [],
            "adoption": [],
            "reconstruction": [],
            "adversarial": [],
            "segmentation": []
        }

        for image, target in tqdm(valid_dataloader, desc='validation'):
            image, target = image.to(self.device), target.to(self.device)

            with torch.no_grad():
                prediction = self.segmentation(image).argmax(1).long()

            adoption_cf = self.generator(image, target, prediction)
            counterfactual = image + adoption_cf
            adoption_rec = self.generator(counterfactual, prediction, target)
            reconstruction = counterfactual + adoption_rec
            rec_loss = self.rec_loss_fn(reconstruction, image)

            per_loss = torch.mean(torch.abs(adoption_cf)) + torch.mean(torch.abs(adoption_rec))

            adv_pred = self.adversarial(counterfactual, target)
            adv_loss = self.dis_loss_fn(adv_pred,
                                        torch.zeros(counterfactual.shape[0], dtype=torch.long,
                                                    device=self.device))

            seg_pred = self.segmentation(counterfactual)
            seg_loss = self.seg_loss_fn(seg_pred, target)

            loss = (self.lambda_adpt * rec_loss +
                    self.lambda_adv * adv_loss +
                    self.lambda_seg * seg_loss +
                    self.lambda_per * per_loss)

            # Store results
            val_losses["loss"].append(loss.item())
            val_losses["adoption"].append(per_loss.item())
            val_losses["reconstruction"].append(rec_loss.item())
            val_losses["adversarial"].append(adv_loss.item())
            val_losses["segmentation"].append(seg_loss.item())

        # Log averages to Tensorboard
        avg_loss = sum(val_losses["loss"]) / len(val_losses["loss"])
        self.writer.add_scalar("val/loss", avg_loss, global_step=self.epoch)
        self.writer.add_scalar("val/adoption",
                               sum(val_losses["adoption"]) / len(val_losses["adoption"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/reconstruction",
                               sum(val_losses["reconstruction"]) / len(val_losses["reconstruction"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/adversarial",
                               sum(val_losses["adversarial"]) / len(val_losses["adversarial"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/segmentation",
                               sum(val_losses["segmentation"]) / len(val_losses["segmentation"]),
                               global_step=self.epoch)

        seg_org = self.segmentation(image)
        print(f"{seg_org.shape=}")
        print(f"{seg_pred.shape=}")
        self.save_visual_explanation(image.cpu(),
                                     counterfactual.cpu(),
                                     reconstruction.cpu(),
                                     target.cpu(),
                                     seg_pred.argmax(1).cpu(),
                                     seg_org.argmax(1).cpu())

    @torch.no_grad()
    def save_visual_explanation(self, img, cf, reconstruction, target_mask, seg_org, seg_cf):
        """
        img, adoption, cf shape: (4, H, W) - [T1, T1c, T2, FLAIR]
        target_mask shape: (1, H, W) or (C, H, W)
        """
        img = visualize_mri(img)
        cf = visualize_mri(cf)
        reconstruction = visualize_mri(reconstruction)

        for i, seq in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            visual = torch.stack([img[:, i], cf[:, i], reconstruction[:, i]], dim=1).flatten(0, 1)
            self.writer.add_image(f'image/{seq}',
                                  make_grid(visual, nrow=3, padding=2),
                                  global_step=self.epoch)

        target_map = visualize_seg(target_mask).permute(3, 0, 1, 2)
        orig_map = visualize_seg(seg_org).permute(3, 0, 1, 2)
        cf_map = visualize_seg(seg_cf).permute(3, 0, 1, 2)
        self.writer.add_image('image/maps',
                              make_grid([target_map, orig_map, cf_map], nrow=3, padding=2),
                              global_step=self.epoch)
