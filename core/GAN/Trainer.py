import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from core.Trainer import SuperTrainer
from models.GAN import Generator, Adversarial
from models.Losses import MultiTaskLoss
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
                 lambda_adpt: float = 1.0,
                 lambda_per: float = 1.0,
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

        self.dis_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = nn.CrossEntropyLoss()
        self.rec_loss_fn = nn.MSELoss()
        self.loss_fn = MultiTaskLoss(4)

        self.generator.to(self.device)
        self.adversarial.to(self.device)
        self.segmentation.to(self.device)

        for param in self.segmentation.parameters():
            param.requires_grad = False

        self.generator_optimizer = AdamW(self.generator.parameters(), lr=self.lr_generator)
        self.adversarial_optimizer = AdamW(self.adversarial.parameters(), lr=self.lr_adversarial)

        self.losses = {}
        self.reset_loss()


    def reset_loss(self):
        self.losses = {
            "generation loss": {
                "loss": [],
                "adoption": [],
                "reconstruction": [],
                "adversarial": [],
                "segmentation": []
            },
            "adversarial loss": [],
            "val loss": {
                "loss": [],
                "adoption": [],
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
        self.generator.train()
        self.adversarial.eval()
        self.segmentation.eval()

        for image, target in tqdm(train_dataloader, total=len(train_dataloader),
                                  desc='train generator'):
            image, target = image.to(self.device), target.to(self.device)

            counterfactual = self.train_generator(image, target)
            self.train_adversarial(image, target, counterfactual)


    def train_generator(self,
                        image: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction_org = self.segmentation(image).argmax(1).long()

        self.generator.train()
        self.adversarial.eval()
        self.segmentation.eval()
        self.generator.zero_grad()

        adoption_cf = self.generator(image, target, prediction_org)
        counterfactual = image + adoption_cf
        adoption_rec = self.generator(counterfactual, prediction_org, target)
        reconstruction = counterfactual + adoption_rec
        rec_loss = self.rec_loss_fn(reconstruction, image)

        per_loss = torch.mean(torch.abs(adoption_cf)) + torch.mean(torch.abs(adoption_rec))

        adv_pred = self.adversarial(counterfactual, target)
        adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(image.shape[0], dtype=torch.long,
                                                          device=self.device))

        prediction_cf = self.segmentation(counterfactual)
        seg_loss = self.seg_loss_fn(prediction_cf, target)

        loss = self.loss_fn(torch.tensor([rec_loss, per_loss, adv_loss, seg_loss]))
        loss.backward()
        self.generator_optimizer.step()

        self.losses["generation loss"]["loss"].append(loss.detach().cpu().item())
        self.losses["generation loss"]["adoption"].append(per_loss.detach().cpu().item())
        self.losses["generation loss"]["reconstruction"].append(rec_loss.detach().cpu().item())
        self.losses["generation loss"]["adversarial"].append(adv_loss.detach().cpu().item())
        self.losses["generation loss"]["segmentation"].append(seg_loss.detach().cpu().item())

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
        d_loss_real = self.dis_loss_fn(real_pred, torch.zeros(image.size(0), dtype=torch.long,
                                                              device=self.device))

        fake_pred = self.adversarial(counterfactual.detach(), target)
        d_loss_fake = self.dis_loss_fn(fake_pred, torch.ones(image.size(0), dtype=torch.long,
                                                             device=self.device))

        loss = d_loss_real + d_loss_fake

        loss.backward()
        self.adversarial_optimizer.step()

        self.losses["adversarial loss"].append(loss.detach().cpu().item())

    @torch.no_grad()
    def evaluation(self, valid_dataloader: DataLoader):
        self.generator.eval()
        self.adversarial.eval()
        self.segmentation.eval()

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

            prediction_cf = self.segmentation(counterfactual)
            seg_loss = self.seg_loss_fn(prediction_cf, target)

            loss = self.loss_fn(torch.tensor([rec_loss, per_loss, adv_loss, seg_loss]))

            # Store results
            self.losses["val loss"]["loss"].append(loss.item())
            self.losses["val loss"]["adoption"].append(per_loss.item())
            self.losses["val loss"]["reconstruction"].append(rec_loss.item())
            self.losses["val loss"]["adversarial"].append(adv_loss.item())
            self.losses["val loss"]["segmentation"].append(seg_loss.item())

        # Log averages to Tensorboard
        self.save_visual_explanation(image.cpu(),
                                     adoption_cf.cpu(),
                                     counterfactual.cpu(),
                                     reconstruction.cpu(),
                                     target.cpu(),
                                     prediction.cpu(),
                                     prediction_cf.argmax(1).cpu())

        # log loss weights
        self.log_loss_weights()

    @torch.no_grad()
    def save_visual_explanation(self, img, adoption, cf, reconstruction, target_mask, seg_org, seg_cf):
        img = visualize_mri(img)
        cf = visualize_mri(cf)
        reconstruction = visualize_mri(reconstruction)
        adoption = visualize_adaption(adoption)
        orig_map = visualize_seg(seg_org).permute(0, 3, 1, 2)
        cf_map = visualize_seg(seg_cf).permute(0, 3, 1, 2)
        target_map = visualize_seg(target_mask).permute(0, 3, 1, 2)

        for i, seq in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            visual = torch.stack([img[:, i], adoption[:, i], cf[:, i], reconstruction[:, i]], dim=1).flatten(0, 1)
            self.writer.add_image(f'image/{seq}',
                                  make_grid(visual, nrow=4, padding=2),
                                  global_step=self.epoch)


        visual = torch.stack([orig_map, cf_map, target_map], dim=1).flatten(0, 1)
        self.writer.add_image('image/maps',
                              make_grid(visual, nrow=3, padding=2),
                              global_step=self.epoch)

    def log_losses(self):
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

        avg_loss = sum(self.losses["val loss"]["loss"]) / len(self.losses["val loss"]["loss"])
        self.writer.add_scalar("val/loss", avg_loss, global_step=self.epoch)
        self.writer.add_scalar("val/adoption",
                               sum(self.losses["val loss"]["adoption"]) / len(self.losses["val loss"]["adoption"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/reconstruction",
                               sum(self.losses["val loss"]["reconstruction"]) / len(
                                   self.losses["val loss"]["reconstruction"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/adversarial",
                               sum(self.losses["val loss"]["adversarial"]) / len(self.losses["val loss"]["adversarial"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/segmentation",
                               sum(self.losses["val loss"]["segmentation"]) / len(self.losses["val loss"]["segmentation"]),
                               global_step=self.epoch)

    def log_loss_weights(self):
        self.writer.add_scalar("MultiTaskLoss/weight_0", self.loss_fn.log_vars[0], global_step=self.epoch)
        self.writer.add_scalar("MultiTaskLoss/weight_1", self.loss_fn.log_vars[1], global_step=self.epoch)
        self.writer.add_scalar("MultiTaskLoss/weight_2", self.loss_fn.log_vars[2], global_step=self.epoch)
        self.writer.add_scalar("MultiTaskLoss/weight_3", self.loss_fn.log_vars[3], global_step=self.epoch)
