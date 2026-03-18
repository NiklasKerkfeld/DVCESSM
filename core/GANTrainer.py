import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import visualize_mri, visualize_seg


class GANTrainer:
    def __init__(self,
                 generator: nn.Module,
                 adversarial: nn.Module,
                 segmentation: nn.Module,
                 device: torch.device,
                 name='GANModel',
                 lr_generator: float = 0.001,
                 lr_adversarial: float = 0.001,
                 lambda_adv: float = 1,
                 lambda_seg: float = 10,
                 lambda_adpt: float = 10,
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

        self.epoch = 0

        self.dis_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = nn.CrossEntropyLoss()

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

    def train_epoch(self, train_dataloader: DataLoader, generator_optimizer,
                    adversarial_optimizer):
        self.generator.train()
        self.adversarial.train()
        self.segmentation.eval()

        self.losses = {
            "generation loss": {
                "loss": [],
                "adoption": [],
                "adversarial": [],
                "segmentation": []
            },
            "adversarial loss": []
        }
        for image, target in tqdm(train_dataloader, total=len(train_dataloader),
                                  desc='train generator'):
            image, target = image.to(self.device), target.to(self.device)

            # train generator
            counterfactual = self.train_generator(image, target, generator_optimizer)
            self.train_discriminator(image, counterfactual, adversarial_optimizer)

        self.writer.add_scalar("generation losses/loss",
                               torch.mean(torch.tensor(self.losses["generation loss"]["loss"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation losses/adoption",
                               torch.mean(torch.tensor(self.losses["generation loss"]["adoption"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation losses/discriminator",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["adversarial"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation losses/segmentation",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["segmentation"])),
                               global_step=self.epoch)
        self.writer.add_scalar("adversarial losses/loss",
                               torch.mean(torch.tensor(self.losses["adversarial loss"])),
                               global_step=self.epoch)

    def train_generator(self, image: torch.Tensor, target: torch.Tensor,
                        optimizer: AdamW) -> torch.Tensor:
        adoption = self.generator(image, target)
        adpt_loss = torch.mean(torch.abs(adoption))
        counterfactual = image + adoption

        adv_pred = self.adversarial(counterfactual)
        adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(image.shape[0], dtype=torch.long, device=self.device))

        seg_pred = self.segmentation(counterfactual)
        seg_loss = self.seg_loss_fn(seg_pred, target)

        loss = self.lambda_adpt * adpt_loss + self.lambda_adv * adv_loss + self.lambda_seg * seg_loss
        loss.backward()
        optimizer.step()
        self.generator.zero_grad()

        self.losses["generation loss"]["loss"].append(loss.detach().cpu().item())
        self.losses["generation loss"]["adoption"].append(adpt_loss.detach().cpu().item())
        self.losses["generation loss"]["adversarial"].append(adv_loss.detach().cpu().item())
        self.losses["generation loss"]["segmentation"].append(seg_loss.detach().cpu().item())

        return counterfactual

    def train_discriminator(self, image: torch.Tensor, counterfactual: torch.Tensor, optimizer):
        self.adversarial.train()

        real_pred = self.adversarial(image)
        d_loss_real = self.dis_loss_fn(real_pred, torch.zeros(image.size(0), dtype=torch.long, device=self.device))

        fake_pred = self.adversarial(counterfactual.detach())
        d_loss_fake = self.dis_loss_fn(fake_pred, torch.ones(image.size(0), dtype=torch.long, device=self.device))

        loss = d_loss_real + d_loss_fake

        loss.backward()
        optimizer.step()

        self.adversarial.zero_grad()

        self.losses["adversarial loss"].append(loss.detach().cpu().item())

    @torch.no_grad()
    def evaluation(self, valid_dataloader: DataLoader):
        self.generator.eval()
        self.adversarial.eval()  # Paper refers to D(x) and Da(x) [cite: 38]
        self.segmentation.eval()  # This is H(.) in the paper [cite: 43]

        val_losses = {
            "loss": [],
            "adoption": [],
            "adversarial": [],
            "segmentation": []
        }

        for image, target in tqdm(valid_dataloader, desc='validation'):
            image, target = image.to(self.device), target.to(self.device)

            adoption = self.generator(image, target)
            adpt_loss = torch.mean(torch.abs(adoption))

            counterfactual = image + adoption

            adv_pred = self.adversarial(counterfactual)
            adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(counterfactual.shape[0], dtype=torch.long, device=self.device))

            seg_pred = self.segmentation(counterfactual)
            seg_loss = self.seg_loss_fn(seg_pred, target)

            total_loss = self.lambda_adpt * adpt_loss + self.lambda_adv * adv_loss + self.lambda_seg * seg_loss

            # Store results
            val_losses["loss"].append(total_loss.item())
            val_losses["adoption"].append(adpt_loss.item())
            val_losses["adversarial"].append(adv_loss.item())
            val_losses["segmentation"].append(seg_loss.item())

        # Log averages to Tensorboard
        avg_loss = sum(val_losses["loss"]) / len(val_losses["loss"])
        self.writer.add_scalar("val/loss", avg_loss, global_step=self.epoch)
        self.writer.add_scalar("val/adoption",
                               sum(val_losses["adoption"]) / len(val_losses["adoption"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/adversarial",
                               sum(val_losses["adversarial"]) / len(val_losses["adversarial"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/segmentation",
                               sum(val_losses["segmentation"]) / len(val_losses["segmentation"]),
                               global_step=self.epoch)

        seg_org = self.segmentation(image)
        self.save_visual_explanation(image[0].cpu(), adoption[0].cpu(), counterfactual[0].cpu(), target[0].cpu(), seg_pred[0].argmax(1).cpu(), seg_org[0].argmax(1).cpu())

    @torch.no_grad()
    def save_visual_explanation(self, img, adoption, cf, target_mask, seg_org, seg_cf):
        """
        img, adoption, cf shape: (4, H, W) - [T1, T1c, T2, FLAIR]
        target_mask shape: (1, H, W) or (C, H, W)
        """
        img = visualize_mri(img[None])[0]
        cf = visualize_mri(cf[None])[0]
        print(f"{img.shape=}") # img.shape=torch.Size([4, 3, 128, 128])
        print(f"{cf.shape=}") # cf.shape=torch.Size([4, 3, 128, 128])

        all_rows = []
        for i in range(4):
            residual_mod = adoption[i]
            heatmap = torch.zeros((3, adoption.shape[1], adoption.shape[2]))
            heatmap[0] = torch.clamp(-residual_mod, min=0)
            heatmap[2] = torch.clamp(residual_mod, min=0)
            heatmap = heatmap / (heatmap.max() + 1e-5)

            row = torch.stack([img[i], heatmap, cf[i]])
            all_rows.append(make_grid(row, nrow=3, padding=2))

        final_grid = torch.cat(all_rows, dim=1)
        self.writer.add_image('val/visualization', final_grid, global_step=self.epoch)

        print(f"{target_mask.shape=}")
        print(f"{seg_org.shape=}")
        print(f"{seg_cf.shape=}")
        target_map = visualize_seg(target_mask[None])
        orig_map = visualize_seg(seg_org[None])
        cf_map = visualize_seg(seg_cf[None])
        self.writer.add_image('val/maps', make_grid([target_map, orig_map, cf_map], nrow=3, padding=2), global_step=self.epoch)


