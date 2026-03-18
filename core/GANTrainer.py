import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm


class GANTrainer:
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
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
        self.adversarial = discriminator
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
                                   torch.tensor(self.losses["generation loss"]["discriminator"])),
                               global_step=self.epoch)
        self.writer.add_scalar("generation losses/segmentation",
                               torch.mean(
                                   torch.tensor(self.losses["generation loss"]["segmentation"])),
                               global_step=self.epoch)
        self.writer.add_scalar("discriminator losses/loss",
                               torch.mean(torch.tensor(self.losses["discriminator loss"])),
                               global_step=self.epoch)

    def train_generator(self, image: torch.Tensor, target: torch.Tensor,
                        optimizer: AdamW) -> torch.Tensor:
        adoption = self.generator(image, target)
        adpt_loss = torch.mean(torch.abs(adoption))
        counterfactual = image + adoption

        adv_pred = self.adversarial(counterfactual)
        adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(image.shape[0], device=self.device))

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
        d_loss_real = self.dis_loss_fn(real_pred, torch.zeros(image.size(0)).to(self.device))

        fake_pred = self.adversarial(counterfactual.detach())
        d_loss_fake = self.dis_loss_fn(fake_pred, torch.ones(image.size(0)).to(self.device))

        loss = d_loss_real + d_loss_fake

        loss.backward()
        optimizer.step()

        self.adversarial.zero_grad()

        self.losses["adversarial loss"].append(loss.detach().cpu().item())

    @torch.no_grad()
    def evaluation(self, valid_dataloader: DataLoader):
        self.generator.eval()
        self.discriminator.eval()  # Paper refers to D(x) and Da(x) [cite: 38]
        self.segmentation.eval()  # This is H(.) in the paper [cite: 43]

        val_losses = {
            "total": [],
            "perturbation": [],
            "explanation": [],
            "adversarial": []
        }

        for image, target in tqdm(valid_dataloader, desc='validation'):
            image, target = image.to(self.device), target.to(self.device)

            adoption = self.generator(image, target)
            adpt_loss = torch.mean(torch.abs(adoption))

            counterfactual = image + adoption

            adv_pred = self.discriminator(counterfactual)
            adv_loss = self.dis_loss_fn(adv_pred, torch.zeros(counterfactual.shape[0], device=self.device))

            seg_pred = self.segmentation(counterfactual)
            seg_loss = self.seg_loss_fn(seg_pred, target)

            total_loss = self.lambda_adpt * adpt_loss + self.lambda_adv * adv_loss + self.lambda_seg * seg_loss

            # Store results
            val_losses["total"].append(total_loss.item())
            val_losses["perturbation"].append(adpt_loss.item())
            val_losses["adversarial"].append(adv_loss.item())
            val_losses["segmentation"].append(seg_loss.item())

        # Log averages to Tensorboard
        avg_total = sum(val_losses["total"]) / len(val_losses["total"])
        self.writer.add_scalar("val/total_loss", avg_total, global_step=self.epoch)
        self.writer.add_scalar("val/perturbation",
                               sum(val_losses["perturbation"]) / len(val_losses["perturbation"]),
                               global_step=self.epoch)
        self.writer.add_scalar("val/explanation",
                               sum(val_losses["explanation"]) / len(val_losses["explanation"]),
                               global_step=self.epoch)

        self.save_visual_explanation(image[0], adoption[0], counterfactual[0], target[0])

    def save_visual_explanation(self, img, res, cf, target):
        """
        Implements the visualization style from Figure 1 & 3[cite: 46, 137].
        Red markings = negative values (erased), Blue = positive (added)[cite: 158].
        """
        # Simple logic to visualize residual as described in paper [cite: 158]
        res_vis = torch.zeros_like(img)
        res_vis[0] = torch.clamp(res, min=0)  # Blue channel (positive)
        res_vis[2] = torch.abs(torch.clamp(res, max=0))  # Red channel (negative)

        grid = make_grid([img, res_vis, cf])
        self.writer.add_image('val/explanation_sample', grid, global_step=self.epoch)


