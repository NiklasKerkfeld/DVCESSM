import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm


from .CounterfactualGAN import CounterfactualGAN


class CounterfactualGANTrainer:
    def __init__(
        self,
        gan: CounterfactualGAN,
        lambda_adv: float = 1.0,
        lambda_seg: float = 1.0,
        lambda_perturbation: float = 0.01,
        log_dir: str="./runs/GAN",
        name: str="GAN",
    ):
        self.lambda_adv = lambda_adv
        self.lambda_seg = lambda_seg
        self.lambda_perturbation = lambda_perturbation
        self.log_dir = log_dir
        self.name = name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.gan = gan.to(self.device)
        self.generator_optimizer = AdamW(
            self.gan.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )
        self.discriminator_optimizer = AdamW(
            self.gan.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.segmentation_loss = nn.CrossEntropyLoss()

        self.gan.target_model.eval()
        for parameter in self.gan.target_model.parameters():
            parameter.requires_grad_(False)

        self.writer = SummaryWriter(f'{self.log_dir}/{self.name}')


    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device).long()

        # Generate the adoption for the discriminator step.
        adoption = self.gan(x, y)

        # ---------------------------------------------------------
        # Train discriminator
        # ---------------------------------------------------------
        for parameter in self.gan.discriminator.parameters():
            parameter.requires_grad_(True)
        self.discriminator_optimizer.zero_grad()

        real_pred = self.gan.discriminator(x)
        fake_pred = self.gan.discriminator((x + adoption).detach())

        real_loss = self.adversarial_loss(
            real_pred,
            torch.ones_like(real_pred),
        )
        fake_loss = self.adversarial_loss(
            fake_pred,
            torch.zeros_like(fake_pred),
        )

        adv_loss = 0.5 * (real_loss + fake_loss)
        adv_loss.backward()
        self.discriminator_optimizer.step()

        # ---------------------------------------------------------
        # Train generator
        # ---------------------------------------------------------
        for parameter in self.gan.discriminator.parameters():
            parameter.requires_grad_(False)

        self.generator_optimizer.zero_grad()

        # Recompute after the discriminator update.
        adoption = self.gan(x, y)
        adv_pred = self.gan.adversarial(x + adoption)
        seg_pred = self.gan.target_model(x + adoption)

        gen_adv_loss = self.adversarial_loss(
            adv_pred,
            torch.ones_like(adv_pred),
        )

        # Untargeted: encourage a segmentation different from y.
        gen_seg_loss = -self.segmentation_loss(seg_pred, y)
        gen_adp_loss = torch.mean(adoption ** 2)

        gen_loss = (
                self.lambda_adv * gen_adv_loss +
                self.lambda_seg * gen_seg_loss +
                self.lambda_perturbation * gen_adp_loss
        )

        gen_loss.backward()
        self.generator_optimizer.step()

        return adv_loss.item(), gen_loss.item()

    @torch.no_grad()
    def eval_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device).long()

        adoption = self.gan(x, y)

        # ---------------------------------------------------------
        # Train discriminator
        # ---------------------------------------------------------
        real_pred = self.gan.discriminator(x)
        fake_pred = self.gan.discriminator((x + adoption).detach())

        real_loss = self.adversarial_loss(
            real_pred,
            torch.ones_like(real_pred),
        )
        fake_loss = self.adversarial_loss(
            fake_pred,
            torch.zeros_like(fake_pred),
        )

        adv_loss = 0.5 * (real_loss + fake_loss)

        adv_pred = torch.cat([torch.argmax(real_pred), torch.argmax(fake_pred)], dim=0)
        adv_target = torch.zeros_like(adv_pred)
        adv_target[:len(real_pred)] = 1.0

        adv_acc = sum(adv_pred == adv_target) / len(adv_pred)

        # ---------------------------------------------------------
        # Train generator
        # ---------------------------------------------------------
        # Recompute after the discriminator update.
        adoption = self.gan(x, y)
        adv_pred = self.gan.adversarial(x + adoption)
        seg_pred = self.gan.target_model(x + adoption)

        gen_adv_loss = self.adversarial_loss(
            adv_pred,
            torch.ones_like(adv_pred),
        )

        # Untargeted: encourage a segmentation different from y.
        gen_seg_loss = -self.segmentation_loss(seg_pred, y)
        gen_adp_loss = torch.mean(adoption ** 2)

        gen_loss = (
                self.lambda_adv * gen_adv_loss +
                self.lambda_seg * gen_seg_loss +
                self.lambda_perturbation * gen_adp_loss
        )

        return adv_loss.item(), gen_loss.item(), adv_acc

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 1,
    ):
        for e in trange(epochs):
            adv_loss_lst, gen_loss_lst = [], []
            for x, y in tqdm(train_loader, desc=f"training epoch {e}"):
                adv_loss, gen_loss = self.train_step(x, y)
                adv_loss_lst.append(adv_loss)
                gen_loss_lst.append(gen_loss)

            self.writer.add_scalar('train/adv_loss', torch.mean(adv_loss_lst), e)
            self.writer.add_scalar('train/gen_loss', torch.mean(gen_loss_lst), e)

            self.gan.eval()
            adv_loss_lst, gen_loss_lst, adv_acc_lst = [], [], []
            for x, y in tqdm(test_loader, desc=f"testing epoch {e}"):
                adv_loss, gen_loss, adv_acc = self.eval_step(x, y)
                adv_loss_lst.append(adv_loss)
                gen_loss_lst.append(gen_loss)
                adv_acc_lst.append(adv_acc)

            self.writer.add_scalar('eval/adv_loss', torch.mean(adv_loss_lst), e)
            self.writer.add_scalar('eval/gen_loss', torch.mean(gen_loss_lst), e)
            self.writer.add_scalar('eval/adv_acc', torch.mean(adv_acc_lst), e)

            self.gan.train()