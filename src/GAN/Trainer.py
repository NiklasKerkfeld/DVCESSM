import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from diff_cf.core import BaseDataset
from .CounterfactualGAN import CounterfactualGAN


class CounterfactualGANTrainer:
    def __init__(
            self,
            gan: CounterfactualGAN,
            train_dataset: BaseDataset,
            test_dataset: BaseDataset,
            lambda_adv: float = 1.0,
            lambda_seg: float = 10.0,
            lambda_perturbation: float = 0.01,
            batch_size: int = 32,
            num_workers: int = 4,
            device: torch.device = torch.device("cpu"),
            log_dir: str = "./runs/GAN",
            name: str = "GAN",
    ):
        self.lambda_adv = lambda_adv
        self.lambda_seg = lambda_seg
        self.lambda_perturbation = lambda_perturbation
        self.device = device
        self.log_dir = log_dir
        self.name = name

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           pin_memory=self.device.type == "cuda",
                                           persistent_workers=num_workers > 1,
                                           drop_last=True)

        self.test_dataloader = DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=self.device.type == "cuda",
                                          persistent_workers=num_workers > 1,
                                          drop_last=False)

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

    @staticmethod
    def _colorize_masks(masks: torch.Tensor) -> torch.Tensor:
        """Convert [B, H, W] class masks to [B, 3, H, W] RGB images."""
        palette = torch.tensor([
            [0., 0., 0.],  # Class 0
            [1., 0., 0.],  # Class 1
            [0., 1., 0.],  # Class 2
            [0., 0., 1.],  # Class 3
            [0., 1., 1.],  # Class 4
            [1., 0., 1.],  # Class 5
            [1., 1., 0.],  # Class 6
            [.5, 0., 0.],  # Class 7
            [0., .5, 0.],  # Class 8
            [0., 0., .5],  # Class 9
            [0., .5, .5],  # Class 10
            [.5, 0., .5],  # Class 11
            [.5, .5, 0.],  # Class 12
            [.25, 0., 0.],  # Class 13
            [0., .25, 0.],  # Class 14
            [0., 0., .25],  # Class 15
            [0., 1., .5],  # Class 16
            [1., 0., .5],  # Class 17
            [1., .5, 0.],  # Class 18
            [0., .5, 1.],  # Class 19
            [.25, 0., 1.],  # Class 20
            [.25, 1., 0.],  # Class 21
            [0., 1., .25],  # Class 22
            [1., 0., .25],  # Class 23
            [1., .25, 0.],  # Class 24
            [0., .25, 1.],  # Class 25
            [.25, 0., 1.],  # Class 26
            [.25, 1., 0.]  # Class 27
        ], dtype=torch.float32,
            device=masks.device)

        masks = masks.long()
        colored = palette[masks.remainder(len(palette))]
        return colored.permute(0, 3, 1, 2)

    @staticmethod
    def _get_examples(
            data_loader: DataLoader,
            number_of_examples: int = 5,
    ):
        """Collect a fixed set of examples, even if the batch size is below 5."""
        images = []
        targets = []
        collected = 0

        for x, y in data_loader:
            remaining = number_of_examples - collected
            number_to_take = min(remaining, len(x))

            images.append(x[:number_to_take])
            targets.append(y[:number_to_take])
            collected += number_to_take

            if collected >= number_of_examples:
                break

        if collected == 0:
            raise ValueError("The test data loader is empty.")

        return torch.cat(images, dim=0), torch.cat(targets, dim=0)

    def _log_counterfactual_examples(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            step: int,
    ):
        x = x.to(self.device)
        y = y.to(self.device).long()

        self.gan.eval()

        # This must run with gradients because the GAN computes its
        # gradient prior from the target model.
        with torch.enable_grad():
            adoption = self.gan(x, y)

        counterfactual = (x + adoption).detach()

        with torch.no_grad():
            segmentation_logits = self.gan.target_model(counterfactual)
            segmentation_prediction = segmentation_logits.argmax(dim=1)

        self.writer.add_images(
            "examples/counterfactual",
            self.train_dataset.denormalize(counterfactual).cpu() / 256,
            step,
        )
        self.writer.add_images(
            "examples/counterfactual_prediction",
            self._colorize_masks(segmentation_prediction).cpu(),
            step,
        )

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
        adv_pred = self.gan.discriminator(x + adoption)
        seg_pred = self.gan.target_model(x + adoption)

        gen_adv_loss = self.adversarial_loss(
            adv_pred,
            torch.ones_like(adv_pred),
        )

        # Untargeted: encourage a segmentation different from y.
        gen_seg_loss = self.segmentation_loss(seg_pred, y)
        gen_adp_loss = torch.mean(adoption ** 2)

        gen_loss = (
                self.lambda_adv * gen_adv_loss +
                self.lambda_seg * gen_seg_loss +
                self.lambda_perturbation * gen_adp_loss
        )

        gen_loss.backward()
        self.generator_optimizer.step()

        return adv_loss.item(), gen_loss.item(), gen_adv_loss.item(), gen_seg_loss.item(), gen_adp_loss.item()

    @torch.no_grad()
    def eval_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device).long()

        with torch.enable_grad():
            adoption = self.gan(x, y).detach()

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

        real_acc = (real_pred >= 0).float().mean()
        fake_acc = (fake_pred < 0).float().mean()
        adv_acc = 0.5 * (real_acc + fake_acc)

        # ---------------------------------------------------------
        # Train generator
        # ---------------------------------------------------------
        # Recompute after the discriminator update.
        adv_pred = self.gan.discriminator(x + adoption)
        seg_pred = self.gan.target_model(x + adoption)

        gen_adv_loss = self.adversarial_loss(
            adv_pred,
            torch.ones_like(adv_pred),
        )

        # Untargeted: encourage a segmentation different from y.
        gen_seg_loss = self.segmentation_loss(seg_pred, y)
        gen_adp_loss = torch.mean(adoption ** 2)

        gen_loss = (
                self.lambda_adv * gen_adv_loss +
                self.lambda_seg * gen_seg_loss +
                self.lambda_perturbation * gen_adp_loss
        )

        return adv_loss.item(), gen_loss.item(), gen_adv_loss.item(), gen_seg_loss.item(), gen_adp_loss.item(), adv_acc.item()

    def train(
            self,
            epochs: int = 1,
    ):
        example_x, example_y = self._get_examples(
            self.test_dataloader,
            number_of_examples=5,
        )

        self.writer.add_images(
            "examples/original_images",
            self.test_dataset.denormalize(example_x) / 256,
            0,
        )
        self.writer.add_images(
            "examples/target",
            self._colorize_masks(example_y),
            0,
        )

        pred = torch.argmax(self.gan.target_model(example_x.to(self.device)), dim=1)
        self.writer.add_images(
            "examples/original_prediction",
            self._colorize_masks(pred),
            0,
        )

        for e in trange(epochs):
            # training
            self.gan.train()
            self.gan.target_model.eval()
            adv_loss_lst, gen_loss_lst, gen_adv_loss_lst, gen_seg_loss_lst, gen_apt_loss_lst = [], [], [], [], []
            for x, y in tqdm(self.train_dataloader, desc=f"training epoch {e}"):
                adv_loss, gen_loss, gen_adv_loss, gen_seg_loss, gen_apt_loss = self.train_step(x, y)
                adv_loss_lst.append(adv_loss)
                gen_loss_lst.append(gen_loss)
                gen_adv_loss_lst.append(gen_adv_loss)
                gen_seg_loss_lst.append(gen_seg_loss)
                gen_apt_loss_lst.append(gen_apt_loss)

            self.writer.add_scalar('train/adv_loss', sum(adv_loss_lst) / len(adv_loss_lst), e)
            self.writer.add_scalar('train/gen_loss', sum(gen_loss_lst) / len(gen_loss_lst), e)
            self.writer.add_scalar('train/gen_adverarial_loss',
                                   sum(gen_adv_loss_lst) / len(gen_adv_loss_lst), e)
            self.writer.add_scalar('train/gen_segmentation_loss',
                                   sum(gen_seg_loss_lst) / len(gen_seg_loss_lst), e)
            self.writer.add_scalar('train/gen_adaption_loss',
                                   sum(gen_apt_loss_lst) / len(gen_apt_loss_lst), e)

            # validation
            self.gan.eval()
            adv_loss_lst, adv_acc_lst, gen_loss_lst, gen_adv_loss_lst, gen_seg_loss_lst, gen_apt_loss_lst = [], [], [], [], [], []
            for x, y in tqdm(self.test_dataloader, desc=f"testing epoch {e}"):
                adv_loss, gen_loss, gen_adv_loss, gen_seg_loss, gen_apt_loss, adv_acc = self.eval_step(
                    x, y)
                adv_loss_lst.append(adv_loss)
                gen_loss_lst.append(gen_loss)
                gen_adv_loss_lst.append(gen_adv_loss)
                gen_seg_loss_lst.append(gen_seg_loss)
                gen_apt_loss_lst.append(gen_apt_loss)
                adv_acc_lst.append(adv_acc)

            self.writer.add_scalar('eval/adv_loss', sum(adv_loss_lst) / len(adv_loss_lst), e)
            self.writer.add_scalar('eval/gen_loss', sum(gen_loss_lst) / len(gen_loss_lst), e)
            self.writer.add_scalar('eval/gen_adverarial_loss',
                                   sum(gen_adv_loss_lst) / len(gen_adv_loss_lst), e)
            self.writer.add_scalar('eval/gen_segmentation_loss',
                                   sum(gen_seg_loss_lst) / len(gen_seg_loss_lst), e)
            self.writer.add_scalar('eval/gen_adaption_loss',
                                   sum(gen_apt_loss_lst) / len(gen_apt_loss_lst), e)
            self.writer.add_scalar('eval/adv_acc', sum(adv_acc_lst) / len(adv_acc_lst), e)

            self._log_counterfactual_examples(
                example_x,
                example_y,
                step=e + 1,
            )
            self.writer.flush()
