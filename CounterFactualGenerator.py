from typing import Literal

import torchvision

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from architecture.Classifier import Classifier
from architecture.Losses import DiscriminatorLoss, ChangeRegularizationLoss
from Datasets import DiscriminatorDataset
from architecture.Generator import Generator
from Parameter import BATCH_SIZE, LR_GEN, STD_DISC, P_DISC, WD_DISC, LR_DISC, LABEL_SMOOTHING, \
    L2_REG, L1_REG, ITERATIONS, ALPHA, BETA, N_CLASSES, IN_CHANNELS, IN_WIDTH, IN_HEIGHT
from utils import get_one_of_each, random_target


class CounterFactualGenerator:
    def __init__(self,
                 classifier,
                 dataset,
                 batch_size: int = BATCH_SIZE,
                 device: torch.device = torch.device("cpu"),
                 name: str = "CounterFactualGenerator"):

        self.classifier = classifier
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.interation = 0

        # the classifier is never trained
        self.classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False

        self.generator = Generator(1)
        self.generator_dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                               shuffle=True)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=LR_GEN)

        self.discriminator = Classifier(num_classes=1)
        self.discriminator_dataset = DiscriminatorDataset(
            torch.concatenate([self.dataset[i][0] for i in range(len(self.dataset))]),
            std=STD_DISC,
            p=P_DISC)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=LR_DISC,
                                                        weight_decay=WD_DISC)

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = DiscriminatorLoss(label_smoothing=LABEL_SMOOTHING)
        self.regularization_criterion = ChangeRegularizationLoss(alpha=L1_REG, beta=L2_REG)

        self.samples = get_one_of_each(self.dataset)
        self.writer = SummaryWriter(f'runs/GAN/{name}')
        img_grid = torchvision.utils.make_grid(torch.tensor(self.samples).cpu(), nrow=10)
        self.writer.add_image('general/samples', img_grid, global_step=self.interation)

    def train(self, interations: int = ITERATIONS):
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.classifier = self.classifier.to(self.device)

        print(f"using device: {self.device}")

        for i in range(1, interations + 1):
            self.interation = i
            print(f"Iteration {i}:")

            alpha = 0.0 if i == 1 else ALPHA
            self._train_generator(alpha=alpha, beta=BETA)
            self._plot_samples()
            self._train_discriminator()

    def _set_mode(self, mode: Literal["generator", "discriminator"]):
        # activate training of generator
        for param in self.generator.parameters():
            param.requires_grad = mode == "generator"

        # deactivate training of discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = mode == "discriminator"

        if mode == "generator":
            self.generator.train()
            self.discriminator.eval()
        else:
            self.generator.eval()
            self.discriminator.train()

    def _train_generator(self, alpha: float, beta: float):
        self._set_mode("generator")

        losses, discriminator_losses, classifier_losses, regularizations = [], [], [], []
        correct = 0
        like_label = 0
        like_target = 0
        total = 0

        bar = tqdm(self.generator_dataloader, "training", total=len(self.generator_dataloader))
        for images, labels in bar:
            images, labels = images.to(self.device), labels.to(self.device)
            targets = random_target(labels, N_CLASSES)

            self.generator_optimizer.zero_grad()
            changes = self.generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
            counterfactuals = torch.clamp(images + changes, 0, 1)

            pred_class = self.classifier(counterfactuals)
            pred_disc = self.discriminator(counterfactuals).squeeze(1)

            regularization = beta * self.regularization_criterion(changes)
            loss_class = self.classifier_criterion(pred_class, targets)
            loss_disc = alpha * self.discriminator_criterion(pred_disc, torch.zeros_like(labels))
            loss = loss_class + loss_disc + regularization

            for name, param in self.discriminator.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Grads/Discriminator/{name}', param.grad, self.interation)

            loss.backward()
            self.generator_optimizer.step()

            pred_class = torch.argmax(pred_class, dim=1)
            pred_disc = (torch.sigmoid(pred_disc) > 0.5).int()
            correct += (pred_disc == 1.0).float().sum().detach().cpu().item()
            like_label += (pred_class == labels).float().sum().detach().cpu().item()
            like_target += (pred_class == targets).float().sum().detach().cpu().item()
            total += len(images)

            losses.append(loss.detach().cpu().item())
            discriminator_losses.append(loss_disc.detach().cpu().item())
            classifier_losses.append(loss_class.detach().cpu().item())
            regularizations.append(regularization.detach().cpu().item())

            bar.set_description(f"training loss: {loss.item():.4f} "
                                f"(class: {loss_class.item():.4f} "
                                f"disc: {loss_disc.item():.4f} "
                                f"regularization: {regularization.item():.4f})")

        self.writer.add_scalar("generator/loss_classificator", torch.mean(torch.tensor(classifier_losses)).item(), self.interation)
        self.writer.add_scalar("generator/loss_discriminator", torch.mean(torch.tensor(discriminator_losses)).item(), self.interation)
        self.writer.add_scalar("generator/loss_regularization", torch.mean(torch.tensor(regularizations)).item(), self.interation)

        self.writer.add_scalar('generator/loss',
                               torch.mean(torch.tensor(losses)).item(),
                               self.interation)
        self.writer.add_scalar('generator/like_target',
                               like_target / total,
                               self.interation)
        self.writer.add_scalar('generator/like_label',
                               like_label / total,
                               self.interation)
        self.writer.add_scalar('generator/discriminator_accuracy',
                               correct / total,
                               self.interation)

    def _train_discriminator(self):
        self._set_mode("discriminator")

        # generate faces
        new_fakes = torch.zeros(len(self.generator_dataloader.dataset), IN_CHANNELS, IN_WIDTH,
                                IN_HEIGHT)
        bar = tqdm(self.generator_dataloader, "generating", total=len(self.generator_dataloader))
        for i, (images, labels) in enumerate(bar):
            images, labels = images.to(self.device), labels.to(self.device)
            targets = random_target(labels, N_CLASSES)

            with torch.no_grad():
                change = self.generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
                fakes = images + change
                start_idx = i * BATCH_SIZE
                new_fakes[start_idx:start_idx + len(fakes)] = fakes

        self.discriminator_dataset.add_fakes(new_fakes)

        dataloader = DataLoader(self.discriminator_dataset, batch_size=BATCH_SIZE, shuffle=True)

        losses = []
        correct = 0
        total = 0
        positive = 0
        bar = tqdm(dataloader, "training", total=len(dataloader))
        for images, labels in bar:
            images, labels = images.to(self.device), labels.to(self.device)

            pred = self.discriminator(images)[:, 0]

            self.discriminator_optimizer.zero_grad()
            loss = self.discriminator_criterion(pred, labels)
            loss.backward()
            self.discriminator_optimizer.step()

            pred = (pred > 0.5).int()
            correct += (pred == labels).float().sum().detach().cpu().item()
            total += len(images)
            positive += pred.float().sum().detach().cpu().item()
            losses.append(loss.detach().cpu().item())

            bar.set_description(f"training loss: {loss.item():.4f}")

        self.writer.add_scalar('discriminator/loss',
                               torch.mean(torch.tensor(losses)).item(),
                               self.interation)
        self.writer.add_scalar('discriminator/accuracy',
                               correct / total,
                               self.interation)
        self.writer.add_scalar('discriminator/positive_prediction_rate',
                               positive / total,
                               self.interation)

    def _plot_samples(self):
        samples = self.samples.to(self.device)
        self.generator.eval()

        counter_factuals = torch.zeros(N_CLASSES, len(self.samples), IN_WIDTH, IN_HEIGHT)
        visualizations = torch.zeros(N_CLASSES, len(self.samples), 3, IN_WIDTH, IN_HEIGHT)
        for y in range(N_CLASSES):
            targets = torch.full((len(samples),), y)
            targets = targets.to(self.device)

            with torch.no_grad():
                changes = self.generator(samples, F.one_hot(targets, num_classes=10).float())

            changes = changes.cpu()

            # counter factuals
            counter_factuals[y] = (self.samples + changes).squeeze(1)

            # visualization
            visualizations[y] = self.samples.repeat(1, 3, 1, 1)

            red_mask = (changes < -0.01).squeeze(1)
            blue_mask = (changes > 0.01).squeeze(1)

            visualizations[y, :, 0][red_mask] = torch.abs(changes)[changes < -0.01]
            visualizations[y, :, 1][red_mask] = 0.0
            visualizations[y, :, 2][red_mask] = 0.0

            visualizations[y, :, 2][blue_mask] = torch.abs(changes)[changes > 0.01]
            visualizations[y, :, 0][blue_mask] = 0.0
            visualizations[y, :, 1][blue_mask] = 0.0

        counter_factuals[torch.arange(10), torch.arange(10)] = 0.0
        visualizations[torch.arange(10), torch.arange(10)] = 0.0

        img_grid = torchvision.utils.make_grid(counter_factuals.reshape(-1, 1, IN_WIDTH, IN_HEIGHT),
                                               nrow=10)
        self.writer.add_image('images/counterfactuals', img_grid, global_step=self.interation)

        img_grid = torchvision.utils.make_grid(visualizations.reshape(-1, 3, IN_WIDTH, IN_HEIGHT),
                                               nrow=10)
        self.writer.add_image('images/visualization', img_grid, global_step=self.interation)
