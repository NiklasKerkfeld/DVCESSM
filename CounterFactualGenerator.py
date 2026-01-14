import os
from typing import Literal

import torchvision
from dotenv import load_dotenv

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dataset import DiscriminatorDataset
from models import Generator, Classifier, DiscriminatorLoss, ChangeRegularizationLoss
from utils import get_one_of_each, random_target, process, func

load_dotenv()

N_CLASSES = int(os.getenv('N_CLASSES'))
IN_CHANNELS = int(os.getenv('IN_CHANNELS'))
IN_HEIGHT = int(os.getenv('IN_HEIGHT'))
IN_WIDTH = int(os.getenv('IN_WIDTH'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
ITERATIONS = int(os.getenv('ITERATIONS'))
ALPHA = float(os.getenv('ALPHA'))
BETA = float(os.getenv('BETA'))
LR_GEN = float(os.getenv('LR_GEN'))
LR_DISC = float(os.getenv('LR_DISC'))

STD_DISC = float(os.getenv('STD_DISC'))
P_DISC = float(os.getenv('P_DISC'))
B_DISC = float(os.getenv('B_DISC'))
B_H_MAX_DISC = int(os.getenv('B_H_MAX_DISC'))
B_W_MAX_DICS = int(os.getenv('B_W_MAX_DICS'))


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
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LR_DISC)

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = DiscriminatorLoss()
        self.regularization_criterion = ChangeRegularizationLoss()

        self.samples = get_one_of_each(self.dataset)
        self.writer = SummaryWriter(f'runs/{name}')
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
        like_label = 0
        like_target = 0
        total = 0

        bar = tqdm(self.generator_dataloader, "training", total=len(self.generator_dataloader))
        for images, labels in bar:
            images, labels = images.to(self.device), labels.to(self.device)
            targets = random_target(labels, N_CLASSES)

            self.generator_optimizer.zero_grad()
            changes = self.generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
            counterfactuals = images + changes

            pred_class = self.classifier(counterfactuals)
            pred_disc = self.discriminator(counterfactuals)[:, 0]

            regularization = beta * self.regularization_criterion(images, changes)
            loss_class = self.classifier_criterion(pred_class, targets)
            loss_disc = alpha * self.discriminator_criterion(pred_disc, torch.zeros_like(labels))
            loss = loss_class + loss_disc + regularization

            loss.backward()
            self.generator_optimizer.step()

            pred = torch.argmax(pred_class, dim=1)
            like_label += (pred == labels).float().sum().detach().cpu().item()
            like_target += (pred == targets).float().sum().detach().cpu().item()
            total += len(images)

            losses.append(loss.detach().cpu().item())
            discriminator_losses.append(loss_disc.detach().cpu().item())
            classifier_losses.append(loss_class.detach().cpu().item())
            regularizations.append(regularization.detach().cpu().item())

            bar.set_description(f"training loss: {loss.item():.4f} "
                                f"(class: {loss_class.item():.4f} "
                                f"disc: {loss_disc.item():.4f} "
                                f"regularization: {regularization.item():.4f})")

        self.writer.add_scalar('general/generator_loss',
                               torch.mean(torch.tensor(losses)).item(),
                               self.interation)
        self.writer.add_scalar('general/generator_classifier_loss',
                               torch.mean(torch.tensor(classifier_losses)).item(),
                               self.interation)
        self.writer.add_scalar('general/generator_discriminator_loss',
                               torch.mean(torch.tensor(discriminator_losses)).item(),
                               self.interation)
        self.writer.add_scalar('general/generator_regularization',
                               torch.mean(torch.tensor(regularizations)).item(),
                               self.interation)
        self.writer.add_scalar('general/generator_like_target',
                               like_target / total,
                               self.interation)
        self.writer.add_scalar('general/generator_like_label',
                               like_label / total,
                               self.interation)

    def _train_discriminator(self):
        self._set_mode("discriminator")

        # generate faces
        new_fakes = torch.zeros(len(self.generator_dataloader.dataset), IN_CHANNELS, IN_HEIGHT, IN_HEIGHT)
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
        label_positive = 0

        bar = tqdm(dataloader, "training", total=len(dataloader))
        for images, labels in bar:
            images, labels = images.to(self.device), labels.to(self.device)

            images = process(images, std=0.1, p=0.1)

            pred = self.discriminator(images)[:, 0]

            self.discriminator_optimizer.zero_grad()
            loss = self.discriminator_criterion(pred, labels)
            loss.backward()
            self.discriminator_optimizer.step()

            pred = (pred > 0.5).int()
            correct += (pred == labels).float().sum().detach().cpu().item()
            total += len(images)
            positive += pred.float().sum().detach().cpu().item()
            label_positive += labels.float().sum().detach().cpu().item()
            losses.append(loss.detach().cpu().item())

            bar.set_description(f"training loss: {loss.item():.4f}")

        self.writer.add_scalar('general/discriminator_loss',
                               torch.mean(torch.tensor(losses)).item(),
                               self.interation)
        self.writer.add_scalar('general/discriminator_accuracy',
                               correct / total,
                               self.interation)
        self.writer.add_scalar('general/discriminator_positive_prediction_rate',
                               positive / total,
                               self.interation)
        self.writer.add_scalar('general/discriminator_positive_label_rate',
                               label_positive / total,
                               self.interation)

    def _plot_samples(self):
        samples = self.samples.to(self.device)
        self.generator.eval()

        counter_factuals = []
        visualizations = []
        for y in range(10):
            targets = torch.full((len(samples),), y)
            targets = targets.to(self.device)

            with torch.no_grad():
                changes = self.generator(process(samples),
                                         F.one_hot(targets, num_classes=10).float())

            # counter factuals
            counter_factuals.append(samples + changes)

            # visualization
            rgb_images = samples.repeat(1, 3, 1, 1)
            rgb_images[:, 0][(changes < -0.01).squeeze(1)] = torch.abs(changes)[changes < -0.01]
            rgb_images[:, 2][(changes > 0.01).squeeze(1)] = torch.abs(changes)[changes > 0.01]
            visualizations.append(rgb_images)

        img_grid = torchvision.utils.make_grid(torch.concatenate(counter_factuals).cpu(), nrow=10)
        self.writer.add_image('general/counterfactuals', img_grid, global_step=self.interation)
        img_grid = torchvision.utils.make_grid(torch.concatenate(visualizations).cpu(), nrow=10)
        self.writer.add_image('general/visualization', img_grid, global_step=self.interation)
