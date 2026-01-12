import os
from dotenv import load_dotenv

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

from dataset import DiscriminatorDataset
from models import Generator, Classifier, DiscriminatorLoss
from utils import get_one_of_each, random_target, process


load_dotenv()

N_CLASSES = int(os.getenv('N_CLASSES'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
ITERATIONS = int(os.getenv('ITERATIONS'))
ALPHA = float(os.getenv('ALPHA'))
BETA = float(os.getenv('BETA'))


class CounterFactualGenerator:
    def __init__(self,
                 classifier,
                 dataset,
                 batch_size:int = BATCH_SIZE,
                 device: torch.device = torch.device("cpu")):

        self.classifier = classifier
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        # the classifier is never trained
        for param in classifier.parameters():
            param.requires_grad = False

        self.generator = Generator(1)
        self.generator_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)

        self.discriminator = Classifier(self.classifier.input_size, self.classifier.output_size)
        self.discriminator_dataset = DiscriminatorDataset([self.dataset[i][0] for i in range(len(self.dataset))])
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = DiscriminatorLoss()


        self.samples = get_one_of_each(self.dataset)

    def train(self, interations: int = ITERATIONS):
        print(f"using device: {self.device}")

        for i in range(1, interations + 1):
            print(f"Iteration {i}:")
            alpha = 0.0 if i == 1 else ALPHA
            self._train_generator(alpha=alpha, beta=BETA)
            self._plot_samples(title=f"Iteration {i}")
            self._train_discriminator()

    def _train_generator(self, alpha: float, beta: float):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.classifier.to(self.device)

        # activate training of generator
        for param in self.generator.parameters():
            param.requires_grad = True

        # deactivate training of discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False


        bar = tqdm(self.generator_dataloader, "training", total=len(self.generator_dataloader))
        for images, labels in bar:
            images, labels = images.to(self.device), labels.to(self.device)
            targets = random_target(labels, N_CLASSES)

            images = process(images)

            self.generator_optimizer.zero_grad()
            changes = self.generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
            counterfactuals = images + changes

            pred_class = self.classifier(counterfactuals)
            pred_disc = self.discriminator(counterfactuals)[:, 0]

            regularization = beta * torch.mean(torch.abs(changes))
            loss_class = self.classifier_criterion(pred_class, targets)
            loss_disc = alpha * self.discriminator_criterion(pred_disc, torch.zeros_like(labels))
            loss = loss_class + loss_disc + regularization

            loss.backward()
            self.generator_optimizer.step()

            pred = torch.argmax(pred_class, dim=1)
            like_label = (pred == labels).float().sum()
            like_target = (pred == targets).float().sum()
            label_acc = like_label / len(labels)
            target_acc = like_target / len(labels)

            bar.set_description(f"training loss: {loss.item():.4f} "
                                f"(class: {loss_class.item():.4f} "
                                f"disc: {loss_disc.item():.4f} "
                                f"regularization: {regularization.item():.4f} "
                                f"label acc {label_acc.item():.4f} "
                                f"target acc {target_acc.item():.4f})")

    def _train_discriminator(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # deactivate training of generator
        for param in self.generator.parameters():
            param.requires_grad = False

        # activate training of discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = True


        # generate faces
        new_fakes = []
        bar = tqdm(self.generator_dataloader, "generating", total=len(self.generator_dataloader))
        for images, labels in bar:
            images, labels = images.to(self.device), labels.to(self.device)
            targets = random_target(labels, N_CLASSES)

            with torch.no_grad():
                change = self.generator(images, F.one_hot(targets, num_classes=N_CLASSES).float())
                fakes = images + change
                new_fakes.extend([x.cpu() for x in fakes])

        self.discriminator_dataset.add_fakes(new_fakes)

        dataloader = DataLoader(self.discriminator_dataset, batch_size=BATCH_SIZE, shuffle=True)

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
            correct = (pred == labels).float().sum()
            accuracy = correct / len(labels)

            bar.set_description(f"training loss: {loss.item():.4f} "
                                f"accuracy {accuracy.item():.4f})")