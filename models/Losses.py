from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class CounterfactualLoss(nn.Module):
    def __init__(self,
                 target: torch.Tensor,
                 border_width: int = 2,
                 use_mask: bool = True,
                 reduction: Literal['mean', 'sum'] = 'mean'):
        super(CounterfactualLoss, self).__init__()
        self.target = target
        self.border_width = border_width
        self.use_mask = use_mask
        self._set_boarders(target)
        self.reduction = torch.sum if reduction == 'sum' else torch.mean

    def forward(self, logits: torch.Tensor):
        loss = F.cross_entropy(logits, self.target, reduction='none', ignore_index=-1)

        if self.use_mask:
            mask = (logits.argmax(dim=1) != self.target).float()
            loss = loss * mask

        return self.reduction(loss)

    def _set_boarders(self, target: torch.Tensor):
        if self.border_width > 0:
            k_size = self.border_width + 1

            dilated = F.max_pool2d(target.float(), kernel_size=k_size, stride=1,
                                   padding=self.border_width // 2)

            eroded = -F.max_pool2d(-target.float(), kernel_size=k_size, stride=1,
                                   padding=self.border_width // 2)

            mask = (dilated != eroded)

            self.target[mask] = -1


import hydra
from omegaconf import DictConfig, OmegaConf
from core.Datasets.dataset import get_datasets
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    _, dataset = get_datasets(cfg)

    print(f"dataset size: {len(dataset)}")

    target = torch.stack([dataset[1][1]], dim=0)

    loss_fn = CounterfactualLoss(target.repeat(16, 1, 1))

    logits = torch.randn(16, 19, 256, 256)
    print(loss_fn(logits))

    print(f"{loss_fn.target.min(), loss_fn.target.max()}")
    plt.imshow(loss_fn.target[0])
    plt.show()


if __name__ == '__main__':
    main()
