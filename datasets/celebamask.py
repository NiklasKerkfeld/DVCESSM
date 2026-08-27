"""Dataset class for CelebAMask dataset."""

from pathlib import Path
from typing import List, Tuple, Optional

from omegaconf import DictConfig
from torchvision.transforms import v2
import torch

from base import BaseDataset, get_datasets


class CelebAMaskDataset(BaseDataset):
    """Dataset class for CelebAMask dataset with attribute filtering."""

    def __init__(self,
                 samples: List[str],
                 cfg: DictConfig,
                 augmentations: bool = False,
                 label: Optional[str] = None,
                 use_positive: bool = True):
        super().__init__(samples, cfg, augmentations, label, use_positive)
        self.means = torch.tensor(cfg.data.means)
        self.stds = torch.tensor(cfg.data.stds)
        self._normalize_transform = v2.Normalize(mean=self.means, std=self.stds)

    def _filter_samples(self, samples: List[str]) -> List[str]:
        if self.label is None:
            return samples

        anno_path = Path(self.cfg.data.root) / "CelebAMask-HQ" / "CelebAMask-HQ-attribute-anno.txt"
        if not anno_path.exists():
            raise FileNotFoundError(f"Attribute file not found at {anno_path}")

        with open(anno_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()][1:]  # Skip image count line

        attributes = lines[0].split()
        if self.label not in attributes:
            raise ValueError(f"Attribute '{self.label}' not found in attribute file.")

        attr_idx = attributes.index(self.label) + 1  # Offset by 1 for filename column

        valid_stems = set()
        for row in lines[1:]:
            parts = row.split()
            filename, value = parts[0], parts[attr_idx]
            is_pos = (value == '1')

            if is_pos == self.use_positive:
                stem = f"{int(Path(filename).stem):05d}"
                valid_stems.add(stem)

        filtered_samples = [s for s in samples if Path(s).stem in valid_stems]
        return filtered_samples