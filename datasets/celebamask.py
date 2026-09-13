"""Dataset class for CelebAMask dataset."""

from pathlib import Path
from typing import List, Optional

from torchvision.transforms import v2

from base_dataset import BaseDataset


class CelebAMaskDataset(BaseDataset):
    """Dataset class for CelebAMask dataset with attribute filtering."""

    def __init__(self,
                 root: str,
                 samples: List[str],
                 means: list[float],
                 stds: list[float],
                 x_min: float,
                 x_max: float,
                 augmentations: bool = False,
                 probability_blur: float = 0.2,
                 blur_sigma: list[float] = [0.2, 2],
                 blur_kernel: int = 3,
                 probability_noise: float = 0.2,
                 noise_sigma: list[float] = [0.2, 2],
                 label: Optional[str] = None,
                 use_positive: bool = True):
        super().__init__(root, samples, means, stds, x_min, x_max, augmentations, probability_blur,
                         blur_sigma, blur_kernel, probability_noise, noise_sigma)
        self.label = label
        self.use_positive = use_positive

        self.all_samples = samples
        self.files = self._filter_samples(self.all_samples)
        self._normalize_transform = v2.Normalize(mean=self.means, std=self.stds)

    def set_attribute(self, label: Optional[str] = None, use_positive: bool = True) -> None:
        """Dynamically switch attribute filtering without rebuilding the dataset object."""
        self.label = label
        self.use_positive = use_positive
        self.files = self._filter_samples(self.all_samples)

    def _filter_samples(self, samples: List[str]) -> List[str]:
        if self.label is None:
            return samples

        anno_path = Path(self.root) / "CelebAMask-HQ" / "CelebAMask-HQ-attribute-anno.txt"
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