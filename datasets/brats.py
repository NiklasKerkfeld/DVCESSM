"""Dataset class for BraTS dataset."""

from typing import List, Tuple, Union

from datasets.base import BaseDataset


class BraTSDataset(BaseDataset):
    """Dataset class for BraTS dataset."""

    def __init__(self,
                 samples: List[str],
                 prob_blur: float = 0.0,
                 kernel_size: int = 3,
                 sigma: Union[float, Tuple[float, float]] = (0.0, 3.0),
                 use_augmentations: bool = True):
        super().__init__(samples, prob_blur, kernel_size, sigma, use_augmentations)
