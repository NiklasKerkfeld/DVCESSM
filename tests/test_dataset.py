from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.diff_cf.core.base_dataset import BaseDataset, get_splits


class DummyDataset(BaseDataset):
    """Subclass of BaseDataset used to test set_attribute functionality."""

    def __init__(self, *args, **kwargs):
        # Handle parameter naming variations if Hydra passes kwargs
        if "mins" in kwargs:
            kwargs["min"] = kwargs.pop("mins")
        if "maxs" in kwargs:
            kwargs["max"] = kwargs.pop("maxs")
        super().__init__(*args, **kwargs)

        self.files = self._filter_samples(self.all_samples)
        self.attribute_label = None
        self.use_positive = True

    def set_attribute(self, label: str | None = None, use_positive: bool = True) -> None:
        self.attribute_label = label
        self.use_positive = use_positive


@pytest.fixture
def temp_npz_files(tmp_path: Path) -> list[str]:
    """Fixture that creates valid .npz files expected by BaseDataset._load."""
    data_dir = tmp_path / "samples"
    data_dir.mkdir()

    file_paths = []
    for i in range(10):
        file_path = data_dir / f"sample_{i:02d}.npz"
        np.savez(
            file_path,
            image=np.random.randn(3, 32, 32).astype(np.float32),
            label=np.array([1], dtype=np.int64),
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )
        file_paths.append(str(file_path))

    return file_paths


# --- BaseDataset Tests ---


def test_dataset_len_and_filepaths(temp_npz_files: list[str]):
    dataset = BaseDataset(
        root="./data",
        samples=temp_npz_files,
        means=[0.5, 0.5, 0.5],
        stds=[0.2, 0.2, 0.2],
        min=0.0,
        max=1.0,
    )

    assert len(dataset) == len(temp_npz_files)
    assert dataset.get_filepath(0) == temp_npz_files[0]


def test_find_index(temp_npz_files: list[str]):
    dataset = BaseDataset(
        root="./data",
        samples=temp_npz_files,
        means=[0.5, 0.5, 0.5],
        stds=[0.2, 0.2, 0.2],
        min=0.0,
        max=1.0,
    )

    # Find index by full path or stem
    assert dataset.find_index("sample_03") == 3
    assert dataset.find_index(temp_npz_files[5]) == 5
    assert dataset.find_index("non_existent_file") is None


def test_load_item_shapes_and_types(temp_npz_files: list[str]):
    dataset = BaseDataset(
        root="./data",
        samples=temp_npz_files,
        means=[0.5, 0.5, 0.5],
        stds=[0.2, 0.2, 0.2],
        min=0.0,
        max=1.0,
        augmentations=False,
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64


def test_load_item_with_augmentations(temp_npz_files: list[str]):
    dataset = BaseDataset(
        root="./data",
        samples=temp_npz_files,
        means=[0.5, 0.5, 0.5],
        stds=[0.2, 0.2, 0.2],
        min=0.0,
        max=1.0,
        augmentations=True,
        probablity_blur=0.0,  # Force blur augmentation trigger (> 0.0)
        probablity_noise=0.0,  # Force noise augmentation trigger (> 0.0)
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)


def test_denormalize_3d_and_4d(temp_npz_files: list[str]):
    dataset = BaseDataset(
        root="./data",
        samples=temp_npz_files,
        means=[0.5, 0.5, 0.5],
        stds=[0.2, 0.2, 0.2],
        min=0.0,
        max=1.0,
    )

    # Test 3D image tensor [C, H, W]
    img_3d = torch.zeros(3, 16, 16)
    denorm_3d = dataset.denormalize(img_3d)
    assert denorm_3d.shape == (3, 16, 16)
    assert torch.allclose(denorm_3d, torch.tensor(0.5).view(1, 1, 1))

    # Test 4D batched tensor [B, C, H, W]
    img_4d = torch.zeros(2, 3, 16, 16)
    denorm_4d = dataset.denormalize(img_4d)
    assert denorm_4d.shape == (2, 3, 16, 16)

    # Test unsupported tensor dimensions
    with pytest.raises(ValueError, match="Unexpected image tensor shape"):
        dataset.denormalize(torch.zeros(16, 16))


def test_base_dataset_set_attribute_raises_error(temp_npz_files: list[str]):
    dataset = BaseDataset(
        root="./data",
        samples=temp_npz_files,
        means=[0.5, 0.5, 0.5],
        stds=[0.2, 0.2, 0.2],
        min=0.0,
        max=1.0,
    )

    with pytest.raises(NotImplementedError):
        dataset.set_attribute("target_attribute")


# --- get_datasets Tests ---
def test_get_datasets_splitting_and_instantiation(tmp_path: Path):
    # Setup directory structure matching root/preprocessed/*/*.npz
    root_dir = tmp_path / "root"
    sub_dir = root_dir / "preprocessed" / "group1"
    sub_dir.mkdir(parents=True)

    # Create dummy .npz files for dataset loading
    for i in range(20):
        np.savez(
            sub_dir / f"file_{i:02d}.npz",
            image=np.random.randn(3, 32, 32).astype(np.float32),
            label=np.array([1], dtype=np.int64),
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )

    train_ds, test_ds = get_splits(
        root=str(root_dir),
    )

    # Assert real instances were returned with correct settings
    assert isinstance(train_ds, list)
    assert isinstance(test_ds, list)
