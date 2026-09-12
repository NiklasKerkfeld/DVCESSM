from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from src.diff_cf.core.base_dataset import BaseDataset
from src.diff_cf.core.trainer import BaseTrainer


@pytest.fixture
def mock_dataset():
    """Fixture providing a mocked BaseDataset instance."""
    dataset = MagicMock(spec=BaseDataset)
    dataset.__len__.return_value = 100  # Fixes PyTorch RandomSampler check

    # Pass input tensor through denormalize by default
    dataset.denormalize.side_effect = lambda img, idx=None: img
    return dataset


@pytest.fixture
def trainer(mock_dataset, tmp_path):
    """Fixture providing a BaseTrainer instance with mocked SummaryWriter."""
    device = torch.device("cpu")
    log_dir = str(tmp_path / "logs")
    checkpoint_path = str(tmp_path / "checkpoints")

    with patch("src.diff_cf.core.trainer.SummaryWriter"):
        trainer_inst = BaseTrainer(
            dataset=mock_dataset,
            device=device,
            lr=1e-4,
            batch_size=16,
            num_workers=0,
            image_type="rgb",
            name="TestTrainer",
            checkpoint_path=checkpoint_path,
            log_dir=log_dir,
        )
    return trainer_inst


def test_init_attributes_and_dataloader(mock_dataset, tmp_path):
    device = torch.device("cpu")
    log_dir = str(tmp_path / "logs")
    checkpoint_path = str(tmp_path / "checkpoints")

    with patch("src.diff_cf.core.trainer.SummaryWriter") as mock_writer_cls:
        trainer_inst = BaseTrainer(
            dataset=mock_dataset,
            device=device,
            lr=1e-4,
            batch_size=16,
            num_workers=0,
            image_type="rgb",
            name="TestTrainer",
            checkpoint_path=checkpoint_path,
            log_dir=log_dir,
        )

        # Verify configuration attributes
        assert trainer_inst.lr == 1e-4
        assert trainer_inst.batch_size == 16
        assert trainer_inst.num_workers == 0
        assert trainer_inst.name == "TestTrainer"
        assert trainer_inst.epoch == 0
        assert trainer_inst.checkpoint == checkpoint_path
        assert trainer_inst.log_dir == log_dir
        assert trainer_inst.image_type == "rgb"

        # Verify DataLoader initialization
        assert isinstance(trainer_inst.train_dataloader, DataLoader)
        assert trainer_inst.train_dataloader.batch_size == 16

        # Verify SummaryWriter log directory path
        mock_writer_cls.assert_called_once_with(f"{log_dir}/TestTrainer")


def test_abstract_methods_raise_not_implemented(trainer):
    with pytest.raises(NotImplementedError):
        trainer.save_checkpoint()

    with pytest.raises(NotImplementedError):
        trainer.load_checkpoint("path/to/checkpoint.pt")


def test_visualize_image_rgb(trainer, mock_dataset):
    sample_image = torch.randn(3, 32, 32)
    result = trainer._visualize_image(sample_image, index=2)

    mock_dataset.denormalize.assert_called_once_with(sample_image, 2)
    assert torch.equal(result, sample_image)


def test_visualize_image_mri(mock_dataset, tmp_path):
    device = torch.device("cpu")
    with patch("src.diff_cf.core.trainer.SummaryWriter"):
        mri_trainer = BaseTrainer(
            dataset=mock_dataset,
            device=device,
            image_type="mri",
            log_dir=str(tmp_path),
        )

    # Denormalized output is formatted into a grid for MRI slices [C, H, W]
    sample_image = torch.randn(4, 32, 32)
    result = mri_trainer._visualize_image(sample_image)

    mock_dataset.denormalize.assert_called_once_with(sample_image, None)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 3  # Output shape: [3, H_grid, W_grid]


def test_log_images(trainer):
    sample_images = torch.randn(8, 3, 32, 32)
    trainer.epoch = 5

    trainer._log_images(sample_images, label="train/samples")

    trainer.writer.add_image.assert_called_once()
    call_args = trainer.writer.add_image.call_args

    assert call_args[0][0] == "train/samples"
    assert call_args[1]["global_step"] == 5
    assert isinstance(call_args[0][1], torch.Tensor)