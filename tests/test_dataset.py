# tests/test_dataset.py
import torch
from core.DiffusionDataset import DiffusionDataset


def test_dataset_output():
    # Create dummy data (10 images, 1 channel, 32x32)
    dummy_data = torch.randn(10, 1, 32, 32)
    dataset = DiffusionDataset(dummy_data)

    assert len(dataset) == 10

    sample = dataset[0]
    # Check shape
    assert sample.shape == (1, 32, 32)
    # Check if data is passed through correctly
    assert torch.equal(sample, dummy_data[0])