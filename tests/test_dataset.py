# tests/test_dataset.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from omegaconf import OmegaConf


# A simple Mock Dataset that behaves like MNIST
class MockMNIST(Dataset):
    def __init__(self, size=10, shape=(1, 28, 28)):
        self.data = torch.randn(size, *shape)
        self.targets = torch.randint(0, 10, (size,))
        self.transform = transforms.Compose([
            transforms.Pad(2, 0.0),  # 28x28 -> 32x32
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def test_dataset_transform_logic():
    """Verify that Padding and Normalization work."""
    size = 5
    dataset = MockMNIST(size=size)
    img, label = dataset[0]

    # Check shape (MNIST 28x28 + Pad 2 = 32x32)
    assert img.shape == (1, 32, 32), f"Expected (1, 32, 32), got {img.shape}"
    # Check normalization (Mean should be roughly 0 if input was random)
    assert -2.0 < img.mean() < 2.0


def test_dataloader_batching():
    """Verify that the DataLoader produces the correct batch size and shape."""
    batch_size = 4
    dataset = MockMNIST(size=10)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get one batch
    batch_imgs, batch_labels = next(iter(loader))

    # The shape should be (Batch, Channels, Height, Width)
    assert batch_imgs.shape == (batch_size, 1, 32, 32)
    assert batch_labels.shape == (batch_size,)


def test_dataloader_from_config():
    # Simulate a Hydra config
    cfg = OmegaConf.create({
        "data": {
            "batch_size": 2,
            "num_workers": 0  # Use 0 for tests to avoid multiprocessing overhead
        }
    })

    dataset = MockMNIST(size=4)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    batch = next(iter(loader))[0]
    assert batch.shape[0] == cfg.data.batch_size