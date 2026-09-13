import pytest
import torch

from src.diff_cf.models import DiffusionModel


@pytest.fixture
def default_kwargs():
    """Default parameters using pre-calculated normalized z_min and z_max tensors."""
    return {
        "channels": 3,
        "width": 64,
        "height": 64,
        "time_dim": 256,
        "nums_features": [32, 64, 128, 256, 512],
        "use_prior": False,
        "num_classes": 3
    }

def test_output(default_kwargs):
    model = DiffusionModel(**default_kwargs)

    image = torch.randn(1, 3, 64, 64)
    timestep = torch.randint(0, 1000, (1,))

    pred = model(image, timestep)

    assert pred.shape == (1, 3, 64, 64)

def test_prior(default_kwargs):
    kwargs = {**default_kwargs, "use_prior": True}

    model = DiffusionModel(**kwargs)

    image = torch.randn(1, 3, 64, 64)
    timestep = torch.randint(0, 1000, (1,))
    target = torch.randint(0, 3, (1, 64, 64))

    pred = model(image, timestep, target)

    assert pred.shape == (1, 3, 64, 64)
