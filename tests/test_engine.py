import torch
import pytest
from src.cfgan_mnist.core.Diffusion.engine import Engine


@pytest.fixture
def engine():
    """This 'fixture' creates an engine instance for all tests to use."""
    device = torch.device("cpu")
    return Engine(T=200, beta_min=1e-4, beta_max=2e-2, device=device, beta_schedule='linear')


def test_diffusion_forward_shapes(engine):
    batch_size = 4
    images = torch.randn(batch_size, 1, 32, 32)

    # Test the sample method (forward process)
    x_t, noise, t = engine.sample(images)

    assert x_t.shape == images.shape, "Noised image shape must match input"
    assert noise.shape == images.shape, "Noise shape must match input"
    assert t.shape == (batch_size,), "Timestep batch size mismatch"


def test_reverse_step_math(engine):
    batch_size = 2
    x_t = torch.randn(batch_size, 1, 32, 32)
    eps_pred = torch.randn_like(x_t)
    t = torch.tensor([10, 150])  # Test different timesteps

    # Test the reverse method
    mean = engine.reverse(x_t, eps_pred, t)

    assert mean.shape == x_t.shape
    assert not torch.isnan(mean).any(), "Reverse step produced NaNs"


def test_extract_broadcasting(engine):
    # This specifically tests the _extract helper we built
    constant_tensor = torch.linspace(0, 1, 200)
    t = torch.tensor([5, 10, 15])
    images_shape = (3, 1, 32, 32)

    extracted = engine._extract(constant_tensor, t, images_shape)

    # The shape should be (Batch, 1, 1, 1) for 4D images
    assert extracted.shape == (3, 1, 1, 1)