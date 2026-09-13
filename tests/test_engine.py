import pytest
import torch
from src.diff_cf.core.diffusion.engine import Engine, cosine_beta_schedule


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def engine(device: torch.device) -> Engine:
    """Fixture providing a default Engine with cosine schedule and T=100."""
    return Engine(T=100, beta_schedule="cosine", device=device)


@pytest.fixture
def linear_engine(device: torch.device) -> Engine:
    """Fixture providing an Engine with a linear schedule and T=100."""
    return Engine(T=100, beta_schedule="linear", device=device)


@pytest.fixture
def dummy_images() -> torch.Tensor:
    """Fixture providing a dummy batch of images with shape (B, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 32, 32)


# --- Schedule & Initialization Tests ---

def test_cosine_beta_schedule_bounds_and_shape():
    T = 1000
    betas = cosine_beta_schedule(T)

    assert betas.shape == (T,)
    assert torch.all(betas >= 0.0001)
    assert torch.all(betas <= 0.999)
    assert not torch.isnan(betas).any()


def test_engine_init_invalid_schedule():
    with pytest.raises(ValueError, match="Unsupported beta_schedule"):
        Engine(beta_schedule="invalid_schedule")


def test_engine_buffers_and_shapes(engine: Engine):
    T = engine.T
    expected_buffers = [
        "alphas_cumprod",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "coeff1",
        "coeff2",
        "sqrt_beta",
    ]

    for name in expected_buffers:
        buffer = getattr(engine, name, None)
        assert buffer is not None, f"Buffer {name} is missing."
        assert buffer.shape == (T,), f"Buffer {name} shape mismatch."
        assert not torch.isnan(buffer).any(), f"Buffer {name} contains NaNs."


# --- Helper Methods Tests ---

def test_extract_shape():
    a = torch.arange(10, dtype=torch.float32)
    t = torch.tensor([1, 4, 0])
    x_shape = torch.Size([3, 3, 32, 32])

    extracted = Engine._extract(a, t, x_shape)

    assert extracted.shape == (3, 1, 1, 1)
    torch.testing.assert_close(extracted.squeeze(), torch.tensor([1.0, 4.0, 0.0]))


def test_get_sigma(engine: Engine):
    t = torch.tensor([0, 10, 50])
    target_shape = torch.Size([3, 3, 16, 16])

    sigma = engine.get_sigma(t, target_shape)

    assert sigma.shape == (3, 1, 1, 1)
    torch.testing.assert_close(sigma.squeeze(), engine.sqrt_beta[t])


# --- Diffusion & Forward Process Tests ---

def test_diffuse_outputs(engine: Engine, dummy_images: torch.Tensor):
    batch_size = dummy_images.shape[0]
    t = torch.tensor([0, 10, 50, 99])

    noised, noise, snr = engine.diffuse(dummy_images, t)

    assert noised.shape == dummy_images.shape
    assert noise.shape == dummy_images.shape
    assert snr.shape == (batch_size,)
    assert torch.all(snr >= 0)


def test_sample_outputs_and_timesteps(engine: Engine, dummy_images: torch.Tensor):
    batch_size = dummy_images.shape[0]

    noised, noise, t, snr = engine.sample(dummy_images)

    assert noised.shape == dummy_images.shape
    assert noise.shape == dummy_images.shape
    assert t.shape == (batch_size,)
    assert snr.shape == (batch_size,)
    assert torch.all((t >= 0) & (t < engine.T))


def test_snr_values(engine: Engine):
    t = torch.tensor([0, 50, 99])
    snr = engine.snr(t)

    assert snr.shape == (3,)
    # SNR should strictly decrease as timestep t increases
    assert snr[0] > snr[1] > snr[2]


# --- Mathematical Inversion & Approximation Tests ---

def test_image_and_epsilon_approximation_exact_reconstruction(engine: Engine, dummy_images: torch.Tensor):
    """
    Verifies that substituting true noise/image into approximation methods 
    reconstructs the original signals accurately.
    """
    t = torch.tensor([5, 20, 50, 80])
    noised, true_noise, _ = engine.diffuse(dummy_images, t)

    # 1. Recover x_0 from x_t and true noise
    pred_x0 = engine.image_approximation(noised, true_noise, t)
    torch.testing.assert_close(pred_x0, dummy_images, rtol=1e-4, atol=1e-4)

    # 2. Recover epsilon from x_t and true x_0
    pred_noise = engine.epsilon_approximation(noised, dummy_images, t)
    torch.testing.assert_close(pred_noise, true_noise, rtol=1e-4, atol=1e-4)


def test_reverse_single_step(engine: Engine, dummy_images: torch.Tensor):
    t = torch.tensor([10, 10, 10, 10])
    noised, noise, _ = engine.diffuse(dummy_images, t)

    reversed_img = engine.reverse(noised, noise, t, steps=1)
    assert reversed_img.shape == dummy_images.shape


def test_reverse_multi_step(engine: Engine, dummy_images: torch.Tensor):
    t = torch.tensor([20, 20, 20, 20])
    noised, noise, _ = engine.diffuse(dummy_images, t)

    reversed_img = engine.reverse(noised, noise, t, steps=5)
    assert reversed_img.shape == dummy_images.shape