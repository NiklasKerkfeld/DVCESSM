import pytest
import torch

from src.diff_cf.core.diffusion.engine import (
    Engine,
    cosine_beta_schedule,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def engine(device: torch.device) -> Engine:
    """Default Engine with cosine schedule and T=100."""
    return Engine(
        T=100,
        beta_schedule="cosine",
        device=device,
    )


@pytest.fixture
def linear_engine(device: torch.device) -> Engine:
    """Engine with linear schedule and T=100."""
    return Engine(
        T=100,
        beta_schedule="linear",
        device=device,
    )


@pytest.fixture
def dummy_images() -> torch.Tensor:
    """Dummy batch with shape (B, C, H, W)."""
    generator = torch.Generator().manual_seed(42)
    return torch.randn(
        4,
        3,
        32,
        32,
        generator=generator,
    )


def extract_for_image(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        image: torch.Tensor,
) -> torch.Tensor:
    """Extract 1-D schedule values and reshape them for image batches."""
    return values[timesteps].reshape(
        timesteps.shape[0],
        *((1,) * (image.ndim - 1)),
    )


# ---------------------------------------------------------------------
# Schedule and initialization
# ---------------------------------------------------------------------

def test_cosine_beta_schedule_bounds_and_shape():
    num_timesteps = 1000
    betas = cosine_beta_schedule(num_timesteps)

    assert betas.shape == (num_timesteps,)
    assert torch.all(betas >= 0.0001)
    assert torch.all(betas <= 0.999)
    assert torch.isfinite(betas).all()


def test_engine_init_invalid_schedule():
    with pytest.raises(ValueError, match="Unsupported beta_schedule"):
        Engine(beta_schedule="invalid_schedule")


def test_engine_buffers_and_shapes(engine: Engine):
    expected_buffers = [
        "beta",
        "alphas_cumprod",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "coeff1",
        "coeff2",
        "posterior_variance",
    ]

    for name in expected_buffers:
        buffer = getattr(engine, name, None)

        assert buffer is not None, f"Buffer {name} is missing."
        assert buffer.shape == (engine.T,)
        assert torch.isfinite(buffer).all()


def test_posterior_variance_properties(engine: Engine):
    variance = engine.posterior_variance

    assert torch.all(variance >= 0)

    # The transition from timestep 0 to the clean endpoint is
    # deterministic.
    torch.testing.assert_close(
        variance[0],
        torch.tensor(0.0, dtype=variance.dtype),
        atol=1e-8,
        rtol=0.0,
    )

    # The posterior variance should not exceed beta_t.
    assert torch.all(variance <= engine.beta + 1e-7)


# ---------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------

def test_extract_shape():
    values = torch.arange(10, dtype=torch.float32)
    timesteps = torch.tensor([1, 4, 0], dtype=torch.long)
    image_shape = torch.Size([3, 3, 32, 32])

    extracted = Engine._extract(
        values,
        timesteps,
        image_shape,
    )

    assert extracted.shape == (3, 1, 1, 1)
    torch.testing.assert_close(
        extracted.squeeze(),
        torch.tensor([1.0, 4.0, 0.0]),
    )


def test_extract_values(engine: Engine):
    timesteps = torch.tensor([0, 10, 50], dtype=torch.long)
    image_shape = torch.Size([3, 3, 16, 16])

    extracted = engine._extract(
        engine.alphas_cumprod,
        timesteps,
        image_shape,
    )

    assert extracted.shape == (3, 1, 1, 1)
    torch.testing.assert_close(
        extracted.flatten(),
        engine.alphas_cumprod[timesteps],
    )


# ---------------------------------------------------------------------
# Forward diffusion
# ---------------------------------------------------------------------

def test_diffuse_outputs(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    batch_size = dummy_images.shape[0]
    timesteps = torch.tensor([0, 10, 50, 99], dtype=torch.long)

    noised, noise, snr = engine.diffuse(
        dummy_images,
        timesteps,
    )

    assert noised.shape == dummy_images.shape
    assert noise.shape == dummy_images.shape
    assert snr.shape == (batch_size,)

    assert torch.isfinite(noised).all()
    assert torch.isfinite(noise).all()
    assert torch.isfinite(snr).all()
    assert torch.all(snr >= 0)


def test_diffuse_matches_closed_form(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    timesteps = torch.tensor([5, 20, 50, 80], dtype=torch.long)

    noised, noise, _ = engine.diffuse(
        dummy_images,
        timesteps,
    )

    sqrt_alpha_bar = extract_for_image(
        engine.sqrt_alphas_cumprod,
        timesteps,
        dummy_images,
    )
    sqrt_one_minus_alpha_bar = extract_for_image(
        engine.sqrt_one_minus_alphas_cumprod,
        timesteps,
        dummy_images,
    )

    expected = (
        sqrt_alpha_bar * dummy_images
        + sqrt_one_minus_alpha_bar * noise
    )

    torch.testing.assert_close(noised, expected)


def test_sample_outputs_and_timesteps(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    batch_size = dummy_images.shape[0]

    noised, noise, timesteps, snr = engine.sample(dummy_images)

    assert noised.shape == dummy_images.shape
    assert noise.shape == dummy_images.shape
    assert timesteps.shape == (batch_size,)
    assert snr.shape == (batch_size,)

    assert timesteps.dtype == torch.long
    assert timesteps.device == dummy_images.device
    assert torch.all((timesteps >= 0) & (timesteps < engine.T))


def test_snr_values(engine: Engine):
    timesteps = torch.tensor([0, 50, 99], dtype=torch.long)

    snr = engine.snr(timesteps)

    assert snr.shape == (3,)
    assert torch.isfinite(snr).all()
    assert snr[0] > snr[1] > snr[2]


# ---------------------------------------------------------------------
# Image and epsilon reconstruction
# ---------------------------------------------------------------------

def test_image_and_epsilon_approximation_exact_reconstruction(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    timesteps = torch.tensor([5, 20, 50, 80], dtype=torch.long)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        timesteps,
    )

    predicted_x0 = engine.image_approximation(
        noised,
        true_noise,
        timesteps,
    )

    torch.testing.assert_close(
        predicted_x0,
        dummy_images,
        rtol=1e-4,
        atol=1e-4,
    )

    predicted_noise = engine.epsilon_approximation(
        noised,
        dummy_images,
        timesteps,
    )

    torch.testing.assert_close(
        predicted_noise,
        true_noise,
        rtol=1e-4,
        atol=1e-4,
    )


# ---------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------

def test_ddpm_step_shapes(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    timesteps = torch.tensor([10, 10, 10, 10], dtype=torch.long)

    noised, noise, _ = engine.diffuse(
        dummy_images,
        timesteps,
    )

    mean, variance = engine.ddpm_step(
        noised,
        noise,
        timesteps,
    )

    assert mean.shape == dummy_images.shape
    assert variance.shape == (dummy_images.shape[0], 1, 1, 1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert torch.all(variance >= 0)


def test_ddpm_step_matches_posterior(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    timesteps = torch.tensor([1, 10, 50, 99], dtype=torch.long)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        timesteps,
    )

    actual_mean, actual_variance = engine.ddpm_step(
        noised,
        true_noise,
        timesteps,
    )

    beta_t = extract_for_image(
        engine.beta,
        timesteps,
        dummy_images,
    )
    alpha_t = 1.0 - beta_t

    alpha_bar_t = extract_for_image(
        engine.alphas_cumprod,
        timesteps,
        dummy_images,
    )

    previous_timesteps = (timesteps - 1).clamp(min=0)
    alpha_bar_previous = extract_for_image(
        engine.alphas_cumprod,
        previous_timesteps,
        dummy_images,
    )

    # There is no noisy timestep before t=0, so alpha_bar_{-1}=1.
    timestep_zero_mask = (timesteps == 0).reshape(
        timesteps.shape[0],
        *((1,) * (dummy_images.ndim - 1)),
    )
    alpha_bar_previous = torch.where(
        timestep_zero_mask,
        torch.ones_like(alpha_bar_previous),
        alpha_bar_previous,
    )

    expected_variance = (
        beta_t
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
    )

    coefficient_x0 = (
        beta_t
        * torch.sqrt(alpha_bar_previous)
        / (1.0 - alpha_bar_t)
    )
    coefficient_xt = (
        torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
    )

    expected_mean = (
        coefficient_x0 * dummy_images
        + coefficient_xt * noised
    )

    torch.testing.assert_close(
        actual_mean,
        expected_mean,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        actual_variance,
        expected_variance,
        rtol=1e-5,
        atol=1e-6,
    )


def test_ddpm_step_at_zero_returns_clean_mean(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    timesteps = torch.zeros(
        dummy_images.shape[0],
        dtype=torch.long,
    )

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        timesteps,
    )

    mean, variance = engine.ddpm_step(
        noised,
        true_noise,
        timesteps,
    )

    torch.testing.assert_close(
        mean,
        dummy_images,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        variance,
        torch.zeros_like(variance),
        atol=1e-8,
        rtol=0.0,
    )


# ---------------------------------------------------------------------
# DDIM
# ---------------------------------------------------------------------

def test_ddim_deterministic_jump_matches_closed_form(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    source_tensor = torch.full(
        (dummy_images.shape[0],),
        20,
        dtype=torch.long,
    )
    target_tensor = torch.full_like(source_tensor, 10)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        source_tensor,
    )

    actual = engine.ddim_step(
        noised,
        true_noise,
        source_tensor,
        target_tensor,
        eta=0.0,
    )

    alpha_bar_target = extract_for_image(
        engine.alphas_cumprod,
        target_tensor,
        dummy_images,
    )

    expected = (
        torch.sqrt(alpha_bar_target) * dummy_images
        + torch.sqrt(1.0 - alpha_bar_target) * true_noise
    )

    assert actual.shape == dummy_images.shape
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_ddim_clean_endpoint_returns_x0(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    source_tensor = torch.tensor(
        [5, 20, 50, 80],
        dtype=torch.long,
    )
    target_tensor = torch.full_like(source_tensor, -1)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        source_tensor,
    )

    reconstructed = engine.ddim_step(
        noised,
        true_noise,
        source_tensor,
        target_tensor,
        eta=0.0,
    )

    torch.testing.assert_close(
        reconstructed,
        dummy_images,
        rtol=1e-4,
        atol=1e-4,
    )


def test_ddim_eta_one_matches_stochastic_formula(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    source_tensor = torch.full(
        (dummy_images.shape[0],),
        20,
        dtype=torch.long,
    )
    target_tensor = torch.full_like(source_tensor, 19)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        source_tensor,
    )

    alpha_bar_source = extract_for_image(
        engine.alphas_cumprod,
        source_tensor,
        dummy_images,
    )
    alpha_bar_target = extract_for_image(
        engine.alphas_cumprod,
        target_tensor,
        dummy_images,
    )

    sigma_squared = (
        (1.0 - alpha_bar_target)
        / (1.0 - alpha_bar_source)
        * (1.0 - alpha_bar_source / alpha_bar_target)
    ).clamp(min=0.0)

    predicted_noise_coefficient = torch.sqrt(
        (1.0 - alpha_bar_target - sigma_squared).clamp(min=0.0)
    )

    torch.manual_seed(123)
    random_noise = torch.randn_like(dummy_images)

    expected = (
        torch.sqrt(alpha_bar_target) * dummy_images
        + predicted_noise_coefficient * true_noise
        + torch.sqrt(sigma_squared) * random_noise
    )

    # Reset the seed so ddim_step draws the same random noise.
    torch.manual_seed(123)
    actual = engine.ddim_step(
        noised,
        true_noise,
        source_tensor,
        target_tensor,
        eta=1.0,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("eta", [-0.1, 1.1])
def test_ddim_rejects_invalid_eta(
        engine: Engine,
        dummy_images: torch.Tensor,
        eta: float,
):
    source_tensor = torch.full(
        (dummy_images.shape[0],),
        20,
        dtype=torch.long,
    )
    target_tensor = torch.full_like(source_tensor, 10)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        source_tensor,
    )

    with pytest.raises(ValueError, match="eta"):
        engine.ddim_step(
            noised,
            true_noise,
            source_tensor,
            target_tensor,
            eta=eta,
        )

def test_complete_ddim_oracle_trajectory(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    batch_size = dummy_images.shape[0]
    epsilon = torch.randn_like(dummy_images)

    start_t = engine.T - 1
    alpha_start = engine.alphas_cumprod[start_t]

    # Construct a mathematically valid point on a deterministic DDIM path.
    x = (
        torch.sqrt(alpha_start) * dummy_images
        + torch.sqrt(1.0 - alpha_start) * epsilon
    )

    for t in range(start_t, -1, -1):
        source = torch.full(
            (batch_size,),
            t,
            dtype=torch.long,
            device=x.device,
        )
        target = torch.full_like(source, t - 1)

        x = engine.ddim_step(
            images=x,
            epsilon_pred=epsilon,
            t=source,
            target_t=target,
            eta=0.0,
        )

    torch.testing.assert_close(
        x,
        dummy_images,
        rtol=5e-4,
        atol=5e-4,
    )


def test_ddim_rejects_non_decreasing_transition(
        engine: Engine,
        dummy_images: torch.Tensor,
):
    source_tensor = torch.full(
        (dummy_images.shape[0],),
        20,
        dtype=torch.long,
    )
    target_tensor = torch.full_like(source_tensor, 20)

    noised, true_noise, _ = engine.diffuse(
        dummy_images,
        source_tensor,
    )

    with pytest.raises(ValueError, match="target_t"):
        engine.ddim_step(
            noised,
            true_noise,
            source_tensor,
            target_tensor,
            eta=0.0,
        )