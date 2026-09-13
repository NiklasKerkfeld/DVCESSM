import pytest
import torch
import torch.nn.functional as F

from src.diff_cf.utils import cone_projection, AdaptGrad


# =====================================================================
# Test for Cone Projection
# =====================================================================


def compute_tensor_angle_degrees(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Calculates angle in degrees between two (B, C, H, W) tensors per batch item."""
    v1_flat = v1.flatten(start_dim=1)
    v2_flat = v2.flatten(start_dim=1)
    cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    return torch.acos(cos_sim) * (180.0 / torch.pi)


def test_inside_cone_no_op_4d():
    """If the angle between grad_seg and grad_rob is <= alpha, return grad_rob unchanged."""
    B, C, H, W = 2, 3, 32, 32
    grad_seg = torch.randn(B, C, H, W)
    # Add a small perturbation so angle is small (< 30 degrees)
    grad_rob = grad_seg + 0.1 * torch.randn(B, C, H, W)
    alpha_rad = torch.pi / 6  # 30 degrees

    out = cone_projection(grad_seg, grad_rob, alpha=alpha_rad)
    torch.testing.assert_close(out, grad_rob)


def test_outside_cone_projected_to_boundary_4d():
    """If angle > alpha, projected tensor must lie on the boundary at angle alpha."""
    B, C, H, W = 1, 3, 16, 16
    grad_seg = torch.zeros(B, C, H, W)
    grad_seg[:, 0, :, :] = 1.0  # Channel 0 active

    grad_rob = torch.zeros(B, C, H, W)
    grad_rob[:, 1, :, :] = 1.0  # Channel 1 active (orthogonal / 90 degree angle)

    alpha_deg = 30.0
    alpha_rad = torch.deg2rad(torch.tensor(alpha_deg)).item()

    out = cone_projection(grad_seg, grad_rob, alpha=alpha_rad)

    # 1. Output angle relative to grad_seg must equal alpha
    actual_angle = compute_tensor_angle_degrees(grad_seg, out)
    torch.testing.assert_close(actual_angle, torch.tensor([alpha_deg]), atol=1e-4, rtol=1e-4)


def test_ascent_direction_guarantee_4d():
    """Projected gradient must maintain a positive dot product (ascent direction) with grad_seg."""
    torch.manual_seed(42)
    grad_seg = torch.randn(4, 3, 64, 64)
    grad_rob = torch.randn(4, 3, 64, 64)
    alpha_rad = torch.deg2rad(torch.tensor(30.0)).item()

    out = cone_projection(grad_seg, grad_rob, alpha=alpha_rad)

    # Dot product per batch element across (C, H, W)
    dot_products = (out * grad_seg).sum(dim=(1, 2, 3))
    assert torch.all(dot_products > 0), "Projection failed to maintain an ascent direction."


@pytest.mark.parametrize("shape", [
    (1, 3, 256, 256),   # Single image batch
    (4, 3, 512, 512),   # High-res batch
    (8, 1, 64, 64)      # Single-channel mask style
])
def test_tensor_shape_preservation_4d(shape):
    """Output shape must strictly match input (B, C, H, W) shape."""
    grad_seg = torch.randn(*shape)
    grad_rob = torch.randn(*shape)

    out = cone_projection(grad_seg, grad_rob, alpha=0.5)
    assert out.shape == shape


def test_batch_independence_4d():
    """Batch items must be processed independently without cross-batch leakage."""
    grad_seg = torch.randn(2, 3, 16, 16)
    grad_rob = torch.randn(2, 3, 16, 16)
    alpha_rad = torch.pi / 6

    out = cone_projection(grad_seg, grad_rob, alpha=alpha_rad)

    out_b0 = cone_projection(grad_seg[0:1], grad_rob[0:1], alpha=alpha_rad)
    out_b1 = cone_projection(grad_seg[1:2], grad_rob[1:2], alpha=alpha_rad)

    torch.testing.assert_close(out[0:1], out_b0)
    torch.testing.assert_close(out[1:2], out_b1)


def test_opposite_tensors_180_degrees_4d():
    """Handling antipodal 4D tensors (-1 * grad_seg) safely without NaNs."""
    grad_seg = torch.randn(2, 3, 32, 32)
    grad_rob = -1.0 * grad_seg
    alpha_rad = torch.pi / 6

    out = cone_projection(grad_seg, grad_rob, alpha=alpha_rad)
    assert not torch.isnan(out).any(), "NaN detected in antipodal projection."


def test_zero_vector_inputs_4d():
    """Handling zero spatial tensors gracefully."""
    grad_seg = torch.zeros(2, 3, 32, 32)
    grad_rob = torch.randn(2, 3, 32, 32)

    out = cone_projection(grad_seg, grad_rob, alpha=0.5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_autograd_gradient_flow_4d():
    """Ensure autograd gradients flow properly through 4D spatial tensors."""
    grad_seg = torch.randn(2, 3, 16, 16, requires_grad=True)
    grad_rob = torch.randn(2, 3, 16, 16, requires_grad=True)

    out = cone_projection(grad_seg, grad_rob, alpha=0.5)
    loss = out.sum()
    loss.backward()

    assert grad_seg.grad is not None
    assert grad_rob.grad is not None
    assert not torch.isnan(grad_seg.grad).any()
    assert not torch.isnan(grad_rob.grad).any()


# =====================================================================
# Test for AdaptGrad
# =====================================================================


@pytest.fixture
def default_kwargs():
    """Default parameters using pre-calculated normalized z_min and z_max tensors."""
    data_min = torch.tensor([0.0, 0.0, 0.0])
    data_max = torch.tensor([1.0, 1.0, 1.0])
    data_means = torch.tensor([0.485, 0.456, 0.406])
    data_stds = torch.tensor([0.229, 0.224, 0.225])

    # Pre-calculate z_min and z_max with shape (C, 1, 1) so z_min[None] produces (1, C, 1, 1)
    z_min = ((data_min - data_means) / data_stds).view(3, 1, 1)
    z_max = ((data_max - data_means) / data_stds).view(3, 1, 1)

    return {
        "N": 5,
        "c": 0.95,
        "z_min": z_min,
        "z_max": z_max,
    }


# =====================================================================
# 1. Identity & Shape Invariant Tests
# =====================================================================

@pytest.mark.parametrize("n_val", [0, 1, -1])
def test_n_less_than_or_equal_one_returns_identity(default_kwargs, n_val):
    """When N <= 1, forward pass must return the input tensor unchanged."""
    kwargs = {**default_kwargs, "N": n_val}
    module = AdaptGrad(**kwargs)
    x = torch.randn(2, 3, 32, 32)

    out = module(x)
    torch.testing.assert_close(out, x)


def test_output_shape_expansion(default_kwargs):
    """Output batch size must scale to B * N while preserving (C, H, W)."""
    B, C, H, W = 4, 3, 16, 16
    N = default_kwargs["N"]
    module = AdaptGrad(**default_kwargs)
    x = torch.randn(B, C, H, W)

    out = module(x)
    assert out.shape == (B * N, C, H, W)


# =====================================================================
# 2. Boundary & Mathematical Behavior Tests
# =====================================================================

def test_zero_noise_outside_bounds(default_kwargs):
    """Inputs <= z_min or >= z_max must receive zero noise."""
    module = AdaptGrad(**default_kwargs)

    # Construct inputs explicitly outside bounds
    z_below = module.z_min - 0.5
    z_above = module.z_max + 0.5
    x_outside = torch.cat([z_below, z_above], dim=0).expand(-1, -1, 8, 8)  # Shape (2, C, 8, 8)

    out = module(x_outside)  # Shape (2 * N, C, 8, 8)

    # Reshape to (N, B, C, H, W) to inspect noise additions across batch expansions
    out_reshaped = out.view(default_kwargs["N"], 2, 3, 8, 8)
    for n in range(default_kwargs["N"]):
        torch.testing.assert_close(out_reshaped[n], x_outside)


def test_empirical_noise_standard_deviation(default_kwargs):
    """Empirical noise std at a known interior point must match theoretical sigma."""
    N_large = 20_000
    kwargs = {**default_kwargs, "N": N_large}
    module = AdaptGrad(**kwargs)

    # Midpoint inside [z_min, z_max]
    x_mid = (module.z_min + module.z_max) / 2.0  # Shape (1, C, 1, 1)

    # Theoretical sigma: min(z_max - x, |z_min - x|) / denominator
    expected_sigma = (module.z_max - x_mid) / module.denominator

    out = module(x_mid)  # Shape (N_large, C, 1, 1)
    empirical_std = out.std(dim=0, keepdim=True)

    torch.testing.assert_close(empirical_std, expected_sigma, atol=1e-2, rtol=1e-2)


# =====================================================================
# 3. Autograd & Module State Tests
# =====================================================================

def test_autograd_gradient_flow(default_kwargs):
    """Gradient must flow back to input x while noise generation remains non-differentiable."""
    module = AdaptGrad(**default_kwargs)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)

    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()


def test_registered_buffers(default_kwargs):
    """Ensure z_min, z_max, and denominator are registered as persistent buffers."""
    module = AdaptGrad(**default_kwargs)
    buffer_dict = dict(module.named_buffers())

    assert "z_min" in buffer_dict
    assert "z_max" in buffer_dict
    assert "denominator" in buffer_dict
    assert buffer_dict["z_min"].shape == (1, 3, 1, 1)
    assert buffer_dict["z_max"].shape == (1, 3, 1, 1)