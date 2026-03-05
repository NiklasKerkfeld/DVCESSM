# tests/test_generator.py
import torch
import pytest
from core.CounterfactualGenerator import CounterfactualGenerator
from core.DiffusionEngine import DiffusionEngine


class MockModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.randn_like(x)


class MockClassifier(torch.nn.Module):
    def forward(self, x):
        # x shape is (B, C, H, W)
        # Calculate mean per image in the batch
        batch_mean = x.mean(dim=(1, 2, 3))  # Result shape: (B,)

        # Stack into (B, 2) to simulate 2 classes
        # dim=1 now exists because batch_mean is 1D
        return torch.stack([batch_mean, -batch_mean], dim=1)


@pytest.fixture
def cf_setup():
    device = torch.device("cpu")
    model = MockModel()
    classifier = MockClassifier()
    engine = DiffusionEngine(T=200, beta_min=1e-4, beta_max=2e-2, device=device, beta_schedule='linear')

    # Assuming your Generator takes these in __init__
    gen = CounterfactualGenerator(classifier, model, engine, device, weight_classifier=.1, weight_distance=.15)
    return gen


def test_cf_gradient_flow(cf_setup):
    gen = cf_setup
    image = torch.randn(1, 1, 32, 32)
    target = torch.tensor([1])  # Target class 1

    # Run a tiny generation (start at 10% noise for speed)
    cf_image = gen.generate_counterfactual(image, target, start_fraction=0.1)

    assert cf_image.shape == image.shape
    assert not torch.equal(cf_image, image), "CF image should be modified from original"
    assert not torch.isnan(cf_image).any()


def test_g_update_direction(cf_setup):
    gen = cf_setup
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    images = torch.randn(1, 1, 32, 32)
    targets = torch.tensor([1])
    t = torch.tensor([10])
    eps_pred = torch.randn_like(x)

    # Check if the gradient update is actually calculated
    g_update = gen.calc_g_update(x, images, targets, eps_pred, t)

    assert g_update.shape == x.shape
    assert g_update.abs().sum() > 0, "Gradient update should not be zero"