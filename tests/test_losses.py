import pytest
import torch
from dataclasses import FrozenInstanceError

from src.diff_cf.core.losses import (SegmentationLossConfig,
                                     SegmentationLoss,
                                     SegmentationLossFactory)


@pytest.fixture
def dummy_target() -> torch.Tensor:
    """Creates a dummy 2D binary segmentation target tensor [B, H, W]."""
    target = torch.zeros((2, 10, 10), dtype=torch.long)
    target[:, :, 5:] = 1  # Boundary at column index 5
    return target


@pytest.fixture
def dummy_logits() -> torch.Tensor:
    """Creates dummy logits [B, C, H, W] matching 2 classes."""
    # Class 0 logits higher on left, Class 1 logits higher on right
    logits = torch.zeros((2, 2, 10, 10), dtype=torch.float32)
    logits[:, 0, :, :5] = 2.0
    logits[:, 1, :, 5:] = 2.0
    return logits


class TestSegmentationLossConfig:
    def test_default_values(self):
        config = SegmentationLossConfig()
        assert config.border_width == 2
        assert config.use_mask is True
        assert config.reduction == 'mean'

    def test_immutability(self):
        config = SegmentationLossConfig()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            config.border_width = 5


class TestSegmentationLoss:
    def test_target_is_cloned(self, dummy_target):
        config = SegmentationLossConfig(border_width=0)
        loss_fn = SegmentationLoss(target=dummy_target, config=config)

        # Verify target is cloned and not modified in-place on original tensor
        assert loss_fn.target is not dummy_target
        dummy_target.fill_(99)
        assert not torch.all(loss_fn.target == 99)

    def test_border_width_zero(self, dummy_target):
        config = SegmentationLossConfig(border_width=0)
        loss_fn = SegmentationLoss(target=dummy_target, config=config)

        # No pixels should be marked as ignore index (-1)
        assert -1 not in loss_fn.target

    def test_border_masking(self, dummy_target):
        config = SegmentationLossConfig(border_width=2)
        loss_fn = SegmentationLoss(target=dummy_target, config=config)

        # Borders near transition boundary (col index 5) should be set to -1
        assert -1 in loss_fn.target

    def test_forward_perfect_predictions_with_mask(self, dummy_target, dummy_logits):
        config = SegmentationLossConfig(border_width=0, use_mask=True, reduction='mean')
        loss_fn = SegmentationLoss(target=dummy_target, config=config)

        # Since predictions match target perfectly, loss with mask should be 0.0
        loss = loss_fn(dummy_logits)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_forward_without_mask(self, dummy_target, dummy_logits):
        config = SegmentationLossConfig(border_width=0, use_mask=False, reduction='mean')
        loss_fn = SegmentationLoss(target=dummy_target, config=config)

        # Without mask, standard cross-entropy loss should be > 0.0 even if predictions are correct
        loss = loss_fn(dummy_logits)
        assert loss.item() > 0.0

    @pytest.mark.parametrize("reduction_str, expected_fn", [
        ("mean", torch.mean),
        ("sum", torch.sum),
    ])
    def test_reduction_modes(self, dummy_target, dummy_logits, reduction_str, expected_fn):
        config = SegmentationLossConfig(border_width=0, use_mask=False, reduction=reduction_str)
        loss_fn = SegmentationLoss(target=dummy_target, config=config)

        loss = loss_fn(dummy_logits)

        # Check that loss matches the expected reduction behavior
        raw_ce = torch.nn.functional.cross_entropy(dummy_logits, dummy_target, reduction='none')
        assert torch.isclose(loss, expected_fn(raw_ce))


class TestSegmentationLossFactory:
    def test_factory_instantiation(self, dummy_target):
        config = SegmentationLossConfig(border_width=1, use_mask=False)
        factory = SegmentationLossFactory(config=config)

        loss_module = factory(dummy_target)

        assert isinstance(loss_module, SegmentationLoss)
        assert loss_module.border_width == 1
        assert loss_module.use_mask is False