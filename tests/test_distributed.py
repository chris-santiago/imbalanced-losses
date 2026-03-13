"""
Unit tests for proxy_losses.distributed.

These tests cover single-process behaviour (world_size=1) and the guard
conditions. True multi-process all-gather is not tested here — that requires
a torchrun launcher and is validated by integration testing.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

import torch.nn as nn

from proxy_losses import RecallAtQuantileLoss, SmoothAPLoss
from proxy_losses.distributed import all_gather_no_grad, all_gather_with_grad
from proxy_losses.warmup_wrapper import LossWarmupWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_single_process_group():
    """Initialize a single-process gloo group if not already done."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            world_size=1,
            rank=0,
        )


def _destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Guard tests (no dist initialized)
# ---------------------------------------------------------------------------


class TestGuards:
    def test_with_grad_raises_if_not_initialized(self):
        _destroy_process_group()
        with pytest.raises(RuntimeError, match="not initialized"):
            all_gather_with_grad(torch.randn(4, 2))

    def test_no_grad_raises_if_not_initialized(self):
        _destroy_process_group()
        with pytest.raises(RuntimeError, match="not initialized"):
            all_gather_no_grad(torch.randint(0, 4, (4,)))


# ---------------------------------------------------------------------------
# Single-process (world_size=1) tests
# ---------------------------------------------------------------------------


class TestSingleProcess:
    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    def test_with_grad_identity(self):
        """world_size=1: output equals input, gradient flows."""
        x = torch.randn(8, 4, requires_grad=True)
        out = all_gather_with_grad(x)
        assert out.shape == x.shape
        assert out.data_ptr() == x.data_ptr(), "should return the same tensor"

    def test_no_grad_identity(self):
        """world_size=1: output equals input for integer targets."""
        t = torch.randint(0, 4, (8,))
        out = all_gather_no_grad(t)
        assert out.shape == t.shape
        assert torch.equal(out, t)

    def test_with_grad_backward(self):
        """Gradient flows through the output."""
        x = torch.randn(6, 3, requires_grad=True)
        out = all_gather_with_grad(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_no_grad_no_gradient_attr(self):
        """all_gather_no_grad output has no gradient."""
        t = torch.randint(0, 2, (10,))
        out = all_gather_no_grad(t)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# gather_distributed on loss classes (world_size=1 → auto resolves to False)
# ---------------------------------------------------------------------------


class TestLossGatherDistributed:
    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    # --- SmoothAPLoss -------------------------------------------------------

    def test_ap_auto_resolves_false_at_world_size_1(self):
        """Auto-detect: world_size=1 → _gather_resolved becomes False after first forward."""
        loss_fn = SmoothAPLoss(num_classes=1, queue_size=0)
        assert loss_fn._gather_resolved is None  # not yet resolved
        logits = torch.randn(16, 1, requires_grad=True)
        targets = torch.randint(0, 2, (16,))
        loss_fn(logits, targets)
        assert loss_fn._gather_resolved is False

    def test_ap_explicit_false_resolves_false(self):
        """gather_distributed=False always resolves to False even with dist initialized."""
        loss_fn = SmoothAPLoss(num_classes=1, queue_size=0, gather_distributed=False)
        logits = torch.randn(16, 1, requires_grad=True)
        targets = torch.randint(0, 2, (16,))
        loss_fn(logits, targets)
        assert loss_fn._gather_resolved is False

    def test_ap_gather_resolved_cached(self):
        """_gather_resolved is set on first forward and not re-evaluated."""
        loss_fn = SmoothAPLoss(num_classes=1, queue_size=0)
        logits = torch.randn(16, 1, requires_grad=True)
        targets = torch.randint(0, 2, (16,))
        loss_fn(logits, targets)
        first = loss_fn._gather_resolved
        loss_fn(logits.detach().requires_grad_(True), targets)
        assert loss_fn._gather_resolved is first  # same object / value

    def test_ap_output_matches_no_gather_at_world_size_1(self):
        """Auto-gather at world_size=1 is identical to no gather (same data)."""
        torch.manual_seed(0)
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        fn_auto = SmoothAPLoss(num_classes=1, queue_size=0)
        fn_off  = SmoothAPLoss(num_classes=1, queue_size=0, gather_distributed=False)
        loss_auto = fn_auto(logits, targets)
        loss_off  = fn_off(logits, targets)
        assert torch.allclose(loss_auto, loss_off)

    def test_ap_gradient_flows_with_gather_flag(self):
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss_fn = SmoothAPLoss(num_classes=1, queue_size=0)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    # --- RecallAtQuantileLoss -----------------------------------------------

    def test_recall_auto_resolves_false_at_world_size_1(self):
        loss_fn = RecallAtQuantileLoss(num_classes=1, queue_size=0, quantile=0.3)
        assert loss_fn._gather_resolved is None
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss_fn(logits, targets)
        assert loss_fn._gather_resolved is False

    def test_recall_explicit_false_resolves_false(self):
        loss_fn = RecallAtQuantileLoss(
            num_classes=1, queue_size=0, quantile=0.3, gather_distributed=False
        )
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss_fn(logits, targets)
        assert loss_fn._gather_resolved is False

    def test_recall_output_matches_no_gather_at_world_size_1(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        fn_auto = RecallAtQuantileLoss(num_classes=1, queue_size=0, quantile=0.3)
        fn_off  = RecallAtQuantileLoss(
            num_classes=1, queue_size=0, quantile=0.3, gather_distributed=False
        )
        assert torch.allclose(fn_auto(logits, targets), fn_off(logits, targets))

    def test_recall_gradient_flows_with_gather_flag(self):
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss_fn = RecallAtQuantileLoss(num_classes=1, queue_size=0, quantile=0.3)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# LossWarmupWrapper propagates gather_distributed to main_loss
# ---------------------------------------------------------------------------


class TestWrapperGatherDistributed:
    def _make_wrapper(self, gather_distributed=None):
        main = SmoothAPLoss(num_classes=1, queue_size=0)
        return LossWarmupWrapper(
            warmup_loss=nn.BCEWithLogitsLoss(),
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=100,
            gather_distributed=gather_distributed,
        )

    def test_default_none_propagated(self):
        wrapper = self._make_wrapper()
        assert wrapper.main_loss.gather_distributed is None

    def test_false_propagated(self):
        wrapper = self._make_wrapper(gather_distributed=False)
        assert wrapper.main_loss.gather_distributed is False

    def test_true_propagated(self):
        wrapper = self._make_wrapper(gather_distributed=True)
        assert wrapper.main_loss.gather_distributed is True

    def test_no_attr_on_custom_loss_does_not_raise(self):
        """Wrapper silently skips propagation if main_loss has no gather_distributed."""
        class CustomLoss(nn.Module):
            def forward(self, logits, targets):
                return logits.sum() * 0.0

        wrapper = LossWarmupWrapper(
            warmup_loss=nn.BCEWithLogitsLoss(),
            main_loss=CustomLoss(),
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=100,
            gather_distributed=False,
        )
        assert not hasattr(wrapper.main_loss, "gather_distributed")
