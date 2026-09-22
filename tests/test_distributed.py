"""
Unit tests for imbalanced_losses.distributed.

These tests cover single-process behaviour (world_size=1) and the guard
conditions. True multi-process all-gather is not tested here — that requires
a torchrun launcher and is validated by integration testing.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

import torch.nn as nn

from imbalanced_losses import (
    PAUCAtBudgetLoss,
    RecallAtQuantileLoss,
    SigmoidFocalLoss,
    SmoothAPLoss,
    SoftmaxFocalLoss,
)
from imbalanced_losses.distributed import all_gather_no_grad, all_gather_with_grad
from imbalanced_losses.warmup_wrapper import LossWarmupWrapper


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


# ---------------------------------------------------------------------------
# Variable-size gather (mocked multi-rank)
# ---------------------------------------------------------------------------


def _make_gather_mock(rank_tensors: list[torch.Tensor]):
    """
    Build a stateful side_effect for ``dist.all_gather`` that simulates
    multiple ranks.  Each call to the returned callable fills *output_list*
    with the pre-computed tensors for that round.

    The function is called twice per gather invocation:
      1. sizes gather  (1-element int64 tensors)
      2. data gather   (padded data tensors)

    *rank_tensors* are the **original, unpadded** tensors — the helper
    derives the size tensors and padded tensors internally.
    """
    sizes = [torch.tensor([t.size(0)], dtype=torch.int64) for t in rank_tensors]

    max_rows = max(t.size(0) for t in rank_tensors)
    padded = []
    for t in rank_tensors:
        if t.size(0) < max_rows:
            pad = torch.zeros(max_rows, *t.shape[1:], dtype=t.dtype)
            pad[: t.size(0)] = t
            padded.append(pad)
        else:
            padded.append(t.clone())

    call_idx = [0]

    def _side_effect(output_list, input_tensor):
        if call_idx[0] % 2 == 0:
            for i, s in enumerate(sizes):
                output_list[i].copy_(s)
        else:
            for i, p in enumerate(padded):
                output_list[i].copy_(p.detach())
        call_idx[0] += 1

    return _side_effect


class TestVariableSizeGather:
    """
    Test variable dim-0 gathering using mocked ``dist`` calls to simulate
    multi-rank scenarios without launching multiple processes.
    """

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    # -- all_gather_no_grad: variable sizes ---------------------------------

    def test_no_grad_variable_sizes(self):
        """3 ranks with sizes [4, 6, 2] → output has 12 rows, correct values."""
        from unittest.mock import patch

        t0 = torch.arange(8).reshape(4, 2).float()
        t1 = torch.arange(12).reshape(6, 2).float() + 100
        t2 = torch.arange(4).reshape(2, 2).float() + 200
        local_rank = 0
        mock = _make_gather_mock([t0, t1, t2])

        with patch.object(dist, "get_world_size", return_value=3), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_no_grad(t0)

        assert out.shape == (12, 2)
        expected = torch.cat([t0, t1, t2], dim=0)
        assert torch.equal(out, expected)

    def test_no_grad_equal_sizes(self):
        """3 ranks, equal sizes [4, 4, 4] → fast path, correct output."""
        from unittest.mock import patch

        t0 = torch.arange(8).reshape(4, 2).float()
        t1 = torch.arange(8).reshape(4, 2).float() + 100
        t2 = torch.arange(8).reshape(4, 2).float() + 200
        local_rank = 0
        mock = _make_gather_mock([t0, t1, t2])

        with patch.object(dist, "get_world_size", return_value=3), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_no_grad(t0)

        assert out.shape == (12, 2)
        expected = torch.cat([t0, t1, t2], dim=0)
        assert torch.equal(out, expected)

    def test_no_grad_1d_targets(self):
        """1D tensors (targets) with variable sizes."""
        from unittest.mock import patch

        t0 = torch.tensor([0, 1, 1])
        t1 = torch.tensor([0, 0, 1, 1, 0])
        local_rank = 0
        mock = _make_gather_mock([t0, t1])

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_no_grad(t0)

        assert out.shape == (8,)
        expected = torch.cat([t0, t1])
        assert torch.equal(out, expected)

    def test_no_grad_zero_rows_one_rank(self):
        """One rank contributes 0 rows — no crash, output is other rank's data."""
        from unittest.mock import patch

        t0 = torch.zeros(0, 3).float()
        t1 = torch.randn(5, 3)
        local_rank = 0
        mock = _make_gather_mock([t0, t1])

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_no_grad(t0)

        assert out.shape == (5, 3)
        assert torch.equal(out, t1)

    # -- all_gather_with_grad: variable sizes -------------------------------

    def test_with_grad_variable_sizes(self):
        """3 ranks, variable sizes, gradient flows only to local rank."""
        from unittest.mock import patch

        t0 = torch.randn(4, 2, requires_grad=True)
        t1 = torch.randn(6, 2)
        t2 = torch.randn(2, 2)
        local_rank = 0
        mock = _make_gather_mock([t0, t1, t2])

        with patch.object(dist, "get_world_size", return_value=3), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_with_grad(t0)

        assert out.shape == (12, 2)
        out.sum().backward()
        assert t0.grad is not None
        assert t0.grad.shape == (4, 2)

    def test_with_grad_equal_sizes(self):
        """3 ranks, equal sizes, fast path preserves gradient."""
        from unittest.mock import patch

        t0 = torch.randn(4, 2, requires_grad=True)
        t1 = torch.randn(4, 2)
        t2 = torch.randn(4, 2)
        local_rank = 0
        mock = _make_gather_mock([t0, t1, t2])

        with patch.object(dist, "get_world_size", return_value=3), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_with_grad(t0)

        assert out.shape == (12, 2)
        out.sum().backward()
        assert t0.grad is not None
        assert t0.grad.shape == (4, 2)

    def test_with_grad_zero_rows_local_rank(self):
        """Local rank has 0 rows — backward succeeds with empty gradient."""
        from unittest.mock import patch

        t0 = torch.zeros(0, 3, requires_grad=True)
        t1 = torch.randn(5, 3)
        local_rank = 0
        mock = _make_gather_mock([t0, t1])

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            out = all_gather_with_grad(t0)

        assert out.shape == (5, 3)
        out.sum().backward()
        assert t0.grad is not None
        assert t0.grad.shape == (0, 3)


# ---------------------------------------------------------------------------
# Task 5B: PAUCAtBudgetLoss iid_mask DDP-gather
#
# Verify that under a mocked multi-rank process group, iid_mask is gathered
# aligned with logits/targets and that FPR thresholds are driven only by the
# gathered iid negatives (not non-iid rows from other ranks).
#
# Two paths tested:
#  1. Equal-size fast path (both ranks have the same dim-0 size)
#  2. Variable dim-0 path (unequal rows per rank)
# ---------------------------------------------------------------------------


def _make_group_gather_mock(groups: list[list[torch.Tensor]]):
    """
    Build a side_effect for ``dist.all_gather`` that serves one gathered
    tensor per *group*, in order, across the two calls each gather makes.

    ``distributed.py`` gathers every tensor with two collectives: first the
    per-rank dim-0 sizes (int64), then the (zero-padded, when sizes differ)
    data. A forward that gathers G tensors therefore issues 2G calls, and
    this mock routes them by call index::

        group = call_idx // 2   (which tensor: groups[0], groups[1], ...)
        phase = call_idx %  2   (0 = sizes call, 1 = data call)

    Each element of *groups* is the list of per-rank tensors for one
    gathered tensor, in rank order; they must be listed in the same order
    the loss gathers them. The equal-size fast path skips padding in the
    data call but still issues the sizes call, so the routing holds for
    both paths.
    """
    prepared = []
    for tensors in groups:
        sizes = [torch.tensor([t.size(0)], dtype=torch.int64) for t in tensors]
        max_rows = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            if t.size(0) < max_rows:
                pad = torch.zeros(max_rows, *t.shape[1:], dtype=t.dtype)
                pad[: t.size(0)] = t
                padded.append(pad)
            else:
                padded.append(t.clone())
        prepared.append((sizes, padded))

    call_idx = [0]

    def _side_effect(output_list, input_tensor):
        sizes, padded = prepared[(call_idx[0] // 2) % len(prepared)]
        if call_idx[0] % 2 == 0:
            for i, size in enumerate(sizes):
                output_list[i].copy_(size)
        else:
            for i, tensor in enumerate(padded):
                output_list[i].copy_(tensor.detach())
        call_idx[0] += 1

    return _side_effect


def _make_pauc_gather_mock(
    logits_ranks: list[torch.Tensor],
    targets_ranks: list[torch.Tensor],
    iid_ranks: list[torch.Tensor],
    weight_ranks: list[torch.Tensor] | None = None,
):
    """
    Gather mock for ``_QueuedRankingLoss.forward``'s tensor order:
    logits, targets, iid_mask (transported as uint8), and -- only when
    *weight_ranks* is supplied -- sample_weight, in the position it
    occupies right after iid_mask. That makes 6 collectives on an
    unweighted step and 8 on a weighted one.
    """
    groups = [logits_ranks, targets_ranks, [m.to(torch.uint8) for m in iid_ranks]]
    if weight_ranks is not None:
        groups.append(weight_ranks)
    return _make_group_gather_mock(groups)


def _make_focal_gather_mock(
    inputs_ranks: list[torch.Tensor],
    targets_ranks: list[torch.Tensor],
    weight_ranks: list[torch.Tensor] | None = None,
):
    """
    Gather mock for the focal losses' tensor order: inputs, targets, and --
    only when *weight_ranks* is supplied -- sample_weight. That makes 4
    collectives on an unweighted step and 6 on a weighted one.
    """
    groups = [inputs_ranks, targets_ranks]
    if weight_ranks is not None:
        groups.append(weight_ranks)
    return _make_group_gather_mock(groups)


class TestPAUCGatherDistributed:
    """
    Verify PAUCAtBudgetLoss gathers iid_mask alongside logits/targets and
    that thresholds are driven by the globally gathered iid negatives only.
    """

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    def _build_rank_data(self, seed_base=100):
        """
        Two-rank setup:
          Rank 0: 20 rows -- 4 positives (class 1), 16 iid negatives with
                  scores in [-1, 1].
          Rank 1: 20 rows -- 4 positives (class 1), 12 iid negatives in
                  [-1, 1], and 4 NON-iid negatives at score +8 (which would
                  inflate t_alpha to ~8 if erroneously included).
        Concatenated global batch has 40 rows.
        """
        g = torch.Generator().manual_seed(seed_base)

        # Rank 0: all iid.
        r0_neg = torch.rand(16, 1, generator=g) * 2.0 - 1.0
        r0_pos = torch.randn(4, 1, generator=g) + 3.0
        r0_logits = torch.cat([r0_neg, r0_pos], dim=0)
        r0_targets = torch.cat([
            torch.zeros(16, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        ])
        r0_iid = torch.ones(20, dtype=torch.bool)

        # Rank 1: 12 iid negatives + 4 non-iid negatives at +8.
        r1_iid_neg = torch.rand(12, 1, generator=g) * 2.0 - 1.0
        r1_non_iid = torch.full((4, 1), 8.0)
        r1_pos = torch.randn(4, 1, generator=g) + 3.0
        r1_logits = torch.cat([r1_iid_neg, r1_non_iid, r1_pos], dim=0)
        r1_targets = torch.cat([
            torch.zeros(16, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        ])
        r1_iid = torch.cat([
            torch.ones(12, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),   # non-iid
            torch.ones(4, dtype=torch.bool),
        ])

        return (r0_logits, r0_targets, r0_iid), (r1_logits, r1_targets, r1_iid)

    def test_iid_mask_gathered_equal_size_path(self):
        """
        Equal-size fast path (both ranks have 20 rows): iid_mask is gathered
        as uint8 and the resulting thresholds are within the iid-negative range
        (not inflated by the non-iid negatives at +8 on rank 1).
        """
        from unittest.mock import patch

        (r0_l, r0_t, r0_iid), (r1_l, r1_t, r1_iid) = self._build_rank_data()
        assert r0_l.size(0) == r1_l.size(0), "both ranks must be equal size for this test"

        local_rank = 0
        mock = _make_pauc_gather_mock(
            [r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid]
        )

        loss_fn = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.05, beta=0.5,
            surrogate="trapezoid", queue_size=0,
            gather_distributed=True,
            quantile_interpolation="linear",
        )
        # Force gather resolution to True (normally cached on first real forward).
        loss_fn._gather_resolved = True

        r0_l_grad = r0_l.detach().requires_grad_(True)

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            loss, stats = loss_fn(r0_l_grad, r0_t, iid_mask=r0_iid, return_diagnostics=True)

        assert torch.isfinite(loss), "loss should be finite after DDP gather"
        loss.backward()
        assert r0_l_grad.grad is not None, "gradient should flow to local-rank logits"
        assert torch.isfinite(r0_l_grad.grad).all(), "gradient should be finite"

        # t_alpha must not be near +8 (the non-iid score).  Since iid negatives
        # are in [-1, 1], t_alpha = quantile(iid_neg, 1 - 0.05) should be < 2.
        t_alpha = stats["t_alpha"][0].item()
        assert torch.isfinite(stats["t_alpha"][0]), "t_alpha should be finite"
        assert t_alpha < 3.0, (
            f"t_alpha={t_alpha:.3f} is unexpectedly large; suggests non-iid negatives "
            "at +8 contaminated threshold estimation after DDP gather."
        )

    def test_iid_mask_gathered_variable_size_path(self):
        """
        Variable dim-0 path (ranks have different row counts): iid_mask is
        still gathered correctly and thresholds are not contaminated by
        non-iid negatives.
        """
        from unittest.mock import patch

        g = torch.Generator().manual_seed(200)

        # Rank 0: 15 rows -- 3 positives, 12 iid negatives in [-1, 1].
        r0_neg = torch.rand(12, 1, generator=g) * 2.0 - 1.0
        r0_pos = torch.randn(3, 1, generator=g) + 3.0
        r0_l = torch.cat([r0_neg, r0_pos], dim=0)   # [15, 1]
        r0_t = torch.cat([torch.zeros(12, dtype=torch.long), torch.ones(3, dtype=torch.long)])
        r0_iid = torch.ones(15, dtype=torch.bool)

        # Rank 1: 25 rows -- 5 positives, 16 iid negatives in [-1, 1],
        # 4 non-iid negatives at +8.
        r1_iid_neg = torch.rand(16, 1, generator=g) * 2.0 - 1.0
        r1_non_iid = torch.full((4, 1), 8.0)
        r1_pos = torch.randn(5, 1, generator=g) + 3.0
        r1_l = torch.cat([r1_iid_neg, r1_non_iid, r1_pos], dim=0)  # [25, 1]
        r1_t = torch.cat([
            torch.zeros(20, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
        ])
        r1_iid = torch.cat([
            torch.ones(16, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            torch.ones(5, dtype=torch.bool),
        ])

        assert r0_l.size(0) != r1_l.size(0), "ranks must have different sizes for this test"

        local_rank = 0
        mock = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid])

        loss_fn = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.05, beta=0.5,
            surrogate="trapezoid", queue_size=0,
            gather_distributed=True,
            quantile_interpolation="linear",
        )
        loss_fn._gather_resolved = True

        r0_l_grad = r0_l.detach().requires_grad_(True)

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock):
            loss, stats = loss_fn(r0_l_grad, r0_t, iid_mask=r0_iid, return_diagnostics=True)

        assert torch.isfinite(loss), "loss should be finite after variable-size DDP gather"
        loss.backward()
        assert r0_l_grad.grad is not None
        assert torch.isfinite(r0_l_grad.grad).all()

        # Thresholds must not be near +8.
        t_alpha = stats["t_alpha"][0].item()
        assert torch.isfinite(stats["t_alpha"][0])
        assert t_alpha < 3.0, (
            f"t_alpha={t_alpha:.3f} after variable-size gather; suggests non-iid "
            "contamination of thresholds."
        )

    def test_pauc_auto_resolves_false_at_world_size_1(self):
        """Auto-detect: world_size=1 → _gather_resolved becomes False after first forward."""
        loss_fn = PAUCAtBudgetLoss(num_classes=1, alpha=0.1, beta=0.5, queue_size=0)
        assert loss_fn._gather_resolved is None
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss_fn(logits, targets)
        assert loss_fn._gather_resolved is False

    def test_pauc_explicit_false_resolves_false(self):
        """gather_distributed=False always resolves to False."""
        loss_fn = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.1, beta=0.5, queue_size=0, gather_distributed=False
        )
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss_fn(logits, targets)
        assert loss_fn._gather_resolved is False

    def test_pauc_output_matches_no_gather_at_world_size_1(self):
        """Auto-gather at world_size=1 is identical to no-gather (same data)."""
        torch.manual_seed(0)
        logits = torch.randn(32, 1, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        fn_auto = PAUCAtBudgetLoss(num_classes=1, alpha=0.1, beta=0.5, queue_size=0)
        fn_off = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.1, beta=0.5, queue_size=0, gather_distributed=False
        )
        assert torch.allclose(fn_auto(logits, targets), fn_off(logits, targets))


# ---------------------------------------------------------------------------
# Task 6: sample_weight DDP-gather
#
# Verify that sample_weight rides the 4th collective group (after logits,
# targets, iid_mask), that the collective count is exactly 8 on a weighted
# step and 6 on an unweighted one, and that a 3-rank unequal-size weighted
# forward (including a zero-row rank whose [0] weight pads/trims like any
# other tensor) produces the same loss as running the concatenated pool
# through a single, non-gathering instance -- the gather is pure transport.
# ---------------------------------------------------------------------------


class TestPAUCGatherWeightedDistributed:
    """
    Verify PAUCAtBudgetLoss gathers sample_weight alongside logits/targets/
    iid_mask, with the exact collective count the spec commits to (spec
    section 9: 'with weights supplied, the multi-rank mock sees exactly one
    additional no-grad gather in the iid_mask position ... with no weights,
    the collective count is unchanged').
    """

    NUM_CLASSES = 1

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    def _loss_kwargs(self):
        return dict(
            num_classes=self.NUM_CLASSES, alpha=0.1, beta=0.5,
            surrogate="trapezoid", queue_size=0,
            gather_distributed=True, quantile_interpolation="linear",
        )

    def _build_two_rank_data(self, seed):
        """Two 20-row iid ranks (16 negatives + 4 positives each), weighted."""
        g = torch.Generator().manual_seed(seed)

        def _one_rank():
            neg = torch.rand(16, 1, generator=g) * 2.0 - 1.0
            pos = torch.randn(4, 1, generator=g) + 3.0
            logits = torch.cat([neg, pos], dim=0)
            targets = torch.cat([
                torch.zeros(16, dtype=torch.long), torch.ones(4, dtype=torch.long)
            ])
            iid = torch.ones(20, dtype=torch.bool)
            weight = torch.rand(20, generator=g) * 2.0 + 0.5
            return logits, targets, iid, weight

        return _one_rank(), _one_rank()

    def test_weighted_step_issues_eight_collectives(self):
        """A weighted step gathers logits, targets, iid_mask, and weight (8 calls)."""
        from unittest.mock import patch

        (r0_l, r0_t, r0_iid, r0_w), (r1_l, r1_t, r1_iid, r1_w) = self._build_two_rank_data(400)
        mock = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid], [r0_w, r1_w])

        loss_fn = PAUCAtBudgetLoss(**self._loss_kwargs())
        loss_fn._gather_resolved = True

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock) as mock_gather:
            loss_fn(r0_l.detach().requires_grad_(True), r0_t, iid_mask=r0_iid, sample_weight=r0_w)

        assert mock_gather.call_count == 8, (
            f"expected 8 collectives (logits, targets, iid_mask, weight x "
            f"sizes+data) on a weighted step, got {mock_gather.call_count}. "
            f"A wrong count here silently desyncs the mock's group routing "
            f"(group_idx % len(groups)) rather than failing loudly."
        )

    def test_unweighted_step_issues_six_collectives(self):
        """An unweighted step issues no additional collective (6 calls, unchanged)."""
        from unittest.mock import patch

        (r0_l, r0_t, r0_iid, _), (r1_l, r1_t, r1_iid, _) = self._build_two_rank_data(401)
        mock = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid])

        loss_fn = PAUCAtBudgetLoss(**self._loss_kwargs())
        loss_fn._gather_resolved = True

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock) as mock_gather:
            loss_fn(r0_l.detach().requires_grad_(True), r0_t, iid_mask=r0_iid)

        assert mock_gather.call_count == 6, (
            f"expected 6 collectives (logits, targets, iid_mask x sizes+data) "
            f"on an unweighted step, got {mock_gather.call_count}. A wrong "
            f"count here silently desyncs the mock's group routing "
            f"(group_idx % len(groups)) rather than failing loudly."
        )

    def test_three_rank_unequal_size_weighted_matches_concatenated_pool(self):
        """
        3 ranks with unequal sizes -- including a zero-row rank -- pad and
        trim the weight correctly, and the DDP-gathered weighted loss equals
        the same loss computed directly on the manually concatenated pool.
        Gradient flows only to the local rank's (rank 0) logits.
        """
        from unittest.mock import patch

        (r0_l, r0_t, r0_iid, r0_w), (r1_l, r1_t, r1_iid, r1_w) = self._build_two_rank_data(402)

        # Rank 2 contributes zero rows -- an explicitly supported DDP shape
        # (e.g. the last, unequal batch of an epoch). Its [0] weight must
        # pad to the other ranks' size for the collective and trim back to
        # empty, same as its [0] logits/targets/iid_mask.
        r2_l = torch.zeros(0, self.NUM_CLASSES)
        r2_t = torch.zeros(0, dtype=torch.long)
        r2_iid = torch.zeros(0, dtype=torch.bool)
        r2_w = torch.zeros(0)

        logits_ranks = [r0_l, r1_l, r2_l]
        targets_ranks = [r0_t, r1_t, r2_t]
        iid_ranks = [r0_iid, r1_iid, r2_iid]
        weight_ranks = [r0_w, r1_w, r2_w]
        assert len({t.size(0) for t in logits_ranks}) > 1, "ranks must have unequal sizes"

        loss_kwargs = self._loss_kwargs()

        # -- DDP path: rank 0's local view, gathered against ranks 1 and 2. --
        mock = _make_pauc_gather_mock(logits_ranks, targets_ranks, iid_ranks, weight_ranks)
        loss_ddp = PAUCAtBudgetLoss(**loss_kwargs)
        loss_ddp._gather_resolved = True

        r0_l_grad = r0_l.detach().requires_grad_(True)
        with patch.object(dist, "get_world_size", return_value=3), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock):
            loss_from_ddp = loss_ddp(r0_l_grad, r0_t, iid_mask=r0_iid, sample_weight=r0_w)

        assert torch.isfinite(loss_from_ddp), "loss should be finite after weighted DDP gather"
        loss_from_ddp.backward()
        assert r0_l_grad.grad is not None, "gradient should flow to local-rank logits"
        assert r0_l_grad.grad.shape == r0_l.shape
        assert torch.isfinite(r0_l_grad.grad).all()

        # -- Single-process reference: the same rows, already concatenated in
        # rank order, run through the identical loss with gathering disabled.
        pool_logits = torch.cat(logits_ranks, dim=0)
        pool_targets = torch.cat(targets_ranks, dim=0)
        pool_iid = torch.cat(iid_ranks, dim=0)
        pool_weight = torch.cat(weight_ranks, dim=0)

        loss_pool_fn = PAUCAtBudgetLoss(**{**loss_kwargs, "gather_distributed": False})
        loss_from_pool = loss_pool_fn(
            pool_logits, pool_targets, iid_mask=pool_iid, sample_weight=pool_weight
        )

        assert torch.equal(loss_from_ddp.detach(), loss_from_pool), (
            f"DDP-gathered weighted loss ({loss_from_ddp.item():.8f}) must equal the "
            f"loss computed directly on the concatenated single-process pool "
            f"({loss_from_pool.item():.8f}) -- the gather is pure transport and must "
            f"not change the weighted arithmetic."
        )


# ---------------------------------------------------------------------------
# Focal-loss sample_weight under mocked multi-rank DDP gather
#
# At world_size == 1 the gather helpers return their input unchanged, so a
# single-process replay cannot tell whether the weight was gathered at all.
# These tests run against a mocked multi-rank group, where dropping the
# weight gather leaves an ungathered [N] weight against a gathered
# [N x world_size] loss.
# ---------------------------------------------------------------------------


class TestFocalGatherWeightedDistributed:
    """
    Verify both focal losses gather ``sample_weight`` alongside
    ``inputs``/``targets``, with the exact collective count (spec section 9:
    a weighted step adds one no-grad gather; an unweighted step's count is
    unchanged) and correct alignment across unequal per-rank batch sizes.
    """

    C = 3

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    def _sigmoid_rank(self, gen, n):
        inputs = torch.randn(n, self.C, generator=gen)
        targets = torch.randint(0, 2, (n, self.C), generator=gen).float()
        # Per-element weight: SigmoidFocalLoss scores every logit
        # independently, so the weight matches `inputs`, not the row count.
        weight = torch.rand(n, self.C, generator=gen) * 2.0 + 0.5
        return inputs, targets, weight

    def _softmax_rank(self, gen, n):
        inputs = torch.randn(n, self.C, generator=gen)
        targets = torch.randint(0, self.C, (n,), generator=gen)
        weight = torch.rand(n, generator=gen) * 2.0 + 0.5
        return inputs, targets, weight

    def _rank_builder(self, family):
        return self._sigmoid_rank if family == "sigmoid" else self._softmax_rank

    def _loss(self, family, **kwargs):
        cls = SigmoidFocalLoss if family == "sigmoid" else SoftmaxFocalLoss
        loss_fn = cls(reduction="mean", **kwargs)
        return loss_fn

    @pytest.mark.parametrize("family", ["sigmoid", "softmax"])
    def test_weighted_step_issues_six_collectives(self, family):
        """inputs, targets, and weight x (sizes + data)."""
        from unittest.mock import patch

        gen = torch.Generator().manual_seed(500)
        build = self._rank_builder(family)
        r0_x, r0_t, r0_w = build(gen, 12)
        r1_x, r1_t, r1_w = build(gen, 12)
        mock = _make_focal_gather_mock([r0_x, r1_x], [r0_t, r1_t], [r0_w, r1_w])

        loss_fn = self._loss(family, gather_distributed=True)
        loss_fn._gather_resolved = True

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock) as mock_gather:
            loss_fn(r0_x.detach().requires_grad_(True), r0_t, sample_weight=r0_w)

        assert mock_gather.call_count == 6, (
            f"expected 6 collectives (inputs, targets, weight x sizes+data) on "
            f"a weighted focal step, got {mock_gather.call_count}. A missing "
            f"weight gather leaves the weight misaligned against the gathered "
            f"batch on every real multi-rank step."
        )

    @pytest.mark.parametrize("family", ["sigmoid", "softmax"])
    def test_unweighted_step_issues_four_collectives(self, family):
        """An unweighted step issues no additional collective."""
        from unittest.mock import patch

        gen = torch.Generator().manual_seed(501)
        build = self._rank_builder(family)
        r0_x, r0_t, _ = build(gen, 12)
        r1_x, r1_t, _ = build(gen, 12)
        mock = _make_focal_gather_mock([r0_x, r1_x], [r0_t, r1_t])

        loss_fn = self._loss(family, gather_distributed=True)
        loss_fn._gather_resolved = True

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock) as mock_gather:
            loss_fn(r0_x.detach().requires_grad_(True), r0_t)

        assert mock_gather.call_count == 4, (
            f"expected 4 collectives (inputs, targets x sizes+data) on an "
            f"unweighted focal step, got {mock_gather.call_count}."
        )

    @pytest.mark.parametrize("family", ["sigmoid", "softmax"])
    def test_three_rank_unequal_size_weighted_matches_concatenated_pool(self, family):
        """
        3 ranks with unequal sizes -- including a zero-row rank -- pad and
        trim the weight correctly, and the DDP-gathered weighted loss equals
        the same loss computed directly on the manually concatenated pool.
        """
        from unittest.mock import patch

        gen = torch.Generator().manual_seed(502)
        build = self._rank_builder(family)
        r0_x, r0_t, r0_w = build(gen, 12)
        r1_x, r1_t, r1_w = build(gen, 7)
        # A zero-row rank is an explicitly supported DDP shape (the last,
        # unequal batch of an epoch).
        r2_x, r2_t, r2_w = build(gen, 0)

        inputs_ranks = [r0_x, r1_x, r2_x]
        targets_ranks = [r0_t, r1_t, r2_t]
        weight_ranks = [r0_w, r1_w, r2_w]
        assert len({t.size(0) for t in inputs_ranks}) > 1, "ranks must have unequal sizes"

        mock = _make_focal_gather_mock(inputs_ranks, targets_ranks, weight_ranks)
        loss_ddp = self._loss(family, gather_distributed=True)
        loss_ddp._gather_resolved = True

        r0_x_grad = r0_x.detach().requires_grad_(True)
        with patch.object(dist, "get_world_size", return_value=3), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock):
            loss_from_ddp = loss_ddp(r0_x_grad, r0_t, sample_weight=r0_w)

        assert torch.isfinite(loss_from_ddp)
        loss_from_ddp.backward()
        assert r0_x_grad.grad is not None, "gradient should flow to local-rank inputs"
        assert r0_x_grad.grad.shape == r0_x.shape

        loss_pool_fn = self._loss(family, gather_distributed=False)
        loss_from_pool = loss_pool_fn(
            torch.cat(inputs_ranks, dim=0),
            torch.cat(targets_ranks, dim=0),
            sample_weight=torch.cat(weight_ranks, dim=0),
        )
        assert torch.equal(loss_from_ddp.detach(), loss_from_pool), (
            f"DDP-gathered weighted focal loss ({loss_from_ddp.item():.8f}) must "
            f"equal the loss on the concatenated single-process pool "
            f"({loss_from_pool.item():.8f}) -- the gather is pure transport."
        )

    def test_sigmoid_trailing_broadcast_weight_survives_the_gather(self):
        """
        A ``[N, 1]`` weight (broadcast over the channel dim) is gathered on
        dim 0 like any other, so it still lines up with the gathered inputs.
        This is the shape the narrowed dim-0 contract still accepts.
        """
        from unittest.mock import patch

        gen = torch.Generator().manual_seed(503)
        r0_x, r0_t, _ = self._sigmoid_rank(gen, 9)
        r1_x, r1_t, _ = self._sigmoid_rank(gen, 5)
        r0_w = torch.rand(9, 1, generator=gen) + 0.1
        r1_w = torch.rand(5, 1, generator=gen) + 0.1

        mock = _make_focal_gather_mock([r0_x, r1_x], [r0_t, r1_t], [r0_w, r1_w])
        loss_ddp = SigmoidFocalLoss(reduction="mean", gather_distributed=True)
        loss_ddp._gather_resolved = True

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "all_gather", side_effect=mock) as mock_gather:
            loss_from_ddp = loss_ddp(r0_x.detach().requires_grad_(True), r0_t,
                                     sample_weight=r0_w)

        assert mock_gather.call_count == 6
        loss_pool = SigmoidFocalLoss(reduction="mean", gather_distributed=False)(
            torch.cat([r0_x, r1_x], dim=0),
            torch.cat([r0_t, r1_t], dim=0),
            sample_weight=torch.cat([r0_w, r1_w], dim=0),
        )
        assert torch.equal(loss_from_ddp.detach(), loss_pool)


# ---------------------------------------------------------------------------
# is_live plumbing under mocked DDP gather (pos_numerator="live")
#
# Verifies that after a mocked multi-rank gather:
#   - All gathered live rows (from both ranks) have is_live=True.
#   - Queue rows (appended post-gather by the merge) have is_live=False.
#   - The loss runs and backward flows correctly.
# ---------------------------------------------------------------------------


class TestPAUCIsLiveDistributed:
    """
    Verify that is_live is correctly built post-gather:
    all gathered rows are live (is_live=True), queue rows are not.
    """

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    def _build_two_rank_data(self, seed=300):
        """
        Two-rank setup: each rank has 20 rows with some positives and negatives.
        """
        g = torch.Generator().manual_seed(seed)
        r0_neg = torch.rand(16, 1, generator=g) * 2.0 - 1.0
        r0_pos = torch.randn(4, 1, generator=g) + 3.0
        r0_l = torch.cat([r0_neg, r0_pos], dim=0)
        r0_t = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(4, dtype=torch.long)])
        r0_iid = torch.ones(20, dtype=torch.bool)

        r1_neg = torch.rand(16, 1, generator=g) * 2.0 - 1.0
        r1_pos = torch.randn(4, 1, generator=g) + 3.0
        r1_l = torch.cat([r1_neg, r1_pos], dim=0)
        r1_t = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(4, dtype=torch.long)])
        r1_iid = torch.ones(20, dtype=torch.bool)

        return (r0_l, r0_t, r0_iid), (r1_l, r1_t, r1_iid)

    def test_is_live_all_true_for_gathered_rows_no_queue(self):
        """
        With queue_size=0, all rows in the pool come from the gather (live).
        pos_numerator='live' must agree with 'pool' since every row is live.
        Both must run forward/backward cleanly.
        """
        from unittest.mock import patch

        (r0_l, r0_t, r0_iid), (r1_l, r1_t, r1_iid) = self._build_two_rank_data()
        local_rank = 0

        fn_pool = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.1, beta=0.5,
            surrogate="trapezoid", queue_size=0,
            gather_distributed=True,
            pos_numerator="pool",
        )
        fn_live = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.1, beta=0.5,
            surrogate="trapezoid", queue_size=0,
            gather_distributed=True,
            pos_numerator="live",
        )
        fn_pool._gather_resolved = True
        fn_live._gather_resolved = True

        r0_l_p = r0_l.detach().requires_grad_(True)
        r0_l_lv = r0_l.detach().requires_grad_(True)

        # Two forwards need two separate mock call sequences.
        mock_pool = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid])
        mock_live = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid])

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock_pool):
            loss_pool = fn_pool(r0_l_p, r0_t, iid_mask=r0_iid)

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock_live):
            loss_live = fn_live(r0_l_lv, r0_t, iid_mask=r0_iid)

        assert torch.isfinite(loss_pool), "pool loss must be finite"
        assert torch.isfinite(loss_live), "live loss must be finite"

        # With no queue every row is live: pool and live must agree.
        assert torch.equal(loss_pool, loss_live), (
            f"pool={loss_pool.item():.6f} and live={loss_live.item():.6f} "
            "must be equal when queue_size=0 (all rows are live)"
        )

        loss_pool.backward()
        loss_live.backward()
        assert r0_l_p.grad is not None and torch.isfinite(r0_l_p.grad).all()
        assert r0_l_lv.grad is not None and torch.isfinite(r0_l_lv.grad).all()

    def test_is_live_false_for_queue_rows_with_populated_queue(self):
        """
        With a populated queue, gathered rows are live and queue rows are not.
        Under pos_numerator='live', queue positives (enqueued before this step)
        must NOT be in the numerator: the loss must differ from 'pool'.

        This verifies the is_live plumbing is correct post-merge.
        """
        from unittest.mock import patch

        g = torch.Generator().manual_seed(301)

        (r0_l, r0_t, r0_iid), (r1_l, r1_t, r1_iid) = self._build_two_rank_data(seed=302)

        # Pre-seed the queue with positives at HIGH scores (+10) -- these are
        # queue positives that should be excluded from the numerator under "live".
        q_neg = torch.randn(48, 1, generator=g)
        q_pos = torch.full((16, 1), 10.0)
        q_log = torch.cat([q_neg, q_pos], dim=0)
        q_tgt = torch.cat([torch.zeros(48), torch.ones(16)]).long()

        fn_pool = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.1, beta=0.5,
            surrogate="trapezoid", queue_size=64,
            gather_distributed=True,
            pos_numerator="pool",
            temperature=0.5,
        )
        fn_live = PAUCAtBudgetLoss(
            num_classes=1, alpha=0.1, beta=0.5,
            surrogate="trapezoid", queue_size=64,
            gather_distributed=True,
            pos_numerator="live",
            temperature=0.5,
        )
        fn_pool._gather_resolved = True
        fn_live._gather_resolved = True

        with torch.no_grad():
            fn_pool._queue.enqueue(q_log, q_tgt)
            fn_live._queue.enqueue(q_log, q_tgt)

        mock_pool = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid])
        mock_live = _make_pauc_gather_mock([r0_l, r1_l], [r0_t, r1_t], [r0_iid, r1_iid])

        local_rank = 0
        r0_l_p  = r0_l.detach().requires_grad_(True)
        r0_l_lv = r0_l.detach().requires_grad_(True)

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock_pool):
            loss_pool = fn_pool(r0_l_p, r0_t, iid_mask=r0_iid)

        with patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_rank", return_value=local_rank), \
             patch.object(dist, "all_gather", side_effect=mock_live):
            loss_live = fn_live(r0_l_lv, r0_t, iid_mask=r0_iid)

        assert torch.isfinite(loss_pool), "pool loss must be finite"
        assert torch.isfinite(loss_live), "live loss must be finite"

        loss_pool.backward()
        loss_live.backward()
        assert r0_l_p.grad is not None and torch.isfinite(r0_l_p.grad).all()
        assert r0_l_lv.grad is not None and torch.isfinite(r0_l_lv.grad).all()

        # With queue positives at +10 (above band), live restricts numerator
        # to gathered positives only.  Losses MUST differ.
        assert not torch.equal(loss_pool, loss_live), (
            "pool and live must differ when queue holds high-score positives: "
            f"pool={loss_pool.item():.6f}, live={loss_live.item():.6f}"
        )
