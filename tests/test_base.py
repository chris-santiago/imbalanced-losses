"""
Behavior shared by every queued ranking loss (``_QueuedRankingLoss``).

``RANKING_LOSSES`` is the module-level parametrization over the three
concrete losses; later tests in this file reuse it.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from imbalanced_losses import PAUCAtBudgetLoss, RecallAtQuantileLoss, SmoothAPLoss
from imbalanced_losses._base import _to_compute_dtype

# (class, constructor kwargs). PAUC uses a wide band so a small batch has
# a non-degenerate negative band.
RANKING_LOSS_SPECS = {
    "smooth_ap": (SmoothAPLoss, {}),
    "recall_at_quantile": (RecallAtQuantileLoss, {"quantile": 0.2}),
    "pauc_at_budget": (PAUCAtBudgetLoss, {"beta": 0.5}),
}
RANKING_LOSSES = pytest.mark.parametrize(
    "loss_name", list(RANKING_LOSS_SPECS), ids=list(RANKING_LOSS_SPECS)
)
HALF_DTYPES = pytest.mark.parametrize(
    "half", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"]
)
QUEUE_SIZES = pytest.mark.parametrize("queue_size", [0, 64])
NUM_CLASSES = pytest.mark.parametrize("num_classes", [1, 3])

N = 96


def make_loss(loss_name: str, num_classes: int, queue_size: int, **kwargs):
    cls, base_kwargs = RANKING_LOSS_SPECS[loss_name]
    return cls(
        num_classes=num_classes, queue_size=queue_size, **{**base_kwargs, **kwargs}
    )


def make_batch(num_classes: int, seed: int, half: torch.dtype | None = None):
    """Return (logits, targets); logits are rounded to ``half`` when given."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, num_classes, generator=g)
    if num_classes == 1:
        targets = (torch.rand(N, generator=g) < 0.4).long()
    else:
        targets = torch.arange(N) % num_classes
    if half is not None:
        logits = logits.to(half)
    return logits, targets


def _leaf(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


@pytest.fixture
def single_process_group():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29503",
            world_size=1,
            rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class TestHalfPrecisionLogits:
    @RANKING_LOSSES
    @HALF_DTYPES
    @QUEUE_SIZES
    @NUM_CLASSES
    def test_half_equals_float32_cast_over_two_steps(
        self, loss_name, half, queue_size, num_classes
    ):
        loss_h = make_loss(loss_name, num_classes, queue_size)
        loss_f = make_loss(loss_name, num_classes, queue_size)
        for seed in (0, 1):
            logits_h, targets = make_batch(num_classes, seed, half)
            leaf = _leaf(logits_h)
            out = loss_h(leaf, targets)
            ref = loss_f(logits_h.float(), targets)
            assert out.dtype == torch.float32
            assert torch.equal(out, ref)
            out.backward()
            assert leaf.grad is not None
            assert leaf.grad.dtype == half
            assert torch.isfinite(leaf.grad).all()
            assert leaf.grad.abs().sum() > 0

    @RANKING_LOSSES
    @HALF_DTYPES
    @QUEUE_SIZES
    @NUM_CLASSES
    def test_reduction_none_is_float32_per_class(
        self, loss_name, half, queue_size, num_classes
    ):
        loss_h = make_loss(loss_name, num_classes, queue_size, reduction="none")
        loss_f = make_loss(loss_name, num_classes, queue_size, reduction="none")
        logits_h, targets = make_batch(num_classes, 0, half)
        out = loss_h(logits_h, targets)
        ref = loss_f(logits_h.float(), targets)
        assert out.dtype == torch.float32
        assert out.shape == (num_classes,)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)

    @RANKING_LOSSES
    @pytest.mark.parametrize(
        "weight_dtype", [torch.float16, torch.float64], ids=["w_float16", "w_float64"]
    )
    def test_sample_weight_follows_compute_dtype(self, loss_name, weight_dtype):
        # Non-uniform weights, so a weight dropped or applied in the wrong
        # dtype would change the value relative to the unweighted run.
        loss_h = make_loss(loss_name, 1, 64)
        loss_f = make_loss(loss_name, 1, 64)
        loss_u = make_loss(loss_name, 1, 64)
        logits_h, targets = make_batch(1, 0, torch.float16)
        weight = torch.linspace(0.25, 4.0, N).to(weight_dtype)
        out = loss_h(logits_h, targets, sample_weight=weight)
        ref = loss_f(logits_h.float(), targets, sample_weight=weight)
        unweighted = loss_u(logits_h.float(), targets)
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)
        assert not torch.equal(out, unweighted)

    @HALF_DTYPES
    def test_pauc_iid_mask(self, half):
        loss_h = make_loss("pauc_at_budget", 1, 64)
        loss_f = make_loss("pauc_at_budget", 1, 64)
        loss_n = make_loss("pauc_at_budget", 1, 64)
        logits_h, targets = make_batch(1, 0, half)
        iid = torch.arange(N) % 3 != 0
        out = loss_h(logits_h, targets, iid_mask=iid)
        ref = loss_f(logits_h.float(), targets, iid_mask=iid)
        no_mask = loss_n(logits_h.float(), targets)
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)
        assert not torch.equal(out, no_mask)

    @HALF_DTYPES
    @NUM_CLASSES
    def test_pauc_diagnostics_and_per_class(self, half, num_classes):
        loss_h = make_loss("pauc_at_budget", num_classes, 64)
        loss_f = make_loss("pauc_at_budget", num_classes, 64)
        logits_h, targets = make_batch(num_classes, 0, half)
        loss, per_class, valid, stats = loss_h(
            logits_h, targets, return_per_class=True, return_diagnostics=True
        )
        ref_loss, ref_pc, ref_valid, ref_stats = loss_f(
            logits_h.float(), targets, return_per_class=True, return_diagnostics=True
        )
        assert loss.dtype == per_class.dtype == torch.float32
        assert valid.dtype == torch.bool
        for key in ("t_alpha", "t_beta", "tau_eff", "pauc_var"):
            assert stats[key].dtype == torch.float32, key
        for key in ("band_neg_count", "grad_pos_count"):
            assert stats[key].dtype == torch.int64, key
        assert torch.equal(loss, ref_loss)
        assert torch.equal(valid, ref_valid)
        torch.testing.assert_close(per_class, ref_pc, rtol=0, atol=0, equal_nan=True)
        for key, value in stats.items():
            torch.testing.assert_close(
                value, ref_stats[key], rtol=0, atol=0, equal_nan=True
            )

    @RANKING_LOSSES
    @HALF_DTYPES
    def test_inf_and_nan_logits_match_float32_cast(self, loss_name, half):
        logits_h, targets = make_batch(1, 0, half)
        logits_h[3, 0] = float("inf")
        logits_h[7, 0] = float("-inf")
        logits_h[11, 0] = float("nan")
        loss_h = make_loss(loss_name, 1, 64)
        loss_f = make_loss(loss_name, 1, 64)
        out = loss_h(logits_h, targets)
        ref = loss_f(logits_h.float(), targets)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)

    @RANKING_LOSSES
    @HALF_DTYPES
    @NUM_CLASSES
    def test_all_ignored_returns_float32_zero_with_backward(
        self, loss_name, half, num_classes
    ):
        loss = make_loss(loss_name, num_classes, 64)
        logits_h, _ = make_batch(num_classes, 0, half)
        targets = torch.full((N,), loss.ignore_index, dtype=torch.long)
        leaf = _leaf(logits_h)
        out = loss(leaf, targets)
        assert out.dtype == torch.float32
        assert out.shape == ()
        assert out.item() == 0.0
        out.backward()
        assert leaf.grad is not None
        assert leaf.grad.abs().sum() == 0

    @RANKING_LOSSES
    @HALF_DTYPES
    def test_gather_forced_single_process(
        self, loss_name, half, single_process_group
    ):
        loss_h = make_loss(loss_name, 1, 64)
        loss_f = make_loss(loss_name, 1, 64)
        loss_h._gather_resolved = True
        loss_f._gather_resolved = True
        logits_h, targets = make_batch(1, 0, half)
        out = loss_h(_leaf(logits_h), targets)
        ref = loss_f(logits_h.float(), targets)
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)

    @RANKING_LOSSES
    @HALF_DTYPES
    def test_queue_buffer_stays_float32(self, loss_name, half):
        loss = make_loss(loss_name, 1, 64)
        logits_h, targets = make_batch(1, 0, half)
        loss(logits_h, targets)
        assert loss._queue._q_logits.dtype == torch.float32
        # The enqueue really happened (rows landed in the buffer).
        assert torch.equal(loss._queue._q_logits, logits_h[-64:].float())
        assert torch.equal(loss._queue._q_targets, targets[-64:])

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    @RANKING_LOSSES
    def test_mps_autocast_end_to_end(self, loss_name):
        loss = make_loss(loss_name, 1, 64).to("mps")
        torch.manual_seed(0)
        layer = torch.nn.Linear(8, 1).to("mps")
        features = torch.randn(N, 8).to("mps")
        _, targets = make_batch(1, 0)
        with torch.autocast("mps", dtype=torch.float16):
            # Linear is autocast-eligible, so these logits really are half.
            logits = layer(features)
            assert logits.dtype == torch.float16
            out = loss(logits, targets.to("mps"))
        assert out.dtype == torch.float32
        out.backward()
        grad = layer.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


class TestFullPrecisionUnchanged:
    @RANKING_LOSSES
    @NUM_CLASSES
    def test_float64_returns_float64(self, loss_name, num_classes):
        loss = make_loss(loss_name, num_classes, 64)
        logits, targets = make_batch(num_classes, 0)
        out = loss(logits.double(), targets)
        assert out.dtype == torch.float64

    def test_helper_returns_same_object_for_non_half(self):
        for dtype in (torch.float32, torch.float64, torch.int64):
            x = torch.ones(3, dtype=dtype)
            assert _to_compute_dtype(x) is x

    @HALF_DTYPES
    def test_helper_upcasts_half_to_float32(self, half):
        x = torch.ones(3, dtype=half)
        y = _to_compute_dtype(x)
        assert y.dtype == torch.float32
        assert y is not x


def _multiclass_batch(n: int = N):
    """C=3 batch of ``n`` rows with every target in {0, 1, 2}."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 3, generator=g), torch.arange(n) % 3


class TestMultiClassTargetRange:
    @RANKING_LOSSES
    @pytest.mark.parametrize("bad", [5, 3, -1, 2**62])
    def test_out_of_range_target_raises(self, loss_name, bad):
        logits, targets = _multiclass_batch()
        targets[7] = bad
        loss = make_loss(loss_name, 3, 0)
        with pytest.raises(ValueError, match="outside"):
            loss(logits, targets)

    @RANKING_LOSSES
    def test_message_keeps_wording_and_examples(self, loss_name):
        logits, targets = _multiclass_batch()
        targets[7] = 5
        loss = make_loss(loss_name, 3, 0)
        with pytest.raises(
            ValueError,
            match=r"targets contain class ids outside \[0, 3\); examples: \[5\]",
        ):
            loss(logits, targets)

    @RANKING_LOSSES
    def test_in_range_targets_do_not_raise(self, loss_name):
        logits, targets = _multiclass_batch()
        loss = make_loss(loss_name, 3, 0)
        assert torch.isfinite(loss(logits, targets))

    @RANKING_LOSSES
    def test_default_ignore_index_rows_not_flagged(self, loss_name):
        logits, targets = _multiclass_batch()
        targets[7] = -100
        loss = make_loss(loss_name, 3, 0)
        assert targets[7] == loss.ignore_index
        loss(logits, targets)

    @RANKING_LOSSES
    def test_custom_ignore_index_not_flagged_but_default_value_is(self, loss_name):
        logits, targets = _multiclass_batch()
        targets[7] = 999
        loss = make_loss(loss_name, 3, 0, ignore_index=999)
        loss(logits, targets)
        # With a custom ignore_index, -100 is an ordinary bad class id.
        targets[7] = -100
        with pytest.raises(ValueError, match="outside"):
            make_loss(loss_name, 3, 0, ignore_index=999)(logits, targets)

    @RANKING_LOSSES
    def test_binary_mode_warns_and_does_not_raise(self, loss_name):
        logits, targets = make_batch(1, 0)
        targets[7] = 2
        loss = make_loss(loss_name, 1, 0)
        with pytest.warns(UserWarning, match="expects targets in"):
            loss(logits, targets)

    @RANKING_LOSSES
    @pytest.mark.parametrize("max_pool", [20, 2])
    def test_bad_row_raises_every_time_under_subsampling(self, loss_name, max_pool):
        n = 200
        logits, targets = _multiclass_batch(n)
        targets[137] = 5
        assert n > max_pool  # the pool really exceeds max_pool_size
        # The sampler reserves a quota for every observed class, so a lone
        # bad row survives whenever the reserved set fits in max_pool_size
        # (max_pool=20). At max_pool=2 the four reserved rows are randomly
        # truncated to two, so a check that ran after subsampling would miss
        # the bad row in about half the draws.
        for _ in range(20):
            loss = make_loss(loss_name, 3, 0, max_pool_size=max_pool)
            with pytest.raises(ValueError, match="outside"):
                loss(logits, targets)

    @RANKING_LOSSES
    def test_raise_leaves_queue_untouched(self, loss_name):
        logits, targets = _multiclass_batch(30)
        loss = make_loss(loss_name, 3, 64)
        loss(logits, targets)
        queue = loss._queue
        assert int(queue._q_ptr) == 30  # the clean call really enqueued
        ptr = queue._q_ptr.clone()
        q_targets = queue._q_targets.clone()
        q_logits = queue._q_logits.clone()
        bad_targets = targets.clone()
        bad_targets[3] = 5
        with pytest.raises(ValueError, match="outside"):
            loss(logits + 1.0, bad_targets)
        assert torch.equal(queue._q_ptr, ptr)
        assert torch.equal(queue._q_targets, q_targets)
        assert torch.equal(queue._q_logits, q_logits)

    @RANKING_LOSSES
    def test_bad_target_already_in_queue_raises(self, loss_name):
        logits, targets = _multiclass_batch(30)
        loss = make_loss(loss_name, 3, 64)
        loss(logits, targets)
        loss._queue._q_targets[0] = 5
        with pytest.raises(ValueError, match="outside"):
            loss(logits, targets)

    @RANKING_LOSSES
    def test_all_ignored_does_not_raise(self, loss_name):
        logits, _ = _multiclass_batch()
        targets = torch.full((N,), -100, dtype=torch.long)
        make_loss(loss_name, 3, 64)(logits, targets)

    @RANKING_LOSSES
    def test_float_targets_in_range_do_not_raise(self, loss_name):
        logits, targets = _multiclass_batch()
        loss = make_loss(loss_name, 3, 0)
        loss(logits, targets.float())
        bad = targets.float()
        bad[7] = 5.0
        with pytest.raises(ValueError, match="outside"):
            make_loss(loss_name, 3, 0)(logits, bad)

    def test_noop_override_skips_check_and_sees_full_pool(self):
        seen: list[int] = []

        class NoCheck(SmoothAPLoss):
            def _validate_filtered_targets(self, targets):
                seen.append(targets.numel())

        n, max_pool = 200, 20
        logits, targets = _multiclass_batch(n)
        targets[137] = 5
        loss = NoCheck(num_classes=3, queue_size=0, max_pool_size=max_pool)
        with pytest.warns(UserWarning, match="max_pool_size"):
            loss(logits, targets)  # subsampling ran, so the hook ran before it
        assert seen == [n]

    def test_recall_loss_no_longer_overrides_the_hook(self):
        assert "_validate_filtered_targets" not in RecallAtQuantileLoss.__dict__
