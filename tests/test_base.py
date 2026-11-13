"""
Behavior shared by every queued ranking loss (``_QueuedRankingLoss``).

``RANKING_LOSSES`` is the module-level parametrization over the three
concrete losses; later tests in this file reuse it.
"""

from __future__ import annotations

import inspect
import warnings

import pytest
import torch
import torch.distributed as dist

from imbalanced_losses import PAUCAtBudgetLoss, RecallAtQuantileLoss, SmoothAPLoss
from imbalanced_losses._base import _QueuedRankingLoss, _to_compute_dtype

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
        # A second step reaches the queue-merge path when queue_size > 0.
        for seed in (0, 1):
            logits_h, targets = make_batch(num_classes, seed, half)
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
        leaf = _leaf(logits_h)
        out = loss_h(leaf, targets, sample_weight=weight)
        ref = loss_f(logits_h.float(), targets, sample_weight=weight)
        unweighted = loss_u(logits_h.float(), targets)
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)
        assert not torch.equal(out, unweighted)
        out.backward()
        assert leaf.grad is not None
        assert leaf.grad.dtype == torch.float16
        assert torch.isfinite(leaf.grad).all()
        assert leaf.grad.abs().sum() > 0

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
        leaf = _leaf(logits_h)
        loss, per_class, valid, stats = loss_h(
            leaf, targets, return_per_class=True, return_diagnostics=True
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
        loss.backward()
        assert leaf.grad is not None
        assert leaf.grad.dtype == half
        assert torch.isfinite(leaf.grad).all()
        assert leaf.grad.abs().sum() > 0
        torch.testing.assert_close(per_class, ref_pc, rtol=0, atol=0, equal_nan=True)
        for key, value in stats.items():
            torch.testing.assert_close(
                value, ref_stats[key], rtol=0, atol=0, equal_nan=True
            )

    @RANKING_LOSSES
    @HALF_DTYPES
    def test_inf_logits_match_float32_cast(self, loss_name, half):
        logits_h, targets = make_batch(1, 0, half)
        logits_h[3, 0] = float("inf")
        logits_h[7, 0] = float("-inf")
        out = make_loss(loss_name, 1, 64)(logits_h, targets)
        ref = make_loss(loss_name, 1, 64)(logits_h.float(), targets)
        assert out.dtype == torch.float32
        # Finite, so the equality below is a real value comparison, not NaN vs NaN.
        assert torch.isfinite(ref)
        assert torch.equal(out, ref)
        # The inf rows matter: zeroing them would change the loss.
        zeroed = logits_h.clone()
        zeroed[3, 0] = 0.0
        zeroed[7, 0] = 0.0
        assert not torch.equal(make_loss(loss_name, 1, 64)(zeroed, targets), ref)

    @RANKING_LOSSES
    @HALF_DTYPES
    def test_nan_logits_match_float32_cast(self, loss_name, half):
        logits_h, targets = make_batch(1, 0, half)
        logits_h[11, 0] = float("nan")
        out = make_loss(loss_name, 1, 64)(logits_h, targets)
        ref = make_loss(loss_name, 1, 64)(logits_h.float(), targets)
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
            def _validate_filtered_targets(self, targets, **_):
                seen.append(targets.numel())

        n, max_pool = 200, 20
        logits, targets = _multiclass_batch(n)
        targets[137] = 5
        loss = NoCheck(num_classes=3, queue_size=0, max_pool_size=max_pool)
        with pytest.warns(UserWarning, match="max_pool_size"):
            loss(logits, targets)  # subsampling ran, so the hook ran before it
        assert seen == [n]

    @pytest.mark.parametrize("max_pool_size", [None, 20])
    def test_recall_loss_raises_the_shared_message(self, max_pool_size):
        logits, targets = _multiclass_batch(200)
        targets[137] = 5
        loss = RecallAtQuantileLoss(
            num_classes=3, queue_size=0, max_pool_size=max_pool_size
        )
        with pytest.raises(
            ValueError, match=r"targets contain class ids outside \[0, 3\); examples: \[5\]"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loss(logits, targets)


class TestEmptyPoolSentinel:
    """An empty pool returns NaN per class under reduction='none'."""

    IGNORE = -100

    @staticmethod
    def _empty_batch(num_classes, dtype=torch.float32, n=8):
        logits = torch.randn(n, num_classes, dtype=dtype)
        targets = torch.full((n,), TestEmptyPoolSentinel.IGNORE, dtype=torch.long)
        return logits, targets

    @staticmethod
    def _assert_nan_vec(out, num_classes, dtype=torch.float32):
        assert out.shape == (num_classes,)
        assert out.dtype == dtype
        assert torch.isnan(out).all()
        assert out.requires_grad is False

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_all_ignored_none_is_nan(self, loss_name, num_classes):
        loss = make_loss(loss_name, num_classes, 0, reduction="none")
        logits, targets = self._empty_batch(num_classes)
        out = loss(_leaf(logits), targets)
        self._assert_nan_vec(out, num_classes)

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_matches_non_empty_all_degenerate_sentinel(self, loss_name, num_classes):
        # Reference: a non-empty pool with no positives, so every class is degenerate.
        loss = make_loss(loss_name, num_classes, 0, reduction="none")
        logits, _ = self._empty_batch(num_classes)
        logits = _leaf(logits)  # leaf, so requires_grad can differ between the two
        targets = torch.zeros(8, dtype=torch.long)
        ref = loss(logits, targets)
        empty = make_loss(loss_name, num_classes, 0, reduction="none")(
            logits, torch.full((8,), self.IGNORE, dtype=torch.long)
        )
        assert torch.isnan(ref).any()  # degenerate classes use the NaN sentinel
        assert torch.isnan(empty).all()
        assert empty.shape == ref.shape
        assert empty.dtype == ref.dtype
        # Logits are a leaf requiring grad, so a result that kept a path to them
        # would report True. (ref may carry a path through masked_fill; the
        # sentinel itself is a constant.)
        assert empty.requires_grad is False

    @RANKING_LOSSES
    @NUM_CLASSES
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_all_ignored_mean_sum_zero_with_grad_path(
        self, loss_name, num_classes, reduction
    ):
        loss = make_loss(loss_name, num_classes, 0, reduction=reduction)
        logits, targets = self._empty_batch(num_classes)
        leaf = _leaf(logits)
        out = loss(leaf, targets)
        assert out.shape == ()
        assert out.item() == 0.0
        assert out.requires_grad
        out.backward()
        assert leaf.grad is not None
        assert (leaf.grad == 0).all()

    @RANKING_LOSSES
    @NUM_CLASSES
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_return_per_class(self, loss_name, num_classes, reduction):
        loss = make_loss(loss_name, num_classes, 0, reduction=reduction)
        logits, targets = self._empty_batch(num_classes)
        out, per_class, valid = loss(logits, targets, return_per_class=True)
        if reduction == "none":
            self._assert_nan_vec(out, num_classes)
        else:
            assert out.item() == 0.0
        self._assert_nan_vec(per_class, num_classes)
        assert valid.shape == (num_classes,)
        assert valid.dtype == torch.bool
        assert not valid.any()

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_per_class_is_not_aliased_to_loss(self, loss_name, num_classes):
        loss = make_loss(loss_name, num_classes, 0, reduction="none")
        logits, targets = self._empty_batch(num_classes)
        out, per_class, _ = loss(logits, targets, return_per_class=True)
        per_class.fill_(1.0)
        assert torch.isnan(out).all()

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_zero_row_batch_empty_queue(self, loss_name, num_classes):
        loss = make_loss(loss_name, num_classes, 64, reduction="none")
        out = loss(torch.zeros(0, num_classes), torch.zeros(0, dtype=torch.long))
        self._assert_nan_vec(out, num_classes)
        assert int(loss._queue._q_ptr) == 0

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_float64(self, loss_name, num_classes):
        loss = make_loss(loss_name, num_classes, 0, reduction="none")
        logits, targets = self._empty_batch(num_classes, torch.float64)
        self._assert_nan_vec(loss(logits, targets), num_classes, torch.float64)

    @RANKING_LOSSES
    @NUM_CLASSES
    @HALF_DTYPES
    def test_half_logits_give_float32_nan(self, loss_name, num_classes, half):
        loss = make_loss(loss_name, num_classes, 0, reduction="none")
        logits, targets = self._empty_batch(num_classes, half)
        self._assert_nan_vec(loss(logits, targets), num_classes, torch.float32)

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_sample_weight_on_all_ignored(self, loss_name, num_classes):
        logits, targets = self._empty_batch(num_classes)
        weight = torch.linspace(0.5, 2.0, 8)
        none = make_loss(loss_name, num_classes, 0, reduction="none")
        self._assert_nan_vec(none(logits, targets, sample_weight=weight), num_classes)
        for reduction in ("mean", "sum"):
            loss = make_loss(loss_name, num_classes, 0, reduction=reduction)
            leaf = _leaf(logits)
            out = loss(leaf, targets, sample_weight=weight)
            assert out.item() == 0.0
            out.backward()
            assert (leaf.grad == 0).all()

    @NUM_CLASSES
    def test_pauc_diagnostics_on_empty_pool(self, num_classes):
        loss = make_loss("pauc_at_budget", num_classes, 0, reduction="none")
        logits, targets = self._empty_batch(num_classes)
        out, stats = loss(logits, targets, return_diagnostics=True)
        self._assert_nan_vec(out, num_classes)
        for key in ("t_alpha", "t_beta", "tau_eff", "pauc_var"):
            assert stats[key].shape == (num_classes,)
            assert stats[key].dtype.is_floating_point
            assert torch.isnan(stats[key]).all(), key
        for key in ("band_neg_count", "grad_pos_count"):
            assert stats[key].shape == (num_classes,)
            assert (stats[key] == 0).all(), key

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_queue_untouched_after_empty_pool(self, loss_name, num_classes):
        loss = make_loss(loss_name, num_classes, 64, reduction="none")
        logits, targets = self._empty_batch(num_classes)
        loss(logits, targets)
        assert int(loss._queue._q_ptr) == 0
        assert (loss._queue._q_targets == self.IGNORE).all()

    @RANKING_LOSSES
    @NUM_CLASSES
    @pytest.mark.parametrize("reduction", ["none", "mean"])
    def test_no_warning_on_empty_pool(self, loss_name, num_classes, reduction):
        import warnings

        loss = make_loss(loss_name, num_classes, 64, reduction=reduction)
        logits, targets = self._empty_batch(num_classes)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loss(logits, targets, return_per_class=True)

    @RANKING_LOSSES
    @NUM_CLASSES
    def test_all_ignored_live_batch_with_filled_queue_is_not_empty(
        self, loss_name, num_classes
    ):
        # The queue holds real rows, so the pool is non-empty and the general
        # path runs: the result equals ranking the queued rows alone, and the
        # early return (which would give NaN everywhere) is not taken.
        loss = make_loss(loss_name, num_classes, 64, reduction="none")
        logits, targets = make_batch(num_classes, 0)
        loss(logits, targets)  # fills the queue (96 rows > 64, keeps last 64)
        assert (loss._queue._q_targets != self.IGNORE).any()
        ref = make_loss(loss_name, num_classes, 0, reduction="none")
        q_logits = loss._queue._q_logits.clone()
        q_targets = loss._queue._q_targets.clone()
        live_logits, live_targets = self._empty_batch(num_classes)
        out = loss(live_logits, live_targets)
        expected = ref(q_logits, q_targets)
        torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
        assert not torch.isnan(out).all()


class TestQueuedBadTargetIsActionable:
    @RANKING_LOSSES
    def test_error_names_reset_queue_and_reset_clears_it(self, loss_name):
        logits, targets = _multiclass_batch(30)
        loss = make_loss(loss_name, 3, 64)
        loss(logits, targets)
        loss._queue._q_targets[0] = 5  # as if restored from an older checkpoint
        with pytest.raises(ValueError, match="outside") as excinfo:
            loss(logits, targets)
        message = str(excinfo.value)
        assert "reset_queue" in message
        assert "examples: [5]" in message
        loss.reset_queue()
        assert torch.isfinite(loss(logits, targets))


class TestRangeErrorRemedyMatchesSource:
    @RANKING_LOSSES
    @pytest.mark.parametrize("queue_size", [0, 64])
    def test_live_only_bad_row_names_ignore_index_not_reset_queue(
        self, loss_name, queue_size
    ):
        logits, targets = _multiclass_batch(30)
        targets[3] = 5
        loss = make_loss(loss_name, 3, queue_size)
        with pytest.raises(ValueError, match="outside") as excinfo:
            loss(logits, targets)
        message = str(excinfo.value)
        assert "ignore_index" in message
        assert "reset_queue" not in message

    @RANKING_LOSSES
    def test_queue_only_bad_row_names_reset_queue(self, loss_name):
        logits, targets = _multiclass_batch(30)
        loss = make_loss(loss_name, 3, 64)
        loss(logits, targets)
        loss._queue._q_targets[0] = 5
        with pytest.raises(ValueError, match="outside") as excinfo:
            loss(logits, targets)
        message = str(excinfo.value)
        assert "reset_queue" in message
        assert "ignore_index" not in message
        loss.reset_queue()
        assert torch.isfinite(loss(logits, targets))

    @RANKING_LOSSES
    def test_both_sources_name_both_remedies(self, loss_name):
        logits, targets = _multiclass_batch(30)
        loss = make_loss(loss_name, 3, 64)
        loss(logits, targets)
        loss._queue._q_targets[0] = 5
        bad = targets.clone()
        bad[3] = 7
        with pytest.raises(ValueError, match="outside") as excinfo:
            loss(logits, bad)
        message = str(excinfo.value)
        assert "ignore_index" in message
        assert "reset_queue" in message

    def test_hook_callable_with_targets_alone(self):
        loss = make_loss("smooth_ap", 3, 0)
        with pytest.raises(ValueError, match="outside") as excinfo:
            loss._validate_filtered_targets(torch.tensor([0, 5]))
        message = str(excinfo.value)
        assert "ignore_index" in message and "reset_queue" in message


# ---------------------------------------------------------------------------
# Thresholds estimated on the full pool before max_pool_size subsampling
# ---------------------------------------------------------------------------

THRESHOLD_LOSSES = pytest.mark.parametrize(
    "loss_name", ["recall_at_quantile", "pauc_at_budget"]
)
DTYPES = pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)


def _skewed_batch(
    num_classes: int, n: int, n_pos: int, seed: int, dtype: torch.dtype = torch.float32
):
    """Imbalanced batch; ``n_pos`` positives when binary, 80/10/10 when C=3."""
    g = torch.Generator().manual_seed(seed)
    if num_classes == 1:
        targets = torch.zeros(n, dtype=torch.long)
        targets[torch.randperm(n, generator=g)[:n_pos]] = 1
        logits = torch.randn(n, 1, generator=g) + 1.5 * targets.unsqueeze(1)
    else:
        probs = torch.tensor([0.8] + [0.2 / (num_classes - 1)] * (num_classes - 1))
        targets = torch.multinomial(probs, n, replacement=True, generator=g)
        one_hot = torch.nn.functional.one_hot(targets, num_classes).float()
        logits = torch.randn(n, num_classes, generator=g) + 1.5 * one_hot
    return logits.to(dtype), targets


def _spy_estimator(loss):
    """Wrap ``loss._estimate_thresholds``; return the list of recorded calls."""
    calls = []
    original = loss._estimate_thresholds

    def spy(pool):
        fields = [t for t in pool if t is not None]
        before = [(t.data_ptr(), tuple(t.shape), t.clone()) for t in fields]
        result = original(pool)
        calls.append({
            "rows": pool.logits.size(0),
            "result": result,
            "pool_unchanged": all(
                t.data_ptr() == ptr and tuple(t.shape) == shape and torch.equal(t, copy)
                for t, (ptr, shape, copy) in zip(fields, before)
            ),
        })
        return result

    loss._estimate_thresholds = spy
    return calls


def _spy_compute(loss):
    """Wrap ``loss._compute_per_class``; return the keyword dict of every call."""
    seen = []
    original = loss._compute_per_class

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    loss._compute_per_class = spy
    return seen


def _flat_tensors(thresholds) -> list[torch.Tensor]:
    """Every tensor inside a thresholds object (a tensor, or per-class tuples)."""
    if isinstance(thresholds, torch.Tensor):
        return [thresholds]
    out = []
    for entry in thresholds:
        if entry is None:
            continue
        out.extend(f for f in entry if isinstance(f, torch.Tensor))
    return out


def _capped_and_uncapped(loss_name, num_classes, cap, **kwargs):
    capped = make_loss(loss_name, num_classes, 0, max_pool_size=cap, **kwargs)
    uncapped = make_loss(loss_name, num_classes, 0, **kwargs)
    return capped, uncapped


class _PlainRankingLoss(_QueuedRankingLoss):
    """Subclass with neither a hook override nor a ``thresholds`` parameter."""

    def _compute_per_class(self, logits, targets, is_iid, is_live, sample_weight=None):
        return logits.mean(dim=0), torch.ones(
            self.num_classes, dtype=torch.bool, device=logits.device
        )


PAUC_GRID = pytest.mark.parametrize(
    "surrogate,budget_basis,tau_scale",
    [
        (s, b, t)
        for s in ("trapezoid", "pairwise")
        for b in ("fpr", "population")
        for t in ("iqr", "band")
    ],
)


class TestFullPoolThresholds:
    def test_recall_binary_loss_matches_uncapped(self):
        logits, targets = _skewed_batch(1, 4000, 20, seed=0)
        capped = RecallAtQuantileLoss(1, quantile=0.05, queue_size=0, max_pool_size=512)
        uncapped = RecallAtQuantileLoss(1, quantile=0.05, queue_size=0)
        with pytest.warns(UserWarning, match="max_pool_size"):
            out = capped(logits, targets)
        assert abs(out.item() - uncapped(logits, targets).item()) < 1e-6

    def test_recall_multiclass_thresholds_are_full_filtered_pool_quantiles(self):
        logits, targets = _skewed_batch(3, 1200, 0, seed=1)
        # Ignored rows carry extreme scores: unfiltered they would move every quantile.
        targets[::25] = -100
        logits[::25] = 1e3
        loss = RecallAtQuantileLoss(3, quantile=0.05, queue_size=0, max_pool_size=256)
        seen = _spy_compute(loss)
        with pytest.warns(UserWarning, match="max_pool_size"):
            loss(logits, targets)
        kept = logits[targets != -100]
        expected = torch.stack([
            torch.quantile(kept[:, c], 0.95, interpolation="higher") for c in range(3)
        ])
        assert torch.equal(seen[0]["thresholds"], expected)

    @PAUC_GRID
    @NUM_CLASSES
    def test_pauc_diagnostics_equal_uncapped(
        self, surrogate, budget_basis, tau_scale, num_classes
    ):
        logits, targets = _skewed_batch(num_classes, 2000 if num_classes == 1 else 1200, 20, seed=2)
        kwargs = dict(
            beta=0.1, surrogate=surrogate, budget_basis=budget_basis, tau_scale=tau_scale
        )
        capped = PAUCAtBudgetLoss(num_classes, queue_size=0, max_pool_size=256, **kwargs)
        uncapped = PAUCAtBudgetLoss(num_classes, queue_size=0, **kwargs)
        with pytest.warns(UserWarning, match="max_pool_size"):
            _, stats_c = capped(logits, targets, return_diagnostics=True)
        _, stats_u = uncapped(logits, targets, return_diagnostics=True)
        for key in ("t_alpha", "t_beta", "tau_eff"):
            assert torch.isfinite(stats_u[key]).all()
            assert torch.equal(stats_c[key], stats_u[key]), key

    def test_pauc_trapezoid_binary_loss_matches_uncapped(self):
        logits, targets = _skewed_batch(1, 2000, 20, seed=3)
        kwargs = dict(beta=0.05, surrogate="trapezoid", queue_size=0)
        capped = PAUCAtBudgetLoss(1, max_pool_size=512, **kwargs)
        uncapped = PAUCAtBudgetLoss(1, **kwargs)
        with pytest.warns(UserWarning, match="max_pool_size"):
            out = capped(logits, targets)
        assert abs(out.item() - uncapped(logits, targets).item()) < 1e-6

    @NUM_CLASSES
    def test_pauc_non_iid_negatives_keep_full_pool_edges(self, num_classes):
        logits, targets = _skewed_batch(num_classes, 1500, 20, seed=4)
        g = torch.Generator().manual_seed(40)
        iid_mask = torch.rand(logits.size(0), generator=g) < 0.5
        capped, uncapped = _capped_and_uncapped(
            "pauc_at_budget", num_classes, 256, beta=0.1
        )
        with pytest.warns(UserWarning, match="max_pool_size"):
            _, stats_c = capped(logits, targets, iid_mask=iid_mask, return_diagnostics=True)
        _, stats_u = uncapped(logits, targets, iid_mask=iid_mask, return_diagnostics=True)
        assert torch.isfinite(stats_u["t_alpha"]).all()
        assert torch.equal(stats_c["t_alpha"], stats_u["t_alpha"])
        assert torch.equal(stats_c["t_beta"], stats_u["t_beta"])

    def test_pauc_live_numerator_with_queue_is_valid_and_edges_match(self):
        step1 = _skewed_batch(1, 300, 15, seed=5)
        step2 = _skewed_batch(1, 300, 15, seed=6)
        results = {}
        for name, cap in (("capped", 200), ("uncapped", None)):
            loss = PAUCAtBudgetLoss(
                1, beta=0.1, pos_numerator="live", queue_size=300, max_pool_size=cap
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loss(*step1)
                results[name] = loss(*step2, return_per_class=True, return_diagnostics=True)
        _, _, valid, stats_c = results["capped"]
        _, _, _, stats_u = results["uncapped"]
        assert valid.all()
        assert torch.isfinite(stats_c["t_alpha"]).all()
        assert torch.equal(stats_c["t_alpha"], stats_u["t_alpha"])
        assert torch.equal(stats_c["t_beta"], stats_u["t_beta"])

    @pytest.mark.parametrize(
        "pos_numerator,loss_moves", [("live", False), ("pool", True)]
    )
    def test_fpr_live_numerator_ignores_queued_positive_scores(
        self, pos_numerator, loss_moves
    ):
        logits1, targets1 = _skewed_batch(1, 300, 15, seed=7)
        logits1_edited = logits1.clone()
        logits1_edited[targets1 == 1] += 3.0
        live = _skewed_batch(1, 300, 15, seed=8)

        def step2_loss(queued_logits):
            loss = PAUCAtBudgetLoss(
                1, beta=0.1, budget_basis="fpr", pos_numerator=pos_numerator,
                queue_size=300, max_pool_size=200,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loss(queued_logits, targets1)
                torch.manual_seed(123)
                return loss(*live)

        same = torch.equal(step2_loss(logits1), step2_loss(logits1_edited))
        assert same is (not loss_moves)

    def test_class_without_iid_negatives_is_nan_others_valid(self):
        logits, targets = _skewed_batch(3, 1200, 0, seed=9)
        # Class 0's negatives are the rows of classes 1 and 2, none of them iid.
        iid_mask = targets == 0
        loss = PAUCAtBudgetLoss(3, beta=0.1, queue_size=0, reduction="none", max_pool_size=256)
        with pytest.warns(UserWarning, match="max_pool_size"):
            out, _, valid = loss(logits, targets, iid_mask=iid_mask, return_per_class=True)
        assert valid.tolist() == [False, True, True]
        assert torch.isnan(out[0])
        assert torch.isfinite(out[1:]).all()

    def test_full_pool_degenerate_dispersion_is_invalid_and_warns_once(self):
        logits, targets = _skewed_batch(1, 600, 20, seed=10)
        logits[targets == 0] = 0.0  # every negative score equal: IQR is 0
        loss = PAUCAtBudgetLoss(1, beta=0.1, queue_size=0, reduction="none", max_pool_size=128)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out, _, valid = loss(logits, targets, return_per_class=True)
            loss(logits, targets)
        degenerate = [w for w in rec if "near-zero" in str(w.message)]
        assert not valid.any()
        assert torch.isnan(out).all()
        assert len(degenerate) == 1

    def test_smooth_ap_never_receives_thresholds_keyword(self):
        logits, targets = _skewed_batch(3, 1200, 0, seed=11)
        loss = SmoothAPLoss(3, queue_size=0, max_pool_size=256)
        seen = _spy_compute(loss)
        with pytest.warns(UserWarning, match="max_pool_size"):
            loss(logits, targets)
        assert len(seen) == 1
        assert "thresholds" not in seen[0]

    def test_subclass_without_hook_or_thresholds_parameter_runs(self):
        logits, targets = _skewed_batch(3, 600, 0, seed=12)
        loss = _PlainRankingLoss(num_classes=3, queue_size=0, max_pool_size=128)
        with pytest.warns(UserWarning, match="max_pool_size"):
            out = loss(logits, targets)
        assert torch.isfinite(out)

    @RANKING_LOSSES
    @DTYPES
    @pytest.mark.parametrize("cap_kind", ["none", "above", "equal"])
    def test_untriggered_cap_skips_hook_and_is_bitwise_identical(
        self, loss_name, dtype, cap_kind
    ):
        logits, targets = make_batch(3, 0)
        logits = logits.to(dtype)
        cap = {"none": None, "above": N + 50, "equal": N}[cap_kind]
        reference = make_loss(loss_name, 3, 0)
        candidate = make_loss(loss_name, 3, 0, max_pool_size=cap)
        calls = _spy_estimator(candidate)
        leaf_r, leaf_c = _leaf(logits), _leaf(logits)
        out_r = reference(leaf_r, targets)
        out_c = candidate(leaf_c, targets)
        out_r.backward()
        out_c.backward()
        assert calls == []
        assert torch.equal(out_r, out_c)
        assert torch.equal(leaf_r.grad, leaf_c.grad)

    @RANKING_LOSSES
    def test_hook_called_once_per_forward_with_pre_subsampling_rows(self, loss_name):
        logits, targets = _skewed_batch(3, 600, 0, seed=13)
        targets[::15] = -100
        n_kept = int((targets != -100).sum())
        assert n_kept < 600
        loss = make_loss(loss_name, 3, 0, max_pool_size=128)
        calls = _spy_estimator(loss)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss(logits, targets)
            assert [c["rows"] for c in calls] == [n_kept]
            loss(logits, targets)
        assert [c["rows"] for c in calls] == [n_kept, n_kept]

    @RANKING_LOSSES
    def test_hook_leaves_the_pool_unchanged(self, loss_name):
        logits, targets = _skewed_batch(3, 600, 0, seed=14)
        loss = make_loss(loss_name, 3, 0, max_pool_size=128)
        calls = _spy_estimator(loss)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss(logits, targets)
        assert len(calls) == 1
        assert calls[0]["pool_unchanged"]

    @THRESHOLD_LOSSES
    @NUM_CLASSES
    def test_thresholds_carry_no_gradient(self, loss_name, num_classes):
        logits, targets = _skewed_batch(num_classes, 600, 20, seed=15)
        loss = make_loss(loss_name, num_classes, 0, max_pool_size=128)
        calls = _spy_estimator(loss)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss(_leaf(logits), targets)
        tensors = _flat_tensors(calls[0]["result"])
        assert tensors
        assert not any(t.requires_grad for t in tensors)

    @THRESHOLD_LOSSES
    @NUM_CLASSES
    def test_sample_weight_does_not_change_thresholds(self, loss_name, num_classes):
        logits, targets = _skewed_batch(num_classes, 600, 20, seed=16)
        g = torch.Generator().manual_seed(160)
        weight = torch.rand(logits.size(0), generator=g) + 0.1
        results = []
        for sw in (None, weight):
            loss = make_loss(loss_name, num_classes, 0, max_pool_size=128)
            calls = _spy_estimator(loss)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loss(logits, targets, sample_weight=sw)
            results.append(_flat_tensors(calls[0]["result"]))
        assert results[0]
        assert len(results[0]) == len(results[1])
        assert all(torch.equal(a, b) for a, b in zip(*results))

    @THRESHOLD_LOSSES
    @NUM_CLASSES
    def test_half_logits_give_float32_thresholds(self, loss_name, num_classes):
        logits, targets = _skewed_batch(num_classes, 600, 20, seed=17)
        loss = make_loss(loss_name, num_classes, 0, max_pool_size=128)
        calls = _spy_estimator(loss)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = loss(logits.to(torch.float16), targets)
        assert torch.isfinite(out)
        tensors = _flat_tensors(calls[0]["result"])
        assert tensors
        assert all(t.dtype == torch.float32 for t in tensors)

    @THRESHOLD_LOSSES
    @NUM_CLASSES
    def test_nan_logit_gives_the_uncapped_nan_pattern(self, loss_name, num_classes):
        logits, targets = _skewed_batch(num_classes, 600, 120, seed=18)
        # A negative of class 0, so the NaN reaches its threshold whether or
        # not subsampling keeps the row.
        negative_targets = targets == 0 if num_classes == 1 else targets != 0
        logits[int(negative_targets.nonzero()[0]), 0] = float("nan")
        capped, uncapped = _capped_and_uncapped(
            loss_name, num_classes, 128, reduction="none"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_u = uncapped(logits, targets)
            out_c = capped(logits, targets)
        assert torch.isnan(out_u[0])
        assert torch.equal(torch.isnan(out_c), torch.isnan(out_u))

    @THRESHOLD_LOSSES
    def test_forced_gather_in_single_process_runs(self, loss_name, single_process_group):
        logits, targets = _skewed_batch(1, 600, 20, seed=19)
        loss = make_loss(loss_name, 1, 0, max_pool_size=128)
        loss._gather_resolved = True
        calls = _spy_estimator(loss)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = loss(logits, targets)
        assert torch.isfinite(out)
        assert len(calls) == 1

    def test_direct_helper_calls_keep_their_shapes(self):
        loss = RecallAtQuantileLoss(1, quantile=0.2, queue_size=0)
        scores = torch.randn(50, generator=torch.Generator().manual_seed(20))
        is_pos = torch.arange(50) < 5
        recall, valid = loss._soft_recall_at_quantile(scores, is_pos)
        theta = torch.quantile(scores, 0.8, interpolation="higher")
        assert valid
        assert torch.equal(
            recall, torch.sigmoid((scores[is_pos] - theta) / loss.temperature).mean()
        )

        pauc = PAUCAtBudgetLoss(1, beta=0.1, queue_size=0)
        t_k, t_alpha, t_beta, scale = pauc._band_quantiles_and_scale(scores)
        assert t_k.shape == (2,)
        assert torch.equal(t_alpha, torch.quantile(scores, 1.0, interpolation="higher"))
        assert torch.equal(t_beta, torch.quantile(scores, 0.9, interpolation="higher"))
        assert scale.ndim == 0

    @RANKING_LOSSES
    def test_subsample_warning_once_with_class_name_and_cap(self, loss_name):
        logits, targets = make_batch(3, 0)
        loss = make_loss(loss_name, 3, 0, max_pool_size=64)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            loss(logits, targets)
            loss(logits, targets)
        hits = [w for w in rec if "minimum-quota subsampling" in str(w.message)]
        assert len(hits) == 1
        assert type(loss).__name__ in str(hits[0].message)
        assert "max_pool_size=64" in str(hits[0].message)

    @NUM_CLASSES
    @pytest.mark.parametrize("cap", [None, 128])
    def test_degenerate_warning_is_not_attributed_inside_the_package(
        self, num_classes, cap
    ):
        import os

        import imbalanced_losses

        package_dir = os.path.dirname(imbalanced_losses.__file__)
        n = 600
        g = torch.Generator().manual_seed(0)
        if num_classes == 1:
            targets = (torch.rand(n, generator=g) < 0.1).long()
            logits = torch.randn(n, 1, generator=g)
            logits[targets == 0] = 0.0
        else:
            targets = torch.randint(0, num_classes, (n,), generator=g)
            logits = torch.zeros(n, num_classes)
        loss = PAUCAtBudgetLoss(num_classes, beta=0.1, queue_size=0, max_pool_size=cap)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            loss(logits, targets)
        hits = [w for w in rec if "near-zero" in str(w.message)]
        assert len(hits) == 1
        assert not os.path.abspath(hits[0].filename).startswith(package_dir)


# ---------------------------------------------------------------------------
# Only non-ignored rows are enqueued
# ---------------------------------------------------------------------------

IGNORE = -100


def _padded_batch(n, n_valid, seed, num_classes=3, first_id=0):
    """Batch of ``n`` rows, ``n_valid`` of them valid, at random positions.

    Every row's first logit is a unique id (``first_id`` + row index) so a
    stored row can be traced back to the batch row it came from.
    Returns ``(logits, targets)``.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_classes, generator=g)
    logits[:, 0] = torch.arange(first_id, first_id + n, dtype=torch.float32)
    targets = torch.full((n,), IGNORE, dtype=torch.long)
    keep = torch.randperm(n, generator=g)[:n_valid].sort().values
    targets[keep] = torch.arange(n_valid) % num_classes
    return logits, targets


def _queued(loss_name, queue_size, **kwargs):
    loss = make_loss(loss_name, 3, queue_size, **kwargs)
    loss.train()
    return loss


def _forward(loss, logits, targets, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return loss(logits, targets, **kwargs)


def _ptr(loss):
    return int(loss._queue._q_ptr)


class TestEnqueueSkipsIgnoredRows:
    @RANKING_LOSSES
    def test_padded_batches_leave_only_valid_rows_and_pointer_counts_them(self, loss_name):
        loss = _queued(loss_name, 1024)
        for step in range(8):
            _forward(loss, *_padded_batch(256, 26, seed=step, first_id=256 * step))
        stored = loss._queue._q_targets
        assert int((stored != IGNORE).sum()) == 208
        assert _ptr(loss) == 208
        assert bool((stored[:208] != IGNORE).all())
        assert bool((stored[208:] == IGNORE).all())

    @RANKING_LOSSES
    def test_padded_batches_eventually_fill_the_whole_queue_with_valid_rows(self, loss_name):
        loss = _queued(loss_name, 1024)
        for step in range(40):
            _forward(loss, *_padded_batch(256, 26, seed=step, first_id=256 * step))
        assert bool((loss._queue._q_targets != IGNORE).all())

    @RANKING_LOSSES
    def test_batch_without_ignored_rows_is_enqueued_whole_and_in_order(self, loss_name):
        logits, targets = make_batch(3, 0)
        loss = _queued(loss_name, 200)
        _forward(loss, logits, targets)
        assert _ptr(loss) == N
        assert torch.equal(loss._queue._q_logits[:N], logits)
        assert torch.equal(loss._queue._q_targets[:N], targets)
        _forward(loss, logits, targets)
        assert _ptr(loss) == 2 * N

    @RANKING_LOSSES
    def test_overflow_keeps_the_last_queue_size_valid_rows_in_order(self, loss_name):
        logits, targets = _padded_batch(120, 100, seed=1)
        loss = _queued(loss_name, 64)
        _forward(loss, logits, targets)
        valid = targets != IGNORE
        assert _ptr(loss) == 0
        assert torch.equal(loss._queue._q_logits, logits[valid][-64:])
        assert torch.equal(loss._queue._q_targets, targets[valid][-64:])

    @RANKING_LOSSES
    def test_valid_rows_equal_to_queue_size_replace_the_queue_wholesale(self, loss_name):
        loss = _queued(loss_name, 64)
        _forward(loss, *_padded_batch(40, 30, seed=2, first_id=1000))
        assert _ptr(loss) == 30
        logits, targets = _padded_batch(90, 64, seed=3)
        _forward(loss, logits, targets)
        valid = targets != IGNORE
        assert _ptr(loss) == 0
        assert torch.equal(loss._queue._q_logits, logits[valid])
        assert torch.equal(loss._queue._q_targets, targets[valid])

    @RANKING_LOSSES
    def test_wrap_around_pointer_and_order_use_valid_rows_only(self, loss_name):
        loss = _queued(loss_name, 10)
        stream = []
        for step in range(3):
            logits, targets = _padded_batch(9, 4, seed=step, first_id=100 * step)
            _forward(loss, logits, targets)
            stream.append(logits[targets != IGNORE])
        stream = torch.cat(stream)
        assert stream.size(0) == 12
        assert _ptr(loss) == 2
        expected = torch.cat([stream[10:], stream[2:10]])
        assert torch.equal(loss._queue._q_logits, expected)

    @RANKING_LOSSES
    def test_all_ignored_batch_leaves_a_filled_queue_untouched(self, loss_name):
        loss = _queued(loss_name, 64)
        _forward(loss, *_padded_batch(40, 30, seed=4))
        before = {k: v.clone() for k, v in loss._queue.state_dict().items()}
        logits, _ = _padded_batch(20, 0, seed=5, first_id=500)
        _forward(loss, logits, torch.full((20,), IGNORE))
        after = loss._queue.state_dict()
        assert before.keys() == after.keys()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert _ptr(loss) == 30

    @RANKING_LOSSES
    def test_zero_row_batch_leaves_a_filled_queue_untouched(self, loss_name):
        loss = _queued(loss_name, 64)
        _forward(loss, *_padded_batch(40, 30, seed=6))
        before = {k: v.clone() for k, v in loss._queue.state_dict().items()}
        _forward(loss, torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
        after = loss._queue.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert _ptr(loss) == 30

    @RANKING_LOSSES
    def test_stored_weights_align_with_stored_targets(self, loss_name):
        logits, targets = _padded_batch(60, 30, seed=7)
        g = torch.Generator().manual_seed(70)
        weight = torch.rand(60, generator=g) + 0.5
        loss = _queued(loss_name, 64)
        _forward(loss, logits, targets, sample_weight=weight)
        valid = targets != IGNORE
        assert _ptr(loss) == 30
        assert torch.equal(loss._queue._q_weight[:30], weight[valid])
        assert torch.equal(loss._queue._q_logits[:30], logits[valid])
        assert loss._queue.has_weights

    @RANKING_LOSSES
    def test_a_non_unit_weight_on_an_ignored_row_is_never_stored(self, loss_name):
        logits, targets = _padded_batch(60, 30, seed=8)
        weight = torch.ones(60)
        weight[int((targets == IGNORE).nonzero()[0])] = 5.0
        loss = _queued(loss_name, 64)
        _forward(loss, logits, targets, sample_weight=weight)
        assert not loss._queue.has_weights
        assert bool((loss._queue._q_weight == 1).all())

    @RANKING_LOSSES
    def test_stored_iid_flags_align_with_stored_targets(self, loss_name):
        logits, targets = _padded_batch(60, 30, seed=9)
        g = torch.Generator().manual_seed(90)
        iid = torch.rand(60, generator=g) < 0.5
        loss = _queued(loss_name, 64)
        _forward(loss, logits, targets, iid_mask=iid)
        valid = targets != IGNORE
        assert torch.equal(loss._queue._q_iid[:30], iid[valid])
        assert torch.equal(loss._queue._q_logits[:30], logits[valid])

    @RANKING_LOSSES
    def test_eval_mode_enqueues_only_when_update_queue_in_eval(self, loss_name):
        logits, targets = _padded_batch(60, 30, seed=10)
        idle = _queued(loss_name, 64).eval()
        _forward(idle, logits, targets)
        assert _ptr(idle) == 0
        assert bool((idle._queue._q_targets == IGNORE).all())

        updating = _queued(loss_name, 64, update_queue_in_eval=True).eval()
        _forward(updating, logits, targets)
        assert _ptr(updating) == 30
        assert torch.equal(updating._queue._q_targets[:30], targets[targets != IGNORE])

    @RANKING_LOSSES
    def test_custom_ignore_index_rows_are_not_enqueued(self, loss_name):
        logits, targets = _padded_batch(60, 30, seed=11)
        targets[targets == IGNORE] = 999
        loss = _queued(loss_name, 64, ignore_index=999)
        _forward(loss, logits, targets)
        assert _ptr(loss) == 30
        assert bool((loss._queue._q_targets[:30] != 999).all())
        assert bool((loss._queue._q_targets[:30] != IGNORE).all())

    @RANKING_LOSSES
    def test_forced_gather_in_a_single_process_stores_the_same_rows(
        self, loss_name, single_process_group
    ):
        logits, targets = _padded_batch(60, 30, seed=12)
        plain = _queued(loss_name, 64)
        gathered = _queued(loss_name, 64)
        gathered._gather_resolved = True
        _forward(plain, logits, targets)
        _forward(gathered, logits, targets)
        for key, value in plain._queue.state_dict().items():
            assert torch.equal(value, gathered._queue.state_dict()[key]), key
        assert _ptr(gathered) == 30

    @RANKING_LOSSES
    def test_baseline_checkpoint_with_ignored_slots_loads_and_scores_as_empty_slots(
        self, loss_name
    ):
        # A baseline checkpoint stores ignored rows (with their logits) in the
        # queue buffers. They are filtered at merge, so the next loss must
        # equal the loss with those slots empty.
        g = torch.Generator().manual_seed(13)
        q_logits = torch.randn(32, 3, generator=g)
        q_targets = torch.arange(32) % 3
        q_targets[::4] = IGNORE
        state = {
            "_q_logits": q_logits,
            "_q_targets": q_targets,
            "_q_ptr": torch.tensor([20]),
            "_q_iid": torch.ones(32, dtype=torch.bool),
            "_q_weight": torch.ones(32),
        }
        emptied = {k: v.clone() for k, v in state.items()}
        emptied["_q_logits"][q_targets == IGNORE] = 0.0

        live = make_batch(3, 14)
        results = []
        for buffers in (state, emptied):
            loss = _queued(loss_name, 32)
            loss._queue.load_state_dict(buffers)
            results.append(_forward(loss, *live))
        assert torch.isfinite(results[0])
        assert torch.equal(results[0], results[1])


class TestGoldenFixtureCompare:
    @pytest.fixture
    def golden(self):
        import _golden_sample_weight as golden

        return golden

    @pytest.fixture
    def new_fixture(self, golden):
        return torch.load(golden.FIXTURE_PATH, weights_only=True)

    def _run_compare(self, golden, old_fixture, tmp_path):
        import subprocess
        import sys

        old_path = tmp_path / "old.pt"
        torch.save(old_fixture, old_path)
        return subprocess.run(
            [sys.executable, golden.__file__, "--compare", str(old_path)],
            capture_output=True,
            text=True,
        )

    def test_allowed_set_is_the_step_2_entries_of_the_queue_size_64_configs(self, golden):
        allowed = golden.allowed_to_differ()
        configs = [c for c in golden.build_grid() if c.loss_kwargs.get("queue_size") == 64]
        assert len(configs) == 10
        assert len(allowed) == 10 * 2 * 3
        assert all("|step=2|" in k and "queue_size=64" in k for k in allowed)

    def test_perturbing_an_allowed_key_passes_and_is_reported(
        self, golden, new_fixture, tmp_path
    ):
        key = sorted(golden.allowed_to_differ())[0]
        old = dict(new_fixture)
        old[key] = new_fixture[key] + 1.0
        result = self._run_compare(golden, old, tmp_path)
        assert result.returncode == 0, result.stderr
        assert f"differs: {key}" in result.stdout

    def test_identical_fixtures_pass(self, golden, new_fixture, tmp_path):
        result = self._run_compare(golden, dict(new_fixture), tmp_path)
        assert result.returncode == 0, result.stderr
        assert "0 tensor(s) differ" in result.stdout

    @pytest.mark.parametrize("step", ["0", "1"])
    def test_perturbing_a_disallowed_key_fails_and_names_it(
        self, golden, new_fixture, tmp_path, step
    ):
        allowed = golden.allowed_to_differ()
        key = next(
            k for k in sorted(new_fixture)
            if "queue_size=64" in k and f"|step={step}|" in k and k.endswith("|loss")
        )
        assert key not in allowed
        old = dict(new_fixture)
        old[key] = new_fixture[key] + 1.0
        result = self._run_compare(golden, old, tmp_path)
        assert result.returncode != 0
        assert key in result.stderr

    def test_a_queue_size_0_step_2_key_is_not_allowed(self, golden, new_fixture, tmp_path):
        key = next(
            k for k in sorted(new_fixture)
            if "queue_size=0" in k and "|step=2|" in k and k.endswith("|grad")
        )
        old = dict(new_fixture)
        old[key] = new_fixture[key] + 1.0
        result = self._run_compare(golden, old, tmp_path)
        assert result.returncode != 0
        assert key in result.stderr

    def test_a_missing_key_fails(self, golden, new_fixture, tmp_path):
        key = sorted(golden.allowed_to_differ())[0]
        old = {k: v for k, v in new_fixture.items() if k != key}
        result = self._run_compare(golden, old, tmp_path)
        assert result.returncode != 0
        assert key in result.stderr

    def test_an_extra_key_fails(self, golden, new_fixture, tmp_path):
        old = dict(new_fixture)
        old["extra|dtype=float32|step=0|loss"] = torch.zeros(())
        result = self._run_compare(golden, old, tmp_path)
        assert result.returncode != 0
        assert "extra|dtype=float32|step=0|loss" in result.stderr


# ---------------------------------------------------------------------------
# Binary-mode target check (owned by _QueuedRankingLoss._validate_filtered_targets)
# ---------------------------------------------------------------------------

BINARY_WARNING = "expects targets in"
ENTRY_PATHS = pytest.mark.parametrize("entry", ["call", "forward", "wrapper"])


def _binary_batch(n: int = 40, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 1, generator=g)
    targets = (torch.rand(n, generator=g) < 0.4).long()
    targets[0], targets[1] = 0, 1  # both classes present
    return logits, targets


def _binary_warnings(caught):
    return [w for w in caught if BINARY_WARNING in str(w.message)]


def _record(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, _binary_warnings(caught)


class TestBinaryTargetCheck:
    @RANKING_LOSSES
    @ENTRY_PATHS
    def test_bad_target_warns_once_at_the_callers_line(self, loss_name, entry):
        from imbalanced_losses import LossWarmupWrapper

        logits, targets = _binary_batch()
        targets[5] = 2
        loss = make_loss(loss_name, 1, 0)
        wrapper = LossWarmupWrapper(
            warmup_loss=torch.nn.CrossEntropyLoss(), main_loss=loss, warmup_epochs=0
        )
        wrapper.on_train_batch_start(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if entry == "call":
                expected = inspect.currentframe().f_lineno + 1
                loss(logits, targets)
            elif entry == "forward":
                expected = inspect.currentframe().f_lineno + 1
                loss.forward(logits, targets)
            else:
                expected = inspect.currentframe().f_lineno + 1
                wrapper(logits, targets)
        hits = _binary_warnings(caught)
        assert len(hits) == 1
        assert issubclass(hits[0].category, UserWarning)
        assert hits[0].filename == __file__
        assert hits[0].lineno == expected

    @RANKING_LOSSES
    @pytest.mark.parametrize("queue_size", [0, 64])
    def test_valid_binary_targets_do_not_warn(self, loss_name, queue_size):
        logits, targets = _binary_batch()
        loss = make_loss(loss_name, 1, queue_size)
        _, hits = _record(lambda: [loss(logits, targets) for _ in range(2)])
        assert hits == []

    @RANKING_LOSSES
    def test_all_ignored_binary_batch_does_not_warn(self, loss_name):
        logits, _ = _binary_batch()
        targets = torch.full((logits.size(0),), -100, dtype=torch.long)
        loss = make_loss(loss_name, 1, 0)
        _, hits = _record(lambda: loss(logits, targets))
        assert hits == []

    @RANKING_LOSSES
    def test_float_targets_zero_and_one_do_not_warn(self, loss_name):
        logits, targets = _binary_batch()
        loss = make_loss(loss_name, 1, 0)
        _, hits = _record(lambda: loss(logits, targets.float()))
        assert hits == []

    @RANKING_LOSSES
    def test_float_target_half_warns(self, loss_name):
        logits, targets = _binary_batch()
        targets = targets.float()
        targets[5] = 0.5
        loss = make_loss(loss_name, 1, 0)
        _, hits = _record(lambda: loss(logits, targets))
        assert len(hits) == 1
        assert "[0.5]" in str(hits[0].message)

    @RANKING_LOSSES
    def test_minus_one_with_default_ignore_index_warns_and_counts_positive(self, loss_name):
        logits, targets = _binary_batch()
        bad = targets.clone()
        bad[5] = -1
        as_positive = targets.clone()
        as_positive[5] = 1
        out, hits = _record(lambda: make_loss(loss_name, 1, 0, ignore_index=-100)(logits, bad))
        assert len(hits) == 1
        assert "[-1]" in str(hits[0].message)
        assert torch.equal(
            out, make_loss(loss_name, 1, 0, ignore_index=-100)(logits, as_positive)
        )

    @RANKING_LOSSES
    def test_several_bad_rows_give_one_warning_listing_at_most_eight_values(self, loss_name):
        logits, targets = _binary_batch(n=60)
        targets[2:14] = torch.arange(2, 14)  # twelve bad rows
        loss = make_loss(loss_name, 1, 0)
        _, hits = _record(lambda: loss(logits, targets))
        assert len(hits) == 1
        message = str(hits[0].message)
        assert "found values: [2, 3, 4, 5, 6, 7, 8, 9]." in message
        assert message.endswith("Non-zero values are treated as positive.")

    @RANKING_LOSSES
    def test_lone_bad_target_warns_on_every_call_despite_subsampling(self, loss_name):
        g = torch.Generator().manual_seed(3)
        logits = torch.randn(400, 1, generator=g)
        targets = (torch.rand(400, generator=g) < 0.4).long()
        targets[123] = 2
        warned = 0
        for _ in range(30):
            loss = make_loss(loss_name, 1, 0, max_pool_size=2)
            _, hits = _record(lambda: loss(logits, targets))
            warned += len(hits) == 1
        assert warned == 30

    @RANKING_LOSSES
    def test_bad_target_only_in_the_queue_warns(self, loss_name):
        logits, targets = _binary_batch()
        loss = make_loss(loss_name, 1, 64)
        loss(logits, targets)
        loss._queue._q_targets[0] = 2
        _, hits = _record(lambda: loss(logits, targets))
        assert len(hits) == 1

    @RANKING_LOSSES
    def test_bad_target_loss_equals_the_loss_with_that_target_set_to_one(self, loss_name):
        logits, targets = _binary_batch()
        bad = targets.clone()
        bad[5] = 2
        as_positive = targets.clone()
        as_positive[5] = 1
        out, _ = _record(lambda: make_loss(loss_name, 1, 0)(logits, bad))
        assert torch.equal(out, make_loss(loss_name, 1, 0)(logits, as_positive))

    @RANKING_LOSSES
    def test_multiclass_has_no_binary_warning_and_still_raises(self, loss_name):
        logits, targets = _multiclass_batch(30)
        targets[3] = 7
        loss = make_loss(loss_name, 3, 0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="outside"):
                loss(logits, targets)
        assert _binary_warnings(caught) == []

    def test_override_without_super_gets_no_binary_warning(self):
        class NoCheck(SmoothAPLoss):
            def _validate_filtered_targets(self, targets, *, is_live=None):
                pass

        logits, targets = _binary_batch()
        targets[5] = 2
        _, hits = _record(lambda: NoCheck(num_classes=1, queue_size=0)(logits, targets))
        assert hits == []

    @pytest.mark.parametrize("module", ["ap_loss", "recall_loss", "pauc_loss"])
    def test_per_loss_modules_do_not_carry_the_warning_text(self, module):
        import importlib
        import pathlib

        source = pathlib.Path(importlib.import_module(f"imbalanced_losses.{module}").__file__)
        assert "Binary mode (num_classes=1)" not in source.read_text()
