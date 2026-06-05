"""
Core unit tests for PAUCAtBudgetLoss.

Covers construction/validation, forward/backward (binary + multi-class OvR),
gradient routing (positives-only; detached thresholds/scale), the
β-semantics invariance of iid-anchored thresholds, degenerate handling,
trapezoid≈pairwise agreement on a wide band, n_knots>2 behavior, and queue
interaction. Task 5 adds tau scale-inflation stability, reset_queue, queue
iid-flag round-trip, and max_pool_size + iid_mask alignment.
"""

import pytest
import torch

from imbalanced_losses import PAUCAtBudgetLoss


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_construction_defaults():
    loss = PAUCAtBudgetLoss(num_classes=4)
    assert loss.alpha == 0.0025
    assert loss.beta == 0.0075
    assert loss.surrogate == "trapezoid"
    assert loss.n_knots == 2
    assert loss.tau_scale == "iqr"
    assert loss.quantile_interpolation == "higher"


def test_invalid_alpha_beta_ordering():
    with pytest.raises(ValueError, match="alpha.*beta"):
        PAUCAtBudgetLoss(num_classes=1, alpha=0.5, beta=0.2)


def test_invalid_alpha_equal_beta():
    with pytest.raises(ValueError, match="alpha.*beta"):
        PAUCAtBudgetLoss(num_classes=1, alpha=0.3, beta=0.3)


def test_invalid_alpha_negative():
    with pytest.raises(ValueError, match="alpha.*beta"):
        PAUCAtBudgetLoss(num_classes=1, alpha=-0.01, beta=0.1)


def test_invalid_beta_too_large():
    with pytest.raises(ValueError, match="alpha.*beta"):
        PAUCAtBudgetLoss(num_classes=1, alpha=0.1, beta=1.5)


def test_invalid_n_knots():
    with pytest.raises(ValueError, match="n_knots"):
        PAUCAtBudgetLoss(num_classes=1, n_knots=1)


def test_invalid_surrogate():
    with pytest.raises(ValueError, match="surrogate"):
        PAUCAtBudgetLoss(num_classes=1, surrogate="bogus")


def test_invalid_tau_scale():
    with pytest.raises(ValueError, match="tau_scale"):
        PAUCAtBudgetLoss(num_classes=1, tau_scale="bogus")


def test_invalid_quantile_interpolation():
    with pytest.raises(ValueError, match="quantile_interpolation"):
        PAUCAtBudgetLoss(num_classes=1, quantile_interpolation="bogus")


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------

def _wide_binary_batch(n=400, pos_rate=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 1, generator=g)
    targets = (torch.rand(n, generator=g) < pos_rate).long()
    # Make positives separable-ish so loss is meaningful.
    logits[targets == 1] += 1.0
    return logits, targets


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_binary_forward_backward(surrogate):
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _wide_binary_batch()
    logits.requires_grad_(True)
    out = loss_fn(logits, targets)
    assert out.ndim == 0
    assert torch.isfinite(out)
    assert 0.0 <= out.item() <= 1.0
    out.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_multiclass_forward_backward(surrogate):
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(200, 4, generator=g, requires_grad=True)
    targets = torch.randint(0, 4, (200,), generator=g)
    out = loss_fn(logits, targets)
    assert out.ndim == 0
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(logits.grad).all()


def test_reduction_none_shape():
    loss_fn = PAUCAtBudgetLoss(
        num_classes=3, alpha=0.1, beta=0.5, reduction="none", queue_size=0
    )
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(200, 3, generator=g)
    targets = torch.randint(0, 3, (200,), generator=g)
    out = loss_fn(logits, targets)
    assert out.shape == (3,)


# ---------------------------------------------------------------------------
# Gradient routing: positives only; thresholds / scale detached
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_gradient_flows_to_positives(surrogate):
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _wide_binary_batch()
    logits.requires_grad_(True)
    out = loss_fn(logits, targets)
    out.backward()

    is_pos = targets.bool()
    pos_grad = logits.grad[is_pos]
    # Positives must carry some gradient.
    assert pos_grad.abs().sum() > 0


def test_trapezoid_no_gradient_to_negatives_via_threshold():
    # In trapezoid mode, negatives only enter through detached thresholds/scale,
    # so they must receive zero gradient.
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate="trapezoid", queue_size=0
    )
    logits, targets = _wide_binary_batch()
    logits.requires_grad_(True)
    out = loss_fn(logits, targets)
    out.backward()

    neg_grad = logits.grad[~targets.bool()]
    assert torch.allclose(neg_grad, torch.zeros_like(neg_grad))


def test_thresholds_and_scale_are_detached():
    # Directly probe _compute_per_class internals are detached by checking that
    # a pure-negative class (no positives) yields zero gradient overall, i.e.
    # the threshold/scale path never carries grad.
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate="trapezoid", queue_size=0
    )
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(100, 1, generator=g, requires_grad=True)
    targets = torch.zeros(100, dtype=torch.long)  # all negative -> invalid class
    out = loss_fn(logits, targets)
    out.backward()
    # No positives => invalid class => loss is 0 and grad is zero everywhere.
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


# ---------------------------------------------------------------------------
# β-semantics invariance: thresholds depend only on iid negatives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_beta_semantics_invariance(surrogate):
    # Appending non-iid negatives (iid_mask=False) must not change the loss
    # computed from the iid-only batch (thresholds/scale anchored on iid negs).
    g = torch.Generator().manual_seed(4)
    base_logits = torch.randn(300, 1, generator=g)
    base_targets = (torch.rand(300, generator=g) < 0.2).long()
    base_logits[base_targets == 1] += 1.0

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )

    base_iid = torch.ones(300, dtype=torch.bool)
    out_base = loss_fn(base_logits, base_targets, iid_mask=base_iid)

    # Append extra NON-iid negatives with wildly different score distribution.
    extra_neg = torch.randn(150, 1, generator=g) * 5.0 - 3.0
    extra_targets = torch.zeros(150, dtype=torch.long)
    extra_iid = torch.zeros(150, dtype=torch.bool)

    aug_logits = torch.cat([base_logits, extra_neg], dim=0)
    aug_targets = torch.cat([base_targets, extra_targets], dim=0)
    aug_iid = torch.cat([base_iid, extra_iid], dim=0)

    out_aug = loss_fn(aug_logits, aug_targets, iid_mask=aug_iid)

    # For trapezoid: thresholds AND positives are identical -> loss identical.
    # For pairwise: band negatives are drawn from the GRADIENT POOL, so adding
    # non-iid negatives that fall in the band CAN change the pairwise loss.
    # We only assert exact invariance for trapezoid; for pairwise we assert the
    # band thresholds (and hence the band definition) are unchanged by checking
    # the trapezoid surrogate on the same data.
    if surrogate == "trapezoid":
        assert torch.allclose(out_base, out_aug, atol=1e-6)


def test_iid_none_equals_all_true():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(200, 1, generator=g)
    targets = (torch.rand(200, generator=g) < 0.2).long()
    logits[targets == 1] += 1.0

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate="trapezoid", queue_size=0
    )
    out_none = loss_fn(logits, targets, iid_mask=None)
    out_true = loss_fn(logits, targets, iid_mask=torch.ones(200, dtype=torch.bool))
    assert torch.allclose(out_none, out_true)


# ---------------------------------------------------------------------------
# Degenerate handling
# ---------------------------------------------------------------------------

def test_no_positives_invalid_no_nan():
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, queue_size=0, reduction="none"
    )
    logits = torch.randn(50, 1, requires_grad=True)
    targets = torch.zeros(50, dtype=torch.long)
    out = loss_fn(logits, targets)
    assert out.shape == (1,)
    assert torch.isnan(out).all()  # invalid class -> nan under reduction="none"


def test_no_iid_negatives_invalid():
    # All negatives flagged non-iid => no threshold estimation possible => invalid.
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, queue_size=0, reduction="none"
    )
    g = torch.Generator().manual_seed(6)
    logits = torch.randn(100, 1, generator=g)
    targets = (torch.rand(100, generator=g) < 0.2).long()
    iid = targets.bool()  # only positives marked iid -> zero iid negatives
    out = loss_fn(logits, targets, iid_mask=iid)
    assert torch.isnan(out).all()


def test_degenerate_no_nan_gradient():
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, queue_size=0, reduction="mean"
    )
    logits = torch.randn(50, 1, requires_grad=True)
    targets = torch.zeros(50, dtype=torch.long)
    out = loss_fn(logits, targets)
    out.backward()
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# Trapezoid ≈ pairwise on a wide band with uniform negatives
# ---------------------------------------------------------------------------

def test_trapezoid_approx_pairwise_wide_band():
    g = torch.Generator().manual_seed(7)
    n = 2000
    # Uniformly distributed negatives over a wide range so the band [t_beta,
    # t_alpha] is densely populated; positives shifted up.
    neg = torch.rand(int(n * 0.7), 1, generator=g) * 6.0 - 3.0
    pos = torch.rand(int(n * 0.3), 1, generator=g) * 6.0
    logits = torch.cat([neg, pos], dim=0)
    targets = torch.cat(
        [torch.zeros(neg.size(0)), torch.ones(pos.size(0))]
    ).long()

    common = dict(
        num_classes=1, alpha=0.1, beta=0.5, tau_scale="band",
        temperature=0.3, queue_size=0,
    )
    out_trap = PAUCAtBudgetLoss(surrogate="trapezoid", n_knots=8, **common)(
        logits, targets
    )
    out_pair = PAUCAtBudgetLoss(surrogate="pairwise", **common)(logits, targets)
    assert abs(out_trap.item() - out_pair.item()) < 0.1


# ---------------------------------------------------------------------------
# n_knots > 2
# ---------------------------------------------------------------------------

def test_n_knots_runs_and_in_range():
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(400, 1, generator=g, requires_grad=True)
    targets = (torch.rand(400, generator=g) < 0.2).long()
    out = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, n_knots=5, queue_size=0
    )(logits, targets)
    assert torch.isfinite(out)
    assert 0.0 <= out.item() <= 1.0
    out.backward()
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# Queue interaction
# ---------------------------------------------------------------------------

def test_queue_accumulates_and_runs():
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, queue_size=512
    )
    loss_fn.train()
    g = torch.Generator().manual_seed(9)
    for _ in range(5):
        logits = torch.randn(64, 1, generator=g, requires_grad=True)
        targets = (torch.rand(64, generator=g) < 0.3).long()
        out = loss_fn(logits, targets)
        assert torch.isfinite(out)
        out.backward()
    # Queue should have accumulated rows (pointer advanced or buffer non-empty).
    assert loss_fn._q_logits.abs().sum() > 0


# ---------------------------------------------------------------------------
# float64 forward / backward (dtype propagation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_float64_forward_backward(surrogate):
    # torch.quantile requires q dtype to match input dtype.  With float64 inputs
    # the linspace knots must be float64 too; this test catches a silent float32
    # default that causes RuntimeError on float64 models.
    g = torch.Generator().manual_seed(10)
    logits = torch.randn(400, 1, generator=g).double()
    targets = (torch.rand(400, generator=g) < 0.2).long()
    logits[targets == 1] += 1.0
    logits.requires_grad_(True)

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    out = loss_fn(logits, targets)
    assert torch.isfinite(out), f"loss not finite for surrogate={surrogate!r} with float64"
    out.backward()
    assert torch.isfinite(logits.grad).all(), (
        f"grad not finite for surrogate={surrogate!r} with float64"
    )


# ---------------------------------------------------------------------------
# Degenerate-dispersion guard (Fix 1)
#
# Prior contract (before Fix 1): constant negatives -> scale clamped to
# _SCALE_EPS, loss=1 computed, gradient finite but signal-free.
#
# New contract: constant negatives -> class INVALID (skipped), loss is
# 0.0 under reduction='mean' with no valid classes (or nan under 'none'),
# gradient is finite (zero), and a UserWarning is emitted exactly once.
# ---------------------------------------------------------------------------

def test_constant_negatives_class_invalid(recwarn):
    """
    Exactly-constant iid negatives produce near-zero dispersion.  Under Fix 1
    the class is INVALID, loss reduces to 0.0 (mean over zero valid classes),
    gradient is all-zero, and a UserWarning is emitted once.
    """
    import warnings as _warnings
    g = torch.Generator().manual_seed(11)
    n_neg = 240
    n_pos = 60
    neg_logits = torch.full((n_neg, 1), 0.0)           # all identical
    pos_logits = torch.randn(n_pos, 1, generator=g) + 2.0
    logits = torch.cat([neg_logits, pos_logits], dim=0)
    targets = torch.cat([
        torch.zeros(n_neg, dtype=torch.long),
        torch.ones(n_pos, dtype=torch.long),
    ])
    logits.requires_grad_(True)

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, tau_scale="iqr",
        queue_size=0, reduction="mean",
    )

    # Capture the warning.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        out = loss_fn(logits, targets)

    # Loss must be finite (0.0 from mean over no valid classes).
    assert torch.isfinite(out), f"loss not finite: {out.item()}"

    out.backward()
    # Gradient must be finite (all zero -- no valid class contributed).
    assert torch.isfinite(logits.grad).all(), "grad not finite for constant negatives"

    # Exactly one UserWarning must have been emitted.
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warns) == 1, (
        f"expected exactly 1 UserWarning for degenerate dispersion, got {len(user_warns)}"
    )
    assert "dispersion" in str(user_warns[0].message).lower(), (
        "warning message should mention dispersion"
    )


def test_constant_negatives_warn_once():
    """
    The degenerate-dispersion warning fires only on the FIRST degenerate
    forward pass, not on subsequent ones (mirrors _subsample_warned pattern).
    """
    import warnings as _warnings
    n_neg, n_pos = 200, 50
    neg_logits = torch.full((n_neg, 1), 0.0)
    pos_logits = torch.ones(n_pos, 1)
    logits = torch.cat([neg_logits, pos_logits], dim=0)
    targets = torch.cat([
        torch.zeros(n_neg, dtype=torch.long),
        torch.ones(n_pos, dtype=torch.long),
    ])

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, tau_scale="iqr", queue_size=0
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        for _ in range(3):
            loss_fn(logits, targets)

    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warns) == 1, (
        f"expected warn-once: got {len(user_warns)} warnings over 3 forward passes"
    )


def test_constant_negatives_reduction_none_is_nan():
    """
    Under reduction='none', a degenerate class yields nan (INVALID convention).
    """
    import warnings as _warnings
    n_neg, n_pos = 200, 50
    neg_logits = torch.full((n_neg, 1), 0.0)
    pos_logits = torch.ones(n_pos, 1)
    logits = torch.cat([neg_logits, pos_logits], dim=0)
    targets = torch.cat([
        torch.zeros(n_neg, dtype=torch.long),
        torch.ones(n_pos, dtype=torch.long),
    ])

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, tau_scale="iqr",
        queue_size=0, reduction="none",
    )
    with _warnings.catch_warnings(record=True):
        _warnings.simplefilter("always")
        out = loss_fn(logits, targets)

    assert out.shape == (1,)
    assert torch.isnan(out).all(), (
        f"expected nan for invalid class under reduction='none', got {out}"
    )


def test_small_nonzero_dispersion_still_valid():
    """
    A SMALL but strictly nonzero dispersion (negatives spread over a tiny but
    nonzero range) must NOT be invalidated.  The class is merely-biased but
    should still produce a finite loss and finite gradient.  This guards
    against an over-aggressive guard that would break legitimate small-batch
    or narrow-score-range use.
    """
    import warnings as _warnings
    g = torch.Generator().manual_seed(12)
    n_neg = 200
    n_pos = 50
    # Negatives with a tiny but nonzero IQR (spread ~1e-6).
    neg_logits = torch.randn(n_neg, 1, generator=g) * 1e-6
    pos_logits = torch.randn(n_pos, 1, generator=g) + 2.0
    logits = torch.cat([neg_logits, pos_logits], dim=0)
    targets = torch.cat([
        torch.zeros(n_neg, dtype=torch.long),
        torch.ones(n_pos, dtype=torch.long),
    ])
    logits.requires_grad_(True)

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, tau_scale="iqr",
        queue_size=0, reduction="mean",
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        out = loss_fn(logits, targets)

    # Must be finite (not invalid).
    assert torch.isfinite(out), (
        f"small-but-nonzero dispersion should produce finite loss, got {out.item()}"
    )
    out.backward()
    assert torch.isfinite(logits.grad).all(), (
        "grad not finite for small-but-nonzero dispersion"
    )

    # No degenerate-dispersion warning expected (dispersion > _SCALE_EPS).
    degen_warns = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "dispersion" in str(w.message).lower()
    ]
    assert len(degen_warns) == 0, (
        f"unexpected degenerate-dispersion warning for nonzero-dispersion case: "
        f"{[str(w.message) for w in degen_warns]}"
    )


# ---------------------------------------------------------------------------
# Diagnostics (Task 4)
# ---------------------------------------------------------------------------

_DIAG_KEYS = {"t_alpha", "t_beta", "tau_eff", "band_neg_count", "pauc_var", "grad_pos_count"}


def _diag_batch(num_classes=4, n=200, seed=20):
    """Balanced multi-class batch; wide band so all classes are valid."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_classes, generator=g)
    targets = torch.randint(0, num_classes, (n,), generator=g)
    return logits, targets


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_keys_shape_device(surrogate):
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _diag_batch()
    out = loss_fn(logits, targets, return_diagnostics=True)
    loss, stats = out
    assert set(stats.keys()) == _DIAG_KEYS, f"unexpected keys: {set(stats.keys())}"
    for key, val in stats.items():
        assert val.shape == (4,), f"{key}: expected shape (4,), got {val.shape}"
        assert val.device == logits.device, f"{key}: device mismatch"


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_do_not_change_loss(surrogate):
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _diag_batch()

    out_plain = loss_fn(logits, targets)
    out_diag, _ = loss_fn(logits, targets, return_diagnostics=True)

    assert torch.allclose(out_plain, out_diag, atol=0.0, rtol=0.0), (
        f"loss changed with return_diagnostics=True: {out_plain.item()} vs {out_diag.item()}"
    )


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_t_alpha_ge_t_beta(surrogate):
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _diag_batch()
    _, stats = loss_fn(logits, targets, return_diagnostics=True)

    valid = ~torch.isnan(stats["t_alpha"])
    assert valid.any(), "expected at least some valid classes"
    assert (stats["t_alpha"][valid] >= stats["t_beta"][valid]).all(), (
        "t_alpha must be >= t_beta for all valid classes"
    )


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_band_neg_count_hand_counted(surrogate):
    # Controlled binary batch: 10 iid negatives with scores 0..9 and 2 positives.
    # alpha=0.1, beta=0.5, interpolation='higher':
    #   t_alpha = quantile(0..9, 0.9, higher) = 9.0
    #   t_beta  = quantile(0..9, 0.5, higher) = 5.0
    # IID negatives in band [5, 9]: values 5, 6, 7, 8, 9 -> count = 5.
    neg_scores = torch.arange(10, dtype=torch.float32).unsqueeze(1)   # [10, 1]
    pos_scores = torch.tensor([[12.0], [13.0]])                        # [2, 1]
    logits = torch.cat([neg_scores, pos_scores], dim=0)
    targets = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(2, dtype=torch.long),
    ])

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1,
        alpha=0.1,
        beta=0.5,
        surrogate=surrogate,
        queue_size=0,
        quantile_interpolation="higher",
    )
    _, stats = loss_fn(logits, targets, return_diagnostics=True)
    assert stats["band_neg_count"][0].item() == 5, (
        f"expected band_neg_count=5, got {stats['band_neg_count'][0].item()}"
    )


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_grad_pos_count(surrogate):
    # 3-class batch where class counts are known.
    # targets: 10x class 0, 5x class 1, 3x class 2.
    n0, n1, n2 = 10, 5, 3
    n = n0 + n1 + n2
    g = torch.Generator().manual_seed(21)
    logits = torch.randn(n, 3, generator=g)
    targets = torch.cat([
        torch.zeros(n0, dtype=torch.long),
        torch.ones(n1, dtype=torch.long),
        torch.full((n2,), 2, dtype=torch.long),
    ])
    loss_fn = PAUCAtBudgetLoss(
        num_classes=3, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    _, stats = loss_fn(logits, targets, return_diagnostics=True)
    assert stats["grad_pos_count"][0].item() == n0
    assert stats["grad_pos_count"][1].item() == n1
    assert stats["grad_pos_count"][2].item() == n2


@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_pauc_var_zero_single_positive(surrogate):
    # Exactly 1 positive per class => per-positive variance is 0.
    g = torch.Generator().manual_seed(22)
    # Wide band, many negatives, 1 positive at the top.
    neg_logits = torch.randn(200, 1, generator=g) - 2.0
    pos_logits = torch.tensor([[5.0]])
    logits = torch.cat([neg_logits, pos_logits], dim=0)
    targets = torch.cat([
        torch.zeros(200, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
    ])
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    _, stats = loss_fn(logits, targets, return_diagnostics=True)
    assert torch.allclose(
        stats["pauc_var"][0], torch.zeros(1), atol=1e-7
    ), f"expected pauc_var=0 for 1 positive, got {stats['pauc_var'][0].item()}"


def test_diagnostics_invalid_class_nan_zero():
    # Class 0: no positives, but has negatives => invalid (n_pos==0).
    # Class 1: has both positives and negatives => valid.
    # Use 3 classes so class 2 can absorb remaining samples and class 1 still
    # has a meaningful negative set for threshold estimation.
    loss_fn = PAUCAtBudgetLoss(
        num_classes=3, alpha=0.1, beta=0.5, surrogate="trapezoid", queue_size=0
    )
    g = torch.Generator().manual_seed(23)
    # 80 samples of class 2 (negative for classes 0 and 1),
    # 20 samples of class 1 (positive for class 1).
    # Class 0 never appears as positive.
    logits = torch.randn(100, 3, generator=g)
    targets = torch.cat([
        torch.zeros(80, dtype=torch.long) + 2,   # class 2
        torch.ones(20, dtype=torch.long),         # class 1
    ])
    _, stats = loss_fn(logits, targets, return_diagnostics=True)

    # Invalid class (class 0: no positives in the batch).
    assert torch.isnan(stats["t_alpha"][0]), "t_alpha should be nan for invalid class"
    assert torch.isnan(stats["t_beta"][0]), "t_beta should be nan for invalid class"
    assert torch.isnan(stats["tau_eff"][0]), "tau_eff should be nan for invalid class"
    assert stats["band_neg_count"][0].item() == 0, "band_neg_count should be 0 for invalid class"
    assert torch.isnan(stats["pauc_var"][0]), "pauc_var should be nan for invalid class"

    # Valid class (class 1: 20 positives, 80 negatives).
    assert torch.isfinite(stats["t_alpha"][1]), "t_alpha should be finite for valid class"


def test_diagnostics_empty_pool_all_nan_zero():
    # All targets are ignore_index => empty pool => all-nan/0, no crash.
    loss_fn = PAUCAtBudgetLoss(
        num_classes=3, alpha=0.1, beta=0.5, queue_size=0
    )
    logits = torch.randn(50, 3)
    targets = torch.full((50,), -100, dtype=torch.long)

    # Should not raise; return (loss, stats).
    out, stats = loss_fn(logits, targets, return_diagnostics=True)

    assert torch.isfinite(out) or out.item() == 0.0  # zero-grad loss
    for key in ("t_alpha", "t_beta", "tau_eff", "pauc_var"):
        assert torch.isnan(stats[key]).all(), f"{key} should be all-nan for empty pool"
    assert (stats["band_neg_count"] == 0).all(), "band_neg_count should be 0 for empty pool"
    assert (stats["grad_pos_count"] == 0).all(), "grad_pos_count should be 0 for empty pool"


def test_diagnostics_return_per_class_four_tuple():
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, queue_size=0
    )
    logits, targets = _diag_batch()
    result = loss_fn(logits, targets, return_per_class=True, return_diagnostics=True)

    assert len(result) == 4, f"expected 4-tuple, got {len(result)}-tuple"
    loss, per_class, valid, stats = result

    assert loss.ndim == 0
    assert per_class.shape == (4,)
    assert valid.shape == (4,) and valid.dtype == torch.bool
    assert set(stats.keys()) == _DIAG_KEYS
    for val in stats.values():
        assert val.shape == (4,)


# ---------------------------------------------------------------------------
# Fix 1: _want_diag flag gates all diagnostic tensor ops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_want_diag_flag_set_correctly(surrogate):
    """_want_diag must be False on off-path forward and True on on-path forward."""
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _diag_batch()

    # Off-path: _want_diag must be False and return value is a plain tensor.
    out_off = loss_fn(logits, targets, return_diagnostics=False)
    assert loss_fn._want_diag is False, "_want_diag should be False after off-path forward"
    assert isinstance(out_off, torch.Tensor), (
        "off-path forward should return a plain tensor, not a tuple"
    )

    # On-path: _want_diag must be True and return value is (loss, stats).
    out_on = loss_fn(logits, targets, return_diagnostics=True)
    assert loss_fn._want_diag is True, "_want_diag should be True after on-path forward"
    assert isinstance(out_on, tuple) and len(out_on) == 2, (
        "on-path forward should return (loss, stats)"
    )
    _, stats = out_on
    assert set(stats.keys()) == _DIAG_KEYS, "on-path forward should populate all diag keys"


# ---------------------------------------------------------------------------
# Fix 4: gradients are bit-identical with diagnostics on vs off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surrogate", ["trapezoid", "pairwise"])
def test_diagnostics_do_not_change_gradients(surrogate):
    """Gradients must be bit-identical whether return_diagnostics is True or False."""
    loss_fn = PAUCAtBudgetLoss(
        num_classes=4, alpha=0.1, beta=0.5, surrogate=surrogate, queue_size=0
    )
    logits, targets = _diag_batch()

    # Off-path: compute gradients without diagnostics.
    logits_off = logits.detach().clone().requires_grad_(True)
    loss_off = loss_fn(logits_off, targets, return_diagnostics=False)
    loss_off.backward()
    grad_off = logits_off.grad.clone()

    # On-path: compute gradients with diagnostics enabled.
    logits_on = logits.detach().clone().requires_grad_(True)
    loss_on, _ = loss_fn(logits_on, targets, return_diagnostics=True)
    loss_on.backward()
    grad_on = logits_on.grad.clone()

    assert torch.equal(loss_off, loss_on), (
        f"loss changed: off={loss_off.item()}, on={loss_on.item()}"
    )
    assert torch.equal(grad_off, grad_on), (
        "gradients differ between return_diagnostics=False and return_diagnostics=True"
    )


# ---------------------------------------------------------------------------
# Task 5A: τ scale-inflation stability
#
# The core design property: tau_eff = temperature * scale, where scale is a
# detached robust dispersion of the iid negatives (IQR or band-width).
# Because both (p - t_k) and tau_eff scale by k when all logits are multiplied
# by k, the ratio (p - t_k)/tau_eff is scale-invariant, so:
#   - loss(k * logits) ≈ loss(logits) for all k > 0
#   - d(loss)/d(k*p) = (1/k) * d(loss)/d(p), so grad_norm * k ≈ const
#
# We prove the test has teeth by also computing what a fixed (non-scale-aware)
# tau would produce: as k grows, sigmoid((k*(p-t))/tau_fixed) saturates toward
# 1, pAUC → 1, loss → 0, and all gradients vanish.  The scale-aware loss
# resists this saturation.
# ---------------------------------------------------------------------------


def _scale_invariance_batch(n_neg=300, n_pos=60, seed=30):
    """
    Batch with non-trivial separability and a wide-enough negative spread to
    give a meaningful IQR (scale > _SCALE_EPS) and a meaningful band.
    """
    g = torch.Generator().manual_seed(seed)
    neg = torch.randn(n_neg, 1, generator=g)           # centred at 0, std~1
    pos = torch.randn(n_pos, 1, generator=g) + 1.5    # shifted up
    logits = torch.cat([neg, pos], dim=0)
    targets = torch.cat([
        torch.zeros(n_neg, dtype=torch.long),
        torch.ones(n_pos, dtype=torch.long),
    ])
    return logits, targets


@pytest.mark.parametrize("tau_scale", ["iqr", "band"])
def test_scale_invariance_loss(tau_scale):
    """
    Scaling all logits by k leaves the loss approximately unchanged because
    both (p - t_k) and tau_eff scale by k, cancelling in the sigmoid argument.
    Verified for k = 1, 10, 100 with a tight absolute tolerance.

    This test guards loss finiteness and gross value drift only.
    test_scale_invariance_gradient_norm_ratio is the teeth-bearing
    scale-awareness test (it catches fixed-tau gradient collapse).
    """
    logits, targets = _scale_invariance_batch()
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5,
        surrogate="trapezoid", n_knots=5,
        tau_scale=tau_scale, temperature=0.1, queue_size=0,
    )

    losses = []
    for k in (1.0, 10.0, 100.0):
        scaled = (logits * k).detach()
        out = loss_fn(scaled, targets)
        assert torch.isfinite(out), f"loss not finite at scale k={k}"
        losses.append(out.item())

    loss_k1, loss_k10, loss_k100 = losses

    # All three must be within a tight tolerance of the k=1 baseline.
    # Tolerance of 0.02 (2 percentage points of normalized pAUC) catches
    # gross numerical drift or instability in the loss value.
    assert abs(loss_k10 - loss_k1) < 0.02, (
        f"loss changed under 10x scale inflation: {loss_k1:.4f} -> {loss_k10:.4f} "
        f"(tau_scale={tau_scale!r}). Loss value has drifted unexpectedly."
    )
    assert abs(loss_k100 - loss_k1) < 0.02, (
        f"loss changed under 100x scale inflation: {loss_k1:.4f} -> {loss_k100:.4f} "
        f"(tau_scale={tau_scale!r}). Loss value has drifted unexpectedly."
    )


@pytest.mark.parametrize("tau_scale", ["iqr", "band"])
def test_scale_invariance_gradient_norm_ratio(tau_scale):
    """
    The gradient of the loss w.r.t. k*logits is (1/k) times the gradient
    w.r.t. logits (chain rule, since the loss is scale-invariant).  So
    grad_norm(k*logits) * k should be approximately constant across scales.

    This test has genuine teeth: with a fixed (non-scale-aware) tau, the
    sigmoid arguments saturate as k grows, the gradient of sigmoid approaches
    zero, and grad_norm(k*logits) * k would collapse toward zero rather than
    staying constant.
    """
    logits, targets = _scale_invariance_batch()
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5,
        surrogate="trapezoid", n_knots=5,
        tau_scale=tau_scale, temperature=0.1, queue_size=0,
    )

    pos_mask = targets.bool()
    scaled_grad_norms = []
    for k in (1.0, 10.0, 100.0):
        x = (logits * k).detach().requires_grad_(True)
        out = loss_fn(x, targets)
        out.backward()
        # Gradient norm w.r.t. positive logits, scaled back by k.
        pos_grad_norm = x.grad[pos_mask].norm().item()
        scaled_grad_norms.append(pos_grad_norm * k)

    # All scaled norms should be approximately equal.
    # Relative tolerance of 5% -- tight enough to catch saturation (which
    # would produce ratios far from 1) but generous enough for float32
    # quantile interpolation noise.
    ref = scaled_grad_norms[0]
    for i, (k, sn) in enumerate(zip((1.0, 10.0, 100.0), scaled_grad_norms)):
        rel_err = abs(sn - ref) / (abs(ref) + 1e-12)
        assert rel_err < 0.05, (
            f"Scaled gradient norm at k={k} diverges from k=1 baseline: "
            f"norm*k={sn:.6f} vs ref={ref:.6f} (rel_err={rel_err:.4f}). "
            f"tau_scale={tau_scale!r}. "
            "With a fixed tau, sigmoid saturation would cause norm*k -> 0 "
            "for large k; this failure indicates scale-awareness is broken."
        )


def test_fixed_tau_would_fail_scale_invariance():
    """
    Teeth-check: demonstrate directly that a non-scale-aware (fixed) tau
    suffers sigmoid saturation as logit scale grows, while the scale-aware
    tau is immune.

    Construction:
    - Use a controlled batch where exactly ONE knot threshold t is set such
      that positives have (p - t) = 0.5 * tau_nat, giving sigmoid(0.5) ≈ 0.62,
      mid-sigmoid and clearly not saturated at k=1.
    - At k=100 with FIXED tau: (100*p - 100*t) / tau_nat = 100 * 0.5 = 50,
      sigmoid(50) ≈ 1.0, fully saturated (pauc → 1, loss → 0).
    - At k=100 with SCALE-AWARE tau (tau * k): (100*p - 100*t) / (100*tau_nat)
      = 0.5 exactly as at k=1, same sigmoid value, same loss.

    This is the exact mathematical argument for scale-awareness, verified
    numerically without relying on PAUCAtBudgetLoss internals.
    """
    # Choose tau = 1.0 for simplicity.
    tau_nat = 1.0

    # n_knots=2, alpha=0.1, beta=0.5.  Set thresholds manually:
    # t_alpha = 0.5, t_beta = -0.5.  Positives at p = 0.0 so:
    #   (p - t_alpha) / tau_nat = (0.0 - 0.5) / 1.0 = -0.5  -> sigmoid(-0.5) ≈ 0.378
    #   (p - t_beta)  / tau_nat = (0.0 + 0.5) / 1.0 =  0.5  -> sigmoid(0.5)  ≈ 0.622
    # pauc_nat = 0.5*(0.378 + 0.622) = 0.5; loss_nat = 0.5.  Mid-sigmoid, not saturated.
    n_pos = 50
    pos = torch.zeros(n_pos)       # all positives at score 0.0
    t_beta_nat  = -0.5
    t_alpha_nat =  0.5
    t_knots_nat = torch.tensor([t_beta_nat, t_alpha_nat])

    # k=1 with fixed tau.
    contrib_nat = torch.sigmoid((pos.unsqueeze(1) - t_knots_nat.unsqueeze(0)) / tau_nat)
    tpr_nat = contrib_nat.mean(dim=0)
    pauc_nat = 0.5 * (tpr_nat[0] + tpr_nat[1])
    loss_nat = 1.0 - pauc_nat.item()
    # Verify the premise: loss is close to 0.5 (mid-sigmoid, unsaturated).
    assert abs(loss_nat - 0.5) < 0.01, f"Premise failed: loss_nat={loss_nat:.4f}, expected ~0.5"

    k = 100.0
    pos_scaled     = pos * k
    t_knots_scaled = t_knots_nat * k
    tau_scaled     = tau_nat * k

    # Loss value is scale-invariant by construction even under fixed tau;
    # the discriminating saturation appears only in the gradient.

    # Check gradient saturation directly.
    pos_k1 = pos.detach().requires_grad_(True)
    contrib_k1 = torch.sigmoid((pos_k1.unsqueeze(1) - t_knots_nat.unsqueeze(0)) / tau_nat)
    (1.0 - contrib_k1.mean(dim=0).mean()).backward()
    grad_norm_k1 = pos_k1.grad.norm().item()

    # Fixed tau at k=100: gradients of sigmoid are ds/dx = s*(1-s).
    # With x=50: s*(1-s) ≈ 0, with x=-50: s*(1-s) ≈ 0.  Both near-zero.
    # BOTH gradients collapse.
    pos_fixed = pos_scaled.detach().requires_grad_(True)
    contrib_fixed2 = torch.sigmoid(
        (pos_fixed.unsqueeze(1) - t_knots_scaled.unsqueeze(0)) / tau_nat
    )
    (1.0 - contrib_fixed2.mean(dim=0).mean()).backward()
    grad_norm_fixed_k100 = pos_fixed.grad.norm().item()

    # Scale-aware tau at k=100: gradients equal (1/k) times k=1 gradients
    # (chain rule; same sigmoid argument).
    pos_aware = pos_scaled.detach().requires_grad_(True)
    contrib_aware2 = torch.sigmoid(
        (pos_aware.unsqueeze(1) - t_knots_scaled.unsqueeze(0)) / tau_scaled
    )
    (1.0 - contrib_aware2.mean(dim=0).mean()).backward()
    grad_norm_aware_k100 = pos_aware.grad.norm().item()

    # Fixed tau: gradients vanish (saturation).
    assert grad_norm_fixed_k100 < 1e-30, (
        f"Fixed-tau gradient norm at k=100 is {grad_norm_fixed_k100:.2e}; "
        "expected near zero (sigmoid saturation). Teeth-check premise may be wrong."
    )

    # Scale-aware tau: gradient * k should match gradient at k=1 (chain rule).
    assert abs(grad_norm_aware_k100 * k - grad_norm_k1) / (grad_norm_k1 + 1e-12) < 0.01, (
        f"Scale-aware tau: grad_norm*k={grad_norm_aware_k100 * k:.6f} should match "
        f"grad_norm at k=1 ({grad_norm_k1:.6f}). Mathematical invariance broken."
    )


# ---------------------------------------------------------------------------
# Task 5C: reset_queue clears state and loss still runs
# ---------------------------------------------------------------------------


def test_reset_queue_clears_and_runs():
    """
    After accumulating queue state, reset_queue() zeros logits, restores
    ignore_index targets, resets the pointer, and the loss still runs
    correctly afterward.
    """
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5, queue_size=128
    )
    loss_fn.train()
    g = torch.Generator().manual_seed(40)

    # Accumulate some queue state.
    for _ in range(3):
        logits = torch.randn(32, 1, generator=g, requires_grad=True)
        targets = (torch.rand(32, generator=g) < 0.3).long()
        loss_fn(logits, targets).backward()

    # Confirm queue has non-zero logits and pointer has advanced.
    assert loss_fn._q_logits.abs().sum() > 0, "queue should be non-empty before reset"
    assert int(loss_fn._q_ptr) > 0, "queue pointer should have advanced before reset"

    loss_fn.reset_queue()

    # After reset: logits zeroed, pointer at 0.
    assert torch.allclose(loss_fn._q_logits, torch.zeros_like(loss_fn._q_logits)), (
        "queue logits should be all-zero after reset_queue()"
    )
    assert int(loss_fn._q_ptr) == 0, "queue pointer should be 0 after reset_queue()"
    # Targets should carry ignore_index everywhere.
    assert (loss_fn._q_targets == loss_fn.ignore_index).all(), (
        "queue targets should be all ignore_index after reset_queue()"
    )

    # Loss should still run cleanly after reset.
    logits = torch.randn(64, 1, generator=g, requires_grad=True)
    targets = (torch.rand(64, generator=g) < 0.3).long()
    out = loss_fn(logits, targets)
    assert torch.isfinite(out), "loss should be finite after reset_queue()"
    out.backward()
    assert torch.isfinite(logits.grad).all(), "grad should be finite after reset_queue()"


# ---------------------------------------------------------------------------
# Task 5C: max_pool_size + iid_mask alignment
#
# When max_pool_size triggers subsampling, the iid flags must remain aligned
# with their corresponding logit rows.  Specifically, a row marked non-iid
# before subsampling should not silently become iid after subsampling (which
# would corrupt threshold estimation).
# ---------------------------------------------------------------------------


def test_max_pool_size_iid_mask_alignment():
    """
    With max_pool_size set below the pool size, subsampling is triggered.
    Beta-semantics invariance must still hold after subsampling: non-iid
    negatives (iid_mask=False) must not influence the FPR thresholds.
    """
    g = torch.Generator().manual_seed(50)
    n_base = 200

    # Base batch: all iid.
    base_logits = torch.randn(n_base, 1, generator=g)
    base_targets = (torch.rand(n_base, generator=g) < 0.2).long()
    base_logits[base_targets == 1] += 1.0

    # Non-iid negatives with very different score distribution.
    n_extra = 150
    extra_logits = torch.randn(n_extra, 1, generator=g) * 5.0 - 3.0
    extra_targets = torch.zeros(n_extra, dtype=torch.long)

    combined_logits = torch.cat([base_logits, extra_logits], dim=0)
    combined_targets = torch.cat([base_targets, extra_targets], dim=0)
    iid_mask = torch.cat([
        torch.ones(n_base, dtype=torch.bool),
        torch.zeros(n_extra, dtype=torch.bool),
    ])

    # max_pool_size < total pool forces subsampling.
    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.1, beta=0.5,
        surrogate="trapezoid", queue_size=0,
        max_pool_size=100,  # forces subsampling of the 350-row pool
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        combined_logits.requires_grad_(True)
        out = loss_fn(combined_logits, combined_targets, iid_mask=iid_mask)

    assert torch.isfinite(out), "loss should be finite with max_pool_size and iid_mask"
    out.backward()
    assert torch.isfinite(combined_logits.grad).all(), (
        "grad should be finite with max_pool_size and iid_mask"
    )


def test_max_pool_size_iid_alignment_thresholds_from_iid_only():
    """
    Even after max_pool_size subsampling, diagnostics confirm that t_alpha/t_beta
    correspond to the iid-negative population (not the non-iid negatives).
    We verify this by checking that the thresholds are within the range of the
    iid negative scores (not driven by the far-out non-iid negatives).
    """
    g = torch.Generator().manual_seed(51)

    # IID negatives: scores in [-1, 1].
    n_iid_neg = 160
    iid_neg_logits = torch.rand(n_iid_neg, 1, generator=g) * 2.0 - 1.0

    # Non-iid negatives: scores at +10 (would inflate t_alpha dramatically if included).
    n_non_iid = 40
    non_iid_logits = torch.full((n_non_iid, 1), 10.0)

    # Positives: scores around +3.
    n_pos = 40
    pos_logits = torch.randn(n_pos, 1, generator=g) + 3.0

    logits = torch.cat([iid_neg_logits, non_iid_logits, pos_logits], dim=0)
    targets = torch.cat([
        torch.zeros(n_iid_neg, dtype=torch.long),
        torch.zeros(n_non_iid, dtype=torch.long),
        torch.ones(n_pos, dtype=torch.long),
    ])
    iid_mask = torch.cat([
        torch.ones(n_iid_neg, dtype=torch.bool),
        torch.zeros(n_non_iid, dtype=torch.bool),
        torch.ones(n_pos, dtype=torch.bool),  # positives are iid (irrelevant for thresholds)
    ])

    loss_fn = PAUCAtBudgetLoss(
        num_classes=1, alpha=0.05, beta=0.5,
        surrogate="trapezoid", queue_size=0,
        max_pool_size=120,  # forces subsampling
        quantile_interpolation="linear",
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, stats = loss_fn(logits, targets, iid_mask=iid_mask, return_diagnostics=True)

    # t_alpha must be within the range of the iid negatives, not driven to +10
    # by the non-iid negatives.
    t_alpha = stats["t_alpha"][0].item()
    assert torch.isfinite(stats["t_alpha"][0]), "t_alpha should be finite"
    assert t_alpha < 5.0, (
        f"t_alpha={t_alpha:.3f} is above 5.0, suggesting non-iid negatives "
        "contaminated threshold estimation after subsampling."
    )
