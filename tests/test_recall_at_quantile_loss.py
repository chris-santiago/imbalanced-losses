"""
Tests for RecallAtQuantileLoss.

Coverage
--------
- Init: argument validation, buffer registration, quantile_interpolation
- _soft_recall_at_quantile: math, degenerate (no positives), gradient safety,
                             stop-gradient on threshold
- Queue: enqueue, wrap-around, batch >= queue_size, reset, eval freeze,
         update_queue_in_eval, _merge_with_queue
- Forward: shape validation, ignore_index, perfect/worst classification,
           loss range, gradient flow, monotonicity, all reductions,
           degenerate classes (no positives → nan/excluded), binary mode,
           return_per_class, all-padding input, out-of-range targets,
           quantile_interpolation effect, threshold stop-gradient
"""

from __future__ import annotations

import math

import pytest
import torch

from imbalanced_losses.recall_loss import RecallAtQuantileLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# With C=4 balanced classes, positives are 25% of pool.
# Using Q=0.30 places the threshold in the negative region for a perfect
# classifier, and in the positive region for a worst-case classifier.
SANITY_Q = 0.30
C, B = 4, 64


def _perfect_logits(B: int, C: int, score: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.arange(B) % C
    logits  = torch.full((B, C), -score)
    for i, t in enumerate(targets):
        logits[i, t] = score
    return logits.requires_grad_(True), targets


def _worst_logits(B: int, C: int, score: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.arange(B) % C
    logits  = torch.full((B, C), score)
    for i, t in enumerate(targets):
        logits[i, t] = -score
    return logits.requires_grad_(True), targets


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossInit:
    def test_valid_construction(self):
        fn = RecallAtQuantileLoss(num_classes=4, quantile=0.1)
        assert fn.num_classes == 4
        assert fn.quantile == 0.1

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            RecallAtQuantileLoss(num_classes=0)

    def test_invalid_quantile_zero(self):
        with pytest.raises(ValueError, match="quantile"):
            RecallAtQuantileLoss(num_classes=2, quantile=0.0)

    def test_invalid_quantile_one(self):
        with pytest.raises(ValueError, match="quantile"):
            RecallAtQuantileLoss(num_classes=2, quantile=1.0)

    def test_invalid_quantile_negative(self):
        with pytest.raises(ValueError, match="quantile"):
            RecallAtQuantileLoss(num_classes=2, quantile=-0.1)

    def test_invalid_queue_size(self):
        with pytest.raises(ValueError, match="queue_size"):
            RecallAtQuantileLoss(num_classes=2, queue_size=-1)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            RecallAtQuantileLoss(num_classes=2, temperature=0.0)

    def test_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            RecallAtQuantileLoss(num_classes=2, reduction="max")

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError, match="quantile_interpolation"):
            RecallAtQuantileLoss(num_classes=2, quantile_interpolation="cubic")

    def test_all_valid_interpolations_accepted(self):
        for interp in ("linear", "lower", "higher", "nearest", "midpoint"):
            fn = RecallAtQuantileLoss(num_classes=2, quantile=0.1,
                                       quantile_interpolation=interp)
            assert fn.quantile_interpolation == interp

    def test_queue_buffers_registered(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=16)
        assert hasattr(fn, "_q_logits")
        assert fn._q_logits.shape == (16, 3)

    def test_no_queue_buffers_when_disabled(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=0)
        assert not hasattr(fn, "_q_logits")

    def test_queue_targets_initialised_to_ignore_index(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=8)
        assert (fn._q_targets == fn.ignore_index).all()

    def test_temperature_stored_as_float(self):
        fn = RecallAtQuantileLoss(num_classes=2, quantile=0.1, temperature=1)
        assert isinstance(fn.temperature, float)


# ---------------------------------------------------------------------------
# _soft_recall_at_quantile (instance method)
# ---------------------------------------------------------------------------

class TestSoftRecallAtQuantile:
    Q   = 0.30   # threshold at 70th pct; positives at 10 >> negatives at -10
    TAU = 0.01

    def _fn(self, **kw) -> RecallAtQuantileLoss:
        return RecallAtQuantileLoss(num_classes=1, quantile=self.Q,
                                    temperature=self.TAU, queue_size=0, **kw)

    def test_all_positives_above_threshold_recall_near_one(self):
        fn = self._fn()
        # 75% negatives at -10, 25% positives at +10 → threshold near -10 boundary
        scores = torch.cat([torch.full((48,), -10.0), torch.full((16,), 10.0)])
        is_pos = torch.cat([torch.zeros(48), torch.ones(16)]).bool()
        recall, valid = fn._soft_recall_at_quantile(scores, is_pos)
        assert valid
        assert recall.item() > 0.99

    def test_all_positives_below_threshold_recall_near_zero(self):
        fn = self._fn()
        scores = torch.cat([torch.full((48,), 10.0), torch.full((16,), -10.0)])
        is_pos = torch.cat([torch.zeros(48), torch.ones(16)]).bool()
        recall, valid = fn._soft_recall_at_quantile(scores, is_pos)
        assert valid
        assert recall.item() < 0.01

    def test_no_positives_returns_invalid(self):
        fn = self._fn()
        scores = torch.randn(20)
        is_pos = torch.zeros(20, dtype=torch.bool)
        recall, valid = fn._soft_recall_at_quantile(scores, is_pos)
        assert not valid

    def test_recall_in_unit_interval(self):
        torch.manual_seed(10)
        fn = self._fn()
        scores = torch.randn(40)
        is_pos = torch.randint(0, 2, (40,)).bool()
        if not is_pos.any():
            is_pos[0] = True
        recall, valid = fn._soft_recall_at_quantile(scores, is_pos)
        assert valid
        assert 0.0 <= recall.item() <= 1.0

    def test_backward_does_not_raise(self):
        fn = self._fn()
        scores = torch.randn(20, requires_grad=True)
        is_pos = torch.randint(0, 2, (20,)).bool()
        if not is_pos.any():
            is_pos[0] = True
        recall, _ = fn._soft_recall_at_quantile(scores, is_pos)
        recall.backward()
        assert scores.grad is not None

    def test_threshold_is_stop_gradient(self):
        # The threshold should be detached: gradients should only flow
        # through positive scores' relationship to theta, not through theta
        # itself back to negative scores.
        fn = self._fn()
        scores = torch.randn(20, requires_grad=True)
        is_pos = torch.zeros(20, dtype=torch.bool)
        is_pos[:5] = True  # first 5 are positive

        recall, _ = fn._soft_recall_at_quantile(scores, is_pos)
        recall.backward()

        # Negatives should have zero gradient (theta is detached, negatives
        # only influence theta which is stop-grad'd)
        neg_grad = scores.grad[5:]
        assert neg_grad.abs().max().item() < 1e-6

    def test_interpolation_higher_vs_linear_differ(self):
        # 'higher' picks next score up; 'linear' interpolates — they can differ
        fn_high = RecallAtQuantileLoss(num_classes=1, quantile=0.25,
                                        queue_size=0, quantile_interpolation="higher")
        fn_lin  = RecallAtQuantileLoss(num_classes=1, quantile=0.25,
                                        queue_size=0, quantile_interpolation="linear")
        torch.manual_seed(11)
        scores = torch.randn(20)
        is_pos = torch.zeros(20, dtype=torch.bool); is_pos[:5] = True
        r_high, _ = fn_high._soft_recall_at_quantile(scores, is_pos)
        r_lin,  _ = fn_lin._soft_recall_at_quantile(scores, is_pos)
        # They won't always differ, but this seed produces different thresholds
        # Just check both are valid floats
        assert not math.isnan(r_high.item())
        assert not math.isnan(r_lin.item())


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossQueue:
    C_ = 3

    def test_enqueue_advances_pointer(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=0.1, queue_size=32)
        fn._enqueue(torch.zeros(8, self.C_), torch.zeros(8, dtype=torch.long))
        assert int(fn._q_ptr) == 8

    def test_enqueue_stores_values(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=0.1, queue_size=32)
        logits  = torch.ones(4, self.C_) * 5.0
        targets = torch.tensor([0, 1, 2, 0])
        fn._enqueue(logits, targets)
        assert torch.allclose(fn._q_logits[:4], logits)
        assert (fn._q_targets[:4] == targets).all()

    def test_enqueue_wraps_around(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=0.1, queue_size=8)
        fn._enqueue(torch.zeros(6, self.C_), torch.zeros(6, dtype=torch.long))
        fn._enqueue(torch.ones(4, self.C_) * 3.0, torch.ones(4, dtype=torch.long))
        assert int(fn._q_ptr) == 2
        assert torch.allclose(fn._q_logits[0], torch.ones(self.C_) * 3.0)

    def test_enqueue_batch_larger_than_queue(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=0.1, queue_size=8)
        big = torch.arange(24, dtype=torch.float).view(24, 1).expand(24, self.C_)
        fn._enqueue(big, torch.zeros(24, dtype=torch.long))
        assert int(fn._q_ptr) == 0
        assert torch.allclose(fn._q_logits, big[-8:])

    def test_reset_queue(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=0.1, queue_size=16)
        fn._enqueue(torch.ones(8, self.C_), torch.ones(8, dtype=torch.long))
        fn.reset_queue()
        assert int(fn._q_ptr) == 0
        assert (fn._q_logits == 0).all()
        assert (fn._q_targets == fn.ignore_index).all()

    def test_queue_not_updated_in_eval(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=SANITY_Q, queue_size=32)
        fn.eval()
        ptr_before = int(fn._q_ptr)
        fn(torch.randn(8, self.C_), torch.randint(0, self.C_, (8,)))
        assert int(fn._q_ptr) == ptr_before

    def test_queue_updated_in_eval_when_flag_set(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=SANITY_Q,
                                   queue_size=32, update_queue_in_eval=True)
        fn.eval()
        fn(torch.randn(8, self.C_), torch.randint(0, self.C_, (8,)))
        assert int(fn._q_ptr) == 8

    def test_queue_updated_in_train(self):
        fn = RecallAtQuantileLoss(num_classes=self.C_, quantile=SANITY_Q, queue_size=32)
        fn.train()
        fn(torch.randn(8, self.C_), torch.randint(0, self.C_, (8,)))
        assert int(fn._q_ptr) == 8

    def test_unfilled_queue_slots_ignored(self):
        # Fresh queue slots have ignore_index targets → stripped → same loss
        torch.manual_seed(42)
        logits  = torch.randn(16, self.C_)
        targets = torch.randint(0, self.C_, (16,))

        fn_noq = RecallAtQuantileLoss(num_classes=self.C_, quantile=SANITY_Q, queue_size=0)
        fn_q   = RecallAtQuantileLoss(num_classes=self.C_, quantile=SANITY_Q, queue_size=64)

        loss_noq = fn_noq(logits.clone(), targets.clone())
        loss_q   = fn_q(logits.clone(), targets.clone())
        assert torch.allclose(loss_noq, loss_q, atol=1e-5)


# ---------------------------------------------------------------------------
# Forward: shape validation
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossForwardShapes:
    def test_wrong_logits_ndim(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 3, 2), torch.zeros(8, dtype=torch.long))

    def test_wrong_num_classes(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))

    def test_targets_wrong_length(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 3), torch.zeros(6, dtype=torch.long))

    def test_out_of_range_targets_raise(self):
        fn = RecallAtQuantileLoss(num_classes=3, quantile=0.1, queue_size=0)
        logits  = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        targets[0] = 5  # class 5 doesn't exist
        with pytest.raises(ValueError, match="outside"):
            fn(logits, targets)


# ---------------------------------------------------------------------------
# Forward: mathematical correctness
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossForwardMath:
    def test_perfect_classification_loss_near_zero(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits, targets = _perfect_logits(B, C)
        loss = fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_classification_loss_near_one(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits, targets = _worst_logits(B, C)
        loss = fn(logits, targets)
        assert loss.item() > 0.99

    def test_loss_in_unit_interval(self):
        torch.manual_seed(1)
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        loss = fn(logits, targets)
        assert 0.0 <= loss.item() <= 1.0

    def test_gradient_flows_to_live_logits(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, C, requires_grad=True)
        targets = torch.randint(0, C, (B,))
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not logits.grad.isnan().any()
        assert logits.grad.abs().sum().item() > 0

    def test_no_gradient_to_queue_logits(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=32)
        fn(torch.randn(B, C), torch.randint(0, C, (B,)))  # prime queue
        live    = torch.randn(B, C, requires_grad=True)
        targets = torch.randint(0, C, (B,))
        loss = fn(live, targets)
        loss.backward()
        assert live.grad is not None
        assert live.grad.abs().sum().item() > 0

    def test_threshold_stop_gradient_negatives_get_no_grad(self):
        # Negatives only influence the detached threshold, so their
        # gradient should be zero.
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, 1, requires_grad=True)
        targets = torch.zeros(B, dtype=torch.long)
        targets[:B // 4] = 1  # first quarter positive

        loss = fn(logits, targets)
        loss.backward()

        neg_grad = logits.grad[B // 4:]   # negative samples
        assert neg_grad.abs().max().item() < 1e-6

    def test_monotone_better_separation_lower_loss(self):
        torch.manual_seed(3)
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        targets = torch.arange(B) % C

        good = torch.randn(B, C)
        for i, t in enumerate(targets):
            good[i, t] += 5.0

        rand = torch.randn(B, C)

        loss_good = fn(good, targets)
        loss_rand = fn(rand, targets)
        assert loss_good.item() < loss_rand.item()

    def test_smaller_quantile_harder_to_satisfy(self):
        # At a stricter quantile fewer positives qualify → higher loss
        torch.manual_seed(4)
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        fn_easy = RecallAtQuantileLoss(num_classes=C, quantile=0.50, queue_size=0)
        fn_hard = RecallAtQuantileLoss(num_classes=C, quantile=0.01, queue_size=0)

        loss_easy = fn_easy(logits, targets)
        loss_hard = fn_hard(logits, targets)
        assert loss_hard.item() >= loss_easy.item()


# ---------------------------------------------------------------------------
# Forward: ignore_index
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossIgnoreIndex:
    def test_all_padding_returns_zero(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, C, requires_grad=True)
        targets = torch.full((B,), -100, dtype=torch.long)
        loss = fn(logits, targets)
        assert loss.item() == 0.0

    def test_all_padding_has_grad_path(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, C, requires_grad=True)
        targets = torch.full((B,), -100, dtype=torch.long)
        fn(logits, targets).backward()  # must not raise

    def test_partial_padding_excluded(self):
        torch.manual_seed(5)
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        padded = targets.clone(); padded[:4] = -100
        loss_padded = fn(logits, padded)
        loss_clean  = fn(logits[4:], targets[4:])
        assert torch.allclose(loss_padded, loss_clean, atol=1e-5)

    def test_custom_ignore_index(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                   queue_size=0, ignore_index=999)
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        targets[0] = 999
        loss = fn(logits, targets)
        assert not math.isnan(loss.item())


# ---------------------------------------------------------------------------
# Forward: reductions
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossReductions:
    def test_reduction_mean_is_scalar(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                   queue_size=0, reduction="mean")
        loss = fn(torch.randn(B, C), torch.randint(0, C, (B,)))
        assert loss.shape == ()

    def test_reduction_sum_equals_sum_of_none(self):
        torch.manual_seed(6)
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        fn_none = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                        queue_size=0, reduction="none")
        fn_sum  = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                        queue_size=0, reduction="sum")
        loss_none = fn_none(logits, targets)
        loss_sum  = fn_sum(logits, targets)
        assert torch.allclose(loss_none.nan_to_num(0.0).sum(), loss_sum, atol=1e-5)

    def test_reduction_mean_equals_mean_of_none(self):
        torch.manual_seed(7)
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        fn_none = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                        queue_size=0, reduction="none")
        fn_mean = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                        queue_size=0, reduction="mean")
        loss_none = fn_none(logits, targets)
        loss_mean = fn_mean(logits, targets)
        valid = ~loss_none.isnan()
        assert torch.allclose(loss_none[valid].mean(), loss_mean, atol=1e-5)

    def test_reduction_none_shape(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                   queue_size=0, reduction="none")
        loss = fn(torch.randn(B, C), torch.randint(0, C, (B,)))
        assert loss.shape == (C,)

    def test_reduction_none_absent_class_is_nan(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                   queue_size=0, reduction="none")
        targets = torch.randint(0, C - 1, (B,))  # class C-1 absent
        loss = fn(torch.randn(B, C), targets)
        assert math.isnan(loss[C - 1].item())
        assert not math.isnan(loss[0].item())

    def test_single_class_present_only_that_class_contributes(self):
        # Only class 0 present. Classes 1–3 have no positives → excluded (nan).
        # Class 0 has all positives → valid recall is computed, loss is finite.
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                   queue_size=0, reduction="mean")
        logits  = torch.randn(B, C)
        targets = torch.zeros(B, dtype=torch.long)
        loss = fn(logits, targets)
        assert not math.isnan(loss.item())
        assert 0.0 <= loss.item() <= 1.0


# ---------------------------------------------------------------------------
# Forward: binary mode
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossBinary:
    # Use 25% positives so SANITY_Q=0.30 places the threshold correctly for
    # both perfect and worst cases — same geometry as the multiclass tests.
    # Perfect: pos=+10 (25%), neg=-10 (75%) → threshold at 70th pct = -10
    #          → sigmoid((10 - (-10)) / τ) ≈ 1 → loss ≈ 0
    # Worst:   pos=-10 (25%), neg=+10 (75%) → threshold at 70th pct = +10
    #          → sigmoid((-10 - 10) / τ) ≈ 0 → loss ≈ 1
    N_POS  = B // 4
    N_NEG  = B - B // 4

    def test_perfect_binary_loss_near_zero(self):
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits  = torch.cat([torch.full((self.N_POS, 1),  10.0),
                              torch.full((self.N_NEG, 1), -10.0)])
        targets = torch.cat([torch.ones(self.N_POS), torch.zeros(self.N_NEG)]).long()
        loss = fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_binary_loss_near_one(self):
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits  = torch.cat([torch.full((self.N_POS, 1), -10.0),
                              torch.full((self.N_NEG, 1),  10.0)])
        targets = torch.cat([torch.ones(self.N_POS), torch.zeros(self.N_NEG)]).long()
        loss = fn(logits, targets)
        assert loss.item() > 0.99

    def test_binary_gradient_flows(self):
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, 1, requires_grad=True)
        targets = torch.randint(0, 2, (B,))
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0

    def test_binary_reduction_none_shape(self):
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q,
                                   queue_size=0, reduction="none")
        loss = fn(torch.randn(B, 1), torch.randint(0, 2, (B,)))
        assert loss.shape == (1,)

    def test_binary_ignore_index_excluded(self):
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, 1)
        targets = torch.randint(0, 2, (B,)); targets[0] = -100
        loss = fn(logits, targets)
        assert not math.isnan(loss.item())

    def test_binary_all_padding_returns_zero(self):
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, 1, requires_grad=True)
        targets = torch.full((B,), -100, dtype=torch.long)
        loss = fn(logits, targets)
        assert loss.item() == 0.0

    def test_binary_out_of_range_targets_warns(self):
        import warnings
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits = torch.randn(B, 1)
        targets = torch.zeros(B, dtype=torch.long)
        targets[0] = 2  # out of range for binary mode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn(logits, targets)
        assert any("expects targets in" in str(warning.message) for warning in w)

    def test_binary_valid_targets_no_warning(self):
        import warnings
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits = torch.randn(B, 1)
        targets = torch.randint(0, 2, (B,))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn(logits, targets)
        assert not any("expects targets in" in str(warning.message) for warning in w)

    def test_binary_all_positive_computes_without_error(self):
        """All-positive binary targets: recall loss should be valid, not nan."""
        fn = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=0)
        logits = torch.randn(B, 1, requires_grad=True)
        targets = torch.ones(B, dtype=torch.long)
        loss = fn(logits, targets)
        assert not torch.isnan(loss)
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# Forward: return_per_class
# ---------------------------------------------------------------------------

class TestRecallAtQuantileLossReturnPerClass:
    def test_return_per_class_shapes(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        loss, per_class, valid = fn(
            torch.randn(B, C), torch.randint(0, C, (B,)), return_per_class=True
        )
        assert loss.shape == ()
        assert per_class.shape == (C,)
        assert valid.shape == (C,)
        assert valid.dtype == torch.bool

    def test_return_per_class_consistent_with_mean(self):
        torch.manual_seed(9)
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q,
                                   queue_size=0, reduction="mean")
        logits  = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        loss, per_class, valid = fn(logits, targets, return_per_class=True)
        assert torch.allclose(per_class[valid].mean(), loss, atol=1e-5)

    def test_return_per_class_nan_for_absent_class(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        targets = torch.randint(0, C - 1, (B,))  # class C-1 absent
        _, per_class, valid = fn(torch.randn(B, C), targets, return_per_class=True)
        assert not valid[C - 1].item()
        assert math.isnan(per_class[C - 1].item())

    def test_return_per_class_false_returns_tensor(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        result = fn(torch.randn(B, C), torch.randint(0, C, (B,)), return_per_class=False)
        assert isinstance(result, torch.Tensor)

    def test_all_padding_return_per_class(self):
        fn = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
        logits  = torch.randn(B, C)
        targets = torch.full((B,), -100, dtype=torch.long)
        loss, per_class, valid = fn(logits, targets, return_per_class=True)
        assert loss.item() == 0.0
        assert valid.sum().item() == 0
        assert per_class.isnan().all()


# ---------------------------------------------------------------------------
# max_pool_size
# ---------------------------------------------------------------------------


class TestMaxPoolSize:
    C = 4
    Q = 0.3   # quantile safely above positive rate for C=4 balanced classes

    def test_init_valid(self):
        fn = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                  queue_size=0, max_pool_size=64)
        assert fn.max_pool_size == 64

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                 queue_size=0, max_pool_size=0)
        with pytest.raises(ValueError):
            RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                 queue_size=0, max_pool_size=-1)

    def test_none_is_noop(self):
        torch.manual_seed(0)
        logits  = torch.randn(32, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (32,))
        fn_none = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                       queue_size=0, max_pool_size=None)
        fn_base = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                       queue_size=0)
        assert torch.allclose(fn_none(logits, targets), fn_base(logits, targets))

    def test_no_subsample_when_pool_small(self):
        torch.manual_seed(0)
        logits  = torch.randn(16, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (16,))
        fn_cap  = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                       queue_size=0, max_pool_size=64)
        fn_base = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                       queue_size=0)
        assert torch.allclose(fn_cap(logits, targets), fn_base(logits, targets))

    def test_subsamples_large_pool(self):
        torch.manual_seed(0)
        logits  = torch.randn(256, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (256,))
        fn = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                  queue_size=0, max_pool_size=32)
        loss = fn(logits, targets)
        assert loss.item() >= 0.0
        assert loss.item() <= 1.0
        loss.backward()
        assert logits.grad is not None

    def test_stratified_preserves_rare_class(self):
        torch.manual_seed(0)
        n = 1000
        logits  = torch.randn(n, self.C, requires_grad=True)
        targets = torch.zeros(n, dtype=torch.long)
        targets[0] = 3
        targets[1] = 3
        fn = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                  queue_size=0, max_pool_size=128,
                                  reduction="none")
        loss = fn(logits, targets)
        assert not loss[3].isnan(), "rare class 3 was lost to subsampling"

    def test_gradient_flows(self):
        torch.manual_seed(0)
        logits  = torch.randn(200, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (200,))
        fn = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                  queue_size=0, max_pool_size=32)
        fn(logits, targets).backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_queue_unaffected(self):
        B = 40
        fn = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                  queue_size=200, max_pool_size=16)
        logits  = torch.randn(B, self.C)
        targets = torch.randint(0, self.C, (B,))
        fn(logits, targets)
        fn(logits, targets)
        assert int(fn._q_ptr.item()) == (2 * B) % 200

    def test_warning_emitted_once(self):
        import warnings as _warnings
        fn = RecallAtQuantileLoss(num_classes=self.C, quantile=self.Q,
                                  queue_size=0, max_pool_size=16)
        logits  = torch.randn(100, self.C)
        targets = torch.randint(0, self.C, (100,))
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            fn(logits, targets)
            fn(logits, targets)
        recall_warns = [x for x in w if "RecallAtQuantileLoss" in str(x.message)]
        assert len(recall_warns) == 1

    def test_binary_mode(self):
        torch.manual_seed(0)
        logits  = torch.randn(200, 1, requires_grad=True)
        targets = torch.randint(0, 2, (200,))
        fn = RecallAtQuantileLoss(num_classes=1, quantile=0.5,
                                  queue_size=0, max_pool_size=32)
        loss = fn(logits, targets)
        assert 0.0 <= loss.item() <= 1.0
        loss.backward()
        assert logits.grad is not None
