"""
Tests for SmoothAPLoss.

Coverage
--------
- Init: argument validation
- _compute_smooth_ap: math, degenerate inputs, gradient safety
- Queue: enqueue, wrap-around, batch >= queue_size, reset, eval freeze,
         update_queue_in_eval flag, _merge_with_queue
- Forward: shape validation, ignore_index, perfect/worst classification,
           loss range, gradient flow, monotonicity, all reductions,
           degenerate classes, binary mode, return_per_class,
           all-padding input, temperature sensitivity
"""

from __future__ import annotations

import math

import pytest
import torch

from ap_loss import SmoothAPLoss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_logits(
    B: int, C: int, score: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits where logits[i, target_i] >> all other cols."""
    targets = torch.arange(B) % C
    logits = torch.full((B, C), -score)
    for i, t in enumerate(targets):
        logits[i, t] = score
    return logits.requires_grad_(True), targets


def _worst_logits(
    B: int, C: int, score: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverted: logits[i, target_i] << all other cols."""
    targets = torch.arange(B) % C
    logits = torch.full((B, C), score)
    for i, t in enumerate(targets):
        logits[i, t] = -score
    return logits.requires_grad_(True), targets


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestSmoothAPLossInit:
    def test_valid_construction(self):
        fn = SmoothAPLoss(num_classes=4)
        assert fn.num_classes == 4

    def test_invalid_num_classes_zero(self):
        with pytest.raises(ValueError, match="num_classes"):
            SmoothAPLoss(num_classes=0)

    def test_invalid_num_classes_negative(self):
        with pytest.raises(ValueError, match="num_classes"):
            SmoothAPLoss(num_classes=-1)

    def test_invalid_queue_size_negative(self):
        with pytest.raises(ValueError, match="queue_size"):
            SmoothAPLoss(num_classes=2, queue_size=-1)

    def test_invalid_temperature_zero(self):
        with pytest.raises(ValueError, match="temperature"):
            SmoothAPLoss(num_classes=2, temperature=0.0)

    def test_invalid_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature"):
            SmoothAPLoss(num_classes=2, temperature=-0.1)

    def test_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            SmoothAPLoss(num_classes=2, reduction="max")

    def test_queue_buffers_registered_when_queue_size_positive(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=16)
        assert hasattr(fn, "_q_logits")
        assert hasattr(fn, "_q_targets")
        assert hasattr(fn, "_q_ptr")
        assert fn._q_logits.shape == (16, 3)
        assert fn._q_targets.shape == (16,)

    def test_no_queue_buffers_when_queue_size_zero(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=0)
        assert not hasattr(fn, "_q_logits")

    def test_queue_targets_initialised_to_ignore_index(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=8, ignore_index=-100)
        assert (fn._q_targets == -100).all()

    def test_temperature_stored_as_float(self):
        fn = SmoothAPLoss(num_classes=2, temperature=1)
        assert isinstance(fn.temperature, float)


# ---------------------------------------------------------------------------
# _compute_smooth_ap (static method)
# ---------------------------------------------------------------------------


class TestComputeSmoothAP:
    TAU = 0.01

    def test_perfect_separation_ap_near_one(self):
        # Positives score >> negatives → AP ≈ 1
        scores = torch.tensor([10.0, 10.0, -10.0, -10.0])
        is_pos = torch.tensor([True, True, False, False])
        ap, valid = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        assert valid
        assert ap.item() > 0.99

    def test_worst_separation_ap_near_zero(self):
        # 1 positive ranked last among 20 samples → soft-AP ≈ 1/20 = 0.05.
        # Need heavy imbalance; with only 2 pos / 2 neg the minimum AP is ~0.43.
        scores = torch.cat([torch.full((1,), -10.0), torch.full((19,), 10.0)])
        is_pos = torch.cat([torch.ones(1), torch.zeros(19)]).bool()
        ap, valid = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        assert valid
        assert ap.item() < 0.1

    def test_degenerate_all_positive_returns_invalid(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        is_pos = torch.ones(3, dtype=torch.bool)
        ap, valid = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        assert not valid

    def test_degenerate_all_negative_returns_invalid(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        is_pos = torch.zeros(3, dtype=torch.bool)
        ap, valid = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        assert not valid

    def test_ap_in_unit_interval(self):
        torch.manual_seed(0)
        scores = torch.randn(20)
        is_pos = torch.randint(0, 2, (20,)).bool()
        if is_pos.all() or (~is_pos).all():
            is_pos[0] = True
            is_pos[1] = False
        ap, valid = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        assert valid
        assert 0.0 <= ap.item() <= 1.0

    def test_ap_monotone_with_score_gap(self):
        # Wider gap between pos/neg → higher AP.
        # Use τ=1.0 so the sigmoid isn't saturated at small gaps; at τ=0.01
        # even a ±1 gap gives σ(200) ≈ 1, making both cases indistinguishable.
        tau_large = 1.0
        is_pos = torch.tensor([True, True, False, False])
        scores_narrow = torch.tensor([0.1, 0.1, -0.1, -0.1])
        scores_wide = torch.tensor([5.0, 5.0, -5.0, -5.0])
        ap_narrow, _ = SmoothAPLoss._compute_smooth_ap(scores_narrow, is_pos, tau_large)
        ap_wide, _ = SmoothAPLoss._compute_smooth_ap(scores_wide, is_pos, tau_large)
        assert ap_wide.item() > ap_narrow.item()

    def test_backward_does_not_raise(self):
        scores = torch.tensor([2.0, -1.0, 1.0, -2.0], requires_grad=True)
        is_pos = torch.tensor([True, False, True, False])
        ap, _ = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        ap.backward()
        assert scores.grad is not None
        assert not scores.grad.isnan().any()

    def test_gradient_zero_on_negatives_is_not_required_but_positive_grad_exists(self):
        # Use interleaved scores so AP is far from 1 and gradients are non-trivial.
        # Scores: pos=[-0.5, 0.5], neg=[1.0, -1.0] → one pos above one neg, one below.
        scores = torch.tensor([-0.5, 1.0, 0.5, -1.0], requires_grad=True)
        is_pos = torch.tensor([True, False, True, False])
        ap, _ = SmoothAPLoss._compute_smooth_ap(scores, is_pos, tau=1.0)
        ap.backward()
        assert scores.grad[[0, 2]].abs().sum().item() > 0

    def test_single_positive_single_negative(self):
        scores = torch.tensor([1.0, -1.0])
        is_pos = torch.tensor([True, False])
        ap, valid = SmoothAPLoss._compute_smooth_ap(scores, is_pos, self.TAU)
        assert valid
        assert ap.item() > 0.5  # positive scores higher → AP > 0.5


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------


class TestSmoothAPLossQueue:
    C = 3

    def test_enqueue_advances_pointer(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=32)
        fn._enqueue(torch.zeros(8, self.C), torch.zeros(8, dtype=torch.long))
        assert int(fn._q_ptr) == 8

    def test_enqueue_stores_values(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=32)
        logits = torch.ones(4, self.C) * 7.0
        targets = torch.tensor([0, 1, 2, 0])
        fn._enqueue(logits, targets)
        assert torch.allclose(fn._q_logits[:4], logits)
        assert (fn._q_targets[:4] == targets).all()

    def test_enqueue_wraps_around(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=8)
        # Fill to near capacity then overflow by 4
        fn._enqueue(torch.zeros(6, self.C), torch.zeros(6, dtype=torch.long))
        fn._enqueue(torch.ones(4, self.C) * 9.0, torch.ones(4, dtype=torch.long))
        # Pointer should be at (6+4) % 8 = 2
        assert int(fn._q_ptr) == 2
        # The 2 wrapped rows should be at indices 0 and 1
        assert torch.allclose(fn._q_logits[0], torch.ones(self.C) * 9.0)
        assert torch.allclose(fn._q_logits[1], torch.ones(self.C) * 9.0)

    def test_enqueue_batch_larger_than_queue_replaces_whole_buffer(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=8)
        big = torch.arange(32, dtype=torch.float).view(32, 1).expand(32, self.C)
        fn._enqueue(big, torch.zeros(32, dtype=torch.long))
        assert int(fn._q_ptr) == 0
        # Buffer should hold last queue_size rows of the big batch
        assert torch.allclose(fn._q_logits, big[-8:])

    def test_enqueue_no_op_when_queue_size_zero(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        fn._enqueue(
            torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long)
        )  # should not raise

    def test_reset_queue_clears_buffer(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=16)
        fn._enqueue(torch.ones(8, self.C), torch.ones(8, dtype=torch.long))
        fn.reset_queue()
        assert int(fn._q_ptr) == 0
        assert (fn._q_logits == 0).all()
        assert (fn._q_targets == fn.ignore_index).all()

    def test_merge_with_queue_no_queue(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        out_l, out_t = fn._merge_with_queue(logits, targets)
        assert out_l is logits
        assert out_t is targets

    def test_merge_with_queue_concatenates(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=8)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        out_l, out_t = fn._merge_with_queue(logits, targets)
        assert out_l.shape == (4 + 8, self.C)
        assert out_t.shape == (4 + 8,)

    def test_queue_not_updated_in_eval_by_default(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=32)
        fn.eval()
        ptr_before = int(fn._q_ptr)
        fn(torch.randn(8, self.C), torch.randint(0, self.C, (8,)))
        assert int(fn._q_ptr) == ptr_before

    def test_queue_updated_in_eval_when_flag_set(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=32, update_queue_in_eval=True)
        fn.eval()
        fn(torch.randn(8, self.C), torch.randint(0, self.C, (8,)))
        assert int(fn._q_ptr) == 8

    def test_queue_updated_in_train_mode(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=32)
        fn.train()
        fn(torch.randn(8, self.C), torch.randint(0, self.C, (8,)))
        assert int(fn._q_ptr) == 8

    def test_unfilled_queue_slots_ignored_in_forward(self):
        # A fresh queue is full of ignore_index targets; loss should equal
        # the no-queue version since those slots are stripped.
        torch.manual_seed(42)
        logits = torch.randn(16, self.C)
        targets = torch.randint(0, self.C, (16,))

        fn_noq = SmoothAPLoss(num_classes=self.C, queue_size=0)
        fn_q = SmoothAPLoss(num_classes=self.C, queue_size=64)

        loss_noq = fn_noq(logits.clone(), targets.clone())
        loss_q = fn_q(logits.clone(), targets.clone())
        assert torch.allclose(loss_noq, loss_q, atol=1e-5)


# ---------------------------------------------------------------------------
# Forward: shape validation
# ---------------------------------------------------------------------------


class TestSmoothAPLossForwardShapes:
    def test_wrong_logits_ndim(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 3, 2), torch.zeros(8, dtype=torch.long))

    def test_wrong_num_classes(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))

    def test_targets_wrong_length(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 3), torch.zeros(6, dtype=torch.long))

    def test_targets_2d_raises(self):
        fn = SmoothAPLoss(num_classes=3, queue_size=0)
        with pytest.raises(ValueError):
            fn(torch.randn(8, 3), torch.zeros(8, 2, dtype=torch.long))


# ---------------------------------------------------------------------------
# Forward: mathematical correctness
# ---------------------------------------------------------------------------


class TestSmoothAPLossForwardMath:
    C, B = 4, 32

    def test_perfect_classification_loss_near_zero(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits, targets = _perfect_logits(self.B, self.C)
        loss = fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_classification_loss_near_one(self):
        # With balanced classes (B/C positives each), the minimum one-vs-rest AP
        # is ~0.15 with C=4, B=32 — loss only reaches ~0.85, not 0.99.
        # Use 1 positive per class so each class has C-1 negatives >> 1 positive.
        # With C=4, B=4: 1 pos + 3 neg per class → worst AP ≈ 1/4 = 0.25,
        # which is still too high. Use C=4 classes with 1 pos each and
        # many extra negatives by repeating the negative rows.
        # Simpler: use C=4, N=40 (1 pos + 9 neg per class) → AP_worst ≈ 1/10.
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        N = 40  # 1 positive + 9 negatives per class
        targets = torch.arange(self.C).repeat_interleave(N // self.C)
        logits = torch.full((N, self.C), 10.0)  # negatives score high
        for i, t in enumerate(targets):
            logits[i, t] = -10.0  # positives score low
        logits = logits.requires_grad_(True)
        loss = fn(logits, targets)
        assert loss.item() > 0.80

    def test_loss_in_unit_interval(self):
        torch.manual_seed(1)
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        loss = fn(logits, targets)
        assert 0.0 <= loss.item() <= 1.0

    def test_loss_monotone_better_model_lower_loss(self):
        # A model that ranks positives higher should have lower loss
        torch.manual_seed(2)
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        _, targets = _perfect_logits(self.B, self.C)

        # Slightly better than random
        good_logits = torch.randn(self.B, self.C)
        for i, t in enumerate(targets):
            good_logits[i, t] += 3.0

        # Random
        rand_logits = torch.randn(self.B, self.C)

        loss_good = fn(good_logits, targets)
        loss_rand = fn(rand_logits, targets)
        assert loss_good.item() < loss_rand.item()

    def test_gradient_flows_to_logits(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not logits.grad.isnan().any()
        assert logits.grad.abs().sum().item() > 0

    def test_no_gradient_to_queue_logits(self):
        # Queue logits are detached; live logits still get gradients.
        fn = SmoothAPLoss(num_classes=self.C, queue_size=32)
        # Prime the queue
        fn(torch.randn(self.B, self.C), torch.randint(0, self.C, (self.B,)))

        live = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        loss = fn(live, targets)
        loss.backward()
        assert live.grad is not None
        assert live.grad.abs().sum().item() > 0

    def test_temperature_lower_gives_sharper_ap(self):
        # Lower τ → better approximation of hard rank → lower loss on
        # a perfectly separated problem
        logits, targets = _perfect_logits(self.B, self.C, score=5.0)
        fn_sharp = SmoothAPLoss(num_classes=self.C, queue_size=0, temperature=0.001)
        fn_soft = SmoothAPLoss(num_classes=self.C, queue_size=0, temperature=1.0)
        loss_sharp = fn_sharp(logits.detach(), targets)
        loss_soft = fn_soft(logits.detach(), targets)
        assert loss_sharp.item() < loss_soft.item()


# ---------------------------------------------------------------------------
# Forward: ignore_index
# ---------------------------------------------------------------------------


class TestSmoothAPLossIgnoreIndex:
    C, B = 3, 24

    def test_all_padding_returns_zero_loss(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.full((self.B,), -100, dtype=torch.long)
        loss = fn(logits, targets)
        assert loss.item() == 0.0

    def test_all_padding_loss_has_grad_path(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.full((self.B,), -100, dtype=torch.long)
        loss = fn(logits, targets)
        loss.backward()  # must not raise

    def test_partial_padding_excluded(self):
        # Loss computed on non-padded subset should match full batch result.
        torch.manual_seed(5)
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))

        padded_targets = targets.clone()
        padded_targets[:4] = -100

        loss_padded = fn(logits, padded_targets)
        loss_clean = fn(logits[4:], targets[4:])
        assert torch.allclose(loss_padded, loss_clean, atol=1e-5)

    def test_custom_ignore_index(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0, ignore_index=999)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        targets[0] = 999
        loss = fn(logits, targets)
        assert not math.isnan(loss.item())


# ---------------------------------------------------------------------------
# Forward: reductions
# ---------------------------------------------------------------------------


class TestSmoothAPLossReductions:
    C, B = 4, 32

    def test_reduction_mean_is_scalar(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="mean")
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        loss = fn(logits, targets)
        assert loss.shape == ()

    def test_reduction_sum_equals_sum_of_none(self):
        torch.manual_seed(7)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))

        fn_none = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="none")
        fn_sum = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="sum")

        loss_none = fn_none(logits, targets)
        loss_sum = fn_sum(logits, targets)
        # nan-safe sum
        assert torch.allclose(loss_none.nan_to_num(0.0).sum(), loss_sum, atol=1e-5)

    def test_reduction_mean_equals_mean_of_none(self):
        torch.manual_seed(8)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))

        fn_none = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="none")
        fn_mean = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="mean")

        loss_none = fn_none(logits, targets)
        loss_mean = fn_mean(logits, targets)
        valid = ~loss_none.isnan()
        assert torch.allclose(loss_none[valid].mean(), loss_mean, atol=1e-5)

    def test_reduction_none_shape(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="none")
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        loss = fn(logits, targets)
        assert loss.shape == (self.C,)

    def test_reduction_none_degenerate_class_is_nan(self):
        # Make one class absent from targets so it's degenerate.
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="none")
        logits = torch.randn(self.B, self.C)
        # Only classes 0, 1, 2 present; class 3 absent.
        targets = torch.randint(0, self.C - 1, (self.B,))
        loss = fn(logits, targets)
        assert math.isnan(loss[self.C - 1].item())
        assert not math.isnan(loss[0].item())

    def test_all_degenerate_returns_zero_loss(self):
        # All same class → all degenerate in one-vs-rest
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="mean")
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.zeros(self.B, dtype=torch.long)  # only class 0
        loss = fn(logits, targets)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Forward: binary mode
# ---------------------------------------------------------------------------


class TestSmoothAPLossBinary:
    B = 32

    def test_perfect_binary_loss_near_zero(self):
        fn = SmoothAPLoss(num_classes=1, queue_size=0)
        logits = torch.cat(
            [torch.full((self.B // 2, 1), 10.0), torch.full((self.B // 2, 1), -10.0)]
        )
        targets = torch.cat([torch.ones(self.B // 2), torch.zeros(self.B // 2)]).long()
        loss = fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_binary_loss_near_one(self):
        # With 50% base rate the minimum AP is ~0.33, not 0.
        # Use 1 positive among 15 negatives → AP_worst ≈ 1/16 ≈ 0.06 → loss > 0.90.
        N_POS, N_NEG = 1, 15
        fn = SmoothAPLoss(num_classes=1, queue_size=0)
        logits = torch.cat(
            [torch.full((N_POS, 1), -10.0), torch.full((N_NEG, 1), 10.0)]
        )
        targets = torch.cat([torch.ones(N_POS), torch.zeros(N_NEG)]).long()
        loss = fn(logits, targets)
        assert loss.item() > 0.85

    def test_binary_gradient_flows(self):
        fn = SmoothAPLoss(num_classes=1, queue_size=0)
        logits = torch.randn(self.B, 1, requires_grad=True)
        targets = torch.randint(0, 2, (self.B,))
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0

    def test_binary_reduction_none_shape(self):
        fn = SmoothAPLoss(num_classes=1, queue_size=0, reduction="none")
        logits = torch.randn(self.B, 1)
        targets = torch.randint(0, 2, (self.B,))
        loss = fn(logits, targets)
        assert loss.shape == (1,)

    def test_binary_ignore_index_excluded(self):
        fn = SmoothAPLoss(num_classes=1, queue_size=0)
        logits = torch.randn(self.B, 1)
        targets = torch.randint(0, 2, (self.B,))
        targets[0] = -100
        loss = fn(logits, targets)
        assert not math.isnan(loss.item())


# ---------------------------------------------------------------------------
# Forward: return_per_class
# ---------------------------------------------------------------------------


class TestSmoothAPLossReturnPerClass:
    C, B = 4, 32

    def test_return_per_class_shapes(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        loss, per_class, valid = fn(logits, targets, return_per_class=True)
        assert loss.shape == ()
        assert per_class.shape == (self.C,)
        assert valid.shape == (self.C,)
        assert valid.dtype == torch.bool

    def test_return_per_class_loss_consistent_with_mean(self):
        torch.manual_seed(9)
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0, reduction="mean")
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        loss, per_class, valid = fn(logits, targets, return_per_class=True)
        assert torch.allclose(per_class[valid].mean(), loss, atol=1e-5)

    def test_return_per_class_nan_for_absent_classes(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C - 1, (self.B,))  # class C-1 absent
        _, per_class, valid = fn(logits, targets, return_per_class=True)
        assert not valid[self.C - 1].item()
        assert math.isnan(per_class[self.C - 1].item())

    def test_return_per_class_false_returns_tensor(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        result = fn(logits, targets, return_per_class=False)
        assert isinstance(result, torch.Tensor)

    def test_all_padding_return_per_class(self):
        fn = SmoothAPLoss(num_classes=self.C, queue_size=0)
        logits = torch.randn(self.B, self.C)
        targets = torch.full((self.B,), -100, dtype=torch.long)
        loss, per_class, valid = fn(logits, targets, return_per_class=True)
        assert loss.item() == 0.0
        assert valid.sum().item() == 0
        assert per_class.isnan().all()
