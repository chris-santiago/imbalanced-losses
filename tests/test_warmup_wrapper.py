"""
Tests for LossWarmupWrapper.

Coverage
--------
- Init: argument validation, warmup_epochs=0 fast-path, missing-temperature warning,
        reset_queue_each_epoch warning when reset_queue absent
- Properties: in_warmup, current_temperature (with and without temperature attr)
- Phase switching: warmup_loss active during warmup, main_loss active after
- on_train_epoch_start: epoch tracking, sentinel set on first main epoch,
                        queue reset called/not called
- on_train_batch_start: switch_step latched on first main-phase batch,
                        geometric decay formula, clamped at temp_end,
                        no-op during warmup or before sentinel set
- Geometric decay: endpoints exact, intermediate values, monotonicity,
                   temp_start > temp_end (decay) and temp_start < temp_end (growth)
- Forward: delegates to warmup_loss during warmup, to main_loss after,
           kwargs forwarded to main_loss only, gradient flows through active loss
- reset_queue_each_epoch: called each main-phase epoch, not during warmup
- Integration with SmoothAPLoss: full epoch loop, backward pass
"""

from __future__ import annotations

import math
import warnings
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from imbalanced_losses.ap_loss import SmoothAPLoss
from imbalanced_losses.warmup_wrapper import LossWarmupWrapper

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_wrapper(
    warmup_epochs: int = 2,
    temp_start: float = 0.1,
    temp_end: float = 0.01,
    temp_decay_steps: int = 100,
    *,
    blend_epochs: int = 0,
    final_main_weight: float = 1.0,
    reset_queue_each_epoch: bool = False,
    main_loss: nn.Module | None = None,
    warmup_loss: nn.Module | None = None,
) -> LossWarmupWrapper:
    if main_loss is None:
        main_loss = SmoothAPLoss(num_classes=4, queue_size=0)
    if warmup_loss is None:
        warmup_loss = nn.CrossEntropyLoss()
    return LossWarmupWrapper(
        warmup_loss=warmup_loss,
        main_loss=main_loss,
        warmup_epochs=warmup_epochs,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=temp_decay_steps,
        blend_epochs=blend_epochs,
        final_main_weight=final_main_weight,
        reset_queue_each_epoch=reset_queue_each_epoch,
    )


def _sim(wrapper: LossWarmupWrapper, epochs: int, steps_per_epoch: int) -> list[tuple]:
    """Run the hook sequence and return (epoch, step, in_warmup, temp) tuples."""
    records = []
    for epoch in range(epochs):
        wrapper.on_train_epoch_start(epoch)
        for s in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + s
            wrapper.on_train_batch_start(global_step)
            records.append(
                (epoch, global_step, wrapper.in_warmup, wrapper.current_temperature)
            )
    return records


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_valid_construction(self):
        w = _make_wrapper()
        assert w.warmup_epochs == 2
        assert w.temp_start == 0.1
        assert w.temp_end == 0.01
        assert w.temp_decay_steps == 100

    def test_negative_warmup_epochs_raises(self):
        with pytest.raises(ValueError, match="warmup_epochs"):
            _make_wrapper(warmup_epochs=-1)

    def test_zero_temp_start_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _make_wrapper(temp_start=0.0)

    def test_negative_temp_end_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _make_wrapper(temp_end=-0.1)

    def test_zero_decay_steps_raises(self):
        with pytest.raises(ValueError, match="temp_decay_steps"):
            _make_wrapper(temp_decay_steps=0)

    def test_negative_decay_steps_raises(self):
        with pytest.raises(ValueError, match="temp_decay_steps"):
            _make_wrapper(temp_decay_steps=-10)

    def test_warmup_epochs_zero_latches_switch_step(self):
        w = _make_wrapper(warmup_epochs=0)
        assert w._switch_step == 0
        assert not w.in_warmup

    def test_warmup_epochs_zero_sets_temp_start(self):
        w = _make_wrapper(warmup_epochs=0, temp_start=0.05)
        assert w.current_temperature == pytest.approx(0.05)

    def test_missing_temperature_warns(self):
        main = nn.Linear(4, 4)  # no .temperature attribute
        with pytest.warns(UserWarning, match="temperature"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=main,
                warmup_epochs=1,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )

    def test_reset_queue_each_epoch_warns_when_absent(self):
        main = nn.Linear(4, 4)  # no reset_queue
        with pytest.warns(UserWarning, match="reset_queue"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=main,
                warmup_epochs=1,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
                reset_queue_each_epoch=True,
            )

    def test_no_warning_when_temperature_present(self):
        with warnings.catch_warnings():
            import warnings as _w
            _w.simplefilter("error", UserWarning)
            _make_wrapper()  # SmoothAPLoss has .temperature — no warning

    def test_gather_distributed_none_does_not_overwrite_main_loss(self):
        """Default gather_distributed=None must not clobber main_loss's explicit setting."""
        main = SmoothAPLoss(num_classes=4, queue_size=0, gather_distributed=False)
        LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=10,
            # gather_distributed not passed → defaults to None
        )
        assert main.gather_distributed is False

    def test_gather_distributed_explicit_overwrites_main_loss(self):
        """Explicit gather_distributed on wrapper should overwrite main_loss's setting."""
        main = SmoothAPLoss(num_classes=4, queue_size=0, gather_distributed=False)
        LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=10,
            gather_distributed=True,
        )
        assert main.gather_distributed is True

    def test_blend_steps_without_warmup_steps_raises(self):
        """blend_steps in epoch mode (no warmup_steps) should raise ValueError."""
        with pytest.raises(ValueError, match="blend_steps requires warmup_steps"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=SmoothAPLoss(num_classes=4, queue_size=0),
                warmup_epochs=5,
                blend_steps=100,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_in_warmup_true_before_switch(self):
        w = _make_wrapper(warmup_epochs=3)
        for epoch in range(3):
            w.on_train_epoch_start(epoch)
            assert w.in_warmup

    def test_in_warmup_false_after_switch(self):
        w = _make_wrapper(warmup_epochs=2)
        w.on_train_epoch_start(2)
        assert not w.in_warmup

    def test_current_temperature_returns_none_when_absent(self):
        main = MagicMock(spec=nn.Module)
        del main.temperature  # ensure attribute is missing
        with pytest.warns(UserWarning):
            w = LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=main,
                warmup_epochs=1,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )
        assert w.current_temperature is None

    def test_current_temperature_reflects_main_loss(self):
        w = _make_wrapper(warmup_epochs=0, temp_start=0.07)
        assert w.current_temperature == pytest.approx(0.07)
        w.main_loss.temperature = 0.03
        assert w.current_temperature == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# on_train_epoch_start
# ---------------------------------------------------------------------------


class TestOnTrainEpochStart:
    def test_epoch_counter_updated(self):
        w = _make_wrapper(warmup_epochs=3)
        for e in range(5):
            w.on_train_epoch_start(e)
            assert w._epoch == e

    def test_sentinel_set_on_first_main_epoch(self):
        w = _make_wrapper(warmup_epochs=2)
        w.on_train_epoch_start(0)
        assert w._switch_step is None
        w.on_train_epoch_start(1)
        assert w._switch_step is None
        w.on_train_epoch_start(2)
        assert w._switch_step == -1  # sentinel

    def test_sentinel_not_overwritten_on_subsequent_epochs(self):
        w = _make_wrapper(warmup_epochs=1)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(10)  # latches switch_step = 10
        w.on_train_epoch_start(2)
        assert w._switch_step == 10  # not reset to -1

    def test_queue_reset_called_in_main_phase(self):
        main = SmoothAPLoss(num_classes=4, queue_size=16)
        w = _make_wrapper(warmup_epochs=1, main_loss=main, reset_queue_each_epoch=True)
        w.on_train_epoch_start(0)  # warmup — no reset
        # Fill queue
        logits = torch.randn(4, 4)
        tgts = torch.randint(0, 4, (4,))
        main.training = True
        main(logits, tgts)
        ptr_after_warmup = int(main._q_ptr)
        assert ptr_after_warmup > 0

        w.on_train_epoch_start(1)  # main phase — reset
        assert int(main._q_ptr) == 0

    def test_queue_reset_not_called_during_warmup(self):
        main = SmoothAPLoss(num_classes=4, queue_size=16)
        w = _make_wrapper(warmup_epochs=3, main_loss=main, reset_queue_each_epoch=True)
        main.training = True
        logits = torch.randn(4, 4)
        tgts = torch.randint(0, 4, (4,))
        main(logits, tgts)
        ptr = int(main._q_ptr)

        w.on_train_epoch_start(0)
        w.on_train_epoch_start(1)
        assert int(main._q_ptr) == ptr  # unchanged

    def test_queue_reset_not_called_when_flag_false(self):
        main = SmoothAPLoss(num_classes=4, queue_size=16)
        w = _make_wrapper(warmup_epochs=1, main_loss=main, reset_queue_each_epoch=False)
        main.training = True
        logits = torch.randn(4, 4)
        tgts = torch.randint(0, 4, (4,))
        main(logits, tgts)
        ptr = int(main._q_ptr)

        w.on_train_epoch_start(1)
        assert int(main._q_ptr) == ptr  # not reset


# ---------------------------------------------------------------------------
# on_train_batch_start — switch_step latching
# ---------------------------------------------------------------------------


class TestOnTrainBatchStartLatching:
    def test_no_op_during_warmup(self):
        w = _make_wrapper(warmup_epochs=2, temp_start=0.1)
        w.on_train_epoch_start(0)
        w.on_train_batch_start(0)
        w.on_train_batch_start(5)
        # temperature not yet set by wrapper (still SmoothAPLoss default 0.01)
        assert w._switch_step is None

    def test_switch_step_latched_on_first_main_batch(self):
        w = _make_wrapper(warmup_epochs=1)
        w.on_train_epoch_start(1)
        assert w._switch_step == -1
        w.on_train_batch_start(global_step=42)
        assert w._switch_step == 42

    def test_temp_set_to_start_on_latch(self):
        w = _make_wrapper(warmup_epochs=1, temp_start=0.08)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(10)
        assert w.current_temperature == pytest.approx(0.08)

    def test_second_batch_begins_decay(self):
        w = _make_wrapper(warmup_epochs=1, temp_start=0.1, temp_end=0.01, temp_decay_steps=100)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(0)   # latch at step 0
        w.on_train_batch_start(1)   # elapsed=1
        expected = 0.1 * math.exp(1 / 100 * math.log(0.01 / 0.1))
        assert w.current_temperature == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Geometric decay correctness
# ---------------------------------------------------------------------------


class TestGeometricDecay:
    def _run(self, temp_start, temp_end, decay_steps, elapsed) -> float:
        w = _make_wrapper(
            warmup_epochs=0,
            temp_start=temp_start,
            temp_end=temp_end,
            temp_decay_steps=decay_steps,
        )
        # switch_step already latched to 0 for warmup_epochs=0
        w.on_train_batch_start(elapsed)
        return w.current_temperature

    def test_at_zero_elapsed_equals_temp_start(self):
        # For warmup_epochs=0, step 0 is the latch step (returns early).
        # Step 1 with elapsed=1 step from switch_step=0 should give approx temp_start.
        w = _make_wrapper(warmup_epochs=0, temp_start=0.1, temp_end=0.01, temp_decay_steps=1000)
        w.on_train_batch_start(0)  # this is the latch call (warmup_epochs=0, switch_step=0 already set)
        # elapsed=0 → frac=0 → temp=temp_start
        assert w.current_temperature == pytest.approx(0.1)

    def test_at_full_decay_equals_temp_end(self):
        t = self._run(0.1, 0.01, 100, 100)
        assert t == pytest.approx(0.01, rel=1e-6)

    def test_beyond_decay_steps_clamped_at_temp_end(self):
        t = self._run(0.1, 0.01, 100, 500)
        assert t == pytest.approx(0.01, rel=1e-6)

    def test_halfway_geometric_midpoint(self):
        temp_start, temp_end, steps = 0.1, 0.01, 100
        t = self._run(temp_start, temp_end, steps, 50)
        expected = temp_start * math.exp(0.5 * math.log(temp_end / temp_start))
        assert t == pytest.approx(expected, rel=1e-6)

    def test_geometric_midpoint_is_sqrt_of_product(self):
        # Geometric mean property: sqrt(a * b)
        temp_start, temp_end = 0.1, 0.01
        t = self._run(temp_start, temp_end, 100, 50)
        assert t == pytest.approx(math.sqrt(temp_start * temp_end), rel=1e-6)

    def test_monotone_decreasing_when_temp_end_lt_temp_start(self):
        temps = [self._run(0.1, 0.001, 100, s) for s in range(0, 101, 10)]
        for a, b in zip(temps, temps[1:]):
            assert a >= b

    def test_monotone_increasing_when_temp_end_gt_temp_start(self):
        # Inverted: growing temperature (less common but valid)
        temps = [self._run(0.001, 0.1, 100, s) for s in range(0, 101, 10)]
        for a, b in zip(temps, temps[1:]):
            assert a <= b

    def test_constant_when_temp_start_equals_temp_end(self):
        temps = [self._run(0.05, 0.05, 100, s) for s in [0, 25, 50, 100, 200]]
        for t in temps:
            assert t == pytest.approx(0.05, rel=1e-6)

    def test_switch_step_offset_applied(self):
        """Schedule is relative to switch_step, not global step 0."""
        w = _make_wrapper(warmup_epochs=1, temp_start=0.1, temp_end=0.01, temp_decay_steps=100)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(50)   # latch: switch_step=50
        w.on_train_batch_start(100)  # elapsed=50
        expected = 0.1 * math.exp(0.5 * math.log(0.01 / 0.1))
        assert w.current_temperature == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# forward — delegation
# ---------------------------------------------------------------------------


class TestForward:
    C, B = 4, 16

    def _batch(self):
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        return logits, targets

    def test_warmup_loss_called_during_warmup(self):
        mock_warmup = MagicMock(return_value=torch.tensor(1.0))
        mock_main = MagicMock(spec=SmoothAPLoss)
        type(mock_main).temperature = 0.01  # satisfy hasattr check
        w = LossWarmupWrapper(
            warmup_loss=mock_warmup,
            main_loss=mock_main,
            warmup_epochs=2,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=10,
        )
        logits, targets = self._batch()
        w.on_train_epoch_start(0)
        w(logits, targets)
        mock_warmup.assert_called_once_with(logits, targets)
        mock_main.assert_not_called()

    def test_main_loss_called_after_warmup(self):
        mock_warmup = MagicMock(return_value=torch.tensor(1.0))
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = LossWarmupWrapper(
            warmup_loss=mock_warmup,
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=10,
        )
        logits, targets = self._batch()
        w.on_train_epoch_start(1)
        w.on_train_batch_start(10)
        loss = w(logits, targets)
        mock_warmup.assert_not_called()
        assert loss.ndim == 0  # scalar from SmoothAPLoss

    def test_kwargs_forwarded_to_main_loss(self):
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = _make_wrapper(warmup_epochs=0, main_loss=main)
        w.on_train_batch_start(0)
        logits, targets = self._batch()
        result = w(logits, targets, return_per_class=True)
        assert isinstance(result, tuple)
        loss, per_class, valid = result
        assert per_class.shape == (self.C,)

    def test_kwargs_not_forwarded_to_warmup_loss(self):
        # CrossEntropyLoss does not accept return_per_class — should not raise
        w = _make_wrapper(warmup_epochs=2)
        w.on_train_epoch_start(0)
        logits, targets = self._batch()
        loss = w(logits, targets, return_per_class=True)
        assert loss.ndim == 0

    def test_gradient_flows_through_warmup_loss(self):
        w = _make_wrapper(warmup_epochs=2)
        w.on_train_epoch_start(0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        loss = w(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.norm() > 0

    def test_gradient_flows_through_main_loss(self):
        w = _make_wrapper(warmup_epochs=1)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        loss = w(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.norm() > 0


# ---------------------------------------------------------------------------
# Integration: full epoch loop with SmoothAPLoss
# ---------------------------------------------------------------------------


class TestIntegration:
    C, B, STEPS = 4, 16, 10

    def _batch(self):
        return torch.randn(self.B, self.C), torch.randint(0, self.C, (self.B,))

    def test_full_loop_no_errors(self):
        wrapper = LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=SmoothAPLoss(num_classes=self.C, queue_size=32),
            warmup_epochs=2,
            temp_start=0.1,
            temp_end=0.005,
            temp_decay_steps=20,
        )
        for epoch in range(5):
            wrapper.on_train_epoch_start(epoch)
            for s in range(self.STEPS):
                global_step = epoch * self.STEPS + s
                wrapper.on_train_batch_start(global_step)
                logits, targets = self._batch()
                logits.requires_grad_(True)
                loss = wrapper(logits, targets)
                loss.backward()

    def test_temperature_monotone_through_loop(self):
        wrapper = _make_wrapper(
            warmup_epochs=1, temp_start=0.1, temp_end=0.005, temp_decay_steps=50
        )
        temps = []
        for epoch in range(3):
            wrapper.on_train_epoch_start(epoch)
            for s in range(5):
                wrapper.on_train_batch_start(epoch * 5 + s)
                t = wrapper.current_temperature
                if not wrapper.in_warmup:
                    temps.append(t)

        assert len(temps) > 0
        for a, b in zip(temps, temps[1:]):
            assert a >= b - 1e-9

    def test_temperature_clamped_after_schedule(self):
        wrapper = _make_wrapper(
            warmup_epochs=0, temp_start=0.1, temp_end=0.01, temp_decay_steps=5
        )
        # Run well past decay_steps
        wrapper.on_train_batch_start(0)  # latch
        for step in range(1, 50):
            wrapper.on_train_batch_start(step)
        assert wrapper.current_temperature == pytest.approx(0.01, rel=1e-6)

    def test_loss_range_in_main_phase(self):
        wrapper = _make_wrapper(warmup_epochs=0)
        wrapper.on_train_batch_start(0)
        logits, targets = torch.randn(self.B, self.C), torch.randint(0, self.C, (self.B,))
        with torch.no_grad():
            loss = wrapper(logits, targets)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_switch_step_invariant_across_epochs(self):
        """switch_step must be set exactly once and not change."""
        wrapper = _make_wrapper(warmup_epochs=2)
        wrapper.on_train_epoch_start(2)
        wrapper.on_train_batch_start(20)
        first_switch = wrapper._switch_step
        for epoch in range(3, 6):
            wrapper.on_train_epoch_start(epoch)
            for s in range(5):
                wrapper.on_train_batch_start(epoch * 5 + s)
        assert wrapper._switch_step == first_switch


# ---------------------------------------------------------------------------
# Queue reset at switch point
# ---------------------------------------------------------------------------


class TestQueueResetAtSwitch:
    def test_queue_reset_on_phase_switch(self):
        main = SmoothAPLoss(num_classes=4, queue_size=16)
        w = _make_wrapper(warmup_epochs=1, main_loss=main)
        # Fill queue during warmup (direct call, no wrapper involvement)
        main.training = True
        logits = torch.randn(4, 4)
        tgts = torch.randint(0, 4, (4,))
        main(logits, tgts)
        assert int(main._q_ptr) > 0

        # Trigger switch
        w.on_train_epoch_start(1)   # sentinel = -1
        w.on_train_batch_start(10)  # latch + reset
        assert int(main._q_ptr) == 0

    def test_queue_reset_only_once(self):
        main = SmoothAPLoss(num_classes=4, queue_size=16)
        w = _make_wrapper(warmup_epochs=1, main_loss=main)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(10)  # latch + reset
        # Simulate a few main-phase batches filling the queue
        main.training = True
        logits = torch.randn(4, 4)
        tgts = torch.randint(0, 4, (4,))
        main(logits, tgts)
        ptr_after_fill = int(main._q_ptr)
        assert ptr_after_fill > 0

        # Next batch: no second reset
        w.on_train_batch_start(11)
        assert int(main._q_ptr) == ptr_after_fill

    def test_no_reset_when_no_reset_queue_method(self):
        main = nn.Linear(4, 4)  # no reset_queue
        with pytest.warns(UserWarning):
            w = LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=main,
                warmup_epochs=1,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )
        w.on_train_epoch_start(1)
        w.on_train_batch_start(0)  # should not raise


# ---------------------------------------------------------------------------
# blend_epochs — validation and properties
# ---------------------------------------------------------------------------


class TestBlendEpochs:
    def test_negative_blend_epochs_raises(self):
        with pytest.raises(ValueError, match="blend_epochs"):
            _make_wrapper(blend_epochs=-1)

    def test_zero_blend_epochs_default(self):
        w = _make_wrapper(warmup_epochs=2, blend_epochs=0)
        w.on_train_epoch_start(2)
        assert not w.in_blend
        assert w.main_weight == 1.0

    def test_in_blend_true_during_blend_epochs(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=2)
        w.on_train_epoch_start(1)
        assert w.in_blend
        w.on_train_epoch_start(2)
        assert w.in_blend

    def test_in_blend_false_after_blend_epochs(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=2)
        w.on_train_epoch_start(3)
        assert not w.in_blend

    def test_in_blend_false_during_warmup(self):
        w = _make_wrapper(warmup_epochs=2, blend_epochs=2)
        w.on_train_epoch_start(0)
        assert not w.in_blend
        w.on_train_epoch_start(1)
        assert not w.in_blend

    def test_main_weight_zero_during_warmup(self):
        w = _make_wrapper(warmup_epochs=2, blend_epochs=2)
        for epoch in range(2):
            w.on_train_epoch_start(epoch)
            assert w.main_weight == 0.0

    def test_main_weight_ramp_during_blend(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=3)
        # blend_epoch_index 0,1,2 → weights 1/4, 2/4, 3/4
        w.on_train_epoch_start(1)
        assert w.main_weight == pytest.approx(1 / 4)
        w.on_train_epoch_start(2)
        assert w.main_weight == pytest.approx(2 / 4)
        w.on_train_epoch_start(3)
        assert w.main_weight == pytest.approx(3 / 4)

    def test_main_weight_one_after_blend(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=2)
        w.on_train_epoch_start(3)
        assert w.main_weight == 1.0

    def test_main_weight_one_with_no_blend(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=0)
        w.on_train_epoch_start(1)
        assert w.main_weight == 1.0


# ---------------------------------------------------------------------------
# forward — blending
# ---------------------------------------------------------------------------


class TestForwardBlend:
    C, B = 4, 16

    def _batch(self):
        logits = torch.randn(self.B, self.C)
        targets = torch.randint(0, self.C, (self.B,))
        return logits, targets

    def test_blend_arithmetic(self):
        """Blended loss equals (1-w)*warmup + w*main up to floating point."""
        warmup = nn.CrossEntropyLoss()
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = LossWarmupWrapper(
            warmup_loss=warmup,
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=100,
            blend_epochs=3,
        )
        w.on_train_epoch_start(1)   # blend epoch 0 → main_weight = 1/4
        w.on_train_batch_start(10)
        logits, targets = self._batch()
        with torch.no_grad():
            blended = w(logits, targets)
            wt = w.main_weight
            expected = (1 - wt) * warmup(logits, targets) + wt * main(logits, targets)
        assert blended == pytest.approx(expected.item(), rel=1e-5)

    def test_kwargs_forwarded_after_blend(self):
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = _make_wrapper(warmup_epochs=1, blend_epochs=1, main_loss=main)
        w.on_train_epoch_start(2)   # past blend → pure main
        w.on_train_batch_start(20)
        logits, targets = self._batch()
        result = w(logits, targets, return_per_class=True)
        assert isinstance(result, tuple)

    def test_gradient_flows_through_both_during_blend(self):
        warmup = nn.CrossEntropyLoss()
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = LossWarmupWrapper(
            warmup_loss=warmup,
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=100,
            blend_epochs=2,
        )
        w.on_train_epoch_start(1)
        w.on_train_batch_start(0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        loss = w(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.norm() > 0

    def test_temperature_active_during_blend(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=2, temp_start=0.1, temp_end=0.01, temp_decay_steps=100)
        w.on_train_epoch_start(1)
        w.on_train_batch_start(0)   # latch
        w.on_train_batch_start(50)  # half-way through decay
        expected = 0.1 * math.exp(0.5 * math.log(0.01 / 0.1))
        assert w.current_temperature == pytest.approx(expected, rel=1e-6)

    def test_integration_full_loop_with_blend(self):
        wrapper = LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=SmoothAPLoss(num_classes=self.C, queue_size=32),
            warmup_epochs=2,
            temp_start=0.1,
            temp_end=0.005,
            temp_decay_steps=20,
            blend_epochs=2,
        )
        for epoch in range(6):
            wrapper.on_train_epoch_start(epoch)
            for s in range(5):
                global_step = epoch * 5 + s
                wrapper.on_train_batch_start(global_step)
                logits = torch.randn(self.B, self.C, requires_grad=True)
                targets = torch.randint(0, self.C, (self.B,))
                loss = wrapper(logits, targets)
                loss.backward()


# ---------------------------------------------------------------------------
# Step mode — validation and basic behaviour
# ---------------------------------------------------------------------------


def _make_step_wrapper(
    warmup_steps: int = 5,
    temp_start: float = 0.1,
    temp_end: float = 0.01,
    temp_decay_steps: int = 100,
    *,
    blend_steps: int = 0,
    final_main_weight: float = 1.0,
    main_loss: nn.Module | None = None,
) -> LossWarmupWrapper:
    if main_loss is None:
        main_loss = SmoothAPLoss(num_classes=4, queue_size=0)
    return LossWarmupWrapper(
        warmup_loss=nn.CrossEntropyLoss(),
        main_loss=main_loss,
        warmup_steps=warmup_steps,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=temp_decay_steps,
        blend_steps=blend_steps,
        final_main_weight=final_main_weight,
    )


class TestStepModeInit:
    def test_step_mode_flag_set(self):
        w = _make_step_wrapper(warmup_steps=10)
        assert w._step_mode is True

    def test_epoch_mode_flag_unset(self):
        w = _make_wrapper(warmup_epochs=2)
        assert w._step_mode is False

    def test_conflicting_warmup_params_raises(self):
        with pytest.raises(ValueError, match="warmup_epochs.*warmup_steps|warmup_steps.*warmup_epochs"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=SmoothAPLoss(num_classes=4, queue_size=0),
                warmup_epochs=5,
                warmup_steps=100,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )

    def test_conflicting_blend_params_raises(self):
        with pytest.raises(ValueError, match="blend_epochs.*blend_steps|blend_steps.*blend_epochs"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=SmoothAPLoss(num_classes=4, queue_size=0),
                warmup_steps=10,
                blend_epochs=2,
                blend_steps=20,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )

    def test_negative_warmup_steps_raises(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=SmoothAPLoss(num_classes=4, queue_size=0),
                warmup_steps=-1,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )

    def test_negative_blend_steps_raises(self):
        with pytest.raises(ValueError, match="blend_steps"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=SmoothAPLoss(num_classes=4, queue_size=0),
                warmup_steps=5,
                blend_steps=-1,
                temp_start=0.1,
                temp_end=0.01,
                temp_decay_steps=10,
            )

    def test_warmup_steps_zero_fast_path(self):
        w = _make_step_wrapper(warmup_steps=0)
        assert w._switch_step == 0
        assert not w.in_warmup

    def test_warmup_steps_zero_sets_temp_start(self):
        w = _make_step_wrapper(warmup_steps=0, temp_start=0.07)
        assert w.current_temperature == pytest.approx(0.07)


class TestStepModeInWarmup:
    def test_in_warmup_true_before_warmup_steps(self):
        w = _make_step_wrapper(warmup_steps=5)
        for step in range(5):
            w.on_train_batch_start(step)
            assert w.in_warmup

    def test_in_warmup_false_at_warmup_steps(self):
        w = _make_step_wrapper(warmup_steps=5)
        w.on_train_batch_start(5)
        assert not w.in_warmup

    def test_in_warmup_false_after_warmup_steps(self):
        w = _make_step_wrapper(warmup_steps=5)
        w.on_train_batch_start(100)
        assert not w.in_warmup

    def test_switch_step_latched_on_first_main_batch(self):
        w = _make_step_wrapper(warmup_steps=5)
        for step in range(5):
            w.on_train_batch_start(step)
        assert w._switch_step is None
        # In step mode the sentinel is set and consumed within the same call,
        # so _switch_step goes directly from None to the latched value.
        w.on_train_batch_start(5)
        assert w._switch_step == 5

    def test_switch_step_not_overwritten(self):
        w = _make_step_wrapper(warmup_steps=3)
        w.on_train_batch_start(3)  # latch at step 3
        assert w._switch_step == 3
        for step in range(4, 10):
            w.on_train_batch_start(step)
        assert w._switch_step == 3


class TestStepModeTemperature:
    def test_temp_set_to_start_on_switch(self):
        w = _make_step_wrapper(warmup_steps=3, temp_start=0.08)
        w.on_train_batch_start(3)
        assert w.current_temperature == pytest.approx(0.08)

    def test_temp_decays_after_switch(self):
        w = _make_step_wrapper(
            warmup_steps=0, temp_start=0.1, temp_end=0.01, temp_decay_steps=100
        )
        w.on_train_batch_start(50)
        expected = 0.1 * math.exp(0.5 * math.log(0.01 / 0.1))
        assert w.current_temperature == pytest.approx(expected, rel=1e-6)

    def test_temp_clamped_at_temp_end(self):
        w = _make_step_wrapper(
            warmup_steps=0, temp_start=0.1, temp_end=0.01, temp_decay_steps=10
        )
        w.on_train_batch_start(999)
        assert w.current_temperature == pytest.approx(0.01, rel=1e-6)

    def test_temp_decay_relative_to_switch_step(self):
        w = _make_step_wrapper(
            warmup_steps=50, temp_start=0.1, temp_end=0.01, temp_decay_steps=100
        )
        w.on_train_batch_start(50)   # latch: switch_step=50
        w.on_train_batch_start(100)  # elapsed=50
        expected = 0.1 * math.exp(0.5 * math.log(0.01 / 0.1))
        assert w.current_temperature == pytest.approx(expected, rel=1e-6)


class TestStepModeBlend:
    def test_in_blend_true_during_blend_steps(self):
        w = _make_step_wrapper(warmup_steps=5, blend_steps=10)
        for step in range(5, 15):
            w.on_train_batch_start(step)
            assert w.in_blend, f"expected in_blend at step {step}"

    def test_in_blend_false_after_blend_steps(self):
        w = _make_step_wrapper(warmup_steps=5, blend_steps=10)
        w.on_train_batch_start(15)
        assert not w.in_blend

    def test_in_blend_false_during_warmup(self):
        w = _make_step_wrapper(warmup_steps=5, blend_steps=10)
        for step in range(5):
            w.on_train_batch_start(step)
            assert not w.in_blend

    def test_main_weight_zero_during_warmup(self):
        w = _make_step_wrapper(warmup_steps=5, blend_steps=5)
        for step in range(5):
            w.on_train_batch_start(step)
            assert w.main_weight == 0.0

    def test_main_weight_ramp_during_blend(self):
        w = _make_step_wrapper(warmup_steps=0, blend_steps=3)
        # blend_step_index 0,1,2 → weights 1/4, 2/4, 3/4
        w.on_train_batch_start(0)
        assert w.main_weight == pytest.approx(1 / 4)
        w.on_train_batch_start(1)
        assert w.main_weight == pytest.approx(2 / 4)
        w.on_train_batch_start(2)
        assert w.main_weight == pytest.approx(3 / 4)

    def test_main_weight_one_after_blend(self):
        w = _make_step_wrapper(warmup_steps=0, blend_steps=3)
        w.on_train_batch_start(3)
        assert w.main_weight == 1.0

    def test_main_weight_one_with_no_blend_steps(self):
        w = _make_step_wrapper(warmup_steps=2, blend_steps=0)
        w.on_train_batch_start(2)
        assert w.main_weight == 1.0


class TestStepModeForward:
    C, B = 4, 16

    def _batch(self):
        return torch.randn(self.B, self.C), torch.randint(0, self.C, (self.B,))

    def test_warmup_loss_during_warmup(self):
        mock_warmup = MagicMock(return_value=torch.tensor(1.0))
        mock_main = MagicMock(spec=SmoothAPLoss)
        type(mock_main).temperature = 0.01
        w = LossWarmupWrapper(
            warmup_loss=mock_warmup,
            main_loss=mock_main,
            warmup_steps=5,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=10,
        )
        w.on_train_batch_start(3)
        logits, targets = self._batch()
        w(logits, targets)
        mock_warmup.assert_called_once_with(logits, targets)
        mock_main.assert_not_called()

    def test_main_loss_after_warmup(self):
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = _make_step_wrapper(warmup_steps=5, main_loss=main)
        w.on_train_batch_start(5)
        logits, targets = self._batch()
        loss = w(logits, targets)
        assert loss.ndim == 0

    def test_gradient_flows_during_warmup(self):
        w = _make_step_wrapper(warmup_steps=10)
        w.on_train_batch_start(0)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        w(logits, targets).backward()
        assert logits.grad is not None and logits.grad.norm() > 0

    def test_gradient_flows_after_warmup(self):
        w = _make_step_wrapper(warmup_steps=5)
        w.on_train_batch_start(5)
        logits = torch.randn(self.B, self.C, requires_grad=True)
        targets = torch.randint(0, self.C, (self.B,))
        w(logits, targets).backward()
        assert logits.grad is not None and logits.grad.norm() > 0

    def test_blend_arithmetic(self):
        warmup = nn.CrossEntropyLoss()
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = LossWarmupWrapper(
            warmup_loss=warmup,
            main_loss=main,
            warmup_steps=0,
            blend_steps=3,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=100,
        )
        w.on_train_batch_start(1)  # blend step 1 → main_weight = 2/4 = 0.5
        logits, targets = self._batch()
        with torch.no_grad():
            blended = w(logits, targets)
            wt = w.main_weight
            expected = (1 - wt) * warmup(logits, targets) + wt * main(logits, targets)
        assert blended == pytest.approx(expected.item(), rel=1e-5)


class TestStepModeIntegration:
    C, B = 4, 16

    def _batch(self):
        return torch.randn(self.B, self.C), torch.randint(0, self.C, (self.B,))

    def test_full_step_loop_no_errors(self):
        wrapper = LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=SmoothAPLoss(num_classes=self.C, queue_size=32),
            warmup_steps=10,
            blend_steps=5,
            temp_start=0.1,
            temp_end=0.005,
            temp_decay_steps=20,
        )
        for step in range(40):
            wrapper.on_train_batch_start(step)
            logits = torch.randn(self.B, self.C, requires_grad=True)
            targets = torch.randint(0, self.C, (self.B,))
            wrapper(logits, targets).backward()

    def test_temperature_monotone_in_step_loop(self):
        wrapper = _make_step_wrapper(
            warmup_steps=5, temp_start=0.1, temp_end=0.005, temp_decay_steps=50
        )
        temps = []
        for step in range(60):
            wrapper.on_train_batch_start(step)
            if not wrapper.in_warmup:
                temps.append(wrapper.current_temperature)
        assert len(temps) > 0
        for a, b in zip(temps, temps[1:]):
            assert a >= b - 1e-9

    def test_switch_step_invariant(self):
        wrapper = _make_step_wrapper(warmup_steps=5)
        wrapper.on_train_batch_start(5)
        first_switch = wrapper._switch_step
        for step in range(6, 30):
            wrapper.on_train_batch_start(step)
        assert wrapper._switch_step == first_switch

    def test_queue_reset_at_step_switch(self):
        main = SmoothAPLoss(num_classes=4, queue_size=16)
        w = _make_step_wrapper(warmup_steps=5, main_loss=main)
        main.training = True
        logits = torch.randn(4, 4)
        tgts = torch.randint(0, 4, (4,))
        main(logits, tgts)
        assert int(main._q_ptr) > 0
        w.on_train_batch_start(5)  # triggers reset
        assert int(main._q_ptr) == 0


# ---------------------------------------------------------------------------
# final_main_weight
# ---------------------------------------------------------------------------


class TestFinalMainWeight:
    C, B = 4, 16

    def _batch(self):
        return torch.randn(self.B, self.C), torch.randint(0, self.C, (self.B,))

    # --- validation ---

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="final_main_weight"):
            _make_wrapper(warmup_epochs=1, final_main_weight=0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="final_main_weight"):
            _make_wrapper(warmup_epochs=1, final_main_weight=-0.5)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="final_main_weight"):
            _make_wrapper(warmup_epochs=1, final_main_weight=1.1)

    def test_exactly_one_ok(self):
        w = _make_wrapper(warmup_epochs=1, final_main_weight=1.0)
        assert w.final_main_weight == 1.0

    # --- epoch mode ---

    def test_main_weight_caps_at_final_after_blend_epoch(self):
        w = _make_wrapper(warmup_epochs=1, blend_epochs=2, final_main_weight=0.75)
        w.on_train_epoch_start(3)  # past blend
        assert w.main_weight == pytest.approx(0.75)

    def test_main_weight_caps_at_final_no_blend_epoch(self):
        w = _make_wrapper(warmup_epochs=1, final_main_weight=0.6)
        w.on_train_epoch_start(1)
        assert w.main_weight == pytest.approx(0.6)

    def test_blend_ramp_scales_to_final_epoch(self):
        # blend_epochs=3, final_main_weight=0.6
        # ramp: 1/4*0.6, 2/4*0.6, 3/4*0.6
        w = _make_wrapper(warmup_epochs=1, blend_epochs=3, final_main_weight=0.6)
        w.on_train_epoch_start(1)
        assert w.main_weight == pytest.approx(0.6 / 4)
        w.on_train_epoch_start(2)
        assert w.main_weight == pytest.approx(2 * 0.6 / 4)
        w.on_train_epoch_start(3)
        assert w.main_weight == pytest.approx(3 * 0.6 / 4)
        w.on_train_epoch_start(4)  # past blend
        assert w.main_weight == pytest.approx(0.6)

    def test_forward_blended_at_final_weight_epoch(self):
        """After warmup with no blend, forward computes (1-w)*warmup + w*main."""
        warmup = nn.CrossEntropyLoss()
        main = SmoothAPLoss(num_classes=self.C, queue_size=0)
        w = LossWarmupWrapper(
            warmup_loss=warmup,
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.1,
            temp_end=0.01,
            temp_decay_steps=100,
            final_main_weight=0.7,
        )
        w.on_train_epoch_start(1)
        w.on_train_batch_start(10)
        logits, targets = self._batch()
        with torch.no_grad():
            result = w(logits, targets)
            wt = w.main_weight  # should be 0.7
            expected = (1 - wt) * warmup(logits, targets) + wt * main(logits, targets)
        assert wt == pytest.approx(0.7)
        assert result == pytest.approx(expected.item(), rel=1e-5)

    # --- step mode ---

    def test_main_weight_caps_at_final_after_blend_steps(self):
        w = _make_step_wrapper(warmup_steps=0, blend_steps=3, final_main_weight=0.75)
        w.on_train_batch_start(3)  # past blend
        assert w.main_weight == pytest.approx(0.75)

    def test_main_weight_caps_at_final_no_blend_steps(self):
        w = _make_step_wrapper(warmup_steps=2, final_main_weight=0.5)
        w.on_train_batch_start(2)
        assert w.main_weight == pytest.approx(0.5)

    def test_blend_ramp_scales_to_final_steps(self):
        w = _make_step_wrapper(warmup_steps=0, blend_steps=3, final_main_weight=0.6)
        w.on_train_batch_start(0)
        assert w.main_weight == pytest.approx(0.6 / 4)
        w.on_train_batch_start(1)
        assert w.main_weight == pytest.approx(2 * 0.6 / 4)
        w.on_train_batch_start(2)
        assert w.main_weight == pytest.approx(3 * 0.6 / 4)
        w.on_train_batch_start(3)  # past blend
        assert w.main_weight == pytest.approx(0.6)
