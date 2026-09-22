"""
Locks the current (pre-``sample_weight``) numerical behavior of every loss.

``TestUnweightedBitwise`` replays the exact grid ``tests/_golden_sample_weight.py``
used to build ``tests/fixtures/sample_weight_golden.pt`` and asserts bitwise
(``torch.equal``) identity against it. This is the guarantee behind spec S7
"Bitwise unweighted identity"
(``.claude/output/specs/2026-09-22-sample-weight-design.md``): every task in
the ``sample_weight`` plan must keep this test green suite-wide, because none
of them may change what runs when ``sample_weight`` is never supplied on the
call and the queue has never held a weighted row.

The grid is built with the library's CURRENT forward signatures -- no
``sample_weight`` keyword exists yet at capture time -- and this test calls
those same signatures, so it stays valid unchanged both before and after the
``sample_weight`` feature lands.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from _golden_sample_weight import (
    DTYPES,
    FIXTURE_PATH,
    SEED,
    build_grid,
    run_entry,
)

from imbalanced_losses._base import _QueuedRankingLoss
from imbalanced_losses.pauc_loss import PAUCAtBudgetLoss


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).rsplit(".", maxsplit=1)[-1]


@pytest.fixture(scope="module")
def golden() -> dict[str, object]:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"{FIXTURE_PATH} is missing. Regenerate on the UNMODIFIED tree with "
            f"'uv run python tests/_golden_sample_weight.py'."
        )
    return torch.load(FIXTURE_PATH, weights_only=True)


class TestUnweightedBitwise:
    """Every (config, dtype, step, field) replays to a bit-identical tensor."""

    def test_fixture_metadata(self, golden):
        assert golden["_meta_seed"] == SEED
        assert isinstance(golden["_meta_torch_version"], str)
        assert golden["_meta_num_configs"] == len(build_grid())
        assert golden["_meta_num_tensor_entries"] > 0

    @pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_str)
    def test_replay_matches_fixture(self, golden, dtype):
        configs = build_grid()
        dtype_str = _dtype_str(dtype)

        expected_keys_for_dtype = {
            k for k in golden if not k.startswith("_meta_") and f"|dtype={dtype_str}|" in k
        }
        checked_keys: set[str] = set()

        for config_index, config in enumerate(configs):
            step_results = run_entry(config, config_index, dtype)
            for step_key, step_result in step_results.items():
                prefix = f"{config.name}|dtype={dtype_str}|step={step_key}"
                for field_name, actual in step_result.items():
                    key = f"{prefix}|{field_name}"
                    assert key in golden, f"fixture is missing key {key!r}"
                    expected = golden[key]
                    assert actual.shape == expected.shape, (
                        f"{key}: shape drifted from the golden fixture "
                        f"({tuple(actual.shape)} vs {tuple(expected.shape)})."
                    )
                    assert actual.dtype == expected.dtype, (
                        f"{key}: dtype drifted from the golden fixture "
                        f"({actual.dtype} vs {expected.dtype})."
                    )
                    assert torch.equal(actual, expected), (
                        f"{key}: replay diverged from the golden fixture -- "
                        f"the unweighted code path changed at the bit level."
                    )
                    checked_keys.add(key)

        # Every fixture entry for this dtype was exercised, and nothing more.
        assert checked_keys == expected_keys_for_dtype


class _RecordingRankingLoss(_QueuedRankingLoss):
    """
    Minimal concrete ``_QueuedRankingLoss`` that records exactly what
    ``_compute_per_class`` receives as ``sample_weight``, instead of
    computing a real ranking objective.  Used to prove the transport rail
    (Task 2) without depending on the weighted arithmetic (Task 3).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.received_sample_weight: torch.Tensor | None = "unset"  # type: ignore[assignment]

    def _compute_per_class(self, logits, targets, is_iid, is_live, sample_weight):
        self.received_sample_weight = sample_weight
        loss_vec = logits.sum(dim=0) * 0.0
        valid_vec = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
        return loss_vec, valid_vec


class TestTransport:
    """
    ``sample_weight`` rides the same rail as ``iid_mask`` through
    ``_QueuedRankingLoss.forward``: ``_compute_per_class`` receives ``None``
    on the unweighted path and the pooled ``[M]`` weight vector (live rows
    at their supplied weight, queue rows at their stored weight,
    ``ignore_index`` rows dropped) on the weighted path.
    """

    NUM_CLASSES = 3
    N = 8

    @staticmethod
    def _targets() -> torch.Tensor:
        return torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    def test_unweighted_call_receives_none(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        loss(logits, self._targets())
        assert loss.received_sample_weight is None

    def test_weighted_call_receives_tensor(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        weight = torch.rand(self.N)
        loss(logits, self._targets(), sample_weight=weight)
        assert loss.received_sample_weight is not None
        assert torch.allclose(loss.received_sample_weight, weight)

    def test_weighted_call_pools_live_and_queue_weights(self):
        # queue_size == N so the first enqueue wholesale-replaces the buffer
        # with exactly weight1 (no wraparound arithmetic to reason about).
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        loss.train()
        weight1 = torch.rand(self.N)
        loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=weight1)

        weight2 = torch.rand(self.N)
        loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=weight2)

        received = loss.received_sample_weight
        assert received is not None
        assert received.shape == (2 * self.N,)
        # Live rows (first N) carry this call's own weight.
        assert torch.allclose(received[: self.N], weight2)
        # Queue rows (last N) carry the previous step's live weight.
        assert torch.allclose(received[self.N :], weight1)

    def test_unweighted_step_after_weighted_step_still_receives_tensor(self):
        # Once the queue holds a weighted row, has_weights gates the pooled
        # path even on a call that supplies no weight of its own.
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        loss.train()
        weight1 = torch.rand(self.N)
        loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=weight1)

        loss(torch.randn(self.N, self.NUM_CLASSES), self._targets())  # no sample_weight

        received = loss.received_sample_weight
        assert received is not None
        # This call's live rows default to weight 1 (materialized, not supplied).
        assert torch.allclose(received[: self.N], torch.ones(self.N))
        assert torch.allclose(received[self.N :], weight1)

    def test_reset_queue_returns_to_none(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        loss.train()
        weight1 = torch.rand(self.N)
        loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=weight1)

        loss.reset_queue()

        loss(torch.randn(self.N, self.NUM_CLASSES), self._targets())
        assert loss.received_sample_weight is None

    def test_ignore_index_rows_dropped_from_pooled_weight(self):
        loss = _RecordingRankingLoss(
            num_classes=self.NUM_CLASSES, queue_size=0, ignore_index=-100
        )
        targets = self._targets().clone()
        targets[0] = -100
        weight = torch.arange(self.N, dtype=torch.float)
        loss(torch.randn(self.N, self.NUM_CLASSES), targets, sample_weight=weight)
        received = loss.received_sample_weight
        assert received.shape == (self.N - 1,)
        assert torch.allclose(received, weight[1:])

    def test_wrong_shape_raises_value_error(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        bad_weight = torch.rand(self.N + 1)
        with pytest.raises(ValueError, match="sample_weight must be"):
            loss(logits, self._targets(), sample_weight=bad_weight)

    def test_negative_weight_raises_value_error(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        weight = torch.rand(self.N)
        weight[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            loss(logits, self._targets(), sample_weight=weight)

    def test_whole_zero_weight_warns_once(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        zero_weight = torch.zeros(self.N)
        with pytest.warns(UserWarning, match="sample_weight"):
            loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=zero_weight)
        # Second call must not warn again (one-shot per instance).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=zero_weight)

    def test_sample_weight_detached_no_grad(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        weight = torch.rand(self.N, requires_grad=True)
        loss(logits, self._targets(), sample_weight=weight)
        assert loss.received_sample_weight.requires_grad is False

    def test_zero_row_batch_with_zero_length_weight_queue_size_zero(self):
        # A correctly-shaped [0] sample_weight on a zero-row batch (an
        # explicitly supported DDP shape -- an unequal-last-batch rank can
        # see N=0) must land on the existing empty-pool branch: no
        # RuntimeError from an empty-tensor reduction, no spurious
        # whole-zero warning, and _compute_per_class is never reached.
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.zeros(0, self.NUM_CLASSES)
        targets = torch.zeros(0, dtype=torch.long)
        weight = torch.zeros(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = loss(logits, targets, sample_weight=weight)
        assert torch.equal(out, torch.zeros(()))
        assert loss.received_sample_weight == "unset"

    def test_sample_weight_positional_fifth_slot_rejected_on_base(self):
        # sample_weight is keyword-only on _QueuedRankingLoss.forward (spec
        # section 6 / plan constraint: trailing keyword argument on every
        # forward). Supplying it positionally in the 5th slot -- the slot
        # right after return_per_class -- must be a TypeError, not a silent
        # bind to nothing (or, on a differently-shaped override such as
        # PAUCAtBudgetLoss, a silent bind to the wrong parameter).
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        weight = torch.rand(self.N)
        with pytest.raises(TypeError):
            loss(logits, self._targets(), None, False, weight)

    def test_sample_weight_positional_sixth_slot_rejected_on_pauc(self):
        # Same guarantee on PAUCAtBudgetLoss.forward, whose extra
        # return_diagnostics parameter shifts sample_weight's positional
        # slot relative to the base class -- this is exactly the class the
        # adjudicated fix (keyword-only sample_weight) targets, since a
        # positional call here is the shape that would otherwise silently
        # bind a weight tensor to return_diagnostics.
        loss = PAUCAtBudgetLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        weight = torch.rand(self.N)
        with pytest.raises(TypeError):
            loss(logits, self._targets(), None, False, False, weight)

    def test_zero_row_batch_with_zero_length_weight_queue_size_positive(self):
        # Same guarantee with a fresh (never-enqueued) positive-size queue:
        # the merged pool is still empty after ignore_index filtering
        # (every stored slot is unfilled), so this also hits the
        # empty-pool branch rather than _compute_per_class.
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=4)
        logits = torch.zeros(0, self.NUM_CLASSES)
        targets = torch.zeros(0, dtype=torch.long)
        weight = torch.zeros(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = loss(logits, targets, sample_weight=weight)
        assert torch.equal(out, torch.zeros(()))
        assert loss.received_sample_weight == "unset"
