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
import torch.distributed as dist

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


def _init_single_process_group() -> None:
    """Initialize a single-process gloo group if not already done."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29502",
            world_size=1,
            rank=0,
        )


def _destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class TestUnweightedBitwiseDDPSingleProcess:
    """
    Replays the golden grid a second time with the real DDP gather call
    sites forced active (``_gather_resolved = True``) under a single-process
    ``gloo`` group, and asserts bitwise identity against the same fixture
    ``TestUnweightedBitwise`` uses.

    Spec S9 (``.claude/output/specs/2026-09-22-sample-weight-design.md``)
    lists "the DDP single-process path" among the bitwise-identity cases,
    but the golden grid itself is captured with no process group
    initialized at all, so ``_should_gather()`` resolves to ``False`` and
    ``all_gather_with_grad`` / ``all_gather_no_grad`` are never called during
    capture. Forcing ``_gather_resolved = True`` here makes every forward
    call the real (unmocked) gather helpers; at ``world_size == 1`` both
    return their input unchanged (see ``distributed.py``), so the replay
    must still match the fixture bit-for-bit. This does not regenerate the
    fixture -- it replays the exact same grid and inputs as
    ``TestUnweightedBitwise``, only with gathering forced on.
    """

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    @pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_str)
    def test_replay_matches_fixture_with_gather_forced(self, golden, dtype):
        dtype_str = _dtype_str(dtype)

        for config_index, config in enumerate(build_grid()):
            step_results = run_entry(config, config_index, dtype, force_gather=True)
            for step_key, step_result in step_results.items():
                prefix = f"{config.name}|dtype={dtype_str}|step={step_key}"
                for field_name, actual in step_result.items():
                    assert torch.equal(actual, golden[f"{prefix}|{field_name}"]), (
                        f"{prefix}|{field_name}: DDP single-process (gather "
                        f"forced, world_size=1) replay diverged from the "
                        f"golden fixture."
                    )


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
        self.received_logits: torch.Tensor | None = None

    def _compute_per_class(self, logits, targets, is_iid, is_live, sample_weight=None):
        self.received_sample_weight = sample_weight
        self.received_logits = logits
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

    def test_wrong_dim0_raises_value_error(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        bad_weight = torch.rand(self.N + 1)
        with pytest.raises(ValueError, match="sample_weight must match logits dim-0"):
            loss(logits, self._targets(), sample_weight=bad_weight)

    def test_wrong_trailing_shape_raises_value_error(self):
        # Dim 0 is right but the weight is [N, 1], not [N]: the ranking
        # contract is exact, no broadcasting.
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        logits = torch.randn(self.N, self.NUM_CLASSES)
        with pytest.raises(ValueError, match="sample_weight must be"):
            loss(logits, self._targets(), sample_weight=torch.rand(self.N, 1))

    def test_nan_weight_raises_value_error(self):
        # A NaN weight is the same class of caller-visible misuse as a
        # negative one: unchecked, it silently produces a NaN loss.
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        weight = torch.rand(self.N)
        weight[2] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=weight)

    def test_inf_weight_raises_value_error(self):
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=0)
        weight = torch.rand(self.N)
        weight[0] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            loss(torch.randn(self.N, self.NUM_CLASSES), self._targets(), sample_weight=weight)

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

    def test_weighted_pool_survives_max_pool_size_subsampling(self):
        # The weighted subsample rail: with max_pool_size below the pool
        # size, the weight must be re-indexed by the same selection as the
        # logits, not dropped or left in the pre-subsample order.
        n, c = 40, self.NUM_CLASSES
        loss = _RecordingRankingLoss(num_classes=c, queue_size=0, max_pool_size=20)
        # Encode each row's original index in logits column 0, exactly as
        # tests/test_sampling.py does, so a returned row identifies itself.
        logits = torch.arange(n * c, dtype=torch.float).reshape(n, c)
        targets = torch.arange(n) % c
        weight = torch.arange(n, dtype=torch.float) * 0.5

        with pytest.warns(UserWarning, match="max_pool_size"):
            loss(logits, targets, sample_weight=weight)

        received = loss.received_sample_weight
        assert received is not None
        assert received.shape == (20,)
        assert loss.received_logits.shape == (20, c)
        for row in range(20):
            orig_idx = int(loss.received_logits[row, 0].item()) // c
            assert received[row].item() == pytest.approx(weight[orig_idx].item()), (
                f"row {row}: pooled weight is misaligned with the subsampled "
                f"logits (expected weight[{orig_idx}])"
            )

    @staticmethod
    def _probe_path(loss, logits, targets) -> bool:
        """Run one unweighted call in eval mode (no enqueue) and report
        whether the weighted arithmetic path was selected."""
        loss.eval()
        loss(logits, targets)
        return loss.received_sample_weight is not None

    def test_checkpoint_round_trip_preserves_the_active_arithmetic_path(self):
        # Which path runs must be a property of the stored rows, so a
        # save/load cycle can never move an instance between the weighted
        # and unweighted op sequences (the ULP-level drift the None guard
        # exists to prevent).
        logits = torch.randn(self.N, self.NUM_CLASSES)
        targets = self._targets()

        saved = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        saved.train()
        saved(logits, targets, sample_weight=torch.rand(self.N) + 0.5)

        restored = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        restored.load_state_dict(saved.state_dict(), strict=True)

        probe_logits = torch.randn(self.N, self.NUM_CLASSES)
        before = self._probe_path(saved, probe_logits, targets)
        after = self._probe_path(restored, probe_logits, targets)
        assert before is True, "a weighted queue must select the weighted path"
        assert after == before, (
            "a checkpoint round trip changed which arithmetic path runs"
        )

    def test_checkpoint_round_trip_of_an_unweighted_queue_stays_unweighted(self):
        # The mirror case: an all-ones weight is arithmetically unweighted,
        # so neither the live instance nor the reloaded one may switch paths.
        logits = torch.randn(self.N, self.NUM_CLASSES)
        targets = self._targets()

        saved = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        saved.train()
        saved(logits, targets, sample_weight=torch.ones(self.N))

        restored = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        restored.load_state_dict(saved.state_dict(), strict=True)

        probe_logits = torch.randn(self.N, self.NUM_CLASSES)
        before = self._probe_path(saved, probe_logits, targets)
        after = self._probe_path(restored, probe_logits, targets)
        assert before is False, "an all-ones weight must not activate the weighted path"
        assert after == before, (
            "a checkpoint round trip changed which arithmetic path runs"
        )

    def test_unweighted_path_restored_once_no_weighted_row_remains(self):
        # queue_size == N, so each step's enqueue replaces the buffer
        # wholesale. Once the weighted rows are gone, the loss returns to
        # the unweighted op sequence rather than paying for it forever.
        loss = _RecordingRankingLoss(num_classes=self.NUM_CLASSES, queue_size=self.N)
        loss.train()
        targets = self._targets()
        loss(torch.randn(self.N, self.NUM_CLASSES), targets,
             sample_weight=torch.rand(self.N) + 0.5)

        loss(torch.randn(self.N, self.NUM_CLASSES), targets)
        assert loss.received_sample_weight is not None  # weighted rows still pooled

        loss(torch.randn(self.N, self.NUM_CLASSES), targets)
        assert loss.received_sample_weight is None, (
            "every weighted row has been overwritten by weight-1 rows, so the "
            "unweighted path must be active again"
        )

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
