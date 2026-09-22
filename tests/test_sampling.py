"""Unit tests for imbalanced_losses._sampling.subsample_pool."""

import torch
import pytest

from imbalanced_losses._queue import PooledBatch
from imbalanced_losses._sampling import subsample_pool


def _make_pool(n: int, n_classes: int, seed: int = 0) -> PooledBatch:
    torch.manual_seed(seed)
    logits  = torch.randn(n, n_classes)
    targets = torch.randint(0, n_classes, (n,))
    return PooledBatch(logits, targets)


def _indexed_pool(n: int, n_classes: int) -> PooledBatch:
    """A pool whose logits encode each row's original index in column 0.

    ``orig_idx = int(logits[row, 0]) // n_classes`` recovers where a
    returned row came from, which is how every alignment test below checks
    that a transported tensor was indexed by the same selection.
    """
    logits = torch.arange(n * n_classes, dtype=torch.float).reshape(n, n_classes)
    targets = torch.arange(n) % n_classes
    return PooledBatch(logits, targets)


def _orig_indices(batch: PooledBatch, n_classes: int) -> list[int]:
    return [int(row[0].item()) // n_classes for row in batch.logits]


class TestSubsamplePool:

    def test_noop_when_pool_at_limit(self):
        pool = _make_pool(64, 4)
        out, _ = subsample_pool(pool, max_size=64)
        assert out.logits is pool.logits
        assert out.targets is pool.targets

    def test_noop_when_pool_below_limit(self):
        pool = _make_pool(32, 4)
        out, _ = subsample_pool(pool, max_size=64)
        assert out.logits is pool.logits
        assert out.targets is pool.targets

    def test_output_size_equals_max_size(self):
        out, _ = subsample_pool(_make_pool(1000, 5), max_size=128)
        assert out.logits.size(0) == 128
        assert out.targets.size(0) == 128

    def test_all_observed_classes_preserved(self):
        """Every class present in the input must appear in the output."""
        torch.manual_seed(0)
        pool = PooledBatch(torch.randn(1000, 10), torch.randint(0, 10, (1000,)))
        out, _ = subsample_pool(pool, max_size=100)
        assert set(out.targets.unique().tolist()) == set(pool.targets.unique().tolist())

    def test_rare_class_always_preserved(self):
        """A class with 2 positives out of 1000 rows must survive the sample."""
        torch.manual_seed(7)
        n = 1000
        targets = torch.zeros(n, dtype=torch.long)   # all class 0
        targets[0] = 4                                # class 4 has exactly 1 sample
        targets[1] = 4                                # class 4 has exactly 2 samples
        pool = PooledBatch(torch.randn(n, 5), targets)

        for _ in range(20):  # run multiple times; must always preserve class 4
            out, _ = subsample_pool(pool, max_size=50)
            assert 4 in out.targets.tolist(), "rare class 4 was dropped by subsampling"

    def test_output_targets_are_subset_of_input(self):
        """Every returned target value must have come from the input."""
        pool = _make_pool(500, 6)
        out, _ = subsample_pool(pool, max_size=80)
        assert set(out.targets.tolist()).issubset(set(pool.targets.tolist()))

    def test_logits_and_targets_correspond(self):
        """Returned logits row i must correspond to the same original row as targets[i]."""
        n, c = 200, 4
        pool = _indexed_pool(n, c)
        out, _ = subsample_pool(pool, max_size=50)
        for row, orig_idx in enumerate(_orig_indices(out, c)):
            assert pool.targets[orig_idx].item() == out.targets[row].item()

    def test_empty_input_returns_empty(self):
        pool = PooledBatch(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
        out, _ = subsample_pool(pool, max_size=64)
        assert out.logits.size(0) == 0
        assert out.targets.size(0) == 0

    def test_single_class_input(self):
        pool = PooledBatch(torch.randn(100, 1), torch.zeros(100, dtype=torch.long))
        out, _ = subsample_pool(pool, max_size=20)
        assert out.logits.size(0) == 20
        assert out.targets.unique().tolist() == [0]

    def test_tiny_max_size_still_returns_exact_size(self):
        """max_size=2 with 10 classes: output has exactly 2 rows."""
        out, _ = subsample_pool(_make_pool(100, 10), max_size=2)
        assert out.logits.size(0) == 2
        assert out.targets.size(0) == 2

    def test_gradients_preserved_through_subsampling(self):
        logits  = torch.randn(200, 4, requires_grad=True)
        targets = torch.randint(0, 4, (200,))
        out, _ = subsample_pool(PooledBatch(logits, targets), max_size=50)
        out.logits.sum().backward()
        assert logits.grad is not None
        # Only subsampled rows should have nonzero gradient.
        assert (logits.grad != 0).sum() == 50 * 4


class TestSubsamplePoolTransport:
    """Every tensor the batch carries survives the selection, aligned."""

    def test_optional_fields_absent_stay_absent(self):
        out, is_live = subsample_pool(_make_pool(200, 4, seed=4), max_size=50)
        assert out.is_iid is None
        assert out.sample_weight is None
        assert is_live is None

    def test_noop_path_returns_the_same_objects(self):
        """Below the cap, nothing is copied and nothing is dropped."""
        pool = PooledBatch(
            torch.randn(32, 4),
            torch.randint(0, 4, (32,)),
            torch.ones(32, dtype=torch.bool),
            torch.rand(32),
        )
        is_live = torch.ones(32, dtype=torch.bool)
        out, out_live = subsample_pool(pool, max_size=64, is_live=is_live)
        assert out is pool
        assert out_live is is_live

    def test_every_supplied_tensor_is_returned_when_subsampling(self):
        """The rail cannot silently drop a tensor it was handed."""
        pool = PooledBatch(
            torch.randn(200, 4),
            torch.randint(0, 4, (200,)),
            torch.ones(200, dtype=torch.bool),
            torch.rand(200),
        )
        out, out_live = subsample_pool(
            pool, max_size=50, is_live=torch.ones(200, dtype=torch.bool)
        )
        assert out.is_iid is not None and out.is_iid.size(0) == 50
        assert out.sample_weight is not None and out.sample_weight.size(0) == 50
        assert out_live is not None and out_live.size(0) == 50

    def test_all_tensors_share_one_selection(self):
        """
        is_iid, is_live, and sample_weight are indexed by the same final_idx
        as logits, so every one of them stays aligned with its original row.
        """
        n, c = 200, 4
        torch.manual_seed(81)
        base = _indexed_pool(n, c)
        # Distinct patterns so a swapped or reused index shows up.
        is_iid  = torch.arange(n) < 100
        is_live = (torch.arange(n) >= 50) & (torch.arange(n) < 150)
        sample_weight = torch.arange(n, dtype=torch.float) * 0.1
        pool = base._replace(is_iid=is_iid, sample_weight=sample_weight)

        out, out_live = subsample_pool(pool, max_size=50, is_live=is_live)

        assert out.logits.size(0) == 50
        assert out.is_iid.dtype == torch.bool
        assert out_live.dtype == torch.bool
        for row, orig_idx in enumerate(_orig_indices(out, c)):
            assert out.is_iid[row].item() == is_iid[orig_idx].item(), (
                f"is_iid misaligned at row {row} (orig_idx={orig_idx})"
            )
            assert out_live[row].item() == is_live[orig_idx].item(), (
                f"is_live misaligned at row {row} (orig_idx={orig_idx})"
            )
            assert out.sample_weight[row].item() == pytest.approx(
                sample_weight[orig_idx].item()
            ), f"sample_weight misaligned at row {row} (orig_idx={orig_idx})"

    def test_uniform_flags_survive_unchanged(self):
        """All-True and all-False flag tensors come back as they went in."""
        n = 200
        pool = _make_pool(n, 4, seed=1)._replace(
            is_iid=torch.ones(n, dtype=torch.bool)
        )
        out, out_live = subsample_pool(
            pool, max_size=50, is_live=torch.zeros(n, dtype=torch.bool)
        )
        assert out.is_iid.all()
        assert not out_live.any()

    def test_transported_shapes_match_output_rows(self):
        n = 300
        pool = _make_pool(n, 5, seed=3)._replace(
            is_iid=torch.randint(0, 2, (n,)).bool(),
            sample_weight=torch.rand(n),
        )
        out, out_live = subsample_pool(
            pool, max_size=80, is_live=torch.randint(0, 2, (n,)).bool()
        )
        rows = out.logits.size(0)
        assert out.targets.shape == (rows,)
        assert out.is_iid.shape == (rows,)
        assert out.sample_weight.shape == (rows,)
        assert out_live.shape == (rows,)

    def test_sample_weight_alone_is_transported(self):
        """
        A weight with no iid/live flags alongside it is still carried through;
        the old positional-arity signature dropped it silently.
        """
        n, c = 200, 4
        sample_weight = torch.arange(n, dtype=torch.float) * 0.1
        pool = _indexed_pool(n, c)._replace(sample_weight=sample_weight)
        out, _ = subsample_pool(pool, max_size=50)
        assert out.sample_weight is not None
        for row, orig_idx in enumerate(_orig_indices(out, c)):
            assert out.sample_weight[row].item() == pytest.approx(
                sample_weight[orig_idx].item()
            )

    def test_selection_is_weight_blind(self):
        """
        Row selection does not consult sample_weight (spec section 3): the
        same seed selects the same rows with and without a wildly skewed
        weight attached.
        """
        n, c = 200, 4
        skewed = torch.zeros(n)
        skewed[:5] = 1e6

        torch.manual_seed(123)
        unweighted, _ = subsample_pool(_indexed_pool(n, c), max_size=50)
        torch.manual_seed(123)
        weighted, _ = subsample_pool(
            _indexed_pool(n, c)._replace(sample_weight=skewed), max_size=50
        )
        assert _orig_indices(unweighted, c) == _orig_indices(weighted, c)
