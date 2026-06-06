"""Unit tests for imbalanced_losses._sampling.subsample_pool."""

import torch
import pytest

from imbalanced_losses._sampling import subsample_pool


def _make_pool(n: int, n_classes: int, seed: int = 0):
    torch.manual_seed(seed)
    logits  = torch.randn(n, n_classes)
    targets = torch.randint(0, n_classes, (n,))
    return logits, targets


class TestSubsamplePool:

    def test_noop_when_pool_at_limit(self):
        logits, targets = _make_pool(64, 4)
        l2, t2 = subsample_pool(logits, targets, max_size=64)
        assert l2 is logits
        assert t2 is targets

    def test_noop_when_pool_below_limit(self):
        logits, targets = _make_pool(32, 4)
        l2, t2 = subsample_pool(logits, targets, max_size=64)
        assert l2 is logits
        assert t2 is targets

    def test_output_size_equals_max_size(self):
        logits, targets = _make_pool(1000, 5)
        l2, t2 = subsample_pool(logits, targets, max_size=128)
        assert l2.size(0) == 128
        assert t2.size(0) == 128

    def test_all_observed_classes_preserved(self):
        """Every class present in the input must appear in the output."""
        torch.manual_seed(0)
        logits  = torch.randn(1000, 10)
        targets = torch.randint(0, 10, (1000,))
        l2, t2 = subsample_pool(logits, targets, max_size=100)
        assert set(t2.unique().tolist()) == set(targets.unique().tolist())

    def test_rare_class_always_preserved(self):
        """A class with 2 positives out of 1000 rows must survive the sample."""
        torch.manual_seed(7)
        n = 1000
        logits  = torch.randn(n, 5)
        targets = torch.zeros(n, dtype=torch.long)   # all class 0
        targets[0] = 4                                # class 4 has exactly 1 sample
        targets[1] = 4                                # class 4 has exactly 2 samples

        for _ in range(20):  # run multiple times; must always preserve class 4
            _, t2 = subsample_pool(logits, targets, max_size=50)
            assert 4 in t2.tolist(), "rare class 4 was dropped by subsampling"

    def test_output_targets_are_subset_of_input(self):
        """Every returned target value must have come from the input."""
        logits, targets = _make_pool(500, 6)
        l2, t2 = subsample_pool(logits, targets, max_size=80)
        input_set  = set(targets.tolist())
        output_set = set(t2.tolist())
        assert output_set.issubset(input_set)

    def test_logits_and_targets_correspond(self):
        """Returned logits row i must correspond to the same original row as targets[i]."""
        n, c = 200, 4
        logits  = torch.arange(n * c, dtype=torch.float).reshape(n, c)
        targets = torch.arange(n) % c
        l2, t2 = subsample_pool(logits, targets, max_size=50)
        # Each row in l2 uniquely identifies its original index via the first element.
        for row_idx in range(l2.size(0)):
            orig_idx = int(l2[row_idx, 0].item()) // c
            assert targets[orig_idx].item() == t2[row_idx].item()

    def test_empty_input_returns_empty(self):
        logits  = torch.zeros(0, 4)
        targets = torch.zeros(0, dtype=torch.long)
        l2, t2 = subsample_pool(logits, targets, max_size=64)
        assert l2.size(0) == 0
        assert t2.size(0) == 0

    def test_single_class_input(self):
        logits  = torch.randn(100, 1)
        targets = torch.zeros(100, dtype=torch.long)
        l2, t2 = subsample_pool(logits, targets, max_size=20)
        assert l2.size(0) == 20
        assert t2.unique().tolist() == [0]

    def test_tiny_max_size_still_returns_exact_size(self):
        """max_size=2 with 10 classes: output has exactly 2 rows."""
        logits, targets = _make_pool(100, 10)
        l2, t2 = subsample_pool(logits, targets, max_size=2)
        assert l2.size(0) == 2
        assert t2.size(0) == 2

    def test_gradients_preserved_through_subsampling(self):
        logits  = torch.randn(200, 4, requires_grad=True)
        targets = torch.randint(0, 4, (200,))
        l2, _ = subsample_pool(logits, targets, max_size=50)
        l2.sum().backward()
        assert logits.grad is not None
        # Only subsampled rows should have nonzero gradient.
        assert (logits.grad != 0).sum() == 50 * 4


class TestSubsamplePoolIidAlignment:
    """Tests for the optional is_iid parameter added for iid_mask plumbing."""

    def test_noop_returns_two_tuple_when_no_iid(self):
        """When is_iid is not provided and pool is small, 2-tuple is returned."""
        logits, targets = _make_pool(32, 4)
        result = subsample_pool(logits, targets, max_size=64)
        assert isinstance(result, tuple) and len(result) == 2

    def test_noop_returns_three_tuple_when_iid_provided_small_pool(self):
        """When is_iid is provided and pool is small, 3-tuple is returned."""
        logits, targets = _make_pool(32, 4)
        is_iid = torch.ones(32, dtype=torch.bool)
        result = subsample_pool(logits, targets, max_size=64, is_iid=is_iid)
        assert isinstance(result, tuple) and len(result) == 3
        l2, t2, iid2 = result
        assert l2 is logits
        assert t2 is targets
        assert iid2 is is_iid

    def test_iid_flags_aligned_with_selected_rows(self):
        """is_iid[i] in output must match is_iid[orig_i] for the same original row."""
        n, c = 200, 4
        torch.manual_seed(42)
        logits  = torch.arange(n * c, dtype=torch.float).reshape(n, c)
        targets = torch.arange(n) % c
        # Alternate True/False so we can verify alignment by original index.
        is_iid  = torch.arange(n) % 2 == 0  # True for even rows

        l2, t2, iid2 = subsample_pool(logits, targets, max_size=50, is_iid=is_iid)

        assert l2.size(0) == 50
        assert iid2.size(0) == 50
        assert iid2.dtype == torch.bool

        # Verify alignment: for each selected row, recover original index from
        # logits (encoded as row_idx * c in column 0) and check flag matches.
        for row in range(l2.size(0)):
            orig_idx = int(l2[row, 0].item()) // c
            assert iid2[row].item() == is_iid[orig_idx].item(), (
                f"Row {row}: expected is_iid[{orig_idx}]={is_iid[orig_idx].item()}, "
                f"got {iid2[row].item()}"
            )

    def test_all_true_iid_returns_all_true(self):
        """An all-True iid mask should survive subsampling as all-True."""
        logits, targets = _make_pool(200, 4, seed=1)
        is_iid = torch.ones(200, dtype=torch.bool)
        _, _, iid2 = subsample_pool(logits, targets, max_size=50, is_iid=is_iid)
        assert iid2.all()

    def test_all_false_iid_returns_all_false(self):
        """An all-False iid mask should survive subsampling as all-False."""
        logits, targets = _make_pool(200, 4, seed=2)
        is_iid = torch.zeros(200, dtype=torch.bool)
        _, _, iid2 = subsample_pool(logits, targets, max_size=50, is_iid=is_iid)
        assert not iid2.any()

    def test_iid_shape_matches_output_logits(self):
        """Output is_iid must have same row count as output logits/targets."""
        logits, targets = _make_pool(300, 5, seed=3)
        is_iid = torch.randint(0, 2, (300,)).bool()
        l2, t2, iid2 = subsample_pool(logits, targets, max_size=80, is_iid=is_iid)
        assert iid2.shape == (l2.size(0),)
        assert iid2.shape == t2.shape

    def test_existing_callers_unaffected(self):
        """Callers not passing is_iid must still receive a 2-tuple (no breakage)."""
        logits, targets = _make_pool(200, 4, seed=4)
        result = subsample_pool(logits, targets, max_size=50)
        assert len(result) == 2


class TestSubsamplePoolIsLiveAlignment:
    """Tests for the optional is_live parameter (pos_numerator='live' feature)."""

    def test_noop_returns_four_tuple_when_both_flags_provided_small_pool(self):
        """When pool <= max_size with both flags, 4-tuple is returned."""
        logits, targets = _make_pool(32, 4)
        is_iid  = torch.ones(32, dtype=torch.bool)
        is_live = torch.ones(32, dtype=torch.bool)
        result = subsample_pool(logits, targets, max_size=64, is_iid=is_iid, is_live=is_live)
        assert isinstance(result, tuple) and len(result) == 4
        l2, t2, iid2, live2 = result
        assert l2 is logits
        assert t2 is targets
        assert iid2 is is_iid
        assert live2 is is_live

    def test_is_live_aligned_with_selected_rows(self):
        """
        is_live[i] in output must match is_live[orig_i] for the same original row.
        Encode the row index in logits[:, 0] to recover orig_idx and verify.
        """
        n, c = 200, 4
        torch.manual_seed(80)
        logits  = torch.arange(n * c, dtype=torch.float).reshape(n, c)
        targets = torch.arange(n) % c
        is_iid  = torch.arange(n) % 2 == 0    # True for even rows
        # is_live: True for first 100 rows (live), False for last 100 (queue).
        is_live = torch.arange(n) < 100

        l2, t2, iid2, live2 = subsample_pool(
            logits, targets, max_size=50, is_iid=is_iid, is_live=is_live
        )

        assert l2.size(0) == 50
        assert live2.size(0) == 50
        assert live2.dtype == torch.bool

        for row in range(l2.size(0)):
            orig_idx = int(l2[row, 0].item()) // c
            assert live2[row].item() == is_live[orig_idx].item(), (
                f"Row {row}: expected is_live[{orig_idx}]={is_live[orig_idx].item()}, "
                f"got {live2[row].item()}"
            )

    def test_all_live_rows_survive_as_all_true(self):
        """An all-True is_live mask should survive subsampling as all-True."""
        logits, targets = _make_pool(200, 4, seed=5)
        is_iid  = torch.ones(200, dtype=torch.bool)
        is_live = torch.ones(200, dtype=torch.bool)
        _, _, _, live2 = subsample_pool(
            logits, targets, max_size=50, is_iid=is_iid, is_live=is_live
        )
        assert live2.all()

    def test_all_queue_rows_survive_as_all_false(self):
        """An all-False is_live mask should survive subsampling as all-False."""
        logits, targets = _make_pool(200, 4, seed=6)
        is_iid  = torch.ones(200, dtype=torch.bool)
        is_live = torch.zeros(200, dtype=torch.bool)
        _, _, _, live2 = subsample_pool(
            logits, targets, max_size=50, is_iid=is_iid, is_live=is_live
        )
        assert not live2.any()

    def test_is_live_shape_matches_output_logits(self):
        """Output is_live must have same row count as output logits/targets."""
        logits, targets = _make_pool(300, 5, seed=7)
        is_iid  = torch.randint(0, 2, (300,)).bool()
        is_live = torch.randint(0, 2, (300,)).bool()
        l2, t2, iid2, live2 = subsample_pool(
            logits, targets, max_size=80, is_iid=is_iid, is_live=is_live
        )
        assert live2.shape == (l2.size(0),)
        assert live2.shape == t2.shape

    def test_is_live_without_is_iid_is_not_supported_by_signature(self):
        """
        is_live is only threaded when is_iid is also provided (both are
        present in the base forward; the API doesn't support is_live alone).
        This test confirms no regression: without is_iid we still get a 2-tuple.
        """
        logits, targets = _make_pool(200, 4, seed=8)
        result = subsample_pool(logits, targets, max_size=50)
        assert len(result) == 2

    def test_is_live_iid_both_flags_alignment_independent(self):
        """
        Both is_iid and is_live are returned indexed by the same final_idx,
        so they remain mutually aligned.  Encode both in logits and verify.
        """
        n, c = 200, 4
        torch.manual_seed(81)
        logits  = torch.arange(n * c, dtype=torch.float).reshape(n, c)
        targets = torch.arange(n) % c
        # is_iid: True for rows 0..99; is_live: True for rows 50..149.
        is_iid  = torch.arange(n) < 100
        is_live = (torch.arange(n) >= 50) & (torch.arange(n) < 150)

        l2, t2, iid2, live2 = subsample_pool(
            logits, targets, max_size=50, is_iid=is_iid, is_live=is_live
        )

        for row in range(l2.size(0)):
            orig_idx = int(l2[row, 0].item()) // c
            assert iid2[row].item()  == is_iid[orig_idx].item(),  (
                f"is_iid misaligned at row {row} (orig_idx={orig_idx})"
            )
            assert live2[row].item() == is_live[orig_idx].item(), (
                f"is_live misaligned at row {row} (orig_idx={orig_idx})"
            )
