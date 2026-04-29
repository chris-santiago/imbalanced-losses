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
