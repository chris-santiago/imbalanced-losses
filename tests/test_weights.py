"""
Unit tests for imbalanced_losses._weights, the shared sample_weight mixin.

Coverage
--------
- One-shot whole-zero UserWarning: attributed to the caller's own frame, not
  to a frame inside the library or inside torch's module-call machinery,
  across every entry-point depth the library has (focal forward, ranking
  forward, a ranking forward that delegates through ``super().forward``, and
  a call mediated by ``LossWarmupWrapper``).
"""

from __future__ import annotations

import inspect
import pathlib
import warnings

import pytest
import torch

import imbalanced_losses
from imbalanced_losses import (
    PAUCAtBudgetLoss,
    SigmoidFocalLoss,
    SmoothAPLoss,
    SoftmaxFocalLoss,
)
from imbalanced_losses.warmup_wrapper import LossWarmupWrapper

SEED = 42

_SRC_DIR = str(pathlib.Path(imbalanced_losses.__file__).parent)
_TORCH_DIR = str(pathlib.Path(torch.__file__).parent)


def _zero_weight_warning(records) -> warnings.WarningMessage:
    """The single whole-zero sample_weight warning out of *records*."""
    hits = [
        record for record in records
        if issubclass(record.category, UserWarning)
        and "sample_weight is entirely zero" in str(record.message)
    ]
    assert len(hits) == 1, f"expected exactly one whole-zero warning, got {len(hits)}"
    return hits[0]


class TestWholeZeroWarningAttribution:
    """
    The one-shot misconfiguration warning must point at the line that called
    the loss, not at the library's own internals and not at torch's
    ``Module.__call__`` wrappers.

    Each test records the exact line number of its triggering call and
    asserts the warning carries it, so an off-by-one ``stacklevel`` fails
    rather than merely looking different in a terminal.  The cases below
    span every frame depth the library produces: the ladder differs by one
    between ``SmoothAPLoss`` and ``PAUCAtBudgetLoss`` (which delegates
    through ``super().forward``), by more again through
    ``LossWarmupWrapper``, and is two frames shorter when ``forward`` is
    called directly.  That spread is why no single hard-coded ``stacklevel``
    can be correct for all of them.
    """

    N, C = 8, 4

    @staticmethod
    def _binary_targets(n: int) -> torch.Tensor:
        targets = torch.zeros(n, dtype=torch.long)
        targets[::2] = 1
        return targets

    def test_sigmoid_focal_warning_points_at_the_caller(self):
        torch.manual_seed(SEED)
        loss_fn = SigmoidFocalLoss()
        logits = torch.randn(self.N, self.C)
        targets = torch.randint(0, 2, (self.N, self.C)).float()
        weight = torch.zeros(self.N, self.C)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lineno = inspect.currentframe().f_lineno + 1
            loss_fn(logits, targets, sample_weight=weight)

        record = _zero_weight_warning(records)
        assert record.filename == __file__
        assert record.lineno == lineno

    def test_softmax_focal_warning_points_at_the_caller(self):
        torch.manual_seed(SEED)
        loss_fn = SoftmaxFocalLoss()
        logits = torch.randn(self.N, self.C)
        targets = torch.randint(0, self.C, (self.N,))
        weight = torch.zeros(self.N)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lineno = inspect.currentframe().f_lineno + 1
            loss_fn(logits, targets, sample_weight=weight)

        record = _zero_weight_warning(records)
        assert record.filename == __file__
        assert record.lineno == lineno

    def test_ranking_loss_warning_points_at_the_caller(self):
        torch.manual_seed(SEED)
        loss_fn = SmoothAPLoss(num_classes=1, queue_size=0)
        logits = torch.randn(self.N, 1)
        targets = self._binary_targets(self.N)
        weight = torch.zeros(self.N)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lineno = inspect.currentframe().f_lineno + 1
            loss_fn(logits, targets, sample_weight=weight)

        record = _zero_weight_warning(records)
        assert record.filename == __file__
        assert record.lineno == lineno

    def test_delegating_forward_warning_points_at_the_caller(self):
        # PAUCAtBudgetLoss.forward delegates to _QueuedRankingLoss.forward
        # via super(), so its ladder is one frame deeper than SmoothAPLoss's.
        torch.manual_seed(SEED)
        loss_fn = PAUCAtBudgetLoss(num_classes=1, queue_size=0)
        logits = torch.randn(self.N, 1)
        targets = self._binary_targets(self.N)
        weight = torch.zeros(self.N)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lineno = inspect.currentframe().f_lineno + 1
            loss_fn(logits, targets, sample_weight=weight)

        record = _zero_weight_warning(records)
        assert record.filename == __file__
        assert record.lineno == lineno

    def test_wrapped_call_site_warning_points_at_the_caller(self):
        # Through LossWarmupWrapper the ladder is deeper again. This is the
        # case the original review named: a hand-tuned stacklevel "silently
        # breaks if either call site is wrapped".
        torch.manual_seed(SEED)
        wrapper = LossWarmupWrapper(
            warmup_loss=SmoothAPLoss(num_classes=1, queue_size=0),
            main_loss=SmoothAPLoss(num_classes=1, queue_size=0),
            warmup_epochs=0,
        )
        logits = torch.randn(self.N, 1)
        targets = self._binary_targets(self.N)
        weight = torch.zeros(self.N)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lineno = inspect.currentframe().f_lineno + 1
            wrapper(logits, targets, sample_weight=weight)

        record = _zero_weight_warning(records)
        assert record.filename == __file__
        assert record.lineno == lineno

    def test_direct_forward_call_warning_points_at_the_caller(self):
        # Calling forward() directly skips torch's __call__ wrappers, so the
        # ladder is two frames shorter than every case above.
        torch.manual_seed(SEED)
        loss_fn = SigmoidFocalLoss()
        logits = torch.randn(self.N, self.C)
        targets = torch.randint(0, 2, (self.N, self.C)).float()
        weight = torch.zeros(self.N, self.C)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lineno = inspect.currentframe().f_lineno + 1
            loss_fn.forward(logits, targets, sample_weight=weight)

        record = _zero_weight_warning(records)
        assert record.filename == __file__
        assert record.lineno == lineno

    @pytest.mark.parametrize(
        "builder",
        [
            lambda: SigmoidFocalLoss(),
            lambda: SmoothAPLoss(num_classes=1, queue_size=0),
        ],
        ids=["sigmoid", "smooth_ap"],
    )
    def test_warning_never_points_inside_the_library_or_torch(self, builder):
        torch.manual_seed(SEED)
        loss_fn = builder()
        if isinstance(loss_fn, SigmoidFocalLoss):
            logits = torch.randn(self.N, self.C)
            targets = torch.randint(0, 2, (self.N, self.C)).float()
            weight = torch.zeros(self.N, self.C)
        else:
            logits = torch.randn(self.N, 1)
            targets = self._binary_targets(self.N)
            weight = torch.zeros(self.N)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            loss_fn(logits, targets, sample_weight=weight)

        filename = _zero_weight_warning(records).filename
        assert not filename.startswith(_SRC_DIR), (
            f"the warning blames the library's own code ({filename}); a user "
            f"cannot act on that"
        )
        assert not filename.startswith(_TORCH_DIR), (
            f"the warning blames torch's module-call machinery ({filename}); "
            f"a user cannot act on that either"
        )
