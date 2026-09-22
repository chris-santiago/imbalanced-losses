"""
Shared ``sample_weight`` validation and degenerate-mass helpers.

``sample_weight`` is a cross-cutting concern: both loss families accept it
(the queued ranking losses via ``_QueuedRankingLoss``, the pointwise focal
losses via ``SigmoidFocalLoss``/``SoftmaxFocalLoss``), and both need the
same three things done to it before any arithmetic runs -- detach, validate,
cast.  This module owns that, so neither family imports validation from the
other and no function reaches into another object's private attributes.

Private module -- nothing here is part of the public API.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

# Directories whose frames a user-facing warning must never be blamed on:
# this library's own modules, and torch's Module.__call__ machinery.
_LIBRARY_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__))


def _caller_stacklevel() -> int:
    """Frames to skip so a warning lands on the caller's own code.

    ``warnings.warn``'s ``stacklevel`` is a frame count, and no single
    constant is correct here, because the ladder between the warning and
    the user differs per entry point:

    - ``SigmoidFocalLoss(...)(x, t)`` goes user -> ``Module.__call__`` ->
      ``_call_impl`` -> ``forward`` -> contract method -> here;
    - ``PAUCAtBudgetLoss.forward`` adds one more frame, since it delegates
      to ``_QueuedRankingLoss.forward`` through ``super()``;
    - ``LossWarmupWrapper`` adds more again;
    - a direct ``loss.forward(...)`` call skips torch's two wrappers.

    A fixed value therefore has to be wrong somewhere, and a value tuned
    for the common path blamed ``pauc_loss.py`` itself for a caller's
    misconfiguration.  Walking out to the first frame that belongs to
    neither this library nor torch is correct for every one of those
    ladders, and stays correct when a new wrapper is added between them.

    Returns
    -------
    int
        The ``stacklevel`` to pass to ``warnings.warn`` from the *calling*
        frame, counted the way ``warnings`` counts it (1 = the caller of
        this function).
    """
    level = 1
    frame = sys._getframe(1)  # the frame that will issue the warning
    while frame is not None:
        if not frame.f_code.co_filename.startswith((_LIBRARY_DIR, _TORCH_DIR)):
            return level
        frame = frame.f_back
        level += 1
    # Every frame on the stack belongs to the library or to torch, which
    # happens only when there is no user frame to blame (an import-time
    # call).  The loop has advanced ``level`` one past the last frame, so
    # step back to point at the outermost frame rather than guessing.
    return level - 1


class _SampleWeightMixin:
    """Mixin giving a loss one ``sample_weight`` validation routine.

    Exposes one method per shape contract -- per-row, element-for-element,
    and trailing-dim-broadcastable -- each taking the tensor it validates
    against, so a loss names its contract at the call site and the
    ``None`` fast path never touches the reference tensor.  All three share
    :meth:`_accept_sample_weight` for the detach, the value checks, the
    one-shot warning, and the dtype cast.

    Owns the one-shot warning flag (:attr:`_sample_weight_warned`) so the
    validation routine never reaches into an object it does not own.  The
    flag is a class-level default: the first (and only) whole-zero warning
    per instance writes an instance attribute that shadows it.
    """

    _sample_weight_warned: bool = False

    # -- one method per shape contract; all three share the value checks --

    def _check_sample_weight_per_row(
        self,
        sample_weight: torch.Tensor | None,
        logits: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Validate a weight carrying one value per row of *logits*.

        The ranking-loss contract: shape exactly ``[logits.size(0)]``.
        """
        if sample_weight is None:
            return None
        n = logits.size(0)
        self._check_dim0(sample_weight, n, "logits")
        if tuple(sample_weight.shape) != (n,):
            raise ValueError(
                f"sample_weight must be {(n,)} matching logits, "
                f"got {tuple(sample_weight.shape)} vs N={n}"
            )
        return self._accept_sample_weight(sample_weight, dtype)

    def _check_sample_weight_like(
        self,
        sample_weight: torch.Tensor | None,
        targets: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Validate a weight matching *targets* element for element.

        ``SoftmaxFocalLoss``'s contract: shape exactly ``targets.shape``,
        no broadcasting.
        """
        if sample_weight is None:
            return None
        self._check_dim0(sample_weight, targets.size(0), "targets")
        if sample_weight.shape != targets.shape:
            raise ValueError(
                f"sample_weight must be {tuple(targets.shape)} matching targets, "
                f"got {tuple(sample_weight.shape)} vs N={targets.size(0)}"
            )
        return self._accept_sample_weight(sample_weight, dtype)

    def _check_sample_weight_broadcastable(
        self,
        sample_weight: torch.Tensor | None,
        inputs: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Validate a weight that broadcasts over *inputs*' trailing dims.

        ``SigmoidFocalLoss``'s contract: dim 0 exactly ``inputs.size(0)``,
        trailing dims broadcastable (``[N, 1, H, W]`` against
        ``[N, C, H, W]``).  Dim 0 is exact rather than broadcastable
        because ``gather_distributed`` concatenates the weight along dim 0.
        """
        if sample_weight is None:
            return None
        self._check_dim0(sample_weight, inputs.size(0), "inputs")
        try:
            sample_weight.expand(inputs.shape)
        except RuntimeError:
            raise ValueError(
                f"sample_weight must be broadcastable to inputs, "
                f"got {tuple(sample_weight.shape)} vs inputs {tuple(inputs.shape)}"
            ) from None
        return self._accept_sample_weight(sample_weight, dtype)

    # -- shared pieces -----------------------------------------------------

    @staticmethod
    def _check_dim0(sample_weight: torch.Tensor, dim0: int, reference: str) -> None:
        """Require the weight's dim-0 extent to be exactly *dim0*.

        Shared by all three contracts, and checked first in each, because
        it is the invariant a DDP all-gather depends on: the gather
        concatenates along dim 0, so a weight whose dim 0 is not the batch
        size passes validation on one rank and then misaligns against the
        gathered batch on the next.
        """
        if sample_weight.ndim == 0 or sample_weight.size(0) != dim0:
            raise ValueError(
                f"sample_weight must match {reference} dim-0, "
                f"got {tuple(sample_weight.shape)} vs N={dim0}"
            )

    def _accept_sample_weight(
        self, sample_weight: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Detach, value-check, and cast an already shape-checked weight.

        Parameters
        ----------
        sample_weight : torch.Tensor
            The caller's weight, shape already validated.
        dtype : torch.dtype
            Loss dtype.  The returned weight is cast to it, so the weight's
            own dtype never decides the dtype of the value that gets
            backpropagated.

        Returns
        -------
        torch.Tensor
            The detached, validated, cast weight.

        Raises
        ------
        ValueError
            On any negative, ``NaN``, or ``inf`` entry.  All three are
            caller-visible misuse: a ``NaN`` weight silently produces a
            ``NaN`` loss, exactly the failure the negative-weight check
            exists to prevent.
        """
        # Detach immediately -- sample_weight never carries or receives
        # gradient, and every downstream touch point assumes this.  The
        # value checks below run on the caller's dtype, before the cast, so
        # a value that only becomes representable-as-valid after narrowing
        # (a float64 -1e-300 flushing to -0.0 in float32) still raises.
        sample_weight = sample_weight.detach()

        # Every reduction below is undefined or vacuous on an empty tensor
        # (.min() raises; .all() is trivially True).  A zero-row call is an
        # explicitly supported shape -- e.g. an unequal-last-batch DDP rank
        # -- and must fall through to the existing empty-batch path exactly
        # like an unweighted zero-row call does, not raise or spuriously
        # warn here.
        if sample_weight.numel() > 0:
            if not bool(torch.isfinite(sample_weight).all()):
                raise ValueError(
                    "sample_weight must be finite, got NaN or inf"
                )
            min_val = sample_weight.min()
            if min_val < 0:
                raise ValueError(
                    f"sample_weight must be non-negative, got min {min_val.item()}"
                )
            if bool((sample_weight == 0).all()) and not self._sample_weight_warned:
                warnings.warn(
                    f"{type(self).__name__}: sample_weight is entirely zero for this "
                    f"call, a likely misconfiguration; the objective falls through to "
                    f"its zero-weight-mass degenerate path, contributing no gradient. "
                    f"(This warning is shown once per instance.)",
                    UserWarning,
                    stacklevel=_caller_stacklevel(),
                )
                self._sample_weight_warned = True
        return sample_weight.to(dtype=dtype)


def _positive_mass(
    sample_weight: torch.Tensor, positives: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    """Select the numerator positives' weights and flag a zero total mass.

    Shared by the three ranking losses, which all take the same early
    return when the positives they are about to average over carry no
    weight at all: that class is degenerate in exactly the way a class
    with no positives is, so it is marked invalid rather than dividing by
    zero (spec section 4.3).

    Parameters
    ----------
    sample_weight : torch.Tensor, shape [M]
        Pooled per-row weight.
    positives : torch.Tensor
        Boolean mask or index tensor selecting the positives that enter
        the objective's numerator.

    Returns
    -------
    weights : torch.Tensor
        ``sample_weight`` restricted to *positives*.
    is_zero : bool
        True when those weights sum to exactly zero.
    """
    weights = sample_weight[positives]
    return weights, float(weights.sum()) == 0.0
