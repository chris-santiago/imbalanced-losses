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

import warnings

import torch

# Frame budget for the one-shot whole-zero UserWarning.  Counting outward
# from ``warnings.warn`` inside ``_check_sample_weight``:
#   1 = _check_sample_weight (this module)
#   2 = the ``forward`` that called it
#   3 = ``forward``'s caller
# 3 therefore attributes the warning to whoever called ``forward``, which is
# what a user needs to see.  One value for every loss family: the previous
# per-family values (2 inline in ``_base.forward``, 3 in the focal helper)
# differed only because of the extra frame, and both aimed at this same
# target frame.
_WARN_STACKLEVEL = 3


class _SampleWeightMixin:
    """Mixin giving a loss one ``sample_weight`` validation routine.

    Owns the one-shot warning flag (:attr:`_sample_weight_warned`) so the
    validation routine never reaches into an object it does not own.  The
    flag is a class-level default: the first (and only) whole-zero warning
    per instance writes an instance attribute that shadows it.
    """

    _sample_weight_warned: bool = False

    def _check_sample_weight(
        self,
        sample_weight: torch.Tensor | None,
        *,
        dim0: int,
        dtype: torch.dtype,
        reference: str,
        exact_shape: tuple[int, ...] | torch.Size | None = None,
        broadcast_shape: tuple[int, ...] | torch.Size | None = None,
    ) -> torch.Tensor | None:
        """Detach, validate, and cast a caller-supplied ``sample_weight``.

        Parameters
        ----------
        sample_weight : torch.Tensor or None
            The caller's weight.  ``None`` returns ``None``: that is the
            structural unweighted path and nothing here may touch it.
        dim0 : int
            Required dim-0 extent.  This is the invariant that makes the
            weight concatenate correctly under a DDP all-gather, which
            concatenates along dim 0: a weight whose dim 0 is not the
            batch size survives validation on one rank and then misaligns
            against the gathered batch on the next.
        dtype : torch.dtype
            Loss dtype.  The returned weight is cast to it, so the weight's
            own dtype never decides the dtype of the value that gets
            backpropagated.
        reference : str
            Name of the tensor the weight is validated against
            (``"logits"``, ``"targets"``, ``"inputs"``), used in error
            messages only.
        exact_shape : tuple[int, ...], optional
            When given, the weight's full shape must equal it.
        broadcast_shape : tuple[int, ...], optional
            When given, the weight must be broadcastable to it (trailing
            dims only -- *dim0* is still checked exactly).  Mutually
            exclusive with *exact_shape*.

        Returns
        -------
        torch.Tensor or None
            The detached, validated, cast weight, or ``None``.

        Raises
        ------
        ValueError
            On a shape mismatch, or on any negative, ``NaN``, or ``inf``
            entry.  All three are caller-visible misuse: a ``NaN`` weight
            silently produces a ``NaN`` loss, exactly the failure the
            negative-weight check exists to prevent.
        """
        if sample_weight is None:
            return None
        if exact_shape is not None and broadcast_shape is not None:
            raise TypeError("pass at most one of exact_shape / broadcast_shape")

        shape = tuple(sample_weight.shape)
        if sample_weight.ndim == 0 or sample_weight.size(0) != dim0:
            raise ValueError(
                f"sample_weight must match {reference} dim-0, "
                f"got {shape} vs N={dim0}"
            )
        if exact_shape is not None and shape != tuple(exact_shape):
            raise ValueError(
                f"sample_weight must be {tuple(exact_shape)} matching {reference}, "
                f"got {shape} vs N={dim0}"
            )
        if broadcast_shape is not None:
            try:
                sample_weight.expand(broadcast_shape)
            except RuntimeError:
                raise ValueError(
                    f"sample_weight must be broadcastable to {reference}, "
                    f"got {shape} vs {reference} {tuple(broadcast_shape)}"
                ) from None

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
                    stacklevel=_WARN_STACKLEVEL,
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
