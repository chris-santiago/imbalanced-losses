"""
Focal Loss variants for classification tasks.

Implements sigmoid (binary/multi-label) and softmax (mutually-exclusive
multiclass) focal losses, both with optional DDP all-gather support so that
positive-count-based normalisations are computed over the global batch rather
than just the local rank's slice.

References
----------
Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imbalanced_losses._base import _resolve_gather
from imbalanced_losses._weights import _SampleWeightMixin
from imbalanced_losses.distributed import all_gather_no_grad, all_gather_with_grad


# ---------------------------------------------------------------------------
# Sigmoid Focal Loss
# ---------------------------------------------------------------------------

class SigmoidFocalLoss(_SampleWeightMixin, nn.Module):
    """
    Sigmoid Focal Loss as used in RetinaNet.

    Binary / multi-label variant operating on raw logits with sigmoid activation.
    Supports optional DDP all-gather so that the global batch is seen when
    computing mean/sum reductions.

    .. note::
        **Multi-label vs. multiclass:** This loss treats every output logit as an
        *independent* binary prediction (sigmoid per element).  Use it when a
        sample can belong to *multiple* classes simultaneously (multi-label), or
        for a single yes/no prediction (binary).  If your classes are
        *mutually exclusive* — each sample belongs to exactly one class — use
        :class:`SoftmaxFocalLoss` instead, which couples the outputs via softmax.

    Parameters
    ----------
    alpha : float
        Weighting factor in [0, 1] to balance positives vs negatives, or -1 to
        ignore. Default: 0.25.
    gamma : float
        Exponent of the modulating factor (1 - p_t). Default: 2.
    reduction : str
        'none' | 'mean' | 'sum'. Default: 'mean'.
    gather_distributed : bool or None, optional
        Whether to all-gather inputs and targets across DDP workers before
        computing the loss.  ``None`` (default) auto-detects: gathers when
        ``torch.distributed`` is initialized with world_size > 1.  Set to
        ``False`` to opt out.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        gather_distributed: bool | None = None,
    ):
        super().__init__()
        if not (0.0 <= alpha <= 1.0) and alpha != -1:
            raise ValueError(
                f"Invalid alpha value: {alpha}. alpha must be in [0, 1] or -1 for ignore."
            )
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.gather_distributed = gather_distributed
        self._gather_resolved: bool | None = None

    def _should_gather(self) -> bool:
        if self._gather_resolved is None:
            self._gather_resolved = _resolve_gather(self.gather_distributed)
        return self._gather_resolved

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor
            Raw logits, arbitrary shape.
        targets : Tensor
            Same shape, float 0/1 labels.
        sample_weight : Tensor, optional
            Per-element weight. Dim 0 must equal ``inputs.size(0)``;
            trailing dims may broadcast (e.g. ``[N, 1, H, W]`` against
            ``[N, C, H, W]``). ``None`` (default) leaves the objective
            unweighted -- when no weight has ever been supplied to this
            instance, the loss and its gradient are bitwise identical to
            the pre-``sample_weight`` release. When provided, must be
            non-negative and finite; it is detached on entry, cast to the
            loss dtype, and never carries gradient. The full dim-0 extent
            is required because ``gather_distributed`` concatenates the
            weight along dim 0: a weight that broadcast over dim 0 would
            validate on one rank and then misalign against the gathered
            batch. Under DDP, all ranks must agree on whether
            ``sample_weight`` is supplied on a given step -- a mismatch
            desynchronizes the collectives across ranks.

        Returns
        -------
        Tensor
            Scalar or per-element loss depending on ``reduction``.
        """
        if sample_weight is not None:
            # Guarded: inputs.size(0) is only defined for a batched input,
            # and an unweighted call must reach the pre-sample_weight path
            # without touching the input's shape at all.
            sample_weight = self._check_sample_weight(
                sample_weight,
                dim0=inputs.size(0),
                dtype=inputs.dtype,
                reference="inputs",
                broadcast_shape=inputs.shape,
            )

        if self._should_gather():
            inputs  = all_gather_with_grad(inputs)
            targets = all_gather_no_grad(targets)
            if sample_weight is not None:
                sample_weight = all_gather_no_grad(sample_weight)

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # No validity mask exists for the sigmoid case: every element of
        # `inputs` is a real prediction, so 'mean' averages over all of them.
        if self.reduction == "none":
            return _reduce_none(loss, sample_weight)
        if self.reduction == "mean":
            return _reduce_mean_all(loss, sample_weight)
        if self.reduction == "sum":
            return _reduce_sum(loss, sample_weight)
        raise _invalid_reduction(self.reduction)


# ---------------------------------------------------------------------------
# Softmax Focal Loss
# ---------------------------------------------------------------------------

class SoftmaxFocalLoss(_SampleWeightMixin, nn.Module):
    """
    Softmax Focal Loss for mutually-exclusive multiclass classification.

    Generalises focal loss from the binary sigmoid case to C classes using
    softmax probabilities and standard cross-entropy as the base loss.
    Supports optional DDP all-gather so that positive-count-based
    normalisations (``mean_positive``) reflect the global batch.

    Parameters
    ----------
    alpha : Tensor or list[float] or None
        Per-class weighting factors of shape (C,).  Typically set to the
        inverse class frequency or similar.  ``None`` disables class
        weighting.  When provided, each sample's loss is scaled by
        ``alpha[y]`` where ``y`` is the ground-truth class.
    gamma : float
        Focusing exponent.  ``gamma=0`` recovers vanilla CE.  Default: 2.0.
    reduction : str
        'none' | 'mean' | 'mean_positive' | 'sum'.  Default: 'mean'.

        - 'mean': average over all valid (non-ignored) positions.
        - 'mean_positive': sum over ALL valid positions divided by the number
          of positive (non-background, non-ignored) positions.  This is the
          RetinaNet convention and stabilises the loss scale when the vast
          majority of samples are background.
    label_smoothing : float
        Label-smoothing epsilon forwarded to ``F.cross_entropy``.
        Default: 0.0.
    ignore_index : int
        Class index to ignore (passed through to ``F.cross_entropy``).
        Default: -100.
    background_class : int
        Class index treated as background/negative for the
        ``'mean_positive'`` reduction denominator.  Default: 0.
    gather_distributed : bool or None, optional
        Whether to all-gather inputs and targets across DDP workers before
        computing the loss.  ``None`` (default) auto-detects: gathers when
        ``torch.distributed`` is initialized with world_size > 1.  Set to
        ``False`` to opt out.

    Notes
    -----
    In DDP, ``mean_positive`` normalization is most affected by gathering: if
    positives are rare and unevenly distributed across ranks, the local
    positive count is noisy.  Gathering ensures the denominator reflects the
    true global positive count.
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        background_class: int = 0,
        gather_distributed: bool | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.background_class = background_class
        self.gather_distributed = gather_distributed
        self._gather_resolved: bool | None = None

        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.ndim != 1:
                raise ValueError("alpha must be a 1-D tensor of shape (C,).")
            self.register_buffer("alpha", alpha)
        else:
            self.alpha: torch.Tensor | None = None

    def _should_gather(self) -> bool:
        if self._gather_resolved is None:
            self._gather_resolved = _resolve_gather(self.gather_distributed)
        return self._gather_resolved

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor
            Raw logits of shape ``(N, C)`` or ``(N, C, *)``.
        targets : Tensor
            Integer class labels of shape ``(N,)`` or ``(N, *)``.
            Values in ``[0, C)`` (plus ``ignore_index``).
        sample_weight : Tensor, optional
            Per-element weight, shape matching ``targets``. ``None``
            (default) leaves the objective unweighted -- when no weight has
            ever been supplied to this instance, the loss and its gradient
            are bitwise identical to the pre-``sample_weight`` release. When
            provided, must be non-negative, finite, and match ``targets``
            exactly; it is detached on entry, cast to the loss dtype, and
            never carries gradient. ``ignore_index`` rows contribute
            nothing regardless of weight. Under DDP
            (``gather_distributed``), all ranks must agree on whether
            ``sample_weight`` is supplied on a given step -- a mismatch
            desynchronizes the collectives across ranks.

        Returns
        -------
        Tensor
            Scalar or per-sample loss depending on ``reduction``.
        """
        sample_weight = self._check_sample_weight(
            sample_weight,
            dim0=targets.size(0),
            dtype=inputs.dtype,
            reference="targets",
            exact_shape=targets.shape,
        )

        if self._should_gather():
            inputs  = all_gather_with_grad(inputs)
            targets = all_gather_no_grad(targets)
            if sample_weight is not None:
                sample_weight = all_gather_no_grad(sample_weight)

        # ---- 1. Unreduced CE: shape matches targets ---------------------------
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        # ---- 2. Softmax probabilities → p_t for the true class ---------------
        log_probs = F.log_softmax(inputs, dim=1)  # (N, C, ...)

        # Reshape targets for gather along dim=1: (N, ...) → (N, 1, ...)
        gather_idx = targets.unsqueeze(1)

        # Clamp ignore_index entries so gather doesn't go out-of-bounds;
        # zero them out via valid_mask afterwards.
        valid_mask = targets != self.ignore_index
        safe_idx = gather_idx.clamp(0, inputs.size(1) - 1)

        log_p_t = log_probs.gather(1, safe_idx).squeeze(1)  # (N, ...)
        p_t = log_p_t.exp()  # probability assigned to the true class

        # ---- 3. Focal modulator: (1 - p_t)^gamma -----------------------------
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        # ---- 4. Per-class alpha weighting ------------------------------------
        if self.alpha is not None:
            safe_targets = targets.clamp(0, self.alpha.size(0) - 1)
            alpha_t = self.alpha[safe_targets]
            loss = alpha_t * loss

        # ---- 5. Mask out padding / ignored positions -------------------------
        # Always apply unconditionally — when no positions match ignore_index,
        # valid_mask is all-True and this is a no-op.
        loss = loss * valid_mask

        # ---- 6. Reduction ----------------------------------------------------
        # `loss` already has ignore_index positions zeroed (step 5), so every
        # mode's numerator is the full sum; the modes differ only in which
        # index set normalises it.
        if self.reduction == "none":
            return _reduce_none(loss, sample_weight)
        if self.reduction == "mean":
            return _reduce_mean_masked(loss, valid_mask, sample_weight)
        if self.reduction == "mean_positive":
            positive_mask = valid_mask & (targets != self.background_class)
            return _reduce_mean_masked(loss, positive_mask, sample_weight)
        if self.reduction == "sum":
            return _reduce_sum(loss, sample_weight)
        raise _invalid_reduction(self.reduction)


# ---------------------------------------------------------------------------
# Shared reduction helpers
#
# One function per reduction mode, each with a single `sample_weight is None`
# guard.  The `None` branch is the pre-``sample_weight`` arithmetic, run
# op-for-op; the weighted branch is always "weighted sum over a weight-mass
# denominator".  Which index set the denominator covers is the caller's
# decision -- it is the caller that knows whether 'mean' means "over every
# element" (sigmoid) or "over the valid ones" (softmax), and that
# `mean_positive` normalises by positives while still summing negatives into
# the numerator (the RetinaNet asymmetry).
# ---------------------------------------------------------------------------

def _floor_zero_mass(mass: torch.Tensor) -> torch.Tensor:
    """Floor a weighted-denominator mass to 1 wherever it is exactly zero.

    Mirrors the unweighted reduction's `.clamp(min=1)` convention: an
    all-background (or otherwise structurally empty) batch still divides by
    1, not by a near-zero epsilon, so a nonzero RetinaNet-asymmetry
    numerator (negatives' weighted loss, still present when the positive
    mask is all-False) does not blow up against a near-zero denominator.
    Legitimate sub-unit weight masses (e.g. positives summing to 0.3) are
    left untouched -- only a mass that is exactly zero is floored.
    """
    return torch.where(mass > 0, mass, torch.ones_like(mass))


def _reduce_none(
    loss: torch.Tensor, sample_weight: torch.Tensor | None
) -> torch.Tensor:
    """Per-element loss, scaled by the weight when there is one."""
    if sample_weight is None:
        return loss
    return loss * sample_weight


def _reduce_sum(
    loss: torch.Tensor, sample_weight: torch.Tensor | None
) -> torch.Tensor:
    """Total loss, weighted when there is a weight."""
    if sample_weight is None:
        return loss.sum()
    return (loss * sample_weight).sum()


def _reduce_mean_all(
    loss: torch.Tensor, sample_weight: torch.Tensor | None
) -> torch.Tensor:
    """Mean over every element of *loss* (no validity mask in play).

    Unweighted this is ``loss.mean()``; weighted it is the weighted sum over
    the weight mass of the same elements.
    """
    if sample_weight is None:
        return loss.mean()
    mass = sample_weight.expand_as(loss).sum()
    return (loss * sample_weight).sum() / _floor_zero_mass(mass)


def _reduce_mean_masked(
    loss: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor | None
) -> torch.Tensor:
    """Sum of *loss* over a denominator restricted to *mask*.

    The numerator is always the full ``loss`` sum -- callers zero out the
    elements that must not contribute before calling.  Only the denominator
    is restricted to *mask*: the element count under *mask* unweighted, the
    weight mass under *mask* weighted.  ``mask=valid_mask`` gives 'mean';
    ``mask=positive_mask`` gives 'mean_positive' with its numerator still
    summing the (non-masked-out) negatives.
    """
    if sample_weight is None:
        return loss.sum() / mask.sum().clamp(min=1)
    mass = (sample_weight * mask).sum()
    return (loss * sample_weight).sum() / _floor_zero_mass(mass)


def _invalid_reduction(reduction: str) -> ValueError:
    """Build the error raised for an unsupported ``reduction`` string."""
    return ValueError(
        f"Invalid reduction: '{reduction}'. "
        "Supported modes: 'none', 'mean', 'mean_positive', 'sum'."
    )
