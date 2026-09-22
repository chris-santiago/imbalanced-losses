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

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from imbalanced_losses._base import _resolve_gather
from imbalanced_losses.distributed import all_gather_no_grad, all_gather_with_grad


# ---------------------------------------------------------------------------
# Sigmoid Focal Loss
# ---------------------------------------------------------------------------

class SigmoidFocalLoss(nn.Module):
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
        self._sample_weight_warned = False

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
            Per-element weight, broadcastable to ``inputs``. ``None``
            (default) leaves the objective unweighted -- when no weight has
            ever been supplied to this instance, the loss and its gradient
            are bitwise identical to the pre-``sample_weight`` release. When
            provided, must be non-negative and broadcastable to ``inputs``;
            it is detached on entry and never carries gradient. Under DDP
            (``gather_distributed``), all ranks must agree on whether
            ``sample_weight`` is supplied on a given step -- a mismatch
            desynchronizes the collectives across ranks.

        Returns
        -------
        Tensor
            Scalar or per-element loss depending on ``reduction``.
        """
        if sample_weight is not None:
            try:
                sample_weight.expand(inputs.shape)
            except RuntimeError:
                raise ValueError(
                    f"sample_weight must be broadcastable to inputs, "
                    f"got {tuple(sample_weight.shape)} vs inputs {tuple(inputs.shape)}"
                ) from None
        sample_weight = _validate_sample_weight(self, sample_weight)

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

        return _reduce(loss, self.reduction, sample_weight=sample_weight)


# ---------------------------------------------------------------------------
# Softmax Focal Loss
# ---------------------------------------------------------------------------

class SoftmaxFocalLoss(nn.Module):
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
        self._sample_weight_warned = False

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
            provided, must be non-negative and match ``targets`` exactly; it
            is detached on entry and never carries gradient. ``ignore_index``
            rows contribute nothing regardless of weight. Under DDP
            (``gather_distributed``), all ranks must agree on whether
            ``sample_weight`` is supplied on a given step -- a mismatch
            desynchronizes the collectives across ranks.

        Returns
        -------
        Tensor
            Scalar or per-sample loss depending on ``reduction``.
        """
        if sample_weight is not None and sample_weight.shape != targets.shape:
            raise ValueError(
                f"sample_weight must match targets shape, "
                f"got {tuple(sample_weight.shape)} vs targets {tuple(targets.shape)}"
            )
        sample_weight = _validate_sample_weight(self, sample_weight)

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
        # 'mean' is routed through _reduce like every other mode (rather than
        # short-circuited inline) so its denominator picks up sample_weight
        # the same way 'mean_positive' does below.
        if self.reduction == "mean_positive":
            positive_mask = valid_mask & (targets != self.background_class)
            return _reduce(loss, "mean_positive", valid_mask, positive_mask, sample_weight=sample_weight)

        return _reduce(loss, self.reduction, valid_mask, sample_weight=sample_weight)


# ---------------------------------------------------------------------------
# Shared reduction helper
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


def _reduce(
    loss: torch.Tensor,
    reduction: str,
    valid_mask: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply reduction, handling valid/positive masks and an optional weight.

    For 'mean_positive', the numerator sums over ALL valid positions (negatives
    included) but the denominator counts only positive positions.  This matches
    the RetinaNet convention where alpha-weighted negative loss still contributes
    but the normalisation is anchored to the positive count.

    When `sample_weight` is supplied (already broadcastable/aligned with
    `loss`, per spec S4.1), every count-normalised denominator becomes the
    weight mass over the same index set the count covered: valid elements
    for 'mean', positive elements for 'mean_positive'.  `sample_weight is
    None` is the structural unweighted path -- it runs the exact
    pre-``sample_weight`` arithmetic below, unchanged.

    A zero weight mass is floored to 1, mirroring the unweighted
    convention's `.clamp(min=1)` (see `_floor_zero_mass`) rather than an
    `eps` floor -- an `eps` floor blows up under the RetinaNet asymmetry,
    where `mean_positive`'s numerator can be nonzero (negatives' weighted
    loss) even when the positive weight mass is structurally zero (an
    all-background batch).
    """
    if reduction == "none":
        if sample_weight is not None:
            return loss * sample_weight
        return loss
    elif reduction == "mean":
        if sample_weight is not None:
            weight_mass = (
                sample_weight * valid_mask
                if valid_mask is not None
                else sample_weight.expand_as(loss)
            )
            denom = _floor_zero_mass(weight_mass.sum())
            return (loss * sample_weight).sum() / denom
        if valid_mask is not None:
            return loss.sum() / valid_mask.sum().clamp(min=1)
        return loss.mean()
    elif reduction == "mean_positive":
        if sample_weight is not None:
            weight_mass = sample_weight * positive_mask
            denom = _floor_zero_mass(weight_mass.sum())
            return (loss * sample_weight).sum() / denom
        n_positive = positive_mask.sum().clamp(min=1)
        return loss.sum() / n_positive
    elif reduction == "sum":
        if sample_weight is not None:
            return (loss * sample_weight).sum()
        return loss.sum()
    raise ValueError(
        f"Invalid reduction: '{reduction}'. "
        "Supported modes: 'none', 'mean', 'mean_positive', 'sum'."
    )


# ---------------------------------------------------------------------------
# Shared sample_weight validation
# ---------------------------------------------------------------------------

def _validate_sample_weight(
    module: nn.Module,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor | None:
    """Detach, reject negatives, and one-shot-warn on an all-zero weight.

    Shared by ``SigmoidFocalLoss`` and ``SoftmaxFocalLoss`` after each has
    already checked its own shape contract (broadcastable-to-inputs for
    Sigmoid, exact ``targets`` shape for Softmax) and raised its own
    shape-specific ``ValueError``. Mirrors the validation/warning idiom in
    ``_base.py::_QueuedRankingLoss.forward`` (detach on entry, raise on any
    negative value, one-shot ``UserWarning`` on a wholly-zero tensor)
    without importing from it -- focal losses have their own forward flow
    and gather block.
    """
    if sample_weight is None:
        return None
    # Detach immediately -- sample_weight never carries or receives
    # gradient, and every downstream touch point assumes this.
    sample_weight = sample_weight.detach()
    # Both reductions below are undefined/vacuous on an empty tensor
    # (.min() raises; .all() is trivially True). A zero-row call must fall
    # through untouched, not raise or spuriously warn here.
    if sample_weight.numel() > 0:
        min_val = sample_weight.min()
        if min_val < 0:
            raise ValueError(
                f"sample_weight must be non-negative, got min {min_val.item()}"
            )
        if bool((sample_weight == 0).all()):
            if not module._sample_weight_warned:
                warnings.warn(
                    f"{type(module).__name__}: sample_weight is entirely zero for this "
                    f"call, a likely misconfiguration; the reduction denominator is "
                    f"floored, yielding a zero loss. "
                    f"(This warning is shown once per instance.)",
                    UserWarning,
                    # One frame deeper than _base.py's direct warnings.warn
                    # call (this helper is invoked from forward, not inlined
                    # in it), so stacklevel=3 -- not 2 -- is what points the
                    # warning at the user's call site rather than at forward.
                    stacklevel=3,
                )
                module._sample_weight_warned = True
    return sample_weight
