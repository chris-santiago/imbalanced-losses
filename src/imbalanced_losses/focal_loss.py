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
import torch.distributed as dist

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

    def _should_gather(self) -> bool:
        if self._gather_resolved is None:
            self._gather_resolved = (
                self.gather_distributed is not False
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
        return self._gather_resolved

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor
            Raw logits, arbitrary shape.
        targets : Tensor
            Same shape, float 0/1 labels.

        Returns
        -------
        Tensor
            Scalar or per-element loss depending on ``reduction``.
        """
        if self._should_gather():
            inputs  = all_gather_with_grad(inputs)
            targets = all_gather_no_grad(targets)

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return _reduce(loss, self.reduction)


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

        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.ndim != 1:
                raise ValueError("alpha must be a 1-D tensor of shape (C,).")
            self.register_buffer("alpha", alpha)
        else:
            self.alpha: torch.Tensor | None = None

    def _should_gather(self) -> bool:
        if self._gather_resolved is None:
            self._gather_resolved = (
                self.gather_distributed is not False
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
        return self._gather_resolved

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor
            Raw logits of shape ``(N, C)`` or ``(N, C, *)``.
        targets : Tensor
            Integer class labels of shape ``(N,)`` or ``(N, *)``.
            Values in ``[0, C)`` (plus ``ignore_index``).

        Returns
        -------
        Tensor
            Scalar or per-sample loss depending on ``reduction``.
        """
        if self._should_gather():
            inputs  = all_gather_with_grad(inputs)
            targets = all_gather_no_grad(targets)

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
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)

        if self.reduction == "mean_positive":
            positive_mask = valid_mask & (targets != self.background_class)
            return _reduce(loss, "mean_positive", valid_mask, positive_mask)

        return _reduce(loss, self.reduction, valid_mask)


# ---------------------------------------------------------------------------
# Shared reduction helper
# ---------------------------------------------------------------------------

def _reduce(
    loss: torch.Tensor,
    reduction: str,
    valid_mask: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply reduction, handling valid/positive masks.

    For 'mean_positive', the numerator sums over ALL valid positions (negatives
    included) but the denominator counts only positive positions.  This matches
    the RetinaNet convention where alpha-weighted negative loss still contributes
    but the normalisation is anchored to the positive count.
    """
    if reduction == "none":
        return loss
    elif reduction == "mean":
        if valid_mask is not None:
            return loss.sum() / valid_mask.sum().clamp(min=1)
        return loss.mean()
    elif reduction == "mean_positive":
        n_positive = positive_mask.sum().clamp(min=1)
        return loss.sum() / n_positive
    elif reduction == "sum":
        return loss.sum()
    raise ValueError(
        f"Invalid reduction: '{reduction}'. "
        "Supported modes: 'none', 'mean', 'mean_positive', 'sum'."
    )
