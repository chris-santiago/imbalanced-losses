"""
Differentiable Recall-at-Quantile loss with a memory queue.

For a given quantile q (e.g. 0.005 = top 50bps), the threshold θ is the
(1-q) quantile of ALL scores in the pool (live batch + queue). Recall@q is
then approximated as the fraction of positives scoring above θ:

    θ = quantile(scores, 1 - q)                     [detached — no grad]
    soft_recall = (1/|P|) · Σ_{i∈P} σ((s_i − θ) / τ)
    loss = 1 − soft_recall

Gradient flows only through the positive scores, pushing them above the
cutoff. The threshold is treated as a fixed constant each forward pass,
analogous to a stop-gradient in contrastive losses.

Multi-class: one-vs-rest per class, same convention as SmoothAPLoss.
Binary:      num_classes=1, logits [N,1], targets 0/1.
Seq2seq:     flatten to [N, C] / [N] upstream.
Padding:     ignore_index=-100 rows are dropped before threshold estimation
             and recall computation.

Note: This is an original loss design, not from a published paper. It
combines quantile-based threshold estimation (stop-gradient) with sigmoid
soft recall in a way that, to our knowledge, has not appeared in prior
literature.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch

from imbalanced_losses._base import _QueuedRankingLoss


class RecallAtQuantileLoss(_QueuedRankingLoss):
    """
    Differentiable Recall-at-Quantile loss with an optional memory queue.

    For a given quantile q, a threshold θ is estimated from the pooled score
    distribution (live batch + queue) without gradient, then soft recall over
    positives is computed per class:

        θ = quantile(scores, 1 - q)                 [stop-gradient]
        soft_recall = mean_{i∈P} σ((s_i − θ) / τ)
        loss = 1 − soft_recall

    Multi-class: one-vs-rest per class using logits[:, c], then reduce.
    Binary:      logits[:, 0] with targets in {0, 1}.

    Inherits queue management, DDP gather, ignore-index filtering,
    subsampling, and reduction logic from ``_QueuedRankingLoss``.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Use 1 for binary mode.
    quantile : float, optional
        Fraction of the score distribution treated as the alert region.
        E.g. 0.005 = top 50 bps, 0.01 = top 100 bps. Must be in (0, 1).
        Default: 0.005.
    queue_size : int, optional
        Circular buffer size (rows). Larger queues stabilise the quantile
        estimate — at 50 bps you need at least ~200 samples for a
        meaningful 99.5th percentile. Set to 0 to disable. Default: 1024.

        **DDP note:** when ``gather_distributed=True``, the all-gather runs
        *before* the enqueue, so each rank stores global-batch rows. The
        effective pool per forward pass is already
        ``global_batch_size + queue_size``. At large global batches the
        quantile is already well-estimated from the live batch alone;
        consider setting ``queue_size=0`` to reduce memory overhead.
    temperature : float, optional
        Sigmoid sharpness τ around the threshold. Larger values give
        smoother gradients but less precise recall estimates. Default: 0.01.
    reduction : {'mean', 'sum', 'none'}, optional
        How to aggregate per-class losses.
        - 'mean': scalar average over valid classes.
        - 'sum':  scalar sum over valid classes.
        - 'none': tensor of shape [C]; classes with no positives are nan.
        Default: 'mean'.
    ignore_index : int, optional
        Target value marking padded positions. Excluded from threshold
        estimation and recall. Default: -100.
    update_queue_in_eval : bool, optional
        If False (default), the queue is frozen during eval mode. Default: False.
    gather_distributed : bool or None, optional
        Whether to all-gather logits and targets across DDP workers before
        computing the loss. ``None`` (default) auto-detects: gathers when
        ``torch.distributed`` is initialized with world_size > 1. Set
        ``False`` to explicitly disable. Resolved once on first forward call,
        so safe to construct before ``dist.init_process_group``. Default: None.
    quantile_interpolation : str, optional
        Interpolation method passed to torch.quantile. 'higher' is the
        conservative default — the threshold never undershoots the true
        cutoff. One of ('linear', 'lower', 'higher', 'nearest', 'midpoint').
        Default: 'higher'.
    max_pool_size : int or None, optional
        Maximum number of rows in the ranking pool (live batch + queue after
        ignore_index filtering).  When the pool exceeds this value,
        minimum-quota subsampling caps it at ``max_pool_size`` rows: each
        observed class is guaranteed an equal quota (``max_pool_size //
        (2 * n_classes)``), then the remaining budget is filled uniformly at
        random.  This is not proportional sampling — rare classes are
        over-represented relative to their natural frequency.  Effective
        ``|P_c| ≈ max_pool_size // (2 * n_classes)``; size accordingly.
        ``None`` (default) disables the cap.

        Use this for seq2seq tasks with very large flattened pool sizes.
        Recommended: 2048–4096.  Subsampling also introduces noise into the
        quantile threshold estimate, so use the largest value your GPU allows.

    Examples
    --------
    >>> loss_fn = RecallAtQuantileLoss(num_classes=4, quantile=0.005, queue_size=512)
    >>> logits  = torch.randn(32, 4)
    >>> targets = torch.randint(0, 4, (32,))
    >>> loss = loss_fn(logits, targets)
    >>> loss.backward()

    Notes
    -----
    The quantile must exceed the positive class fraction for the threshold
    to fall in the negative region under perfect classification. With C=4
    balanced classes (25% positives), use quantile > 0.25 for sanity tests.

    In DDP, the all-gather runs *before* the enqueue, so every rank stores
    identical global-batch rows and queues stay in sync automatically. The
    effective pool per step is ``global_batch_size + queue_size``. At large
    global batch sizes the queue contribution may be negligible; prefer
    ``queue_size=0`` when the global batch already provides a stable quantile
    estimate.
    """

    _VALID_INTERPOLATIONS = ("linear", "lower", "higher", "nearest", "midpoint")

    def __init__(
        self,
        num_classes: int,
        quantile: float = 0.005,
        queue_size: int = 1024,
        temperature: float = 0.01,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
        update_queue_in_eval: bool = False,
        gather_distributed: bool | None = None,
        quantile_interpolation: str = "higher",
        max_pool_size: int | None = None,
    ) -> None:
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if quantile_interpolation not in self._VALID_INTERPOLATIONS:
            raise ValueError(
                f"quantile_interpolation must be one of {self._VALID_INTERPOLATIONS}, "
                f"got '{quantile_interpolation}'"
            )

        super().__init__(
            num_classes=num_classes,
            queue_size=queue_size,
            temperature=temperature,
            reduction=reduction,
            ignore_index=ignore_index,
            update_queue_in_eval=update_queue_in_eval,
            gather_distributed=gather_distributed,
            max_pool_size=max_pool_size,
        )

        self.quantile = float(quantile)
        self.quantile_interpolation = quantile_interpolation

    # ------------------------------------------------------------------
    # Backward-compatible access to queue internals
    # ------------------------------------------------------------------
    # Tests and external code may access _q_logits, _q_targets, _q_ptr
    # directly on the loss instance. These properties delegate to the
    # nested _MemoryQueue submodule.

    @property
    def _q_logits(self):
        return self._queue._q_logits

    @_q_logits.setter
    def _q_logits(self, value):
        self._queue._q_logits = value

    @property
    def _q_targets(self):
        return self._queue._q_targets

    @_q_targets.setter
    def _q_targets(self, value):
        self._queue._q_targets = value

    @property
    def _q_ptr(self):
        return self._queue._q_ptr

    @_q_ptr.setter
    def _q_ptr(self, value):
        self._queue._q_ptr = value

    # ------------------------------------------------------------------
    # Backward-compatible queue methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _enqueue(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Delegate to the internal ``_MemoryQueue``."""
        self._queue.enqueue(logits, targets)

    def _merge_with_queue(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate to the internal ``_MemoryQueue``."""
        return self._queue.merge(logits, targets)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _soft_recall_at_quantile(
        self,
        scores: torch.Tensor,
        is_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """
        Compute soft recall above the score quantile for one class.

        Parameters
        ----------
        scores : torch.Tensor, shape [M]
            Pooled scores for one class (live + queue, padding stripped).
            The threshold is computed from all scores (positives and
            negatives), then applied only to positives.
        is_pos : torch.Tensor, shape [M], dtype=bool
            Positive mask for this class.

        Returns
        -------
        recall : torch.Tensor, scalar
            Soft recall estimate in [0, 1].
        valid : bool
            False if there are no positives in the pool. Classes with
            no positives are excluded from the reduction rather than
            contributing a misleading 0.

        Notes
        -----
        The threshold θ is detached before use. Gradient flows only
        through the positive scores, pushing them above the cutoff.
        """
        n_pos = int(is_pos.sum())
        if n_pos == 0:
            return scores.new_zeros(()), False

        theta = torch.quantile(
            scores.detach(),
            1.0 - self.quantile,
            interpolation=self.quantile_interpolation,
        )
        soft_above = torch.sigmoid((scores[is_pos] - theta) / self.temperature)
        return soft_above.mean(), True

    # ------------------------------------------------------------------
    # Subclass validation hook
    # ------------------------------------------------------------------

    def _validate_filtered_targets(self, targets: torch.Tensor) -> None:
        """
        Validate target range after ignore-index filtering.

        In multi-class mode (num_classes > 1), checks that all targets
        are in [0, num_classes) and raises ValueError if any are out of
        range.
        """
        if self.num_classes > 1:
            bad = targets[(targets < 0) | (targets >= self.num_classes)]
            if bad.numel() > 0:
                raise ValueError(
                    f"targets contain class ids outside [0, {self.num_classes}); "
                    f"examples: {bad[:8].tolist()}"
                )

    # ------------------------------------------------------------------
    # Per-class dispatch (required by _QueuedRankingLoss)
    # ------------------------------------------------------------------

    def _compute_per_class(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 1 - recall for each class via one-vs-rest decomposition.

        Parameters
        ----------
        logits : torch.Tensor, shape [M, C]
            Pooled logits (live batch + queue, ignore-index rows removed,
            subsampling applied).
        targets : torch.Tensor, shape [M]
            Corresponding integer targets.

        Returns
        -------
        loss_vec : torch.Tensor, shape [C]
            Per-class loss values (1 - soft_recall).
        valid_vec : torch.Tensor, shape [C], dtype=bool
            True for classes with at least one positive in the pool.
        """
        if self.num_classes == 1:
            # Binary mode: warn on out-of-range targets
            bad = targets[(targets != 0) & (targets != 1)]
            if bad.numel() > 0:
                warnings.warn(
                    f"Binary mode (num_classes=1) expects targets in {{0, 1}}, "
                    f"but found values: {bad[:8].tolist()}. "
                    "Non-zero values are treated as positive.",
                    UserWarning,
                    stacklevel=4,
                )
            recall, is_valid = self._soft_recall_at_quantile(
                logits[:, 0], targets.bool()
            )
            loss_vals = [1.0 - recall]
            valid_mask = [is_valid]
        else:
            loss_vals, valid_mask = [], []
            for c in range(self.num_classes):
                recall, is_valid = self._soft_recall_at_quantile(
                    logits[:, c], targets == c
                )
                loss_vals.append(1.0 - recall)
                valid_mask.append(is_valid)

        loss_vec = torch.stack(loss_vals)
        valid_vec = torch.tensor(valid_mask, device=logits.device)
        return loss_vec, valid_vec
