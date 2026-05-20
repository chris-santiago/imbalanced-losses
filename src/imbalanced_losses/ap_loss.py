"""
Smooth-AP loss (Brown et al., 2020) with a memory queue.

For each positive i, soft ranks are estimated via:
    ŝ_i   = 1 + Σ_{j≠i}       σ((s_j − s_i) / τ)   # overall
    ŝ_i^+ = 1 + Σ_{j≠i, j∈P} σ((s_j − s_i) / τ)   # among positives
    AP ≈ (1/|P|) · Σ_{i∈P}  ŝ_i^+ / ŝ_i

Multi-class: one-vs-rest per class (scores = logits[:, c]).
Binary:      num_classes=1, logits [N, 1], targets in {0, 1}.
Seq2seq:     flatten to [N, C] / [N] upstream.
Padding:     ignore_index=-100 rows dropped before ranking.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch

from imbalanced_losses._base import _QueuedRankingLoss


class SmoothAPLoss(_QueuedRankingLoss):
    """
    Differentiable Average Precision loss with an optional memory queue.

    Approximates AP using soft sigmoid-based rank estimation (Smooth-AP,
    Brown et al. 2020). Supports multi-class (one-vs-rest) and binary
    (num_classes=1) classification. Expects logits [N, C] and targets [N];
    this class is agnostic to sequence structure — flatten upstream.

    Inherits queue management, DDP gather, ignore-index filtering,
    subsampling, and reduction logic from ``_QueuedRankingLoss``.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Use 1 for binary mode.
    queue_size : int, optional
        Number of (logits, targets) rows stored in the circular buffer.
        Larger queues give more stable AP estimates at the cost of O(|P|×M)
        memory in _compute_smooth_ap, where |P| is the number of positives.
        Set to 0 to disable. Default: 1024.

        **DDP note:** when ``gather_distributed=True``, the all-gather runs
        *before* the enqueue, so each rank stores global-batch rows. The
        effective pool per forward pass is already
        ``global_batch_size + queue_size``. At large global batches
        (e.g. 8 GPUs × 1 500 samples = 12 K) the default queue adds < 10 %
        to the pool. Consider setting ``queue_size=0`` in that regime to
        avoid storing redundant data and reduce memory overhead.
    temperature : float, optional
        Sigmoid sharpness τ. Smaller values approximate the true
        discontinuous rank more closely but produce harder gradients.
        Typical range: 0.005–0.05. Default: 0.01.
    reduction : {'mean', 'sum', 'none'}, optional
        How to aggregate per-class losses.
        - 'mean': scalar average over valid classes.
        - 'sum':  scalar sum over valid classes.
        - 'none': tensor of shape [C]; degenerate classes are nan.
        Default: 'mean'.
    ignore_index : int, optional
        Target value marking padded positions. Matching rows are excluded
        from ranking and the positive set. Default: -100.
    update_queue_in_eval : bool, optional
        If False (default), the queue is frozen during eval mode. Set to
        True to allow queue updates during validation. Default: False.
    gather_distributed : bool or None, optional
        Whether to all-gather logits and targets across DDP workers before
        computing the loss.  ``None`` (default) auto-detects: gathers when
        ``torch.distributed`` is initialized with world_size > 1.  Set
        ``False`` to explicitly disable. Resolved once on first forward call,
        so safe to construct before ``dist.init_process_group``. Default: None.
    max_pool_size : int or None, optional
        Maximum number of rows in the ranking pool (live batch + queue after
        ignore_index filtering).  When the pool exceeds this value,
        minimum-quota subsampling is applied: each observed class is guaranteed
        an equal quota of rows (``max_pool_size // (2 * n_classes)``), then
        the remaining budget is filled uniformly at random.  This is not
        proportional sampling — rare classes are over-represented relative to
        their natural frequency.  Effective ``|P_c| ≈ max_pool_size //
        (2 * n_classes)``; size accordingly.  ``None`` (default) disables
        the cap.

        Use this for seq2seq tasks where flattened inputs produce very large
        pools. The pairwise matrix in ``_compute_smooth_ap`` is ``[P, M]``
        where ``M`` is the pool size — at M=15 000 the gradient memory is
        O(M^2) and easily OOMs.  Recommended: 2048–4096 for seq2seq.

        .. note::
            Subsampling is a stochastic approximation — the loss value will
            vary across steps even for the same batch.  Use the largest value
            your GPU allows for the most stable gradient estimates.

    Examples
    --------
    >>> loss_fn = SmoothAPLoss(num_classes=4, queue_size=512)
    >>> logits  = torch.randn(32, 4)
    >>> targets = torch.randint(0, 4, (32,))
    >>> loss = loss_fn(logits, targets)
    >>> loss.backward()

    Notes
    -----
    Complexity of _compute_smooth_ap is O(|P| × M), where |P| is the number
    of positives in the pool and M = batch_size + queue_size. At low positive
    rates this is much cheaper than the naive O(M²) formulation.

    In DDP, set ``gather_distributed=False`` to opt out; otherwise the loss
    auto-detects and all-gathers on first forward when world_size > 1.
    Because the gather happens *before* the enqueue, every rank stores
    identical global-batch rows — queues stay in sync automatically, but
    the pool per step is ``global_batch_size + queue_size``. At large
    global batch sizes the queue contribution may be negligible; prefer
    ``queue_size=0`` when the global batch already provides a stable pool.
    """

    def __init__(
        self,
        num_classes: int,
        queue_size: int = 1024,
        temperature: float = 0.01,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
        update_queue_in_eval: bool = False,
        gather_distributed: bool | None = None,
        max_pool_size: int | None = None,
    ) -> None:
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

    @staticmethod
    def _compute_smooth_ap(
        scores: torch.Tensor,
        is_pos: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, bool]:
        """
        Compute Smooth-AP for a single binary partition of the pool.

        Parameters
        ----------
        scores : torch.Tensor, shape [M]
            Raw scores for one class across the full pool (live + queue).
            Gradients flow through live-batch scores only; queue scores
            are detached before being passed in.
        is_pos : torch.Tensor, shape [M], dtype=bool
            True for positive samples (target == c for class c).
        tau : float
            Sigmoid temperature. See class docstring.

        Returns
        -------
        ap : torch.Tensor, scalar
            Smooth-AP estimate in [0, 1]. Zero (no gradient) for
            degenerate cases.
        valid : bool
            False if the class is degenerate (all-positive or all-negative
            in the pool). Degenerate classes are excluded from the
            mean/sum reduction rather than contributing a misleading 0.

        Notes
        -----
        Pairwise soft rank (computed only for positive rows):
            diff[k, j]    = s_j - s_pos_k            k ∈ P, j ∈ [M]
            soft_gt[k, j] ≈ P(s_j > s_pos_k) = σ(diff[k,j] / τ)
            rank_all[k]   = 1 + Σ_j soft_gt[k, j]   (self zeroed)
            rank_pos[k]   = 1 + Σ_{j∈P} soft_gt[k, j]
            AP            = mean_{k∈P} rank_pos[k] / rank_all[k]

        Complexity is O(|P| × M) rather than O(M²), reducing memory and
        compute by roughly 1/pos_rate (e.g. ~200× at 0.5% positives).
        """
        m     = scores.size(0)
        n_pos = int(is_pos.sum())

        if n_pos == 0 or n_pos == m:
            return scores.new_zeros(()), False

        # Only compute rows for positives: [|P|, M] instead of [M, M].
        # Reduces memory/compute by ~1/pos_rate (e.g. 200× at 0.5% positives).
        pos_idx  = is_pos.nonzero(as_tuple=False).squeeze(1)           # [P]
        diff_pos = scores.unsqueeze(0) - scores[pos_idx].unsqueeze(1)  # [P, M]; diff[k,j] = s_j - s_pos_k
        soft_gt  = torch.sigmoid(diff_pos / tau)                        # [P, M]
        # Zero self-comparisons without in-place ops (would break autograd).
        self_mask = torch.zeros(n_pos, m, device=scores.device, dtype=torch.bool)
        self_mask[torch.arange(n_pos, device=scores.device), pos_idx] = True
        soft_gt   = soft_gt.masked_fill(self_mask, 0.0)

        rank_all = 1.0 + soft_gt.sum(dim=1)            # [P]
        rank_pos = 1.0 + soft_gt[:, is_pos].sum(dim=1) # [P]

        ap = (rank_pos / rank_all).mean()
        return ap, True

    # ------------------------------------------------------------------
    # Per-class dispatch (required by _QueuedRankingLoss)
    # ------------------------------------------------------------------

    def _compute_per_class(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 1 - AP for each class via one-vs-rest decomposition.

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
            Per-class loss values (1 - AP).
        valid_vec : torch.Tensor, shape [C], dtype=bool
            True for classes with at least one positive and one negative.
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
            ap, is_valid = self._compute_smooth_ap(
                logits[:, 0], targets.bool(), self.temperature
            )
            loss_vals = [1.0 - ap]
            valid_mask = [is_valid]
        else:
            loss_vals, valid_mask = [], []
            for c in range(self.num_classes):
                ap, is_valid = self._compute_smooth_ap(
                    logits[:, c], targets == c, self.temperature
                )
                loss_vals.append(1.0 - ap)
                valid_mask.append(is_valid)

        loss_vec = torch.stack(loss_vals)
        valid_vec = torch.tensor(valid_mask, device=logits.device)
        return loss_vec, valid_vec
