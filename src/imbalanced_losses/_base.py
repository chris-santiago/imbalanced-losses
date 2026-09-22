"""
Abstract base class for queued ranking losses.

Captures the shared forward-flow of SmoothAPLoss and RecallAtQuantileLoss:
shape validation, DDP gather, queue merge, ignore_index filtering,
subsampling, per-class compute dispatch, enqueue, reduction, and
return_per_class handling.

Private module -- nothing here is part of the public API.
"""

from __future__ import annotations

import abc
import warnings
from typing import Literal

import torch
import torch.nn as nn

from imbalanced_losses._queue import _MemoryQueue
from imbalanced_losses._sampling import subsample_pool
from imbalanced_losses._weights import _SampleWeightMixin
from imbalanced_losses.distributed import all_gather_no_grad, all_gather_with_grad


# ---------------------------------------------------------------------------
# Module-level helper -- reusable by focal losses without subclassing
# ---------------------------------------------------------------------------

def _resolve_gather(gather_distributed: bool | None) -> bool:
    """
    Determine whether DDP all-gather should run.

    Returns True when ``gather_distributed`` is not explicitly ``False``
    *and* ``torch.distributed`` is available, initialized, with
    ``world_size > 1``.  Safe to call before ``dist.init_process_group``
    -- simply returns False until dist is initialized.

    Parameters
    ----------
    gather_distributed : bool or None
        User-supplied flag.  ``None`` means auto-detect.

    Returns
    -------
    bool
    """
    if gather_distributed is False:
        return False
    import torch.distributed as dist
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class _QueuedRankingLoss(_SampleWeightMixin, nn.Module, abc.ABC):
    """
    Abstract base for ranking losses that use a memory queue.

    Subclasses implement :meth:`_compute_per_class` (and optionally
    :meth:`_validate_filtered_targets`); everything else -- validation,
    DDP gather, queue merge, ignore-index filtering, subsampling,
    reduction, and ``return_per_class`` -- is handled here.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Use 1 for binary mode.
    queue_size : int
        Circular buffer capacity (rows).  0 disables.
    temperature : float
        Sigmoid sharpness parameter exposed to subclasses and
        ``LossWarmupWrapper``.
    reduction : ``'mean'`` | ``'sum'`` | ``'none'``
        How to aggregate per-class losses.
    ignore_index : int
        Target sentinel for padded positions.
    update_queue_in_eval : bool
        Whether the queue updates during ``eval()`` mode.
    gather_distributed : bool or None
        DDP all-gather flag (``None`` = auto-detect).
    max_pool_size : int or None
        Cap on ranking-pool rows after ignore-index filtering.
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
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if queue_size < 0:
            raise ValueError(f"queue_size must be >= 0, got {queue_size}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction '{reduction}'")
        if max_pool_size is not None and max_pool_size <= 0:
            raise ValueError(f"max_pool_size must be positive, got {max_pool_size}")

        self.num_classes = num_classes
        self.queue_size = queue_size
        self.temperature = float(temperature)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.update_queue_in_eval = update_queue_in_eval
        self.gather_distributed = gather_distributed
        self.max_pool_size = max_pool_size

        self._gather_resolved: bool | None = None
        self._subsample_warned = False

        # Delegate all queue state to _MemoryQueue.
        self._queue = _MemoryQueue(queue_size, num_classes, ignore_index)

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Remap legacy flat-buffer keys (pre-refactor checkpoints stored
        # buffers directly on the loss) to the nested _queue.* prefix.
        for suffix in ("_q_logits", "_q_targets", "_q_ptr", "_q_weight"):
            old_key = prefix + suffix
            new_key = prefix + "_queue." + suffix
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # ------------------------------------------------------------------
    # Gather / queue helpers
    # ------------------------------------------------------------------

    def _should_gather(self) -> bool:
        """
        Return True if logits/targets should be all-gathered before this forward.

        Resolved once on first call and cached.
        """
        if self._gather_resolved is None:
            self._gather_resolved = _resolve_gather(self.gather_distributed)
        return self._gather_resolved

    def _should_update_queue(self) -> bool:
        """
        Return True if the queue should be updated on this forward pass.
        """
        return self.queue_size > 0 and (self.training or self.update_queue_in_eval)

    @torch.no_grad()
    def reset_queue(self) -> None:
        """
        Clear the circular buffer.

        Resets all stored logits to zero, all stored targets to
        ignore_index, and the write pointer to 0.  Typically called
        between training and evaluation epochs.
        """
        self._queue.reset()

    # ------------------------------------------------------------------
    # Abstract / hook methods for subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute_per_class(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_iid: torch.Tensor,
        is_live: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and validity for each class.

        Parameters
        ----------
        logits : torch.Tensor, shape [M, C]
            Pooled logits (live batch + queue, ignore-index rows already
            removed, subsampling already applied).
        targets : torch.Tensor, shape [M]
            Corresponding integer targets.
        is_iid : torch.Tensor, shape [M], dtype=bool
            Per-row flag indicating whether the row is an iid sample
            eligible for FPR threshold estimation (used by
            ``PAUCAtBudgetLoss``; existing losses ignore it).
        is_live : torch.Tensor, shape [M], dtype=bool
            Per-row flag indicating whether the row came from the live
            batch (True) or the memory queue (False).  Used by
            ``PAUCAtBudgetLoss`` when ``pos_numerator="live"``; existing
            losses accept and ignore it.
        sample_weight : torch.Tensor, shape [M], optional
            Pooled per-row weight aligned row-for-row with *logits*
            (live rows at their supplied weight, queue rows at their
            stored weight).  ``None`` iff the unweighted code path is
            active for this call -- no weight was supplied and the queue
            holds no weighted row.  A tensor is the signal to run the
            weighted arithmetic; only positives' weights participate in
            the ranking objective, negatives' weights are never used, and
            thresholds/ranks/bands never see it.

        Returns
        -------
        loss_vec : torch.Tensor, shape [C]
            Per-class loss values.  Degenerate classes should use 0.0
            (they will be masked to nan downstream when reduction='none').
        valid_vec : torch.Tensor, shape [C], dtype=bool
            True for classes that had a meaningful computation (e.g. at
            least one positive and one negative).
        """

    def _validate_filtered_targets(self, targets: torch.Tensor) -> None:
        """
        Optional hook called after ignore-index filtering and subsampling,
        before ``_compute_per_class``.

        Default is a no-op.  ``RecallAtQuantileLoss`` overrides this to
        validate target ranges.

        Parameters
        ----------
        targets : torch.Tensor, shape [M]
            Filtered targets (no ignore-index values).
        """
        pass

    # ------------------------------------------------------------------
    # Forward template
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        iid_mask: torch.Tensor | None = None,
        return_per_class: bool = False,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the ranking loss.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Raw (un-normalised) class scores.
        targets : torch.Tensor, shape [N]
            Integer class labels.  Positions equal to ``ignore_index``
            are excluded from ranking.
        iid_mask : torch.Tensor, shape [N], dtype=bool, optional
            Per-row flag indicating which live-batch rows are iid-eligible
            for FPR threshold estimation (used by ``PAUCAtBudgetLoss``).
            ``None`` (default) treats all rows as iid, giving identical
            behavior to passing an explicit all-True mask.  Must match
            ``logits`` dim-0 when provided.
        return_per_class : bool, optional
            If True, also return per-class losses and a validity mask.
        sample_weight : torch.Tensor, shape [N], optional
            Per-observation weight.  ``None`` (default) leaves the
            objective unweighted -- when no weight has ever been supplied
            to this instance and the memory queue holds no weighted row,
            the loss and its gradient are bitwise identical to the
            pre-``sample_weight`` release.  When provided, must be
            non-negative and match ``logits`` dim-0; it is detached on
            entry and never carries gradient.  Only positives' weights
            enter the ranking objective (thresholds, ranks, and the pAUC
            band never see it); negatives' weights are ignored.  Under
            DDP (``gather_distributed``), all ranks must agree on whether
            ``sample_weight`` is supplied on a given step -- a mismatch
            desynchronizes the collectives across ranks.

        Returns
        -------
        loss : torch.Tensor
            Scalar (reduction='mean' or 'sum') or shape [C]
            (reduction='none').
        per_class_loss : torch.Tensor, shape [C]
            Only returned when ``return_per_class=True``.
        valid_classes : torch.Tensor, shape [C], dtype=bool
            Only returned when ``return_per_class=True``.
        """
        # --- shape validation ------------------------------------------------
        if targets.ndim == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if logits.ndim != 2 or logits.size(1) != self.num_classes:
            raise ValueError(
                f"Expected logits [N, {self.num_classes}], got {tuple(logits.shape)}"
            )
        if targets.ndim != 1 or targets.size(0) != logits.size(0):
            raise ValueError(
                f"targets must be [N] matching logits, got {tuple(targets.shape)}"
            )
        if iid_mask is not None and (
            iid_mask.ndim != 1 or iid_mask.size(0) != logits.size(0)
        ):
            raise ValueError(
                f"iid_mask must be [N] matching logits dim-0, "
                f"got {tuple(iid_mask.shape)} vs N={logits.size(0)}"
            )
        sample_weight = self._check_sample_weight_per_row(
            sample_weight, logits, dtype=logits.dtype
        )

        # Materialise all-True iid_mask when None so downstream is uniform.
        # Detached always — iid_mask is never differentiable.
        if iid_mask is None:
            live_iid: torch.Tensor = torch.ones(
                logits.size(0), dtype=torch.bool, device=logits.device
            )
        else:
            live_iid = iid_mask.detach().bool()

        # --- DDP gather ------------------------------------------------------
        if self._should_gather():
            logits   = all_gather_with_grad(logits)
            targets  = all_gather_no_grad(targets)
            # Gather iid_mask no-grad alongside targets.  Cast to uint8 for
            # transport (bool support varies by backend) and cast back.
            live_iid = all_gather_no_grad(live_iid.to(torch.uint8)).bool()
            # Gather sample_weight only when supplied on this call -- an
            # unweighted call issues no additional collective.  Float, no
            # uint8 cast (unlike iid_mask, weight is not boolean).
            if sample_weight is not None:
                sample_weight = all_gather_no_grad(sample_weight)

        # --- build is_live before merge (N_live = post-gather live-batch size) -
        # The merge places live rows first, so is_live is True for the first
        # N_live rows and False for all queue rows.  Built no-grad here; never
        # enters autograd.
        n_live = logits.size(0)

        # The weighted path activates when this call supplied a weight, or
        # the queue already holds a weighted row from a previous call. This
        # is the structural guard behind bitwise unweighted identity: when
        # both are false, no weight tensor is ever materialized and merge()
        # is called exactly as it was before sample_weight existed.
        weighted_active = sample_weight is not None or self._queue.has_weights

        # --- merge with queue ------------------------------------------------
        # A materialized all-ones live weight is what lets a weighted queue
        # row weight a later unweighted call; on the unweighted path
        # live_weight stays None and no weight tensor exists at all.
        live_weight: torch.Tensor | None = None
        if weighted_active:
            live_weight = (
                sample_weight
                if sample_weight is not None
                else torch.ones(n_live, dtype=logits.dtype, device=logits.device)
            )
        pool = self._queue.merge(
            logits, targets, is_iid=live_iid, sample_weight=live_weight
        )

        # is_live: True for the N_live live rows (placed first by merge), False
        # for queue rows.  Aligned row-for-row with the pooled tensors.  Rides
        # outside PooledBatch because it is a property of this call, not of
        # the queue's contents.
        with torch.no_grad():
            all_is_live = torch.cat([
                torch.ones(n_live, dtype=torch.bool, device=logits.device),
                torch.zeros(pool.logits.size(0) - n_live, dtype=torch.bool, device=logits.device),
            ])

        # --- filter ignore_index ---------------------------------------------
        valid = pool.targets != self.ignore_index
        pool = pool.index(valid)
        all_is_live = all_is_live[valid]

        # --- subsample if pool exceeds max_pool_size -------------------------
        if self.max_pool_size is not None and pool.logits.size(0) > self.max_pool_size:
            if not self._subsample_warned:
                warnings.warn(
                    f"{type(self).__name__}: pool size {pool.logits.size(0)} exceeds "
                    f"max_pool_size={self.max_pool_size}; applying "
                    f"minimum-quota subsampling. Loss is now a stochastic approximation. "
                    f"(This warning is shown once per instance.)",
                    UserWarning,
                    stacklevel=2,
                )
                self._subsample_warned = True
            pool, all_is_live = subsample_pool(
                pool, self.max_pool_size, is_live=all_is_live
            )

        # --- empty pool check ------------------------------------------------
        if pool.logits.size(0) == 0:
            out = logits.sum() * 0.0
            if self.reduction == "none":
                out = out.expand(self.num_classes)
            if return_per_class:
                return (
                    out,
                    logits.new_full((self.num_classes,), float("nan")),
                    torch.zeros(
                        self.num_classes, dtype=torch.bool, device=logits.device
                    ),
                )
            return out

        # --- subclass validation hook ----------------------------------------
        self._validate_filtered_targets(pool.targets)

        # --- per-class compute -----------------------------------------------
        loss_vec, valid_vec = self._compute_per_class(
            pool.logits, pool.targets, pool.is_iid, all_is_live,
            sample_weight=pool.sample_weight,
        )

        # --- enqueue live-batch (post-gather; runs before next merge) ---------
        if self._should_update_queue():
            self._queue.enqueue(logits, targets, is_iid=live_iid, sample_weight=sample_weight)

        # --- reduction -------------------------------------------------------
        if self.reduction == "none":
            out = loss_vec.masked_fill(~valid_vec, float("nan"))
        else:
            valid_losses = loss_vec[valid_vec]
            if valid_losses.numel() == 0:
                out = logits.sum() * 0.0
            elif self.reduction == "sum":
                out = valid_losses.sum()
            else:
                out = valid_losses.mean()

        if return_per_class:
            per_class = loss_vec.masked_fill(~valid_vec, float("nan"))
            return out, per_class, valid_vec
        return out
