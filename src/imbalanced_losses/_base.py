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

class _QueuedRankingLoss(nn.Module, abc.ABC):
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
        for suffix in ("_q_logits", "_q_targets", "_q_ptr"):
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
        return_per_class: bool = False,
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
        return_per_class : bool, optional
            If True, also return per-class losses and a validity mask.

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

        # --- DDP gather ------------------------------------------------------
        if self._should_gather():
            logits  = all_gather_with_grad(logits)
            targets = all_gather_no_grad(targets)

        # --- merge with queue ------------------------------------------------
        all_logits, all_targets = self._queue.merge(logits, targets)

        # --- filter ignore_index ---------------------------------------------
        valid = all_targets != self.ignore_index
        all_logits, all_targets = all_logits[valid], all_targets[valid]

        # --- subsample if pool exceeds max_pool_size -------------------------
        if self.max_pool_size is not None and all_logits.size(0) > self.max_pool_size:
            if not self._subsample_warned:
                warnings.warn(
                    f"{type(self).__name__}: pool size {all_logits.size(0)} exceeds "
                    f"max_pool_size={self.max_pool_size}; applying "
                    f"minimum-quota subsampling. Loss is now a stochastic approximation. "
                    f"(This warning is shown once per instance.)",
                    UserWarning,
                    stacklevel=2,
                )
                self._subsample_warned = True
            all_logits, all_targets = subsample_pool(
                all_logits, all_targets, self.max_pool_size
            )

        # --- empty pool check ------------------------------------------------
        if all_logits.size(0) == 0:
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
        self._validate_filtered_targets(all_targets)

        # --- per-class compute -----------------------------------------------
        loss_vec, valid_vec = self._compute_per_class(all_logits, all_targets)

        # --- enqueue live-batch (post-gather; runs before next merge) ---------
        if self._should_update_queue():
            self._queue.enqueue(logits, targets)

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
