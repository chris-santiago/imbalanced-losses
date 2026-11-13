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

from imbalanced_losses._queue import PooledBatch, _MemoryQueue
from imbalanced_losses._sampling import subsample_pool
from imbalanced_losses._weights import _SampleWeightMixin, _caller_stacklevel
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


def _to_compute_dtype(logits: torch.Tensor) -> torch.Tensor:
    """
    Return *logits* in the dtype the ranking losses compute in.

    ``float16`` and ``bfloat16`` become ``float32`` (``torch.quantile`` has
    no half-precision kernel and half-precision rank sums lose accuracy).
    Every other dtype, float64 included, is returned as the same tensor
    object.  The cast is differentiable, so gradient reaches the caller's
    tensor in its original dtype.
    """
    if logits.dtype in (torch.float16, torch.bfloat16):
        return logits.to(torch.float32)
    return logits


# Largest input ``torch.quantile`` accepts; one more element raises
# ``RuntimeError: quantile() input tensor is too large``.
_QUANTILE_MAX_NUMEL = 2**24


def _quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    interpolation: str = "linear",
) -> torch.Tensor:
    """
    ``torch.quantile`` over the flattened *input*, without its size limit.

    Up to ``_QUANTILE_MAX_NUMEL`` elements this is ``torch.quantile`` itself.
    Above it, the same order statistics are read from one sort.  The two
    routes follow the same conventions: *q* is first cast to the input's
    dtype, the rank is ``q * (n - 1)``, ``"lower"``/``"higher"`` take its
    floor/ceil, ``"nearest"`` rounds half to even, ``"linear"`` and
    ``"midpoint"`` interpolate between the neighbouring order statistics
    (weight ``rank - floor(rank)`` and ``0.5``), and any NaN in *input*
    makes every result NaN.

    Callers pass a float or 1-D tensor *q* in ``[0, 1]``, a float32 or
    float64 input and one of the five interpolation names.  Above the limit
    the sort route checks only the range of *q*: it does not reject a NaN
    *q*, an unknown interpolation name (treated as ``"lower"``), a *q* tensor
    of another dtype or with more than one dimension, or a float16/bfloat16
    input.  It also forms the rank in float64 (on the CPU), because a float32
    rank cannot represent integers above 2**24, whereas ``torch.quantile``
    uses the input dtype.  For a float32 input and a *q* whose float32
    product with ``n - 1`` is not exact, the two can select the neighbouring
    order statistic.

    Parameters
    ----------
    input : torch.Tensor
        Scores of any shape; flattened.
    q : float or torch.Tensor
        Level in ``[0, 1]``: a Python float (scalar result) or a 0-d / 1-D
        tensor of levels (result of the same shape).
    interpolation : ``'linear'`` | ``'lower'`` | ``'higher'`` | ``'nearest'`` | ``'midpoint'``
        How to resolve a rank between two order statistics.

    Returns
    -------
    torch.Tensor
        In the input's dtype and on its device.
    """
    if input.numel() <= _QUANTILE_MAX_NUMEL:
        return torch.quantile(input, q, interpolation=interpolation)

    # The few levels are handled on the CPU in float64 (MPS has no float64);
    # only the integer indices and the weights move to the input's device.
    q_t = torch.as_tensor(q, dtype=input.dtype).cpu()
    if bool(((q_t < 0) | (q_t > 1)).any()):
        raise RuntimeError("quantile() q values must be in the range [0, 1]")
    ordered = input.reshape(-1).sort().values
    last = ordered.numel() - 1
    ranks = q_t.to(torch.float64) * last
    if bool(torch.isnan(ordered[-1])):  # NaN sorts last
        ranks = torch.full_like(ranks, last)
    if interpolation == "lower":
        ranks = ranks.floor()
    elif interpolation == "higher":
        ranks = ranks.ceil()
    elif interpolation == "nearest":
        ranks = ranks.round()
    below = ranks.to(torch.long)
    values = ordered[below.to(input.device)]
    if interpolation in ("linear", "midpoint"):
        weights = (
            torch.full_like(ranks, 0.5) if interpolation == "midpoint" else ranks - below
        )
        above = ranks.ceil().to(torch.long)
        values = torch.lerp(
            values,
            ordered[above.to(input.device)],
            weights.to(device=input.device, dtype=input.dtype),
        )
    return values


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class _QueuedRankingLoss(_SampleWeightMixin, nn.Module, abc.ABC):
    """
    Abstract base for ranking losses that use a memory queue.

    Subclasses implement :meth:`_compute_per_class`; everything else --
    validation, DDP gather, queue merge, ignore-index filtering,
    subsampling, reduction, and ``return_per_class`` -- is handled here.
    The target-range rule lives in :meth:`_validate_filtered_targets`,
    which subclasses may override; an override that adds checks should
    call ``super()._validate_filtered_targets(...)`` to keep the range
    rule.

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
        Cap on ranking-pool rows after ignore-index filtering.  Losses
        with pool-level thresholds (see :meth:`_estimate_thresholds`)
        estimate them from the full pool before the cap is applied.
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
        thresholds: object | None = None,
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
        thresholds : object, optional
            Whatever :meth:`_estimate_thresholds` returned for this call.
            Passed as a keyword only when subsampling ran and the hook
            returned a non-``None`` value, so an implementation that never
            overrides the hook need not accept it.  When absent, compute
            the thresholds from *logits* itself.

        Returns
        -------
        loss_vec : torch.Tensor, shape [C]
            Per-class loss values.  Degenerate classes should use 0.0
            (they will be masked to nan downstream when reduction='none').
        valid_vec : torch.Tensor, shape [C], dtype=bool
            True for classes that had a meaningful computation (e.g. at
            least one positive and one negative).
        """

    def _estimate_thresholds(self, pool: PooledBatch) -> object | None:
        """
        Estimate score thresholds from the full pool before subsampling.

        Called at most once per forward, and only when ``max_pool_size``
        subsampling is about to run, with the ``ignore_index``-filtered,
        pre-subsampling pool.  A loss whose thresholds depend on the pool
        overrides this so that capping the pool does not move its operating
        point; the result is handed to :meth:`_compute_per_class` as the
        ``thresholds`` keyword.  Implementations must not modify *pool*, and
        the result must carry no gradient.

        Returns
        -------
        object or None
            ``None`` (the default) means the loss has no pool-level
            thresholds, and ``_compute_per_class`` is called without the
            ``thresholds`` keyword.
        """
        return None

    def _validate_filtered_targets(
        self, targets: torch.Tensor, *, is_live: torch.Tensor | None = None
    ) -> None:
        """
        Validate target range on the pooled, ``ignore_index``-filtered rows.

        Called once per forward, after ignore-index filtering and before
        ``max_pool_size`` subsampling, so the check sees every live and
        queued row.  With ``num_classes > 1``, raises ``ValueError`` if
        any target lies outside ``[0, num_classes)``; such a row would
        count as a negative for every class.  Queued rows are checked
        along with the live batch, so a queue restored from a checkpoint
        saved before this check existed raises until :meth:`reset_queue`
        drops its rows.

        With ``num_classes == 1``, a target outside ``{0, 1}`` emits one
        ``UserWarning`` per call (listing at most eight of the offending
        values) and the row still counts as positive.  The warning is
        attributed to the first frame outside this library and torch, so it
        points at the caller's own line.  An empty pool does not warn.

        Overrides that add checks should call
        ``super()._validate_filtered_targets(targets, is_live=is_live)``
        to keep this rule, and must accept (and may ignore) ``is_live``.

        Parameters
        ----------
        targets : torch.Tensor, shape [M]
            Filtered targets (no ignore-index values); may be empty.
        is_live : torch.Tensor, shape [M], dtype=bool, optional
            True for rows from the current batch, False for queued rows.
            Selects which remedy the error message names; when omitted,
            both remedies are named.
        """
        if self.num_classes == 1:
            bad = targets[(targets != 0) & (targets != 1)]
            if bad.numel() > 0:
                warnings.warn(
                    f"Binary mode (num_classes=1) expects targets in {{0, 1}}, "
                    f"but found values: {bad[:8].tolist()}. "
                    "Non-zero values are treated as positive.",
                    UserWarning,
                    stacklevel=_caller_stacklevel(),
                )
        elif self.num_classes > 1:
            is_bad = (targets < 0) | (targets >= self.num_classes)
            bad = targets[is_bad]
            if bad.numel() > 0:
                if is_live is None:
                    in_live = in_queue = True
                else:
                    in_live = bool(is_live[is_bad].any())
                    in_queue = bool((~is_live[is_bad]).any())
                remedies = []
                if in_live:
                    remedies.append(
                        "mark rows in the current batch with ignore_index "
                        "or fix their labels"
                    )
                if in_queue:
                    remedies.append(
                        "call reset_queue() to drop queued rows restored "
                        "from an older checkpoint"
                    )
                raise ValueError(
                    f"targets contain class ids outside [0, {self.num_classes}); "
                    f"examples: {bad[:8].tolist()}. "
                    + "; ".join(remedies).capitalize() + "."
                )

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

        ``float16`` and ``bfloat16`` logits are computed in ``float32`` and
        the returned loss is ``float32``; gradient reaches the original
        tensor in its own dtype.  Other dtypes (``float32``, ``float64``)
        are computed as given.  A supplied ``sample_weight`` follows the
        compute dtype.

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
            (reduction='none').  When the pool is empty (every row is
            ``ignore_index``, or a zero-row batch with an empty queue),
            'none' returns ``nan`` for every class, matching the
            all-degenerate case; 'mean' and 'sum' return ``0.0`` with a
            gradient path.
        per_class_loss : torch.Tensor, shape [C]
            Only returned when ``return_per_class=True``.
        valid_classes : torch.Tensor, shape [C], dtype=bool
            Only returned when ``return_per_class=True``.
        """
        logits = _to_compute_dtype(logits)

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

        # --- validate the whole pool, before subsampling can hide a row -------
        self._validate_filtered_targets(pool.targets, is_live=all_is_live)

        # --- subsample if pool exceeds max_pool_size -------------------------
        thresholds = None
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
            # Thresholds come from the whole filtered pool, so the cap does
            # not move the operating point; the per-class compute then runs
            # over the subsample against them.
            thresholds = self._estimate_thresholds(pool)
            pool, all_is_live = subsample_pool(
                pool, self.max_pool_size, is_live=all_is_live
            )

        # --- empty pool check ------------------------------------------------
        # _compute_per_class is never called on an empty pool; the result is
        # the same one a pool of all-degenerate classes reduces to.
        if pool.logits.size(0) == 0:
            return self._reduce(
                logits.new_zeros((self.num_classes,)),
                torch.zeros(self.num_classes, dtype=torch.bool, device=logits.device),
                logits,
                return_per_class,
            )

        # --- per-class compute -----------------------------------------------
        # The keyword is passed only when the hook produced something, so a
        # _compute_per_class without a ``thresholds`` parameter keeps working.
        extra = {} if thresholds is None else {"thresholds": thresholds}
        loss_vec, valid_vec = self._compute_per_class(
            pool.logits, pool.targets, pool.is_iid, all_is_live,
            sample_weight=pool.sample_weight, **extra,
        )

        # --- enqueue live-batch (post-gather; runs before next merge) ---------
        # Only rows the loss can use are stored: ignore_index rows would
        # occupy queue slots that merge() filters out again. The mask depends
        # only on the gathered targets, so every DDP rank enqueues the same
        # rows. A batch with nothing to store is a no-op in enqueue().
        if self._should_update_queue():
            keep = targets != self.ignore_index
            self._queue.enqueue(
                logits.detach()[keep],
                targets[keep],
                is_iid=live_iid[keep],
                sample_weight=None if sample_weight is None else sample_weight[keep],
            )

        return self._reduce(loss_vec, valid_vec, logits, return_per_class)

    def _reduce(
        self,
        loss_vec: torch.Tensor,
        valid_vec: torch.Tensor,
        logits: torch.Tensor,
        return_per_class: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reduce per-class losses to the value ``forward`` returns.

        Shared by the general path and the empty-pool early return.
        ``reduction='none'`` masks invalid classes to ``nan`` (so a result
        with no valid class is a ``nan`` vector with no gradient path unless
        *loss_vec* has one).  ``'mean'`` and ``'sum'`` reduce over valid
        classes only; with none valid they return ``logits.sum() * 0.0``,
        a zero that keeps a gradient path to the caller's logits.  The
        ``per_class`` element is always a fresh tensor, never the loss
        itself.

        Parameters
        ----------
        loss_vec : torch.Tensor, shape [C]
            Per-class losses; entries of invalid classes are ignored.
        valid_vec : torch.Tensor, shape [C], dtype=bool
            True for classes with a meaningful loss.
        logits : torch.Tensor
            Compute-dtype logits of this call; anchors the zero returned
            when no class is valid.
        return_per_class : bool
            Also return the masked per-class vector and *valid_vec*.
        """
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
