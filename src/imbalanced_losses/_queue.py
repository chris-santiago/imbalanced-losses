"""
Circular memory queue for ranking losses.

Used internally by SmoothAPLoss and RecallAtQuantileLoss to accumulate
(logits, targets) pairs across batches for stable gradient estimates at
low positive rates.

Also defines :class:`PooledBatch`, the fixed shape every pooled
(live batch + queue) rail passes around: ``_MemoryQueue.merge`` produces
it, ``_sampling.subsample_pool`` re-indexes it, and
``_QueuedRankingLoss.forward`` consumes it.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class PooledBatch(NamedTuple):
    """One pooled ranking batch: live rows first, queue rows after.

    Every field is aligned row-for-row.  The optional fields are ``None``
    when that tensor is not in play for this call -- notably
    ``sample_weight`` is ``None`` exactly when the unweighted code path is
    active -- and every rail that re-indexes a batch must carry each
    supplied tensor through with the same index (see :meth:`index`), so a
    tensor can never be silently dropped by a rail that did not expect it.

    Attributes
    ----------
    logits : torch.Tensor, shape [M, C]
    targets : torch.Tensor, shape [M]
    is_iid : torch.Tensor, shape [M], dtype=bool, optional
        Per-row iid-eligibility flag.
    sample_weight : torch.Tensor, shape [M], optional
        Per-row weight.  ``None`` iff the unweighted path is active.
    """

    logits: torch.Tensor
    targets: torch.Tensor
    is_iid: torch.Tensor | None = None
    sample_weight: torch.Tensor | None = None

    def index(self, idx: torch.Tensor) -> PooledBatch:
        """Select rows by *idx* from every tensor this batch carries.

        Parameters
        ----------
        idx : torch.Tensor
            Boolean mask or index tensor applied to dim 0 of every field.
        """
        return PooledBatch(
            logits=self.logits[idx],
            targets=self.targets[idx],
            is_iid=None if self.is_iid is None else self.is_iid[idx],
            sample_weight=(
                None if self.sample_weight is None else self.sample_weight[idx]
            ),
        )


class _MemoryQueue(nn.Module):
    """
    Circular buffer that stores (logits, targets) pairs across batches.

    Registers all state as named buffers so the queue participates in
    ``state_dict()`` serialisation and ``.to(device)`` / ``.to(dtype)``
    device transfers automatically.

    Parameters
    ----------
    queue_size : int
        Number of rows in the buffer.  Use 0 to create a no-op queue
        (``enqueue`` and ``reset`` are no-ops; ``merge`` returns inputs
        unchanged).
    num_classes : int
        Width of the logits buffer (second dimension).
    ignore_index : int, optional
        Sentinel written into unfilled target slots.  Downstream losses
        must filter these out.  Default: -100.
    """

    def __init__(
        self,
        queue_size: int,
        num_classes: int,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Whether the buffer currently holds a row whose stored weight is
        # not 1.  One definition, evaluated from the buffer itself at every
        # point the buffer can change (enqueue, reset, checkpoint load), so
        # it is a property of the stored rows rather than of the call
        # history: overwriting every weighted row with unweighted rows
        # returns the queue -- and the loss -- to the unweighted path, and
        # a state_dict round trip cannot move an instance between the
        # weighted and unweighted op sequences.  Not itself persisted.
        self.has_weights = False

        if queue_size > 0:
            # Unfilled slots carry ignore_index targets and are stripped naturally.
            self.register_buffer("_q_logits",  torch.zeros(queue_size, num_classes))
            self.register_buffer("_q_targets", torch.full((queue_size,), ignore_index, dtype=torch.long))
            self.register_buffer("_q_ptr",     torch.zeros(1, dtype=torch.long))
            # Unfilled slots default to True (treated as iid) so legacy
            # checkpoints/rows with no recorded flag are treated as iid.
            self.register_buffer("_q_iid",     torch.ones(queue_size, dtype=torch.bool))
            # Unfilled slots default to weight 1.0 so legacy checkpoints/rows
            # with no recorded weight behave as unweighted.
            self.register_buffer("_q_weight",  torch.ones(queue_size))

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # When loading a checkpoint saved before _q_iid/_q_weight were
        # introduced, inject their all-True / all-one defaults so the
        # missing keys never trigger a strict-mode error and the buffers
        # are initialized to the correct semantic default.
        if self.queue_size > 0:
            iid_key = prefix + "_q_iid"
            if iid_key not in state_dict:
                state_dict[iid_key] = torch.ones(self.queue_size, dtype=torch.bool)
            weight_key = prefix + "_q_weight"
            if weight_key not in state_dict:
                state_dict[weight_key] = torch.ones(self.queue_size)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._refresh_has_weights()

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _refresh_has_weights(self) -> None:
        """Re-evaluate :attr:`has_weights` from the stored weights."""
        self.has_weights = self.queue_size > 0 and bool((self._q_weight != 1).any())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Capacity of the circular buffer (equals ``queue_size``)."""
        return self.queue_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def enqueue(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_iid: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Write a detached batch into the circular buffer.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Live-batch logits to store (detached internally).
        targets : torch.Tensor, shape [N]
            Corresponding integer targets.
        is_iid : torch.Tensor, shape [N], dtype=bool, optional
            Per-row flag indicating whether the row is an iid sample
            eligible for FPR threshold estimation.  When ``None``, the
            whole batch is treated as iid (all True).  Stored detached,
            consistent with logits/targets handling.
        sample_weight : torch.Tensor, shape [N], optional
            Per-row weight to store alongside the batch.  When ``None``,
            the whole batch is stored at weight 1.0 via a scalar fill (no
            same-shape ones tensor is allocated).  Stored detached.
            :attr:`has_weights` is re-evaluated from the buffer afterwards,
            so an explicit all-ones weight leaves it ``False`` and a batch
            that overwrites the last non-unit row clears it.

        Notes
        -----
        If N >= queue_size the buffer is replaced wholesale with the last
        queue_size rows of the batch and the pointer is reset to 0.
        Wrap-around writes are handled with explicit head/tail slicing.
        """
        if self.queue_size == 0:
            return

        n = logits.size(0)

        # Materialise is_iid once; default to all-True when not provided.
        if is_iid is None:
            iid = logits.new_ones(n, dtype=torch.bool)
        else:
            iid = is_iid.detach()

        # Materialise sample_weight only when explicitly supplied; the
        # unweighted default is written with scalar fills below instead of
        # allocating a same-shape ones tensor purely to copy it in.
        if sample_weight is None:
            weight = None
        else:
            weight = sample_weight.detach().to(dtype=logits.dtype)

        if n >= self.queue_size:
            self._q_logits.copy_(logits.detach()[-self.queue_size:])
            self._q_targets.copy_(targets.detach()[-self.queue_size:])
            self._q_iid.copy_(iid[-self.queue_size:])
            if weight is None:
                self._q_weight.fill_(1.0)
            else:
                self._q_weight.copy_(weight[-self.queue_size:])
            self._q_ptr.zero_()
            self._refresh_has_weights()
            return

        ptr = int(self._q_ptr)
        end = ptr + n

        if end <= self.queue_size:
            self._q_logits[ptr:end]  = logits.detach()
            self._q_targets[ptr:end] = targets.detach()
            self._q_iid[ptr:end]     = iid
            if weight is None:
                self._q_weight[ptr:end].fill_(1.0)
            else:
                self._q_weight[ptr:end] = weight
        else:
            first  = self.queue_size - ptr
            second = n - first
            self._q_logits[ptr:]     = logits.detach()[:first]
            self._q_targets[ptr:]    = targets.detach()[:first]
            self._q_iid[ptr:]        = iid[:first]
            if weight is None:
                self._q_weight[ptr:].fill_(1.0)
                self._q_weight[:second].fill_(1.0)
            else:
                self._q_weight[ptr:]     = weight[:first]
                self._q_weight[:second]  = weight[first:]
            self._q_logits[:second]  = logits.detach()[first:]
            self._q_targets[:second] = targets.detach()[first:]
            self._q_iid[:second]     = iid[first:]

        self._q_ptr.fill_((ptr + n) % self.queue_size)
        self._refresh_has_weights()

    def merge(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_iid: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> PooledBatch:
        """
        Concatenate the live batch with the current queue contents.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
        targets : torch.Tensor, shape [N]
        is_iid : torch.Tensor, shape [N], dtype=bool, optional
            Per-row iid flag for the live batch.  When ``None``, the live
            batch is treated as all-iid and the flag is synthesized.
        sample_weight : torch.Tensor, shape [N], optional
            Per-row weight for the live batch.  When ``None`` (default),
            the returned batch carries ``sample_weight=None`` -- the
            structural unweighted path, in which no weight tensor is
            materialized at all.  When provided, the returned weight holds
            the live weights (detached, cast to ``logits.dtype``) prepended
            to the stored ``_q_weight``.

        Returns
        -------
        PooledBatch
            ``logits``/``targets`` are the live rows followed by the queue
            rows (``[N + Q, C]`` / ``[N + Q]``; Q = queue_size, unfilled
            slots carry ignore_index targets and are filtered downstream).
            ``is_iid`` is always populated and aligned row-for-row.
            ``sample_weight`` is populated exactly when one was supplied.

        Notes
        -----
        When ``queue_size == 0`` the live ``logits``/``targets`` are
        returned unchanged (no copy).
        """
        live_iid = (
            torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
            if is_iid is None
            else is_iid
        )
        live_weight = (
            None
            if sample_weight is None
            else sample_weight.detach().to(dtype=logits.dtype)
        )

        if self.queue_size == 0:
            return PooledBatch(logits, targets, live_iid, live_weight)

        q_logits  = self._q_logits.to(device=logits.device, dtype=logits.dtype)
        q_targets = self._q_targets.to(device=targets.device)
        q_iid     = self._q_iid.to(device=logits.device)
        all_weight = None
        if live_weight is not None:
            q_weight = self._q_weight.to(device=logits.device, dtype=logits.dtype)
            all_weight = torch.cat([live_weight, q_weight], dim=0)

        return PooledBatch(
            logits=torch.cat([logits, q_logits], dim=0),
            targets=torch.cat([targets, q_targets], dim=0),
            is_iid=torch.cat([live_iid, q_iid], dim=0),
            sample_weight=all_weight,
        )

    @torch.no_grad()
    def reset(self) -> None:
        """
        Clear the circular buffer.

        Resets all stored logits to zero, all stored targets to
        ignore_index, all stored iid flags to True, all stored weights to
        1.0 (and :attr:`has_weights` to ``False``), and the write pointer
        to 0.  Typically called between training and evaluation epochs.
        """
        if self.queue_size > 0:
            self._q_logits.zero_()
            self._q_targets.fill_(self.ignore_index)
            self._q_iid.fill_(True)
            self._q_weight.fill_(1.0)
            self._q_ptr.zero_()
        self.has_weights = False
