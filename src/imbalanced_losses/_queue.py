"""
Circular memory queue for ranking losses.

Used internally by SmoothAPLoss and RecallAtQuantileLoss to accumulate
(logits, targets) pairs across batches for stable gradient estimates at
low positive rates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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

        if queue_size > 0:
            # Unfilled slots carry ignore_index targets and are stripped naturally.
            self.register_buffer("_q_logits",  torch.zeros(queue_size, num_classes))
            self.register_buffer("_q_targets", torch.full((queue_size,), ignore_index, dtype=torch.long))
            self.register_buffer("_q_ptr",     torch.zeros(1, dtype=torch.long))
            # Unfilled slots default to True (treated as iid) so legacy
            # checkpoints/rows with no recorded flag are treated as iid.
            self.register_buffer("_q_iid",     torch.ones(queue_size, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # When loading a checkpoint saved before _q_iid was introduced, inject
        # an all-True default so the missing key never triggers a strict-mode
        # error and the buffer is initialized to the correct semantic default.
        if self.queue_size > 0:
            iid_key = prefix + "_q_iid"
            if iid_key not in state_dict:
                state_dict[iid_key] = torch.ones(self.queue_size, dtype=torch.bool)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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

        if n >= self.queue_size:
            self._q_logits.copy_(logits.detach()[-self.queue_size:])
            self._q_targets.copy_(targets.detach()[-self.queue_size:])
            self._q_iid.copy_(iid[-self.queue_size:])
            self._q_ptr.zero_()
            return

        ptr = int(self._q_ptr)
        end = ptr + n

        if end <= self.queue_size:
            self._q_logits[ptr:end]  = logits.detach()
            self._q_targets[ptr:end] = targets.detach()
            self._q_iid[ptr:end]     = iid
        else:
            first  = self.queue_size - ptr
            second = n - first
            self._q_logits[ptr:]     = logits.detach()[:first]
            self._q_targets[ptr:]    = targets.detach()[:first]
            self._q_iid[ptr:]        = iid[:first]
            self._q_logits[:second]  = logits.detach()[first:]
            self._q_targets[:second] = targets.detach()[first:]
            self._q_iid[:second]     = iid[first:]

        self._q_ptr.fill_((ptr + n) % self.queue_size)

    def merge(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_iid: torch.Tensor | None = None,
        return_iid: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Concatenate the live batch with the current queue contents.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
        targets : torch.Tensor, shape [N]
        is_iid : torch.Tensor, shape [N], dtype=bool, optional
            Per-row iid flag for the live batch.  When ``None``, the live
            batch is treated as all-iid.  Only consulted when
            ``return_iid=True``.
        return_iid : bool, optional
            When ``False`` (default), return the 2-tuple
            ``(all_logits, all_targets)`` exactly as before this extension
            (backward compatible).  When ``True``, return the 3-tuple
            ``(all_logits, all_targets, all_is_iid)`` where ``all_is_iid``
            has the live-batch flags (or all-True if ``is_iid=None``)
            prepended to the stored ``_q_iid``, aligned row-for-row with
            the returned logits/targets.

        Returns
        -------
        all_logits : torch.Tensor, shape [N + Q, C]
            Live logits followed by queue logits (cast to matching
            device/dtype). Q = queue_size; unfilled slots have
            ignore_index targets and are filtered downstream.
        all_targets : torch.Tensor, shape [N + Q]
        all_is_iid : torch.Tensor, shape [N + Q], dtype=bool
            Only present when ``return_iid=True``.

        Notes
        -----
        When ``queue_size == 0`` the inputs are returned unchanged (no copy).
        For ``return_iid=True`` with ``queue_size == 0``, the synthesized
        live iid tensor (or the supplied ``is_iid``) is returned as the
        third element.
        """
        if self.queue_size == 0:
            if not return_iid:
                return logits, targets
            live_iid = (
                torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
                if is_iid is None
                else is_iid
            )
            return logits, targets, live_iid

        q_logits  = self._q_logits.to(device=logits.device, dtype=logits.dtype)
        q_targets = self._q_targets.to(device=targets.device)
        all_logits  = torch.cat([logits, q_logits], dim=0)
        all_targets = torch.cat([targets, q_targets], dim=0)

        if not return_iid:
            return all_logits, all_targets

        live_iid = (
            torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
            if is_iid is None
            else is_iid
        )
        q_iid = self._q_iid.to(device=logits.device)
        all_is_iid = torch.cat([live_iid, q_iid], dim=0)
        return all_logits, all_targets, all_is_iid

    @torch.no_grad()
    def reset(self) -> None:
        """
        Clear the circular buffer.

        Resets all stored logits to zero, all stored targets to
        ignore_index, all stored iid flags to True, and the write pointer
        to 0.  Typically called between training and evaluation epochs.
        """
        if self.queue_size > 0:
            self._q_logits.zero_()
            self._q_targets.fill_(self.ignore_index)
            self._q_iid.fill_(True)
            self._q_ptr.zero_()
