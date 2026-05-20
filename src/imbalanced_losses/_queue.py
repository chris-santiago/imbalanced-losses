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
    def enqueue(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Write a detached batch into the circular buffer.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Live-batch logits to store (detached internally).
        targets : torch.Tensor, shape [N]
            Corresponding integer targets.

        Notes
        -----
        If N >= queue_size the buffer is replaced wholesale with the last
        queue_size rows of the batch and the pointer is reset to 0.
        Wrap-around writes are handled with explicit head/tail slicing.
        """
        if self.queue_size == 0:
            return

        n = logits.size(0)

        if n >= self.queue_size:
            self._q_logits.copy_(logits.detach()[-self.queue_size:])
            self._q_targets.copy_(targets.detach()[-self.queue_size:])
            self._q_ptr.zero_()
            return

        ptr = int(self._q_ptr)
        end = ptr + n

        if end <= self.queue_size:
            self._q_logits[ptr:end]  = logits.detach()
            self._q_targets[ptr:end] = targets.detach()
        else:
            first  = self.queue_size - ptr
            second = n - first
            self._q_logits[ptr:]     = logits.detach()[:first]
            self._q_targets[ptr:]    = targets.detach()[:first]
            self._q_logits[:second]  = logits.detach()[first:]
            self._q_targets[:second] = targets.detach()[first:]

        self._q_ptr.fill_((ptr + n) % self.queue_size)

    def merge(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate the live batch with the current queue contents.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
        targets : torch.Tensor, shape [N]

        Returns
        -------
        all_logits : torch.Tensor, shape [N + Q, C]
            Live logits followed by queue logits (cast to matching
            device/dtype). Q = queue_size; unfilled slots have
            ignore_index targets and are filtered downstream.
        all_targets : torch.Tensor, shape [N + Q]

        Notes
        -----
        When ``queue_size == 0`` the inputs are returned unchanged (no copy).
        """
        if self.queue_size == 0:
            return logits, targets
        q_logits  = self._q_logits.to(device=logits.device, dtype=logits.dtype)
        q_targets = self._q_targets.to(device=targets.device)
        return torch.cat([logits, q_logits], dim=0), torch.cat([targets, q_targets], dim=0)

    @torch.no_grad()
    def reset(self) -> None:
        """
        Clear the circular buffer.

        Resets all stored logits to zero, all stored targets to
        ignore_index, and the write pointer to 0. Typically called
        between training and evaluation epochs.
        """
        if self.queue_size > 0:
            self._q_logits.zero_()
            self._q_targets.fill_(self.ignore_index)
            self._q_ptr.zero_()
