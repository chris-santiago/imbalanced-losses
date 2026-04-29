"""
Distributed utilities for imbalanced losses.

All-gather helpers that preserve gradient flow through the local rank's slice,
enabling globally-aware rank estimation in DDP training without breaking
autograd.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _gather_sizes(
    local_size: int, world_size: int, device: torch.device
) -> torch.Tensor:
    """All-gather the dim-0 size from every rank. Returns ``[world_size]`` int64 tensor."""
    local = torch.tensor([local_size], dtype=torch.int64, device=device)
    sizes_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(sizes_list, local)
    return torch.cat(sizes_list)


def _pad_to(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad *tensor* with zeros along dim 0 to *target_rows*."""
    pad_rows = target_rows - tensor.size(0)
    if pad_rows == 0:
        return tensor
    padding = torch.zeros(
        pad_rows, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=0)


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather a tensor across all workers, preserving gradients for the
    local rank's slice.

    Standard ``dist.all_gather`` returns detached tensors. This function
    replaces the local rank's slice in the output with the original tensor,
    so gradients flow back to the local model parameters. Other workers'
    slices remain stop-gradient, matching DDP semantics (each worker
    optimizes its own parameters via all-reduced gradients).

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor to gather. Typically ``[N, C]`` logits from one GPU.
        ``N`` (dim 0) may differ across ranks; all other dimensions must
        match.

    Returns
    -------
    torch.Tensor
        Concatenation of all workers' tensors along dim 0, shape
        ``[sum(N_i), C]``. Gradient flows only through the rows
        contributed by the local rank.

    Notes
    -----
    Dim 0 may vary across ranks (e.g. unequal last-batch sizes). When
    sizes differ, tensors are zero-padded to the max for the collective,
    then trimmed back to their true lengths before concatenation. An
    equal-size fast path skips padding when all ranks contribute the same
    number of rows.

    All workers' queues stay synchronized automatically: since every worker
    calls ``all_gather`` before passing to the loss, every worker enqueues
    the same global-batch data. No extra synchronization is needed.

    Raises
    ------
    RuntimeError
        If ``torch.distributed`` is not available or not initialized.

    Examples
    --------
    Typical usage in a DDP training step::

        from imbalanced_losses.distributed import all_gather_with_grad

        logits_global  = all_gather_with_grad(logits)          # [sum(N_i), C]
        targets_global = all_gather_no_grad(targets)           # [sum(N_i)]
        loss = loss_fn(logits_global, targets_global)
        loss.backward()
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call dist.init_process_group before using all_gather_with_grad."
        )

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    rank = dist.get_rank()
    local_size = tensor.size(0)
    sizes = _gather_sizes(local_size, world_size, tensor.device)

    if sizes.eq(local_size).all():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        gathered[rank] = tensor
        return torch.cat(gathered, dim=0)

    max_size = sizes.max().item()
    padded = _pad_to(tensor, max_size)
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    gathered = [gathered[i][: sizes[i]] for i in range(world_size)]
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def all_gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather a tensor across all workers without gradient tracking.

    Intended for targets / labels, which are integer tensors with no gradient.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor to gather. Typically ``[N]`` integer targets.
        ``N`` (dim 0) may differ across ranks; all other dimensions must
        match.

    Returns
    -------
    torch.Tensor
        Concatenation of all workers' tensors along dim 0, shape
        ``[sum(N_i)]``.

    Notes
    -----
    Dim 0 may vary across ranks (e.g. unequal last-batch sizes). When
    sizes differ, tensors are zero-padded to the max for the collective,
    then trimmed back to their true lengths before concatenation. An
    equal-size fast path skips padding when all ranks contribute the same
    number of rows.

    Raises
    ------
    RuntimeError
        If ``torch.distributed`` is not available or not initialized.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call dist.init_process_group before using all_gather_no_grad."
        )

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    local_size = tensor.size(0)
    sizes = _gather_sizes(local_size, world_size, tensor.device)

    if sizes.eq(local_size).all():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    max_size = sizes.max().item()
    padded = _pad_to(tensor, max_size)
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    gathered = [gathered[i][: sizes[i]] for i in range(world_size)]
    return torch.cat(gathered, dim=0)
