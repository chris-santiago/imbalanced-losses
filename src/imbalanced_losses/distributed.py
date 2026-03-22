"""
Distributed utilities for imbalanced losses.

All-gather helpers that preserve gradient flow through the local rank's slice,
enabling globally-aware rank estimation in DDP training without breaking
autograd.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


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

    Returns
    -------
    torch.Tensor
        Concatenation of all workers' tensors along dim 0, shape
        ``[world_size * N, C]``. Gradient flows only through rows
        ``[rank*N : (rank+1)*N]``.

    Notes
    -----
    All workers must contribute tensors with the same shape. The pre-allocated
    receive buffers use ``torch.zeros_like(tensor)``, which assumes every
    rank's local tensor has identical dimensions. Unequal batch sizes across
    ranks will cause a shape mismatch error. When using
    ``DistributedSampler``, set ``drop_last=True`` to guarantee equal batch
    sizes across ranks.

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

        logits_global  = all_gather_with_grad(logits)          # [world*N, C]
        targets_global = all_gather_no_grad(targets)           # [world*N]
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
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    # Restore gradient connection for the local slice.
    # Other slices are already detached (dist.all_gather does not propagate grad).
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

    Returns
    -------
    torch.Tensor
        Concatenation of all workers' tensors along dim 0, shape
        ``[world_size * N]``.

    Notes
    -----
    All workers must contribute tensors with the same shape. The pre-allocated
    receive buffers use ``torch.zeros_like(tensor)``, which assumes every
    rank's local tensor has identical dimensions. When using
    ``DistributedSampler``, set ``drop_last=True`` to guarantee equal batch
    sizes across ranks.

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

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)
