"""
Pool subsampling utilities for ranking losses.

Used internally by SmoothAPLoss and RecallAtQuantileLoss to cap pool size
when flattened seq2seq inputs would otherwise produce OOM-inducing pairwise
matrices.
"""

from __future__ import annotations

import torch

from imbalanced_losses._queue import PooledBatch


def subsample_pool(
    batch: PooledBatch,
    max_size: int,
    is_live: torch.Tensor | None = None,
) -> tuple[PooledBatch, torch.Tensor | None]:
    """
    Minimum-quota subsample of a ranking pool to at most *max_size* rows.

    When ``M <= max_size`` the inputs are returned unchanged (zero copy).
    When ``M > max_size``, each observed class is guaranteed a minimum quota
    of rows before the remaining budget is filled uniformly at random from
    the unselected rows.

    This is **not** proportional (stratified) sampling.  Proportional sampling
    would allocate rows in proportion to class frequency; this algorithm gives
    every observed class an equal quota regardless of frequency.  A dominant
    class (e.g. background) and a rare class receive the same reserved count.
    The consequence is that rare classes are over-represented relative to their
    natural frequency (intentionally so, to ensure they contribute gradient
    signal), while the remainder of the budget is filled uniformly.

    Selection is weight-blind: ``batch.sample_weight`` is transported, never
    consulted, when deciding which rows to keep (spec section 3).

    Parameters
    ----------
    batch : PooledBatch
        Pool rows (live batch + queue, already filtered for ignore_index).
        ``logits`` may carry gradients; the returned rows preserve them.
        Every tensor the batch carries is re-indexed by the same selection,
        so no supplied tensor can be dropped or left misaligned.
    max_size : int
        Maximum number of rows to return.  Must be positive.
    is_live : torch.Tensor, shape [M], dtype=bool, optional
        Per-row live-batch flag (True = live-batch row, False = queue row).
        Travels outside *batch* because it is built by the caller after the
        merge, not by the queue; it is indexed by the same selection.

    Returns
    -------
    batch_sub : PooledBatch
        The selected rows, ``min(M, max_size)`` of them, with every field
        the input batch carried.
    is_live_sub : torch.Tensor or None
        *is_live* under the same selection, or ``None`` when it was not
        supplied.

    Notes
    -----
    The quota formula is:
        per_class_quota = max(1, max_size // (2 * num_observed_classes))
        reserved_per_class = min(class_count, per_class_quota)

    Half the budget is reserved for class quotas; the other half is
    filled with uniform random draws from remaining indices.  If the
    reserved set already fills *max_size* (only at very small max_size),
    it is randomly truncated.

    Important: because the quota is equal per class, the effective positive
    rate in the subsampled pool can be much higher than in the original pool
    when one class dominates (e.g. a background class at 99%+ frequency).
    This inflates ``|P_c|`` and therefore pairwise matrix memory.  Size
    ``max_pool_size`` accordingly: ``|P_c| ≈ max_pool_size // (2 * n_classes)``.
    """
    logits, targets = batch.logits, batch.targets
    m = logits.size(0)
    if m <= max_size:
        return batch, is_live

    device = targets.device
    classes, inverse = targets.unique(return_inverse=True)
    n_classes = classes.size(0)

    per_class_budget = max(1, max_size // (2 * n_classes))

    reserved: list[torch.Tensor] = []
    for ci in range(n_classes):
        mask = inverse == ci
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        keep = min(idx.size(0), per_class_budget)
        perm = torch.randperm(idx.size(0), device=device)[:keep]
        reserved.append(idx[perm])

    reserved_idx = torch.cat(reserved)

    if reserved_idx.size(0) >= max_size:
        # Edge case: reserved set itself exceeds budget — randomly truncate.
        perm = torch.randperm(reserved_idx.size(0), device=device)[:max_size]
        final_idx = reserved_idx[perm]
    else:
        remaining_budget = max_size - reserved_idx.size(0)
        # Build a mask of already-reserved positions and sample from the rest.
        reserved_set = torch.zeros(m, dtype=torch.bool, device=device)
        reserved_set[reserved_idx] = True
        remaining_idx = (~reserved_set).nonzero(as_tuple=False).squeeze(1)
        perm = torch.randperm(remaining_idx.size(0), device=device)[:remaining_budget]
        extra_idx = remaining_idx[perm]
        final_idx = torch.cat([reserved_idx, extra_idx])

    return (
        batch.index(final_idx),
        None if is_live is None else is_live[final_idx],
    )
