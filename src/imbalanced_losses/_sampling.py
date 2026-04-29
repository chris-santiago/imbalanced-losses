"""
Pool subsampling utilities for ranking losses.

Used internally by SmoothAPLoss and RecallAtQuantileLoss to cap pool size
when flattened seq2seq inputs would otherwise produce OOM-inducing pairwise
matrices.
"""

from __future__ import annotations

import torch


def subsample_pool(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified random subsample of a ranking pool to at most *max_size* rows.

    When ``M <= max_size`` the inputs are returned unchanged (zero copy).
    When ``M > max_size``, stratified sampling guarantees at least one row
    per observed target class before filling the remaining budget uniformly
    at random from the unselected rows.

    Parameters
    ----------
    logits : torch.Tensor, shape [M, C]
        Pool logits (live batch + queue, already filtered for ignore_index).
        May contain gradients — the returned view preserves them.
    targets : torch.Tensor, shape [M]
        Integer class labels corresponding to *logits*.
    max_size : int
        Maximum number of rows to return.  Must be positive.

    Returns
    -------
    logits_sub : torch.Tensor, shape [min(M, max_size), C]
    targets_sub : torch.Tensor, shape [min(M, max_size)]

    Notes
    -----
    The stratified guarantee is:
        per_class_budget = max(1, max_size // (2 * num_observed_classes))
        reserved_per_class = min(class_count, per_class_budget)

    Half the budget is reserved for class guarantees; the other half is
    filled with uniform random draws from remaining indices.  If the
    reserved set already fills *max_size* (only at very small max_size),
    it is randomly truncated.
    """
    m = logits.size(0)
    if m <= max_size:
        return logits, targets

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

    return logits[final_idx], targets[final_idx]
