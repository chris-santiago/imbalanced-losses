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
    is_iid: torch.Tensor | None = None,
    is_live: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    natural frequency — intentionally so, to ensure they contribute gradient
    signal — while the remainder of the budget is filled uniformly.

    Parameters
    ----------
    logits : torch.Tensor, shape [M, C]
        Pool logits (live batch + queue, already filtered for ignore_index).
        May contain gradients — the returned view preserves them.
    targets : torch.Tensor, shape [M]
        Integer class labels corresponding to *logits*.
    max_size : int
        Maximum number of rows to return.  Must be positive.
    is_iid : torch.Tensor, shape [M], dtype=bool, optional
        Per-row iid flag.  When provided, the returned 3-tuple includes the
        flag tensor indexed by the same selected rows as logits/targets.
        When ``None`` (default), the existing 2-tuple is returned, keeping
        backward compatibility with all existing callers.
    is_live : torch.Tensor, shape [M], dtype=bool, optional
        Per-row live-batch flag (True = live-batch row, False = queue row).
        When provided alongside *is_iid*, it is indexed by the same selected
        rows and returned as the fourth element.  When ``None`` (default),
        it does not affect the return arity.

    Returns
    -------
    logits_sub : torch.Tensor, shape [min(M, max_size), C]
    targets_sub : torch.Tensor, shape [min(M, max_size)]
    is_iid_sub : torch.Tensor, shape [min(M, max_size)], dtype=bool
        Only returned when *is_iid* is not ``None``.
    is_live_sub : torch.Tensor, shape [min(M, max_size)], dtype=bool
        Only returned when *is_live* is not ``None``.  Always follows
        *is_iid_sub* in position.

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
    m = logits.size(0)
    if m <= max_size:
        if is_iid is None:
            return logits, targets
        if is_live is None:
            return logits, targets, is_iid
        return logits, targets, is_iid, is_live

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

    if is_iid is None:
        return logits[final_idx], targets[final_idx]
    if is_live is None:
        return logits[final_idx], targets[final_idx], is_iid[final_idx]
    return logits[final_idx], targets[final_idx], is_iid[final_idx], is_live[final_idx]
