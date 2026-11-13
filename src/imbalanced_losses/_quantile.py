"""
Order statistics without ``torch.quantile``'s input size limit.

The threshold computations of RecallAtQuantileLoss and PAUCAtBudgetLoss use it.

Private module -- nothing here is part of the public API.
"""

from __future__ import annotations

import torch

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
