# PAUCAtBudgetLoss

Differentiable partial-AUC-over-an-FPR-band loss with an optional memory queue. Optimizes the normalized partial AUC over a false-positive-rate band `[alpha, beta]` that brackets a target operating point (e.g. FPR ≈ 0.005 / 50 bps), rather than the full AUC or a single-threshold recall. Useful when you care about recall at a fixed, low false-alarm budget (fraud, screening, alerting).

::: imbalanced_losses.pauc_loss.PAUCAtBudgetLoss

## Quick examples

### Optimize pAUC around a 50 bps operating point

```python
from imbalanced_losses import PAUCAtBudgetLoss
import torch

# Band [0.0025, 0.0075] brackets FPR = 0.005 (50 bps).
loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0025, beta=0.0075, queue_size=1024)
logits  = torch.randn(256, 4)
targets = torch.randint(0, 4, (256,))

loss = loss_fn(logits, targets)
loss.backward()
```

### Binary classification

```python
loss_fn = PAUCAtBudgetLoss(num_classes=1, alpha=0.0025, beta=0.0075, queue_size=1024)
logits  = torch.randn(256, 1)
targets = torch.randint(0, 2, (256,))

loss = loss_fn(logits, targets)
```

### Per-class logging

```python
loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
loss.backward()

for c in valid.nonzero(as_tuple=True)[0].tolist():
    print(f"Class {c} pAUC-loss: {per_class[c].item():.4f}")
```

### Diagnostics — detect band starvation

```python
loss, stats = loss_fn(logits, targets, return_diagnostics=True)
# stats: per-class [C] tensors
print(stats["band_neg_count"])   # iid negatives landing in the band
print(stats["grad_pos_count"])   # live positives carrying gradient (rank-local)
print(stats["t_alpha"], stats["t_beta"], stats["tau_eff"], stats["pauc_var"])
```

If `grad_pos_count` sits near 1 and `pauc_var` wanders, the band is starved of gradient signal — increase the effective batch (DDP all-gather) or densify positives upstream.

### Marking densified negatives (advanced)

If a caller densifies negatives by class (e.g. hard-negative mining), pass `iid_mask` so the FPR band edges are still estimated from an iid sample and `beta` keeps meaning population FPR:

```python
# iid_mask[i] = True for rows drawn iid; False for injected/densified rows.
loss = loss_fn(logits, targets, iid_mask=iid_mask)
```

When `iid_mask=None` (the default) every negative is treated as iid — correct for any pipeline that never densifies negatives by class.

## Parameter guidance

| Parameter | Default | Notes |
|---|---|---|
| `num_classes` | required | Use `1` for binary |
| `alpha` | `0.0025` | Lower FPR band edge; `0 <= alpha < beta <= 1` |
| `beta` | `0.0075` | Upper FPR band edge; brackets the operating point (not a ceiling) |
| `surrogate` | `"trapezoid"` | `"trapezoid"` integrates soft-TPR over the band (gradient through positives only); `"pairwise"` compares positives vs band negatives (band negatives carry gradient) — for wide/volatile bands |
| `n_knots` | `2` | Trapezoid FPR knots; `2` is accurate for narrow bands, `>= 3` for wide bands |
| `tau_scale` | `"iqr"` | Scale used for the scale-aware temperature: `"iqr"` (stable bulk statistic; pair with small `temperature`) or `"band"` (sized to the operating region; pair with `temperature` near 1.0) |
| `temperature` | `0.1` | **Dimensionless** multiplier on `tau_eff = temperature * scale` — not raw logits. Larger = smoother/biased-to-0.5; smaller = sharper but risks saturation |
| `queue_size` | `1024` | Larger queues stabilise the tail quantile; at low FPR you need many pooled negatives |
| `reduction` | `"mean"` | `"none"` returns `[C]`; invalid classes are `nan` |
| `ignore_index` | `-100` | Excluded from threshold estimation and the positive set |
| `quantile_interpolation` | `"higher"` | Conservative default for the band edges |
| `max_pool_size` | `None` | Minimum-quota subsampling cap for very large pools (seq2seq) |

## Band selection guidance

The band `[alpha, beta]` brackets the FPR you deploy at. For a 50 bps (FPR = 0.005) operating point, `[0.0025, 0.0075]` centers the band on it. Band **width** is a bias–variance knob: narrower tracks a tighter point (lower bias, higher variance); wider averages over more of the ROC (lower variance, more bias).

The band edges are estimated as score quantiles of the **iid negatives** and approximate *population* FPR only when the pooled iid-negative count substantially exceeds `1/alpha`. Below that, the tail quantile is biased toward the maximum negative score. Use `queue_size` to accumulate enough negatives, and the `band_neg_count` diagnostic as the empirical check. A class whose iid-negative score dispersion is degenerate (≈ 0) is skipped (marked invalid) with a one-time warning.

## Choosing among the ranking losses

`PAUCAtBudgetLoss` sits between the two existing ranking losses on the ROC:

- [`SmoothAPLoss`](smooth-ap-loss.md) optimizes the **whole** precision–recall curve (Average Precision).
- `PAUCAtBudgetLoss` optimizes a **band** of the ROC around your operating point.
- [`RecallAtQuantileLoss`](recall-at-quantile-loss.md) optimizes recall at a **single** score threshold.

Reach for `PAUCAtBudgetLoss` when your business constraint is a fixed false-alarm budget (a region, not the whole curve and not one hard point).
