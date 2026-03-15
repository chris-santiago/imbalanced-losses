# RecallAtQuantileLoss

Differentiable Recall-at-Quantile loss with an optional memory queue. Optimizes recall above a score threshold set at the q-th quantile of the pooled distribution. Useful for alert/detection workloads.

::: imbalanced_losses.recall_loss.RecallAtQuantileLoss

## Quick examples

### Optimize recall at top 0.5%

```python
from imbalanced_losses import RecallAtQuantileLoss
import torch

loss_fn = RecallAtQuantileLoss(num_classes=4, quantile=0.005, queue_size=1024)
logits  = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))

loss = loss_fn(logits, targets)
loss.backward()
```

### Binary classification

```python
loss_fn = RecallAtQuantileLoss(num_classes=1, quantile=0.01, queue_size=512)
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32,))

loss = loss_fn(logits, targets)
```

### Per-class logging

```python
loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
loss.backward()

for c in valid.nonzero(as_tuple=True)[0].tolist():
    print(f"Class {c} recall-loss: {per_class[c].item():.4f}")
```

## Parameter guidance

| Parameter | Default | Notes |
|---|---|---|
| `num_classes` | required | Use `1` for binary |
| `quantile` | `0.005` | Fraction targeted as alert region; must be in `(0, 1)` |
| `queue_size` | `1024` | For `quantile=0.005`, need ≥ 200 pooled samples |
| `temperature` | `0.01` | Larger = smoother gradient; smaller = sharper recall estimate |
| `reduction` | `"mean"` | `"none"` returns `[C]` tensor; classes with no positives are `nan` |
| `ignore_index` | `-100` | Excluded from threshold estimation and recall |
| `update_queue_in_eval` | `False` | Freezes queue during `model.eval()` by default |
| `quantile_interpolation` | `"higher"` | Conservative default; use `"linear"` for a softer threshold |

## Quantile selection guidance

The threshold is the `(1 - quantile)` percentile of all pooled scores. For the threshold to fall in the negative score region under a perfect model, `quantile` must exceed the positive class fraction:

- 4 balanced classes (25% positives per class): use `quantile > 0.25` for sanity tests
- Real-world imbalance (1% positives): `quantile=0.005` is well above the positive fraction
