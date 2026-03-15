# SmoothAPLoss

Differentiable Average Precision loss with an optional memory queue. Approximates AP using sigmoid-based soft rank estimation (Smooth-AP, Brown et al. 2020).

::: imbalanced_losses.ap_loss.SmoothAPLoss

## Quick examples

### Multi-class

```python
from imbalanced_losses import SmoothAPLoss
import torch

loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024, temperature=0.01)
logits  = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))

loss = loss_fn(logits, targets)
loss.backward()
```

### Binary classification

```python
loss_fn = SmoothAPLoss(num_classes=1, queue_size=256)
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32,))

loss = loss_fn(logits, targets)
```

### Per-class logging

```python
loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
loss.backward()

for c in valid.nonzero(as_tuple=True)[0].tolist():
    print(f"Class {c} AP-loss: {per_class[c].item():.4f}")
```

### Queue management

```python
# Reset queue between training and eval
loss_fn.reset_queue()
```

## Parameter guidance

| Parameter | Default | Notes |
|---|---|---|
| `num_classes` | required | Use `1` for binary |
| `queue_size` | `1024` | `0` to disable; keep `batch + queue` ≤ 4096 |
| `temperature` | `0.01` | Range `0.005–0.05`; lower = sharper, closer to true rank |
| `reduction` | `"mean"` | `"none"` returns `[C]` tensor with `nan` for degenerate classes |
| `ignore_index` | `-100` | Excludes padding from ranking and the positive set |
| `update_queue_in_eval` | `False` | Freezes queue during `model.eval()` by default |
| `gather_distributed` | `None` | Auto-detects DDP; set `False` to opt out |

## Complexity note

The core computation is O(|P| × M) where |P| is the number of positives and M = batch_size + queue_size. At low positive rates this is much less than O(M²) — roughly 200× cheaper at 0.5% positives.
