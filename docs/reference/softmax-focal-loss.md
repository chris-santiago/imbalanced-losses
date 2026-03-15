# SoftmaxFocalLoss

Multiclass focal loss with softmax, for mutually-exclusive classification. Supports per-class `alpha` weighting, `mean_positive` reduction (RetinaNet convention), label smoothing, and arbitrary input shapes.

::: imbalanced_losses.focal_loss.SoftmaxFocalLoss

## Quick examples

### Standard multiclass

```python
from imbalanced_losses import SoftmaxFocalLoss
import torch

loss_fn = SoftmaxFocalLoss(gamma=2.0, reduction="mean")
logits  = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))

loss = loss_fn(logits, targets)
loss.backward()
```

### RetinaNet-style detection

```python
loss_fn = SoftmaxFocalLoss(
    gamma=2.0,
    alpha=[0.25] * 10,         # per-class weights
    reduction="mean_positive",  # normalize by foreground count
    background_class=0,
    ignore_index=-100,
)
loss = loss_fn(logits, targets)
```

### Dense prediction (spatial inputs)

```python
# [N, C, H, W] logits, [N, H, W] targets
logits  = torch.randn(4, 10, 64, 64)
targets = torch.randint(0, 10, (4, 64, 64))
loss = loss_fn(logits, targets)
```

## Parameter guidance

| Parameter | Default | Notes |
|---|---|---|
| `alpha` | `None` | Per-class 1-D tensor or list; `None` disables class weighting |
| `gamma` | `2.0` | Higher = harder focus; `0` = vanilla cross-entropy |
| `reduction` | `"mean"` | `"mean_positive"` normalizes by foreground count (detection tasks) |
| `background_class` | `0` | Class excluded from `mean_positive` denominator |
| `ignore_index` | `-100` | Padded positions — zero loss, zero gradient |
| `label_smoothing` | `0.0` | Forwarded to `F.cross_entropy` |

### mean_positive reduction semantics

- **Numerator:** sum of loss over all valid (non-ignored) positions, including background
- **Denominator:** count of non-background, non-ignored positions only
- This matches the original RetinaNet implementation and stabilizes loss scale when positives are rare
