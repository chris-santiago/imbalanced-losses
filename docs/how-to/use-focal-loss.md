# Use Focal Loss

## Binary / multi-label: SigmoidFocalLoss

### Drop-in replacement for BCEWithLogitsLoss

```python
import torch
from imbalanced_losses import SigmoidFocalLoss

loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
loss = loss_fn(logits, targets)
loss.backward()
```

### Disable alpha weighting (focusing only)

Set `alpha=-1` to apply the focusing modulator without re-weighting positives vs. negatives:

```python
loss_fn = SigmoidFocalLoss(alpha=-1, gamma=2.0, reduction="mean")
```

### Multi-label (arbitrary shape)

`SigmoidFocalLoss` accepts any shape for both `logits` and `targets`:

```python
loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
logits  = torch.randn(8, 10)       # batch of 8, 10 labels
targets = torch.randint(0, 2, (8, 10)).float()
loss = loss_fn(logits, targets)
```

**Confirm:** `loss` is a scalar tensor.

---

## Multiclass: SoftmaxFocalLoss

### Standard multiclass (drop-in for CrossEntropyLoss)

```python
from imbalanced_losses import SoftmaxFocalLoss

loss_fn = SoftmaxFocalLoss(gamma=2.0, reduction="mean")
logits  = torch.randn(32, 10)         # [N, C]
targets = torch.randint(0, 10, (32,)) # [N] integer labels
loss = loss_fn(logits, targets)
loss.backward()
```

### Per-class alpha weighting

Pass a list or tensor of length C to weight each class individually:

```python
# Down-weight background (class 0) relative to foreground
alpha = [0.1] + [1.0] * 9   # length 10

loss_fn = SoftmaxFocalLoss(alpha=alpha, gamma=2.0, reduction="mean")
loss = loss_fn(logits, targets)
```

### RetinaNet-style mean_positive reduction

For detection tasks where most samples are background, `mean_positive` normalizes by the number of foreground (non-background) samples rather than the total batch:

```python
loss_fn = SoftmaxFocalLoss(
    gamma=2.0,
    alpha=[0.25] * 10,
    reduction="mean_positive",
    background_class=0,
    ignore_index=-100,
)
loss = loss_fn(logits, targets)
```

**Confirm:** `loss` is a scalar. If all samples are background, the denominator is clamped to 1 to avoid division by zero.

### Ignore padded positions

Set padded target positions to `-100` (or your custom `ignore_index`). They contribute zero loss and zero gradient:

```python
targets[0] = -100   # mark position 0 as padding
loss = loss_fn(logits, targets)
```

### Spatial / sequence inputs

`SoftmaxFocalLoss` handles any shape of the form `(N, C, *)`:

```python
# Dense prediction: [N, C, H, W] logits, [N, H, W] targets
logits  = torch.randn(4, 10, 64, 64)
targets = torch.randint(0, 10, (4, 64, 64))
loss = loss_fn(logits, targets)
```

**Confirm:** `loss` is a scalar.
