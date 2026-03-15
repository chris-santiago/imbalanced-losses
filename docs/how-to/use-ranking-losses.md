# Use Ranking Losses

## SmoothAPLoss

### Multi-class AP loss

```python
import torch
from imbalanced_losses import SmoothAPLoss

loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024, temperature=0.01)
logits  = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))
loss = loss_fn(logits, targets)
loss.backward()
```

**Confirm:** `loss` is a scalar in `[0, 1]`.

### Binary classification

Set `num_classes=1` and pass targets in `{0, 1}`:

```python
loss_fn = SmoothAPLoss(num_classes=1, queue_size=256)
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32,))
loss = loss_fn(logits, targets)
loss.backward()
```

### Disable the memory queue

Set `queue_size=0` to compute AP on the current batch only. Useful for debugging or large batch sizes:

```python
loss_fn = SmoothAPLoss(num_classes=4, queue_size=0)
```

### Reset queue between training and validation

Call `reset_queue()` before running evaluation to avoid contaminating the queue with eval-time data:

```python
loss_fn.reset_queue()
# then run validation loop
```

### Seq2seq / token-level targets

Flatten to `[N, C]` / `[N]` before passing:

```python
B, T, C = 4, 128, 10
logits  = torch.randn(B, T, C).view(-1, C)   # [B*T, C]
targets = torch.randint(0, C, (B, T)).view(-1) # [B*T]
loss = loss_fn(logits, targets)
```

### Queue size guidance

The queue accumulates past batches to give the soft-rank estimator more context. At very low positive rates (e.g. 0.5%), a `queue_size` of at least 512–1024 gives stable AP estimates. The total pool `M = batch_size + queue_size` should stay ≤ ~4096; the implementation is O(|P| × M) where |P| is the positive count, but at higher positive rates this approaches O(M²).

---

## RecallAtQuantileLoss

### Optimize recall at the top 0.5%

```python
from imbalanced_losses import RecallAtQuantileLoss

loss_fn = RecallAtQuantileLoss(num_classes=4, quantile=0.005, queue_size=1024)
logits  = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))
loss = loss_fn(logits, targets)
loss.backward()
```

**Confirm:** `loss` is a scalar in `[0, 1]`.

### Choose the right quantile

`quantile` is the fraction of the score distribution targeted as the alert region:

| Use case | Typical quantile |
|---|---|
| Top 0.5% of scores flagged | `0.005` |
| Top 1% of scores flagged | `0.01` |
| Top 10% of scores flagged | `0.10` |

The quantile must exceed the positive class fraction for the threshold to fall in the negative score region under a perfect model. With balanced 4-class data (25% positives per class), use `quantile > 0.25` for sanity checks.

### Queue size for quantile stability

For `quantile=0.005` (top 50 bps), you need at least ~200 samples in the pool for a meaningful 99.5th percentile estimate. A `queue_size=1024` with a batch of 32 gives 1056 pooled samples — well above this minimum.

### Binary classification

```python
loss_fn = RecallAtQuantileLoss(num_classes=1, quantile=0.01, queue_size=512)
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32,))
loss = loss_fn(logits, targets)
```

### Change quantile interpolation

The threshold is computed with `torch.quantile`. The default `'higher'` interpolation is conservative — the threshold never undershoots the true cutoff. Use `'linear'` for a softer estimate:

```python
loss_fn = RecallAtQuantileLoss(
    num_classes=4,
    quantile=0.01,
    quantile_interpolation="linear",
)
```
