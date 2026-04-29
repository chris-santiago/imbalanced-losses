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
loss_fn = SmoothAPLoss(num_classes=C, queue_size=1024, max_pool_size=4096)
logits  = torch.randn(B, T, C).view(-1, C)    # [B*T, C]
targets = torch.randint(0, C, (B, T)).view(-1) # [B*T]
loss = loss_fn(logits, targets)
```

Without `max_pool_size`, a batch of 30 sequences × 512 tokens produces a pool of 15 360 rows. `SmoothAPLoss` builds a `[|P|, M]` pairwise matrix per class and retains the autograd graph across all classes simultaneously — peak memory is O(M²) and easily OOMs on GPU.

`max_pool_size` caps the pool with stratified random subsampling: at least one row per observed class is guaranteed before the remaining budget is filled uniformly. The queue is unaffected — it continues to accumulate the original full batch.

| Setting | Effect |
|---|---|
| `max_pool_size=None` (default) | No cap; full pool used |
| `max_pool_size=4096` | Pool capped at 4 096 rows; stochastic approximation |

**Choosing a value:** Use the largest `max_pool_size` your GPU memory allows. 2048–4096 is a reasonable starting point for most seq2seq tasks. A one-time `UserWarning` is emitted the first time subsampling is triggered.

### Queue size guidance

The queue accumulates past batches to give the soft-rank estimator more context. At very low positive rates (e.g. 0.5%), a `queue_size` of at least 512–1024 gives stable AP estimates. For seq2seq, use `max_pool_size` to control the effective pool size rather than relying solely on `queue_size`.

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

---

## Queue behavior during validation

By default, both `SmoothAPLoss` and `RecallAtQuantileLoss` freeze the queue when the model is in eval mode (`model.eval()`). This prevents validation-phase logits from contaminating training-phase queue contents:

| Mode | `update_queue_in_eval` | Queue behavior |
|------|----------------------|----------------|
| `model.train()` | (ignored) | Queue always updates |
| `model.eval()` | `False` (default) | Queue frozen — retains training-phase contents |
| `model.eval()` | `True` | Queue updates with current-phase logits |

**Standard training loop (keep the default `False`):** The queue retains its training-batch statistics across the validation phase and resumes cleanly when training restarts. You do not need to call `reset_queue()` between validation and the next training epoch.

**Online inference or streaming (`True`):** Set `update_queue_in_eval=True` when you want the queue to adapt to the current input distribution continuously during inference.

```python
# Training loop pattern — no extra handling needed
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)  # queue updates
        ...

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            ...  # queue frozen, no contamination
```
