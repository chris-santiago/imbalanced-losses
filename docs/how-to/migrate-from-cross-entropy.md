# Migrate from BCE / CrossEntropyLoss

This guide is a drop-in migration checklist for switching from `BCEWithLogitsLoss` or
`CrossEntropyLoss` to the losses in this library. Each section covers one migration path.

---

## BCE → SigmoidFocalLoss

`SigmoidFocalLoss` is a direct replacement for `BCEWithLogitsLoss`. It accepts the same input
shape and handles binary (`[N, 1]`) and multi-label (`[N, C]`) tasks identically.

**Before:**
```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets.float())
```

**After:**
```python
from imbalanced_losses import SigmoidFocalLoss

loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
loss = loss_fn(logits, targets.float())
```

**What changes:**
- Add `alpha` (class balance weight; `alpha=-1` disables it) and `gamma` (focusing exponent)
- Targets must still be `float` in `[0, 1]`
- `reduction` options are the same: `"mean"`, `"sum"`, `"none"` (`"mean_positive"` is a `SoftmaxFocalLoss`-only option)

**What stays the same:**
- Input and target shapes
- `.backward()` call — no changes to your training loop

**Where to start:** `alpha=0.25, gamma=2.0` are the RetinaNet defaults and a reasonable
first try. Set `gamma=0` if you want alpha-weighted BCE without focusing.

---

## CrossEntropyLoss → SoftmaxFocalLoss

`SoftmaxFocalLoss` is a direct replacement for `CrossEntropyLoss`. It accepts `[N, C]` logits
and `[N]` long integer targets.

**Before:**
```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
```

**After:**
```python
from imbalanced_losses import SoftmaxFocalLoss

loss_fn = SoftmaxFocalLoss(gamma=2.0, reduction="mean")
loss = loss_fn(logits, targets)
```

**What changes:**
- Add `gamma` (focusing exponent; `gamma=0` reduces to standard softmax cross-entropy)
- Optionally add `alpha` (per-class weights, length C) to up-weight rare classes

**What stays the same:**
- `ignore_index` is supported with the same semantics as `CrossEntropyLoss`
- Input and target shapes: `[N, C]` logits, `[N]` long integer targets
- Works with spatial/sequence inputs `[N, C, *]` the same way `CrossEntropyLoss` does

**Computing per-class alpha from class counts:**
```python
counts = [(y_train == c).sum().item() for c in range(n_classes)]
total  = sum(counts)
alpha  = [total / (n_classes * max(c, 1)) for c in counts]
alpha  = [a / max(alpha) for a in alpha]  # normalize: max weight = 1.0

loss_fn = SoftmaxFocalLoss(alpha=alpha, gamma=2.0)
```

---

## BCE / CE → SmoothAPLoss, RecallAtQuantileLoss, or PAUCAtBudgetLoss

Ranking losses require a warm start because their gradients are nearly zero when the model is
random. **Do not** switch cold — wrap with `LossWarmupWrapper` to run BCE or CE for the first
few epochs first.

### Binary

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.BCEWithLogitsLoss(),
    main_loss=SmoothAPLoss(num_classes=1, queue_size=512),
    warmup_epochs=5,
    blend_epochs=2,
)

# In your training loop:
global_step = 0
for epoch in range(total_epochs):
    loss_fn.on_train_epoch_start(epoch)
    for xb, yb in loader:
        loss_fn.on_train_batch_start(global_step)
        loss = loss_fn(model(xb), yb.float().unsqueeze(1))  # [N, 1] float targets
        ...
        global_step += 1
```

### Multiclass

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=n_classes, queue_size=1024),
    warmup_epochs=5,
    blend_epochs=2,
)

# CrossEntropyLoss and SmoothAPLoss share the same input format:
# logits [N, C] float, targets [N] long — no conversion needed.
global_step = 0
for epoch in range(total_epochs):
    loss_fn.on_train_epoch_start(epoch)
    for xb, yb in loader:
        loss_fn.on_train_batch_start(global_step)
        loss = loss_fn(model(xb), yb)
        ...
        global_step += 1
```

---

## Common mistakes

### Forgetting the epoch / step hooks

`LossWarmupWrapper` tracks phase transitions in `on_train_epoch_start` and temperature decay
in `on_train_batch_start`. Each hook has its own failure mode. In epoch mode, missing
`on_train_epoch_start` means the loss never transitions from warmup. Missing
`on_train_batch_start` still transitions, but silently disables temperature decay and the
queue reset at the phase switch (the wrapper emits a one-time `UserWarning` when this happens).

```python
# Wrong — loss stays in warmup forever, no temperature decay
for epoch in range(total_epochs):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)

# Correct
global_step = 0
for epoch in range(total_epochs):
    loss_fn.on_train_epoch_start(epoch)   # ← drives the phase transition (epoch mode)
    for xb, yb in loader:
        loss_fn.on_train_batch_start(global_step)  # ← drives temp decay + switch-time queue reset
        loss = loss_fn(model(xb), yb)
        global_step += 1
```

### Target dtype mismatch

`SigmoidFocalLoss` expects **float** targets in `[0, 1]`; passing integer targets raises a
runtime error. `SoftmaxFocalLoss` and `SmoothAPLoss(num_classes > 1)` expect **long** integer
class indices. `SmoothAPLoss(num_classes=1)` is flexible: it accepts float or integer targets
in `{0, 1}` and treats any non-zero value as positive.

```python
# SigmoidFocalLoss — targets must be float
targets_float = targets.float()                     # or .unsqueeze(1) for [N, 1]

# Multiclass — targets must be long integer indices
targets_long = targets.long()
```

### Skipping queue reset between training phases

The memory queue accumulates logits from previous batches. If you switch from a
classifier pre-training phase to fine-tuning without resetting, stale logits from the old
distribution contaminate the AP computation.

```python
# Reset manually when starting a new training phase
loss_fn.main_loss.reset_queue()

# Or configure automatic reset in LossWarmupWrapper
loss_fn = LossWarmupWrapper(
    ...,
    reset_queue_each_epoch=True,   # useful when distribution shifts each epoch
)
```

### Using ranking losses without warmup at all

```python
# Wrong — gradients are flat at random initialization, training stalls
loss_fn = SmoothAPLoss(num_classes=5, queue_size=1024)

# Correct — warm up with a proxy loss first
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=5, queue_size=1024),
    warmup_epochs=5,
)
```

---

## Decision checklist

Use this to decide how far to migrate:

| Situation | Recommended loss |
|---|---|
| Small imbalance (< 10:1), fast iteration | Start with `CrossEntropyLoss` + `alpha` weighting |
| Moderate imbalance, retinaNet-style detection | `SoftmaxFocalLoss(alpha=..., gamma=2.0)` |
| High imbalance (> 50:1), optimize macro-AP | `SmoothAPLoss` with warmup |
| High imbalance, optimize top-k recall | `RecallAtQuantileLoss` with warmup |
| Fixed false-alarm budget / operating-point band | `PAUCAtBudgetLoss` with warmup |
| Per-class monitoring needed | Any ranking loss with `return_per_class=True` |

See [Assumptions and Failure Modes](../explanation/assumptions-and-failure-modes.md) for when
each loss is likely to underperform.
