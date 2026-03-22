# Log Per-Class Metrics

`SmoothAPLoss` and `RecallAtQuantileLoss` both support returning per-class loss values alongside the aggregated scalar, without requiring a second forward pass.

## Retrieve per-class losses

Pass `return_per_class=True`:

```python
import torch
from imbalanced_losses import SmoothAPLoss

loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024)
logits  = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))

loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
loss.backward()

# per_class: shape [C], nan for degenerate classes
# valid:     shape [C], bool — True for classes with at least one pos and one neg
```

## Log in PyTorch Lightning

```python
def training_step(self, batch, batch_idx):
    logits, targets = batch
    loss, per_class, valid = self.loss_fn(logits, targets, return_per_class=True)

    self.log("train/loss", loss)
    for c in valid.nonzero(as_tuple=True)[0].tolist():
        self.log(f"train/ap_loss_class_{c}", per_class[c])

    return loss
```

Only classes in `valid` are logged — degenerate classes (all-positive or all-negative in the current pool) have `nan` values and are skipped automatically by the `valid` mask.

## Use with RecallAtQuantileLoss

The same pattern applies to `RecallAtQuantileLoss`:

```python
from imbalanced_losses import RecallAtQuantileLoss

loss_fn = RecallAtQuantileLoss(num_classes=4, quantile=0.005, queue_size=1024)
loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)

for c in valid.nonzero(as_tuple=True)[0].tolist():
    print(f"Class {c} recall-loss: {per_class[c].item():.4f}")
```

## Use with LossWarmupWrapper

`**kwargs` (including `return_per_class=True`) are forwarded to `main_loss` only when `main_weight >= 1.0` — i.e. `final_main_weight == 1.0` (default) and the blend period has ended. During warmup, blend, or when `final_main_weight < 1.0`, they are silently ignored:

```python
result = self.loss_fn(logits, targets, return_per_class=True)

if isinstance(result, tuple):
    loss, per_class, valid = result
    for c in valid.nonzero(as_tuple=True)[0].tolist():
        self.log(f"train/ap_class_{c}", per_class[c])
else:
    loss = result

return loss
```

**Confirm:** During warmup `result` is a plain scalar tensor. After blend it is a `(loss, per_class, valid)` tuple.
