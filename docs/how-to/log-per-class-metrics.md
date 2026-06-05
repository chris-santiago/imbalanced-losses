# Log Per-Class Metrics

`SmoothAPLoss`, `RecallAtQuantileLoss`, and `PAUCAtBudgetLoss` all support returning per-class loss values alongside the aggregated scalar, without requiring a second forward pass.

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

## Use with PAUCAtBudgetLoss

`PAUCAtBudgetLoss` supports `return_per_class=True` with the same three-value tuple pattern:

```python
from imbalanced_losses import PAUCAtBudgetLoss

loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0025, beta=0.0075, queue_size=1024)
loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)

for c in valid.nonzero(as_tuple=True)[0].tolist():
    print(f"Class {c} pAUC-loss: {per_class[c].item():.4f}")
```

It also supports `return_diagnostics=True`, which returns per-class statistics alongside the loss. You can combine both:

```python
# Diagnostics only
loss, stats = loss_fn(logits, targets, return_diagnostics=True)
# stats: per-class [C] tensors — t_alpha, t_beta, tau_eff, band_neg_count, pauc_var, grad_pos_count

# Both per-class losses and diagnostics
loss, per_class, valid, stats = loss_fn(logits, targets, return_per_class=True, return_diagnostics=True)
```

Log both in a Lightning training step:

```python
def training_step(self, batch, batch_idx):
    logits, targets = batch
    loss, per_class, valid, stats = self.loss_fn(
        logits, targets, return_per_class=True, return_diagnostics=True
    )

    self.log("train/loss", loss)
    for c in valid.nonzero(as_tuple=True)[0].tolist():
        self.log(f"train/pauc_loss_class_{c}", per_class[c])
        self.log(f"train/band_neg_count_class_{c}", stats["band_neg_count"][c])

    return loss
```

`band_neg_count` and `grad_pos_count` are the key health indicators: near-zero values signal band or gradient starvation before the loss itself shows unusual behavior.

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

## See also

[`examples/per_class_metrics_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/per_class_metrics_demo.py) — runnable script demonstrating `return_per_class=True` for `SmoothAPLoss` and `RecallAtQuantileLoss`, including the `valid_mask` guard pattern. `PAUCAtBudgetLoss` uses the same pattern and additionally accepts `return_diagnostics=True`.
