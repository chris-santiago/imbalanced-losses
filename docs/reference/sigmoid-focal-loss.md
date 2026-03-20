# SigmoidFocalLoss

Binary / multi-label focal loss operating on raw logits with a sigmoid activation. Drop-in replacement for `BCEWithLogitsLoss` for imbalanced problems.

!!! note "Multi-label vs. multiclass"
    `SigmoidFocalLoss` applies a **sigmoid independently to each logit**, so every output is a separate binary prediction. Use it when a sample can belong to *multiple* classes at once (multi-label), or for a single yes/no decision (binary). If your classes are *mutually exclusive* — each sample has exactly one correct class — use [`SoftmaxFocalLoss`](softmax-focal-loss.md) instead.

::: imbalanced_losses.focal_loss.SigmoidFocalLoss

## Quick example

```python
from imbalanced_losses import SigmoidFocalLoss
import torch

loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()

loss = loss_fn(logits, targets)
loss.backward()
```

## Parameter guidance

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | `0.25` | Weights positives; set to `-1` to disable |
| `gamma` | `2.0` | Higher = more focus on hard examples; `0` = vanilla BCE |
| `reduction` | `"none"` | `"mean"` averages over elements; `"sum"` for total |
| `gather_distributed` | `None` | Auto-detects DDP; set `False` to opt out |
