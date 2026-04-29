# Use Focal Loss

## Binary / multi-label: SigmoidFocalLoss

> **Multi-label vs. multiclass:** `SigmoidFocalLoss` applies sigmoid *independently* to each logit — every output is a separate binary prediction. This covers binary tasks (one logit) and multi-label tasks (many logits, where a sample can match several classes at once). If your classes are mutually exclusive and each sample has exactly one correct label, use [SoftmaxFocalLoss](#multiclass-softmaxfocalloss) instead.

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

---

## When to prefer weighted cross-entropy over focal loss

Focal loss was introduced to address class imbalance by down-weighting easy, well-classified examples so training focuses on hard ones. The loss for sample $i$ is:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log p_t$$

where $p_t$ is the model's predicted probability for the true class, $\alpha_t$ is a class-balance weight, and $\gamma \geq 0$ is the focusing exponent. At $\gamma = 0$ this reduces to standard weighted cross-entropy.

The focusing term $(1 - p_t)^\gamma$ downweights examples the model classifies confidently and upweights examples it finds difficult. This is most beneficial when easy examples are numerous enough to dominate the gradient — the regime for which focal loss was designed (e.g. RetinaNet, where ~34% of anchors are positive after filtering). **At extreme positive rates (≪ 1%), this mechanism can backfire.**

### Why the focusing term hurts at very low positive rates

**1. It suppresses gradient from confident positives.**

When a positive is well-classified ($p_t$ high), the base gradient is already small. Focal loss suppresses it further by $(1 - p_t)^\gamma$: at $p_t = 0.9$ and $\gamma = 2$, the focal contribution is roughly 1% of the weighted cross-entropy contribution. When per-batch positive counts are in the single digits, every positive gradient signal is statistically meaningful regardless of confidence. Discarding the contribution of well-classified positives is a cost that scales inversely with positive count.

**2. The focusing acts almost entirely on negatives.**

When positives are rare, the hard examples that $\gamma$ upweights are predominantly hard negatives — samples near the decision boundary where the model is uncertain. Whether these are the most informative examples is domain-dependent; in many cases they represent label noise or genuine ambiguity. Either way, the intended purpose of focal loss — amplifying signal from hard *positives* — is structurally undermined when nearly all hard examples are negative by construction.

### Alpha does the real work

In highly imbalanced settings, the $\alpha_t$ term is the primary mechanism correcting for imbalance. It directly rescales the loss contribution of each class by inverse frequency, unconditionally and independently of model confidence. Focal loss adds $\gamma$ on top, but at extreme imbalance the marginal benefit diminishes and the cost — suppressed gradient from an already-scarce positive class — is concrete.

### A practical qualification

This analysis assumes random or globally-stratified sampling, where per-batch positive counts reflect the population rate. When batches are constructed with controlled per-class quotas (e.g. oversampling), the effective within-batch positive rate can be substantially higher than the population rate. With many positives per batch, easy positives are more plausibly redundant and the focusing term recovers its intended function. Evaluate $\gamma$ against **actual per-batch positive counts**, not the population rate alone.

### Rule of thumb

Treat $\gamma$ as a continuous hyperparameter whose optimal value approaches zero as per-batch positive counts fall. At very low positive counts, well-tuned alpha weighting alone is likely sufficient, and $\gamma > 0$ should be motivated empirically rather than assumed to help.

```python
# At extreme imbalance: start here and increase gamma only if it helps empirically
pos_rate = 0.0015
alpha    = [1.0, 1.0 / pos_rate]    # [negative_weight, positive_weight]

loss_fn = SoftmaxFocalLoss(alpha=alpha, gamma=0.0)   # equivalent to weighted CE
```

## See also

[`examples/focal_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/focal_demo.py) — side-by-side AUCPR comparison of BCE, `BCEWithLogitsLoss` with `pos_weight`, `SigmoidFocalLoss(alpha, gamma)`, and `SigmoidFocalLoss(gamma)` (no alpha) on the same imbalanced dataset, so you can isolate the contribution of each component.
