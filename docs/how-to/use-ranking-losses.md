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

### Reset the queue manually

You do **not** need to reset the queue around validation — it is frozen while the model is in
eval mode (see [Queue behavior during validation](#queue-behavior-during-validation)). Call
`reset_queue()` when the queue's contents are genuinely stale: at the start of a new training
phase (e.g. switching from pre-training to fine-tuning), or if you set
`update_queue_in_eval=True` and want to drop eval-time entries before training resumes:

```python
loss_fn.reset_queue()
# then start the new training phase
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

`max_pool_size` caps the pool using minimum-quota subsampling: each observed class receives an equal reserved quota of `max_pool_size // (2 × n_classes)` rows, then the remaining budget is filled uniformly at random. The queue is unaffected — it continues to accumulate the original full batch.

This is **not** proportional sampling. A dominant background class and a rare foreground class receive the same quota, so rare classes are over-represented relative to their natural frequency. The practical consequence: effective `|P_c| ≈ max_pool_size // (2 × n_classes)`, not `max_pool_size × positive_rate`. Size `max_pool_size` from the target positive count, not from memory alone.

| Setting | Effect |
|---|---|
| `max_pool_size=None` (default) | No cap; full pool used |
| `max_pool_size=4096` | Pool capped at 4 096 rows; stochastic approximation |

**Choosing a value:** Use the largest `max_pool_size` your GPU memory allows. 2048–4096 is a reasonable starting point for most seq2seq tasks. A one-time `UserWarning` is emitted the first time subsampling is triggered.

### Queue size guidance

The queue accumulates past batches to give the soft-rank estimator more context. At very low positive rates (e.g. 0.5%), a `queue_size` of 512–1024 is a starting point, not a stability guarantee — a 1024-row pool at 0.5% holds only ~5 expected positives, so treat it as a floor to tune upward if per-batch AP estimates are noisy. For seq2seq, use `max_pool_size` to control the effective pool size rather than relying solely on `queue_size`.

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

## PAUCAtBudgetLoss

### Optimize pAUC around a 50 bps operating point

```python
from imbalanced_losses import PAUCAtBudgetLoss

# Recommended band: alpha=0, beta=budget.
# Contrasts positives against all false-positives above the operating threshold.
loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0, beta=0.005, queue_size=1024)
logits  = torch.randn(256, 4)
targets = torch.randint(0, 4, (256,))
loss = loss_fn(logits, targets)
loss.backward()
```

**Confirm:** `loss` is a scalar in `[0, 1]`.

### Binary classification

Set `num_classes=1` and pass targets in `{0, 1}`:

```python
loss_fn = PAUCAtBudgetLoss(num_classes=1, alpha=0.0, beta=0.005, queue_size=1024)
logits  = torch.randn(256, 1)
targets = torch.randint(0, 2, (256,))
loss = loss_fn(logits, targets)
loss.backward()
```

### Choose a surrogate

The default `"trapezoid"` surrogate integrates soft-TPR over `n_knots` FPR knots within the band. Gradient flows through positives only. Use `"pairwise"` when the band is wide or the FPR boundaries are volatile — it compares positives against the negatives that land inside the band, so band negatives also carry gradient:

```python
# Default band (alpha=0, beta=budget): trapezoid with 2 knots is accurate for narrow bands
loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0, beta=0.005, surrogate="trapezoid", n_knots=2)

# Wide or volatile band: pairwise is more robust
loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0, beta=0.05, surrogate="pairwise")
```

### Undilute the gradient at extreme imbalance

With a memory queue and only a few live positives per batch, the default
`pos_numerator="pool"` averages the soft-TPR numerator over all pooled positives
(live batch + the detached queue), so the live-positive gradient is diluted. Set
`pos_numerator="live"` to compute the numerator over the live positives only (the
queue still feeds the thresholds), restoring an undiluted gradient:

```python
# Few live positives per batch + large queue: undilute the trapezoid numerator
loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0, beta=0.005,
                           surrogate="trapezoid", pos_numerator="live", queue_size=8192)
```

This helps the `"trapezoid"` surrogate most. The `"pairwise"` surrogate usually
prefers the default `"pool"`, since restricting its positive×band-negative contrast
to the few live positives can starve it.

### FPR vs population budget (`budget_basis`)

By default `beta` is a false-positive rate — the band edges are quantiles of the
negatives. Pass `budget_basis="population"` to instead make `beta` a fraction of the
*whole population* (a top-k alert budget over all scores, matching an "action the top
`beta·N` cases" workflow):

```python
loss_fn = PAUCAtBudgetLoss(num_classes=1, alpha=0.0, beta=0.005,
                           budget_basis="population")
```

Prefer the default `"fpr"`. On synthetic contested-top data the two bases are within
seed noise at the recommended `alpha=0` band (with `alpha=0` the upper edge already
spans every top negative), and `alpha=0` beats `"population"` at any other band — see
the [PAUC deep dive](../explanation/pauc-at-budget-deep-dive.md) for the ablation.
(`surrogate="trapezoid"` + `budget_basis="population"` is approximately
`RecallAtQuantileLoss`.)

### Check band health with diagnostics

Pass `return_diagnostics=True` to get per-class statistics alongside the loss. Use `band_neg_count` to confirm the band is populated, and `grad_pos_count` to confirm positives are contributing gradients:

```python
loss, stats = loss_fn(logits, targets, return_diagnostics=True)
# stats keys: t_alpha, t_beta, tau_eff, band_neg_count, pauc_var, grad_pos_count (all [C] tensors)
print(stats["band_neg_count"])  # iid negatives landing in the FPR band
print(stats["grad_pos_count"])  # live positives carrying gradient (rank-local)
```

If `band_neg_count` is near zero, the band is too narrow for the current pool size — increase `queue_size`. If `grad_pos_count` is near 1, gradient signal is weak — increase the effective batch (DDP all-gather or larger `queue_size`), or set `pos_numerator="live"` (trapezoid) so the numerator uses only the gradient-carrying live positives.

With `alpha=0` (recommended), `t_alpha = max(neg_iid)` requires no tail-quantile estimation. Only `t_beta = quantile(neg_iid, 1 - beta)` requires adequate pool coverage; `queue_size=1024` with a batch of 256 gives 1280 negatives, well above the `~1/beta` minimum for `beta=0.005`.

---

## Queue behavior during validation

By default, `SmoothAPLoss`, `RecallAtQuantileLoss`, and `PAUCAtBudgetLoss` all freeze the queue when the model is in eval mode (`model.eval()`). This prevents validation-phase logits from contaminating training-phase queue contents:

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

## See also

[`examples/coverage_at_budget_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/coverage_at_budget_demo.py) — runnable comparison on an extreme-imbalance (<1% positives) problem with a contested top: weighted CE vs `SmoothAPLoss` vs `PAUCAtBudgetLoss` (both surrogates × `pos_numerator` pool/live) on coverage at a 50 bps budget. Shows PAUC pairwise recovering coverage CE leaves behind, why the trapezoid surrogate is the wrong tool for a hard-negative top, and the `pos_numerator` gradient-dilution effect. Requires `numpy` + `scikit-learn`.
