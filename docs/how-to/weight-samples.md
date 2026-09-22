# Weight Samples by Value

Every loss in this library accepts an optional `sample_weight` keyword argument on `forward()`. Supplying it moves the objective from counting samples to summing a per-observation value. The canonical case is a fraud, collections, or claims workload where a dollar amount matters more than a raw hit count. This guide walks through the value-weighted version of `RecallAtQuantileLoss`, then covers the pattern for the other losses.

`sample_weight=None` (the default) leaves every loss bitwise unchanged. This guide only applies once you start passing an explicit weight.

## The problem: count-based recall hides dollar impact

`RecallAtQuantileLoss` optimizes the fraction of positives that score above a threshold. Every positive counts as one unit, regardless of size. In a fraud-detection setting, catching ten $50 cases and catching one $50,000 case both move item-count recall by the same amount, but they are not remotely equivalent outcomes. If the review budget is measured in cases (`quantile=0.005` = top 50 bps of scores), the model should still be pushed to prioritize the positives that carry the most value within that budget.

## Weighted recall at a quantile budget

Compare the unweighted and weighted objectives. Let $P$ be the positive set in the pool and $w_i$ each positive's weight (dollar amount, default $1$):

$$
\text{unweighted: } \frac{1}{|P|}\sum_{i \in P} \sigma\!\Big(\frac{s_i - \theta}{\tau}\Big)
\qquad\qquad
\text{weighted: } \frac{\sum_{i \in P} w_i\, \sigma\!\big((s_i - \theta)/\tau\big)}{\sum_{i \in P} w_i}
$$

The threshold $\theta$ itself is unaffected: it is still the unweighted $(1-q)$-quantile of the pooled scores. Only the recall numerator and denominator become weight mass instead of counts. In practice:

```python
import torch
from imbalanced_losses import RecallAtQuantileLoss

loss_fn = RecallAtQuantileLoss(num_classes=1, quantile=0.005, queue_size=1024)

logits  = torch.randn(256, 1)                 # [N, 1] raw scores
targets = torch.randint(0, 2, (256,))          # [N] binary labels
dollar_weight = torch.rand(256) * 10_000.0     # [N] non-negative, one per row

loss = loss_fn(logits, targets, sample_weight=dollar_weight)
loss.backward()
```

**Confirm:** `loss` is a scalar. Positives with a larger `dollar_weight` now contribute proportionally more gradient toward pushing their score above $\theta$; a $50 case and a $50{,}000$ case are no longer interchangeable.

## Same pattern for `PAUCAtBudgetLoss` and `SmoothAPLoss`

`sample_weight` follows the identical shape contract (`[N]`, float, non-negative, on the logits device) across every ranking loss:

```python
from imbalanced_losses import PAUCAtBudgetLoss, SmoothAPLoss

pauc_loss = PAUCAtBudgetLoss(num_classes=1, alpha=0.0, beta=0.005, queue_size=1024)
pauc_out  = pauc_loss(logits, targets, sample_weight=dollar_weight)

ap_loss = SmoothAPLoss(num_classes=1, queue_size=1024)
ap_out  = ap_loss(logits, targets, sample_weight=dollar_weight)
```

For `PAUCAtBudgetLoss`, only the soft-TPR numerator is weighted; the FPR band edges (`t_alpha`, `t_beta`) remain unweighted order statistics of the iid negatives, so `budget_basis` and `beta` keep their existing meaning regardless of how you weight positives. Under `pos_numerator="live"`, only live positives' weights participate; under `"pool"` (the default), the memory queue's stored weights participate too.

## Focal losses: weighted denominators, same shape as targets

`SigmoidFocalLoss` and `SoftmaxFocalLoss` accept `sample_weight` too: shaped exactly like `targets` for `SoftmaxFocalLoss`, and for `SigmoidFocalLoss` shaped with dim 0 equal to `inputs.size(0)` and trailing dims broadcastable to `inputs` (so `[N, 1, H, W]` over `[N, C, H, W]` works, while `[C]` or a scalar does not: the full dim-0 extent is what keeps the weight aligned after a DDP all-gather):

```python
from imbalanced_losses import SoftmaxFocalLoss

loss_fn = SoftmaxFocalLoss(gamma=2.0, reduction="mean")
targets = torch.randint(0, 10, (32,))
dollar_weight = torch.rand(32) * 10_000.0

loss = loss_fn(logits, targets, sample_weight=dollar_weight)
```

`reduction="mean"` divides by weight mass over valid elements instead of the valid count; `reduction="mean_positive"` keeps its RetinaNet asymmetry: the numerator still sums over all valid elements, and only the denominator restricts to (and is weighted by) non-background positive mass.

## Weights persist through the queue, DDP, and subsampling

A weight you supply on one step is stored in the memory queue and reused on later steps (default weight `1` for rows enqueued without one), gathered across DDP ranks alongside `logits`/`targets`/`iid_mask` (see [DDP All-Gather and Gradients](../explanation/ddp-all-gather.md)), and dropped or reordered in lockstep with every other tensor by `ignore_index` filtering and `max_pool_size` subsampling. See [Memory Queue Design](../explanation/memory-queue.md#stored-sample_weight) for the checkpoint-compatibility details.

## What weighting does *not* change

- **Thresholds, ranks, and band membership stay unweighted.** `RecallAtQuantileLoss`'s $\theta$, `PAUCAtBudgetLoss`'s $t_\alpha, t_\beta, \tau_{\text{eff}}$ and band mask, and `SmoothAPLoss`'s soft ranks are all computed the same way regardless of weight. See [Assumptions and Failure Modes](../explanation/assumptions-and-failure-modes.md#sample_weight-all-losses) for why a zero-weight positive still occupies its place in those computations rather than being treated as absent.
- **Negatives' weights are never used** by any ranking loss. Only positives enter the weighted numerator/denominator.
- **Subsampling under `max_pool_size` is weight-blind.** When the pool exceeds `max_pool_size`, rows are selected uniformly at random within each class's quota, without regard to `sample_weight`. Weight-aware subsampling is a known follow-up, not yet built.

## Validation

- `sample_weight` must be non-negative; a negative value anywhere raises `ValueError`.
- `sample_weight` must be finite; a `NaN` or `inf` anywhere raises `ValueError`. Both would otherwise propagate silently into the loss value and the gradient.
- `sample_weight` must match the expected shape (`[N]` for ranking losses, `targets.shape` for `SoftmaxFocalLoss`, dim-0-exact and trailing-dim broadcastable to `inputs` for `SigmoidFocalLoss`); a mismatch raises `ValueError`.
- `sample_weight` is cast to the loss dtype, so a `float64` weight does not promote a `float32` loss.
- A `sample_weight` tensor that is entirely zero triggers a one-time `UserWarning` per instance, since it is very likely a wiring bug rather than an intentional all-zero batch.
