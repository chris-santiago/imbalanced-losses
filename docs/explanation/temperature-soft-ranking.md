# Temperature and Soft Ranking

## Why the rank function is not differentiable

Average Precision is defined in terms of ranks: for each positive, what fraction of all higher-ranked samples are also positive? A sample's rank is the count of samples with strictly higher scores — a step function. Step functions have zero gradient everywhere and are undefined at ties. You cannot backpropagate through them.

`SmoothAPLoss` replaces the hard rank with a soft approximation using the sigmoid function.

## The sigmoid approximation

The hard indicator `1[s_j > s_i]` (1 if j ranks above i, 0 otherwise) is replaced by:

```
σ((s_j - s_i) / τ)
```

where τ is the temperature. As τ → 0, σ((s_j - s_i) / τ) converges to the hard indicator. As τ → ∞, it converges to 0.5 everywhere — no rank information.

The soft rank of a positive i is then:

```
ŝ_i = 1 + Σ_{j≠i} σ((s_j - s_i) / τ)
```

This is differentiable everywhere with respect to all scores. Gradients push positives to have lower ranks (higher scores relative to negatives).

## What temperature controls in practice

**High temperature (e.g. 0.1–0.5):**
- Soft ranks are a smooth average of many pairwise comparisons
- Gradients are smaller in magnitude, more stable
- Less precise approximation of true AP
- Better for early training when score differences are small

**Low temperature (e.g. 0.005–0.02):**
- Soft ranks are sharper step-function approximations
- Gradients are larger, more informative, but can be unstable
- Closer to true AP but harder to optimize
- Better for late training when the model is already reasonable

The geometric decay schedule in `LossWarmupWrapper` exploits this: start with high temperature for stable early gradients, then decay to low temperature to refine toward the true ranking objective.

## Temperature in RecallAtQuantileLoss

`RecallAtQuantileLoss` uses temperature differently. The threshold θ is computed without gradient (stop-gradient), and temperature controls the sharpness of the sigmoid around the threshold:

```
soft_recall = mean_{i∈P} σ((s_i - θ) / τ)
```

High τ: gradients flow from positives that are far below the threshold (soft push).
Low τ: gradients flow mainly from positives right at the boundary (hard push, can be unstable when positives jump across θ).

## Temperature in PAUCAtBudgetLoss

`PAUCAtBudgetLoss` uses a **scale-aware** temperature rather than a fixed raw-logit value. The effective temperature is:

```
tau_eff = temperature * scale
```

where `scale` is a detached (stop-gradient) robust dispersion of the iid negatives — the IQR of their scores by default (`tau_scale="iqr"`), or the band width `t_alpha - t_beta` (`tau_scale="band"`). Because `scale` is in the same units as the model's logits, `tau_eff` adapts automatically as the score distribution expands or contracts during training.

The `temperature` parameter in `PAUCAtBudgetLoss` is therefore **dimensionless** (default `0.1`), unlike the raw-logit `temperature=0.01` of `SmoothAPLoss` and `RecallAtQuantileLoss`. A temperature of `0.1` means the sigmoid kernel is tuned to a transition region of `0.1 × IQR` in score space — if the IQR is 2.0, the effective tau is 0.2. As training progresses and the IQR widens to 5.0, tau_eff automatically becomes 0.5, maintaining the same relative sharpness in FPR units.

This design prevents a common failure mode in long training runs: a fixed small temperature that worked at initialization becomes too sharp as score scale inflates, saturating the soft kernels and producing nearly zero gradients.

**Choosing `tau_scale`:**

| Setting | Effect | Pair with |
|---|---|---|
| `"iqr"` (default) | Stable bulk statistic; ignores the band specifically | Small `temperature` (0.05–0.2) |
| `"band"` | Sized to the operating region `t_alpha − t_beta` | `temperature` near 1.0 |

Use `"iqr"` for most cases. Switch to `"band"` when the band is very wide and you want the kernel tuned to the band's own scale rather than the full score distribution.

## Practical temperature ranges

| Setting | Recommended τ |
|---|---|
| Early training / warm start | `0.1–0.5` |
| Stable mid-training | `0.02–0.05` |
| Late training refinement | `0.005–0.01` |

The `temp_start=0.5, temp_end=0.01` defaults in `LossWarmupWrapper` examples cover the full range over the main training phase.

## Connection to the discontinuous rank

You can verify the approximation quality by comparing `SmoothAPLoss` with a perfect model (all positives score above all negatives). At τ → 0 with perfect scores, the soft AP should approach 1.0 and the loss should approach 0.0. The tests in `test_smooth_ap_loss.py` confirm this numerically.
