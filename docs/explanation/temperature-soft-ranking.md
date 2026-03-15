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

## Practical temperature ranges

| Setting | Recommended τ |
|---|---|
| Early training / warm start | `0.1–0.5` |
| Stable mid-training | `0.02–0.05` |
| Late training refinement | `0.005–0.01` |

The `temp_start=0.5, temp_end=0.01` defaults in `LossWarmupWrapper` examples cover the full range over the main training phase.

## Connection to the discontinuous rank

You can verify the approximation quality by comparing `SmoothAPLoss` with a perfect model (all positives score above all negatives). At τ → 0 with perfect scores, the soft AP should approach 1.0 and the loss should approach 0.0. The tests in `test_smooth_ap_loss.py` confirm this numerically.
