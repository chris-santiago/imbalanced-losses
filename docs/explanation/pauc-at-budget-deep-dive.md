# PAUCAtBudgetLoss: a deep dive

A technical write-up of `PAUCAtBudgetLoss` — what it optimizes, how each piece
works, where it helps, and where it does not. It is candid about limitations:
on easy problems a well-weighted cross-entropy is hard to beat, and the gains
are regime-specific.

## Abstract

`PAUCAtBudgetLoss` optimizes the **normalized partial AUC over a false-positive-rate
band** `[α, β]` that brackets a deployment operating point (e.g. a 50 bps alert
budget), rather than the whole ROC/PR curve or a single score threshold. The band
edges are estimated as **stop-gradient score quantiles of the iid negatives**, so
`β` keeps meaning population FPR even when a caller densifies the batch. A
**scale-aware temperature** keeps the surrogate's sharpness constant in FPR units as
the model's score scale drifts during training. Two surrogates are provided — a
**trapezoid** estimator (soft-TPR integrated over the band) and a **pairwise**
estimator (band-restricted Smooth-AP) — and a `pos_numerator` switch controls
whether the numerator uses all pooled positives or only the live, gradient-carrying
ones. The loss is an original design; the partial-AUC objective it targets has a
long published lineage (see [References](#references)).

## 1. Motivation

For an alerting or review workload, the metric that matters is **coverage at a
budget**: if you can action the top `b` fraction of scores (say the top 0.5 % —
"50 bps"), what fraction of positives do you catch? Equivalently, recall at a fixed
low false-positive rate. This is decided entirely at the **top of the score
ranking**, and standard objectives do not optimize it directly:

- **Cross-entropy** optimizes calibrated per-sample likelihood. Its gradient is
  spread across the whole distribution and, at extreme imbalance, dominated by the
  mass of easy negatives.
- **AUCPR / Smooth-AP** optimize the *whole* precision–recall curve.
- **Recall-at-Quantile** optimizes a *single* threshold.

A common symptom motivates this loss: improving a model — scaling up capacity, more
training, better features — can **raise mid-range AUCPR while leaving, or even
degrading, coverage at the operating point.** The model gets better in the bulk of
the score distribution and no better, or worse, exactly where it is deployed.
Whole-curve and per-sample objectives have no special pull on the top of the
ranking, so improvements land where the mass is, not where the budget is. When your
objective is a *band* of the ROC around a fixed budget, optimize that band directly.

`PAUCAtBudgetLoss` sits between Smooth-AP (whole curve) and Recall-at-Quantile
(single point): it optimizes a **region** of the ROC.

## 2. Formulation

For one class, let `s` be the scores, `P` the positives, and `N` the negatives.

### 2.1 The target quantity

The normalized partial AUC over the FPR band `[α, β]` has a clean probabilistic
form:

```
pAUC_norm(α, β) = P( s_i > s_j  |  i ∈ P,  s_j ∈ band )
```

where the **band** is the set of negatives whose score falls between the two FPR
edges. Writing `g` for the negative score density and `TPR(t) = P(s_pos > t)`:

```
∫_α^β TPR(u) du  =  ∫_{t_β}^{t_α} P(s_pos > t) g(t) dt  =  P(s_pos > s_neg ∧ s_neg ∈ band)
```

and the band holds exactly `(β − α)` of the negative mass, so dividing by `(β − α)`
gives the conditional probability above. This makes the loss a *consistent plug-in
estimator* of a well-defined integral, not a heuristic.

### 2.2 Band edges (iid-anchored, stop-gradient)

The two edges are score quantiles of the **iid negatives only**, detached:

```
t_α = quantile(neg_iid, 1 − α)        [no grad]
t_β = quantile(neg_iid, 1 − β)        [no grad]      (t_β ≤ t_α since α < β)
```

A negative is "in the band" iff `t_β ≤ s ≤ t_α`. FPR at threshold `t` is the
fraction of negatives scoring above `t`, so this is exactly FPR ∈ `[α, β]`
(empirically: on 2M N(0,1) negatives with `α=0.0025, β=0.0075`, the realized FPR at
the edges is 0.0025 / 0.0075 and the band holds ≈ `β − α` of the mass).

### 2.3 Surrogates

**Trapezoid** (default). Place `n_knots` equally-spaced FPR knots `f_k ∈ [α, β]`
(endpoints included), each with a detached threshold `t_k = quantile(neg_iid, 1 − f_k)`.
Soft-TPR at each knot, then composite-trapezoid integrate:

```
TPR̂_k = mean_{i ∈ P} σ((s_i − t_k) / τ_eff)
pAUC  = ( ½·TPR̂_0 + Σ_mid TPR̂_k + ½·TPR̂_{K−1} ) / (n_knots − 1)
loss  = 1 − pAUC
```

Gradient flows through positives only; negatives enter solely via the detached
thresholds. Cost is `O(|P| × n_knots)`. For a narrow band, `n_knots = 2` is accurate
(trapezoid error scales as `(β − α)³ · TPR''`); use `n_knots ≥ 3` for wide bands.

**Pairwise** (`surrogate="pairwise"`). Band-restricted Smooth-AP — compare positives
against the negatives that land *inside* the band, which carry gradient:

```
pAUC = mean_{i ∈ P, j ∈ band} σ((s_i − s_j) / τ_eff)
loss = 1 − pAUC
```

Cost is `O(|P| × |band|)`. Because it *pushes the band negatives down*, it is the
right tool when the operating point is contested by hard negatives.

### 2.4 Scale-aware temperature

```
τ_eff = temperature × scale          (scale detached)
```

where `scale` is a robust dispersion of the iid negatives — `IQR` by default
(`tau_scale="iqr"`), or the band width `t_α − t_β` (`tau_scale="band"`).
`temperature` is therefore a **dimensionless** multiplier (default `0.1`), unlike the
raw-logit `temperature=0.01` of the sibling losses.

### 2.5 `pos_numerator`

`"pool"` (default) averages the numerator over all pooled positives (live batch +
memory queue); `"live"` averages over the live-batch positives only. See §3.3.

## 3. Mechanisms and properties

### 3.1 β stays population FPR (iid anchoring)

Because the edges depend only on rows flagged `iid_mask=True`, appending non-iid
negatives (caller-side hard-negative mining / class-balanced sampling) does not move
`t_α` / `t_β`. With `iid_mask=None` (the default) every negative is treated as iid —
correct for any pipeline that never densifies negatives by class. The mask is
gathered across DDP ranks and stored per-row in the queue, so the edges are computed
from the **global** iid-negative pool. *(Verified: thresholds and loss are
bit-invariant to injected non-iid negatives.)*

### 3.2 Scale invariance (the τ design)

A fixed raw-logit temperature silently hardens as a model's score scale inflates
during training — the sigmoid saturates and band gradient vanishes exactly when
overfitting begins. Since `τ_eff ∝ scale` (a detached statistic of the negatives),
the ratio `(s − t)/τ_eff` is held roughly constant: scaling all logits by a constant
leaves the normalized loss and the gradient *direction* unchanged. *(Verified: loss
bit-identical and gradient-direction cosine 1.0 under score scaling, for both
`tau_scale` settings and all quantile interpolations.)*

### 3.3 Gradient dilution and `pos_numerator`

The trapezoid numerator is a mean over positives. At extreme imbalance a minibatch
holds only a handful of live (gradient-carrying) positives, while the memory queue
holds many *detached* ones. Under `pos_numerator="pool"` the live gradient is scaled
by `1/|P_pool|` and the soft-TPR value is dominated by stale queue positives — the
trapezoid surrogate underperforms or destabilizes. `pos_numerator="live"` computes
the numerator over the live positives only (the queue still feeds the thresholds),
restoring an undiluted gradient. This is **surrogate-dependent**: it rescues the
trapezoid, but the pairwise surrogate generally prefers `"pool"`, because restricting
its positive × band-negative contrast to a few live positives starves it.

### 3.4 Degenerate-dispersion guard

If the iid-negative dispersion is ≈ 0 (near-constant negatives, or a band collapsed
because too few iid negatives resolve the tail quantile), the class is marked invalid
and excluded from the reduction, with a one-time warning — instead of producing a
signal-free or exploding gradient. Relatedly, the band edges approximate population
FPR only when the pooled iid-negative count **substantially exceeds `1/α`**; below
that the tail quantile is biased toward the maximum. The `band_neg_count` diagnostic
is the empirical check.

### 3.5 Verified correctness

A technical review checked the math against analytic/large-sample references: the
quantile→FPR mapping is exact; the trapezoid normalization equals mean-TPR over the
band; trapezoid and pairwise both estimate `P(s_pos > s_neg | s_neg ∈ band)`; the
band edges and `τ_eff` are detached so gradient flows only through positives
(trapezoid) or positives + band negatives (pairwise); degenerate cases are NaN-free.
No mathematical errors were found.

## 4. Experiments

These are small, controlled, CPU-scale synthetic studies (see
`examples/coverage_at_budget_demo.py`), reported to characterize behavior — **not**
large-scale benchmark claims. Metrics are averaged over seeds because coverage at
50 bps is estimated from few validation positives and is noisy per seed.

### 4.1 Contested-negative top

Data: sub-1% positives; hard "decoy" negatives sit at the operating point,
separable only by a weak non-linear cue; "confounder" negatives carry that cue
without the easy signal (so over-using it costs AUCPR). Weighted CE protects the bulk
and under-resolves the top.

| loss | AUCPR | coverage@50bps |
|---|---|---|
| Weighted CE | ~0.24 | 0.573 ± 0.094 |
| SmoothAP | ~0.47 | 0.704 ± 0.152 |
| PAUC trapezoid | ~0.11 | 0.38 (collapses) |
| **PAUC pairwise** | ~0.54 | **0.772 ± 0.051** |

The pairwise surrogate recovers coverage CE leaves on the table (**+35 %**,
seed-stable). The trapezoid collapses here: it only lifts positives toward a frozen
threshold and cannot suppress the band negatives — the wrong surrogate for a
hard-*negative* top.

### 4.2 The `pos_numerator` rescue (non-contested top)

On easier data where the top is not contested by hard negatives, the trapezoid is
the natural surrogate, but with a large queue and few live positives it is
gradient-diluted under `"pool"`. Switching to `pos_numerator="live"` restores it to
competitive coverage. (On the contested-negative data of §4.1, the opposite holds —
pairwise wants `"pool"`.)

### 4.3 Where the gap closes

On cleanly separable tops — e.g. the Kaggle credit-card fraud data (~17 bps), whose
top is well separated (AUROC ≈ 0.97) — weighted CE already catches ~84 % at 50 bps,
and all losses land within noise. There is simply no contested region for a band loss
to recover. **This is the honest common case for easy data.**

## 5. When to use it (and when not)

**Reach for `PAUCAtBudgetLoss` when:**

- Your deployment metric is recall at a fixed, low false-positive **budget** (alerting,
  screening, fraud review), not whole-curve AP.
- The operating point is **contested by hard negatives** the model must learn to push
  down — use `surrogate="pairwise"`.
- You are at extreme imbalance but the pool (batch + queue, with DDP gather) holds far
  more than `1/α` iid negatives.

**Prefer something else when:**

- The top is already cleanly separable — a well-weighted CE or `SmoothAPLoss` is hard
  to beat, and the extra machinery buys little.
- You care about the **whole** ranking → `SmoothAPLoss`; or a **single hard
  threshold** → `RecallAtQuantileLoss`.
- The pool cannot supply enough iid negatives for a stable tail quantile at your `α`.

**Configuration heuristics:**

- Band `[α, β]` brackets the operating FPR (e.g. `[0.0025, 0.0075]` around 50 bps);
  band width is a bias–variance knob.
- Contested-negative top → `surrogate="pairwise"`, `pos_numerator="pool"`.
- Non-contested top, few live positives + large queue → `surrogate="trapezoid"`,
  `pos_numerator="live"`.
- `tau_scale="iqr"` with small `temperature` (≈0.1) for stability; `"band"` with
  `temperature ≈ 1.0` for wide/volatile bands.
- Watch `band_neg_count` (band populated?) and `grad_pos_count` (positives carrying
  gradient?) via `return_diagnostics=True`.

## 6. Limitations

- **Strong baseline.** Weighted CE is hard to beat on separable tops; the wins are
  regime-specific (contested operating points), and on synthetic data the gaps,
  while real, can be modest and seed-noisy.
- **Surrogate is regime-dependent.** Trapezoid suits non-contested tops (lifting hard
  positives) and collapses on contested-*negative* tops; pairwise suits contested
  tops but needs enough positives in its contrast. There is no single dominant
  surrogate.
- **`pos_numerator` is also regime-dependent** (`"live"` for trapezoid dilution;
  `"pool"` for pairwise), which is a real configuration burden.
- **Tail-quantile sensitivity.** At very low `α` the band edge is defined by few
  negatives; a small pool gives a biased, jittery threshold. Needs pool ≫ `1/α`.
- **No large-scale benchmark validation.** The evidence here is controlled synthetic
  studies plus a small real dataset; the loss has not been validated on large public
  benchmarks or against the deep-pAUC methods in the literature.
- **`pos_numerator="live"` + `max_pool_size`** subsampling are not jointly exercised;
  the live numerator is intended for the minibatch-with-queue regime.

## References

This loss is an original design, but the partial-AUC objective and its estimators
build on:

1. D. K. McClish (1989). *Analyzing a Portion of the ROC Curve.* Medical Decision
   Making 9(3), 190–195.
2. L. E. Dodd and M. S. Pepe (2003). *Partial AUC Estimation and Regression.*
   Biometrics 59(3), 614–623.
3. H. Narasimhan and S. Agarwal (2013). *A Structural SVM Based Approach for
   Optimizing Partial AUC.* ICML 2013. (And the KDD 2013 "tight" variant.)
4. D. Zhu, G. Li, B. Wang, X. Wu, T. Yang (2022). *When AUC meets DRO: Optimizing
   Partial AUC for Deep Learning with Non-Convex Convergence Guarantee.* ICML 2022.

See also the [`PAUCAtBudgetLoss` reference](../reference/pauc-at-budget-loss.md),
[Use Ranking Losses](../how-to/use-ranking-losses.md), and
[Assumptions and Failure Modes](assumptions-and-failure-modes.md).
