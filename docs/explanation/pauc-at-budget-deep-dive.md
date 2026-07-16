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
"50 bps"), what fraction of positives do you catch? At extreme imbalance this is
approximately recall at a fixed low false-positive rate — the two coincide as
positives become a negligible share of the population, but they are not the same
quantity (the top-`b` budget is a fraction of *all* samples, FPR a fraction of
*negatives*; see the technical report §1.3 for the small gap). This is decided
entirely at the **top of the score ranking**, and standard objectives do not
optimize it directly:

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
fraction of negatives scoring above `t`, so this is exactly FPR ∈ `[α, β]`.
With the recommended `α=0`, `t_α = max(neg_iid)` and the band covers all negatives
above the budget threshold (empirically: on 2M N(0,1) negatives with `α=0, β=0.005`,
the band holds ≈ `β` of the mass and includes the top-scoring false-positives).

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
from the **global** iid-negative pool. *(Verified: the thresholds are bit-invariant to
injected non-iid negatives for both surrogates; the loss is bit-invariant for the
trapezoid surrogate only. The pairwise surrogate deliberately contrasts positives
against every negative that lands in the band — iid or not — so injected non-iid
negatives that fall in the band do change its loss.)*

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
FPR only when the pooled iid-negative count comfortably resolves the band's smaller
nonzero edge: **substantially exceeding `1/β`** with the default `α=0` (where
`t_α = max(neg_iid)` needs no tail quantile at all), and additionally `1/α` when
`α > 0`; below that the tail quantile is biased toward the maximum. The
`band_neg_count` diagnostic is the empirical check.

### 3.5 Verified correctness

A technical review checked the math against analytic/large-sample references: the
quantile→FPR mapping is exact; the trapezoid normalization equals mean-TPR over the
band; trapezoid and pairwise both estimate `P(s_pos > s_neg | s_neg ∈ band)`; the
band edges and `τ_eff` are detached so gradient flows only through positives
(trapezoid) or positives + band negatives (pairwise); degenerate cases are NaN-free.
No mathematical errors were found.

## 4. Experiments

These are small, controlled, CPU-scale synthetic studies — **not** large-scale
benchmark claims. They come in two tiers, and the second is the stronger basis for
everything that follows:

1. **The shipped demo** (`examples/coverage_at_budget_demo.py`) — a single,
   reproducible contested-top instance you can run in one command (§4.3).
2. **A controlled investigation** — a CI-backed ablation (8 seeds, bootstrap-over-seed
   paired CIs) that isolates *what* makes the advantage appear and characterizes it
   across cue form, budget, surrogate, and capacity. This is far more informative than
   any single demo run: it identifies the **binding variable** and underwrites the
   mechanism in §5.

Metrics are averaged over seeds because coverage at 50 bps is estimated from few
validation positives and is noisy per seed; the investigation reports a bootstrap CI
on every lift and counts a result only when the CI excludes zero.

### 4.1 The binding variable: is the operating-point cue one cross-entropy under-learns?

A single "pairwise beats CE by X %" number does not tell you *when* the loss helps.
The controlled ablation answers that by holding **the same two discriminative
features** fixed and varying only the **functional form of the cue** that separates
positives from the decoys at the operating point. The cue's linearity — not bulk
difficulty, capacity, or class ratio — is the determinant:

| cue (50 bps, 8 seeds) | CE coverage | PAUC-pairwise | lift [95 % CI] |
|---|---:|---:|---|
| **linear** (same two features) | 0.864 | 0.880 | **+0.015 [+0.005, +0.024]** |
| **nonlinear, product** (`f5·f6`) | 0.575 | 0.791 | **+0.216 [+0.183, +0.257]** |
| **nonlinear, radial** (distinct form) | 0.426 | 0.638 | **+0.212 [+0.177, +0.247]** |

A **linear** cue is cheap for cross-entropy to capture, so its coverage stays high
(0.86) and PAUC adds almost nothing (+0.015). A **nonlinear** cue that is relevant
only to the rare top is one cross-entropy's bulk-dominated gradient never invests in
(coverage 0.43–0.58), and the band-focused PAUC gradient drives the MLP to learn it —
an **order-of-magnitude larger** lift that **replicates across two distinct nonlinear
forms** with intervals that do not overlap the linear cue's. This is the headline
result of the investigation, and §5 explains the mechanism behind it.

Two qualifiers come from the same study:

- **Operating-point specific.** Across these cells AUROC stays ~0.99 — the loss moves
  coverage at the budget without making a globally better-ranked model. The advantage
  is concentrated exactly where you deploy, which is the whole point.
- **Budget-dependent.** The nonlinear-product lift falls from **+0.216 at 50 bps** to
  **+0.045 [+0.035, +0.055] at 100 bps**: a wider budget is easier for cross-entropy,
  leaving less for the band to recover. The advantage is largest at the tightest
  budgets, and is gated on enough capacity (an MLP, to represent the cue at all) plus
  CE warmup and temperature annealing for stable training.

### 4.2 Surrogate specificity: only `pairwise` survives a contested-*negative* top

Within the nonlinear regime the surrogate choice is not a free parameter. The
**pairwise** surrogate carries the entire advantage; the **trapezoid** surrogate
collapses to the trivial floor, because it only lifts positives toward detached
thresholds and never suppresses the band negatives — the wrong tool when the top is
contested by hard *negatives*. `SmoothAPLoss`, a strong whole-curve ranking baseline
that sees the same data, reaches ~0.72 in the favorable regime but is beaten on
coverage@budget by the band-restricted pairwise surrogate (~0.79), exactly as
expected for a whole-curve objective that does not concentrate at one operating point.

### 4.3 The shipped demo: a contested-negative top (reproducible)

The bundled demo is one reproducible instance of the favorable regime: sub-1 %
positives; hard "decoy" negatives at the operating point, separable only by a weak
nonlinear cue; "confounder" negatives that carry the cue without the easy signal (so
over-using it costs AUCPR). Weighted CE protects the bulk and under-resolves the top.

| loss | AUCPR | coverage@50bps |
|---|---|---|
| Weighted CE | ~0.24 | 0.573 ± 0.094 |
| SmoothAP | ~0.47 | 0.704 ± 0.152 |
| PAUC trapezoid | ~0.11 | 0.38 (collapses) |
| **PAUC pairwise** | ~0.54 | **0.772 ± 0.051** |

The pairwise surrogate recovers coverage CE leaves on the table (**+35 %**,
seed-stable) — the same effect the CI-backed ablation in §4.1 isolates, here in a
form you can reproduce in one command.

### 4.4 The `pos_numerator` rescue (non-contested top)

On easier data where the top is not contested by hard negatives, the trapezoid is
the natural surrogate, but with a large queue and few live positives it is
gradient-diluted under `"pool"`. Switching to `pos_numerator="live"` restores it to
competitive coverage. (On the contested-negative data of §4.3, the opposite holds —
pairwise wants `"pool"`.)

### 4.5 Where the gap closes

On cleanly separable tops — e.g. the Kaggle credit-card fraud data (~17 bps), whose
top is well separated (AUROC ≈ 0.97) — weighted CE already catches ~84 % at 50 bps,
and all losses land within noise. (These numbers are an informal, unpublished
observation from ad-hoc runs; unlike §4.1–§4.4 they are not backed by an artifact in
this repository.) There is simply no contested region for a band loss to recover.
**This is the honest common case for easy data.**

## 5. Why it wins: adaptive hard-negative mining at the operating point

§4.1 shows the pairwise surrogate recovering coverage that weighted CE leaves on the
table. This section explains *why*, and the explanation is deflationary in a useful
way: the win is a **gradient-allocation effect**, not a property unique to the
partial-AUC objective. The evidence below comes from a controlled investigation on
contested-top synthetic data (a nonlinear, rarely-relevant cue; 8 seeds, bootstrap
CIs over paired per-seed differences) — the same regime as §4.1, run with the
diagnostics instrumented.

### 5.1 The mechanism: the band *is* a hard-negative miner

The pairwise surrogate contrasts each positive against the negatives whose scores
fall in the FPR band around the budget. **By construction those band negatives are
the decoys** — they are the only negatives that reach the top of the ranking — so
almost all of the loss's gradient is spent on the positives-versus-decoy contrast,
which is exactly the comparison the cue is needed to resolve. Cross-entropy
optimizes average log-loss; it sees the decoys as ~1.2 % of all negatives and never
concentrates there, so the bulk-dominated gradient under-invests in the one
distinction that decides coverage at the budget.

Two diagnostics on **identical data and scores** make this concrete:

- **Band enrichment.** The pairwise band is **~73 % decoys against a 1.2 % base
  rate** — roughly **60× enrichment** — and it selects them **with no decoy
  labels.** The band is a label-free hard-negative miner.
- **Gradient mass.** PAUC places **~96 % of its negative-gradient mass on decoys**,
  versus **~58 % for cross-entropy** scored on the same model outputs.

### 5.2 It is allocation, not capacity — and a label-free rule reproduces it

If the advantage were that PAUC *represents* the cue better, giving cross-entropy the
same gradient concentration would not help. It does. On the contested-top cell
(CE 0.576, pairwise PAUC 0.792):

| variant | coverage@50bps | note |
|---|---|---|
| Weighted CE | 0.576 | baseline |
| CE + oracle decoy up-weight ×10 | 0.767 | uses decoy labels |
| CE + label-free top-score HNM | 0.765 | no labels — mines the top of the negative ranking |
| **PAUC pairwise** | **0.792** | adaptive, label-free |

Concentrating cross-entropy's gradient — whether by an oracle decoy up-weight or a
label-free top-score hard-negative-mining rule — **recovers ~90 % of the advantage**
(the −0.216 gap collapses to ≈ −0.025). A complementary check rules out the capacity
story directly: a **linear probe** separating positives from decoys on the
penultimate activations scores **0.895 (CE) vs 0.910 (PAUC)** — nearly equal, so both
models *encode* the cue and differ only in whether the objective **acts on it at the
budget.** And the concentration must be *bounded*: crude over-concentration degrades a
pointwise loss (oracle ×30 < ×10 in every cell; ×50 → 0.702, ×200 → 0.632), whereas
PAUC's bounded contrast over an adaptively tracking band does not.

The effect **transfers** across cue form (product, radial) and budget (50, 100 bps):
concentrating cross-entropy's gradient closes the CE→PAUC gap in all four cells —
**≈ 89 % in the product/50-bps cell and past 100 % (overshooting PAUC) in the other
three** — and a label-free HNM-CE **matches or beats PAUC in three of the four.** So
the partial-AUC objective has no categorically higher ceiling here. Its distinct
contribution is delivering the concentration **adaptively and stably, without a tuned
up-weight factor and without decoy labels** — the band tracks the operating point as
the score scale drifts, where a hand-set mining factor does not.

### 5.3 Band escape — why the default `α` leaves coverage on the table

The same mechanism explains where the *default* band underperforms, and the fix is
one knob. The decoys pile up at the very top of the negative score distribution, but
the pairwise gradient only reaches the FPR band `[α, β]`. With the older default
`[budget/2, 1.5·budget]`, that band contains only **40–48 % of the decoys** while
**21 % (50 bps) to 41 % (100 bps) escape *above* it** (above `t_α`) and receive no
gradient at all — compared with **92–94 %** captured by a top-2 % miner.

```
default band [budget/2, 1.5·budget]:   t_β        t_α
score axis  ──────────────┼──── band ───┼──────────►  (higher score)
                          │             │
   decoys below        in band      decoys ESCAPE here
   (no gradient)       (gradient)   (no gradient — but they're the worst FPs)
```

Because `α = budget/2` caps the band below the top, the highest-scoring
false-positives — the decoys that most hurt coverage — sit *above* `t_α` and are
never pushed down. **Lowering `α` toward 0 widens the band up to `t_α = max(neg_iid)`,
covering the escaped top decoys.** This improves PAUC in every cell (**+0.017 to
+0.056** over the default) and makes wide-band PAUC match or beat the HNM rule in
three of four cells. A full sweep over both edges and four positive rates finds the
robust optimum at **`α = 0, β = budget`** — precisely the band of false-positives at
the operating point — which is why that is now the recommended default (§6). The gain
concentrates at `pos_rate ≪ budget` and vanishes once `pos_rate ≥ budget`, where
coverage@budget is capped at `budget / pos_rate`.

The takeaway for practice: **a label-free hard-negative-mined cross-entropy and
PAUC with a sufficiently wide band are roughly equivalent implementations of the same
gradient-concentration mechanism.** Reach for the pairwise band when you want that
concentration delivered adaptively and without labels; if you already have a working
HNM pipeline, you are most of the way there.

### 5.4 FPR vs population band basis (`budget_basis`)

The band edges default to quantiles of the iid **negatives** (`budget_basis="fpr"`), so `β` is a false-positive rate. But the deployment metric — coverage@budget — is a top-k over the *whole population* (recall among the top `⌈budget·N⌉` of all scores, positives included). Setting `budget_basis="population"` aligns the loss's band with that metric by taking the edge quantiles over the pooled population (positives + negatives) instead of the negatives only. The band still selects only negatives for the pairwise contrast; only the reference set for the edges moves. (With `surrogate="trapezoid"`, `budget_basis="population"` is approximately `RecallAtQuantileLoss` — a band-averaged soft recall over population quantiles.)

Whether that alignment helps is an empirical question, and the answer is **mostly no** — for the same reason the top-k and FPR *metric* estimators nearly coincide at extreme imbalance. On the nonlinear-product headline cell (8 seeds, the `budget_basis_ab` slice):

| band | `fpr` | `population` | Δ (pop − fpr) |
|---|---:|---:|---:|
| `α=0, β=budget` (recommended) | 0.814 | 0.814 | ≈ 0 |
| `α=budget/2, β=1.5·budget` (older) | 0.792 | 0.803 | +0.012 |

With `α=0`, `t_α = max` spans every top negative regardless of basis, so the population edge changes nothing (Δ ≈ 0). The population basis gives a small lift only at the narrow high-`α` band, where it shifts the band upward toward the escaped top decoys (§5.3) — a weaker version of the band-escape fix that lowering `α` provides in full. `α=0` (either basis) still dominates both. (These are 8-seed point estimates; the +0.012 is plausibly within seed noise, and there is no bootstrap CI on the fpr-vs-population delta.)

**Takeaway.** Keep the default `budget_basis="fpr"`. `"population"` is available for the alert-budget reading (`β` as a fraction of the whole population), but it does not beat the recommended `α=0` band — which already makes the basis moot.

## 6. When to use it (and when not)

**Reach for `PAUCAtBudgetLoss` when:**

- Your deployment metric is recall at a fixed, low false-positive **budget** (alerting,
  screening, fraud review), not whole-curve AP.
- The operating point is **contested by hard negatives** the model must learn to push
  down — use `surrogate="pairwise"`.
- You are at extreme imbalance but the pool (batch + queue, with DDP gather) holds far
  more than `1/β` iid negatives (and far more than `1/α` too, if you set `α > 0`).

**Prefer something else when:**

- The top is already cleanly separable — a well-weighted CE or `SmoothAPLoss` is hard
  to beat, and the extra machinery buys little.
- You care about the **whole** ranking → `SmoothAPLoss`; or a **single hard
  threshold** → `RecallAtQuantileLoss`.
- The pool cannot supply enough iid negatives for a stable band-edge quantile at your
  `β` (or at your `α`, when `α > 0`).

**Configuration heuristics:**

- **Recommended band: `α ≈ 0, β ≈ budget`** (e.g. `alpha=0.0, beta=0.005` for
  50 bps). Sets `t_alpha = max(neg_iid)`, contrasting positives against every
  false-positive above the budget threshold. The older `[budget/2, 1.5·budget]`
  convention excludes the highest-scoring negatives (via `alpha = budget/2`) and
  extends below the threshold (via `beta = 1.5·budget`); a band sweep on
  synthetic contested-top data found it in the poorly-performing high-`alpha`
  region (the worst cell being `alpha=budget/2, beta=2.5·budget`).
  Caveat: evidence is synthetic, 50 bps only, contested-top regime; improvement
  concentrates at `pos_rate ≪ budget`.
- Contested-negative top → `surrogate="pairwise"`, `pos_numerator="pool"`.
- Non-contested top, few live positives + large queue → `surrogate="trapezoid"`,
  `pos_numerator="live"`.
- `tau_scale="iqr"` with small `temperature` (≈0.1) for stability; `"band"` with
  `temperature ≈ 1.0` for wide/volatile bands.
- Watch `band_neg_count` (band populated?) and `grad_pos_count` (positives carrying
  gradient?) via `return_diagnostics=True`.

## 7. Limitations

- **Strong baseline.** Weighted CE is hard to beat on separable tops; the wins are
  regime-specific (contested operating points), and on synthetic data the gaps,
  while real, can be modest and seed-noisy.
- **Surrogate is regime-dependent.** Trapezoid suits non-contested tops (lifting hard
  positives) and collapses on contested-*negative* tops; pairwise suits contested
  tops but needs enough positives in its contrast. There is no single dominant
  surrogate.
- **`pos_numerator` is also regime-dependent** (`"live"` for trapezoid dilution;
  `"pool"` for pairwise), which is a real configuration burden.
- **Tail-quantile sensitivity.** The band's smaller nonzero edge is defined by few
  negatives when the pool is small, giving a biased, jittery threshold. Needs pool
  ≫ `1/β` with the default `α=0` (and ≫ `1/α` when `α > 0`).
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

## Further reading: the regime study

The regime characterization here — when the PAUC win is large versus when it vanishes — is drawn
from a controlled, CI-backed synthetic study. For the full methodology, results, mechanism, and
limitations, and to reproduce the numbers yourself:

- [**Technical report**](https://github.com/chris-santiago/imbalanced-losses/blob/main/lab/pauc_vs_ce_regimes/reports/TECHNICAL_REPORT.md)
  — `PAUCAtBudgetLoss` vs well-tuned cross-entropy on coverage@budget: cue nonlinearity as the binding
  variable, the operating-point-specific mechanism, surrogate and `pos_numerator` ablations, and the
  boundary conditions where the advantage disappears.
- [**Reproducible pipeline**](https://github.com/chris-santiago/imbalanced-losses/tree/main/lab/pauc_vs_ce_regimes)
  — the self-contained Metaflow + Hydra reproducer (`flow/pauc_flow.py`, configs in `conf/`, validator
  at `flow/validate.py`). All evidence is synthetic.
