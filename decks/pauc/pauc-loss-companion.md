# Speaker companion — `PAUCAtBudgetLoss` teaching deck

One section per slide. Deliver the narrative; the "details not on the slide" are numbers and caveats to have ready for questions.

---

## Slide 1 — `PAUCAtBudgetLoss`

Open by naming the gap this loss fills. Most imbalanced-classification losses optimize either the whole ranking (AP/AUC) or per-sample calibration (cross-entropy). This one optimizes a narrow slice of the ROC that corresponds to how the model is actually used: an alert budget. Set expectations honestly up front — this is a loss with a *specific* regime where it wins, and part of the talk is telling you when *not* to use it.

**Details not on the slide**

- The loss is an original design in this library, not a published paper, though the partial-AUC objective has a long lineage (McClish 1989; Dodd & Pepe 2003; Narasimhan & Agarwal 2013; Zhu et al. 2022).
- All empirical evidence in the deck is controlled synthetic study plus one small real dataset — no large-scale benchmark validation.

---

## Slide 2 — The metric you actually deploy on

The core reframing: in alerting/fraud/screening you can only action the top *b* fraction of scores, so the metric that matters is coverage@budget — recall among that top-*b* alert budget. Define it explicitly (it's the term of art the whole deck hangs on). Then make the point that none of the three standard objectives optimizes *that*: CE spreads gradient everywhere, AP optimizes the whole curve, Recall-at-Quantile optimizes one point.

Be precise about the "≈ recall at a fixed low FPR" line, because it invites a fair objection: the top-*b* budget is a fraction of the *whole population*, while FPR is a fraction of *negatives only*. They are not the same quantity — they only coincide at extreme imbalance, where positives are a negligible share so the top-*b* cutoff is set almost entirely by negatives. If asked, that's the honest answer.

**Details not on the slide**

- "50 bps" = 0.5% = top 0.005 fraction of the population. Practitioners in fraud will recognize this as the alert budget.
- The two estimators are not identical even in the limit: because true positives consume some of the top-*b* slots, the top-*k* form runs at a slightly *stricter* effective FPR — the technical report (§1.3) measures ≈0.0038–0.0041 vs the nominal 0.005. The sign of every result is unaffected.
- At extreme imbalance CE's gradient is dominated by the mass of easy negatives — that's the mechanistic reason it under-resolves the top.

---

## Slide 3 — The symptom that motivates a band loss

This is the "aha" that makes people care. You improve the model — more capacity, more data — and AUCPR goes up, but coverage at your operating point stays flat or *drops*. The model got better in the bulk and no better where you deploy. Whole-curve and per-sample objectives have no special pull on the top, so gains land where the mass is.

**Details not on the slide**

- Ask the audience if they've seen this: "mid-range metrics improve, production alert recall doesn't." It's a common, frustrating pattern.
- The fix is conceptual, not just a new loss: optimize the band you deploy on.

---

## Slide 4 — The idea: optimize a region of the ROC

Position the loss between two things the audience already knows: Smooth-AP (whole curve) and Recall-at-Quantile (single threshold). PAUCAtBudgetLoss optimizes a *region* — the FPR band [α, β] at the budget. The figure shows the shaded band and the partial-AUC area under the curve inside it.

**Details not on the slide**

- The band is defined in FPR space, so it tracks the operating point regardless of score scale.
- α=0 is the recommended lower edge (explained two slides later); the figure draws that case.

---

## Slide 5 — What "partial AUC over a band" means

Walk the symbols before the meaning — the slide now carries a "where" legend, so point at it: `s` is the model score, `P` the positives, `s_i`/`s_j` a positive/negative score, the band is the negatives whose FPR lands in [α, β] (score in [t_β, t_α]), and TPR(u) is recall at FPR u. Then give the reading: the normalized partial AUC is the probability a positive outranks a negative *given that negative sits in the band*. Because the band holds exactly (β − α) of the negatives, that probability equals the normalized area under the ROC inside the band — a consistent plug-in estimator (a statistic that converges to the true quantity as data grows), not a heuristic score. The loss is 1 − pAUC.

**Details not on the slide**

- "Consistent plug-in estimator" is worth a sentence for a mixed audience: you estimate the population integral by plugging in the empirical scores, and the estimate converges to the true partial AUC as the sample grows.
- This is the target *quantity*; the next two slides are the two ways to *estimate* it differentiably (trapezoid vs pairwise).

---

## Slide 6 — Band edges: iid-anchored, stop-gradient

The two edges are score quantiles of the iid negatives, detached (no gradient flows through them). Two consequences to stress: (1) iid-anchoring means β keeps meaning population FPR even if you densify the batch with hard negatives — the edges don't move; (2) the recommended α=0 sets t_α = max(neg), so the band covers every false-positive above the budget threshold.

**Details not on the slide**

- "Detached" = `torch.no_grad`/`.detach()` on the thresholds; gradient reaches scores, not the quantile.
- iid_mask=None (default) treats all negatives as iid — correct for any pipeline that doesn't densify negatives by class. The mask is gathered across DDP ranks.
- Reliability condition: the pooled iid-negative count must comfortably exceed 1/β (and 1/α if α>0), or the tail quantile is biased toward the max.

---

## Slide 7 — Two surrogates — and they behave differently

This is the most important mechanistic slide. Lean on the two parallel "right tool when…" captions: trapezoid lifts positives over detached thresholds — gradient flows through positives only, negatives never move — so it's the right tool when the top is contested by hard *positives*, and it collapses when the top is contested by *negatives*. Pairwise contrasts positives against the negatives *inside* the band, and those band negatives carry gradient — so pairwise actively *pushes hard negatives down*, the right tool for a contested-*negative* top. That positives-vs-negatives contrast is the whole story of why only pairwise wins on contested tops (slide 11).

**Details not on the slide**

- Trapezoid cost is O(|P|·n_knots); pairwise is O(|P|·|band|). Neither is O(M²).
- n_knots=2 (trapezoid rule) is accurate for narrow bands; use ≥3 for wide bands.
- The where-legend on the slide glosses σ as a soft 1[s_i > s_j]; if asked, it's a temperature-scaled sigmoid whose sharpness is set by τ_eff (next slide).

---

## Slide 8 — Scale-aware temperature

Explain why τ isn't just a raw-logit knob. Walk the figure left-to-right: both panels show the same inflated late-training score spread. On the left, a fixed raw τ keeps the sigmoid's width constant in raw units, so it becomes a near-step — only a razor-thin slice of pairs sits on the slope, the rest are saturated (σ'≈0) and carry no gradient. On the right, τ_eff ∝ scale widens the sigmoid's transition zone to track the spread, so the gradient zone keeps covering the operating region. That's the whole argument: a fixed raw temperature silently hardens as the logit scale inflates during training — band gradient dies exactly when overfitting starts — while tying τ_eff to a detached dispersion of the negatives (IQR or band width) keeps the ratio (s−t)/τ_eff roughly constant, so loss and gradient direction are invariant to score rescaling.

**Details not on the slide**

- Verified: loss bit-identical and gradient-direction cosine 1.0 under score scaling, for both tau_scale settings.
- `temperature` here is dimensionless (default 0.1), unlike the raw-logit temperature≈0.01 of sibling losses. Pair `tau_scale="iqr"` with small temperature, `tau_scale="band"` with temperature≈1.0.

---

## Slide 9 — Headline: when does it beat well-tuned CE?

This is the money slide. The experiment holds the same two discriminative features fixed and varies only the *functional form* of the cue that separates positives from decoys at the top. Cue linearity is the determinant: CE captures a linear cue cheaply (+0.015 lift), but a nonlinear cue relevant only to the rare top is something CE's bulk-dominated gradient never invests in, and pairwise recovers ~+0.21 — an order of magnitude more, replicated across two distinct nonlinear forms.

**Details not on the slide**

- Exact numbers: linear +0.015 [+0.005, +0.024]; nonlinear-product +0.216 [+0.183, +0.257]; nonlinear-radial +0.212 [+0.177, +0.247]. 8 seeds, bootstrap-over-seed paired CIs.
- CE coverage collapses to 0.43–0.58 on the nonlinear cues while staying 0.86 on the linear one — that gap is what PAUC recovers.
- The comparator is a *well-tuned* CE (pos_weight and weight decay swept, validation-selected). This is best-vs-best, not a strawman.

---

## Slide 10 — Two honest qualifiers (same study)

Immediately temper the headline. Two caveats from the same ablation. Operating-point specific: AUROC stays ~0.99 across all cells — the loss moves coverage at the budget without making a globally better ranker. Budget-dependent: the lift falls from +0.216 at 50 bps to +0.045 at 100 bps, because a wider budget is easier for CE, leaving less to recover.

**Details not on the slide**

- "Operating-point specific" is a *feature*, not a bug — it's the whole point of a band loss — but it means you won't see it in whole-curve metrics.
- The advantage is gated on enough capacity (an MLP to represent the cue at all), CE warmup, and temperature annealing (0.5→0.1).

---

## Slide 11 — Surrogate choice is not free

Within the favorable regime, the surrogate is not a free parameter. Pairwise (green) carries the entire advantage. Trapezoid (red) collapses toward the trivial floor because it only lifts positives and never suppresses the contesting negatives — the wrong tool when the top is contested by hard *negatives*. SmoothAP (purple) is a strong whole-curve baseline that reaches ~0.72 but is beaten by band-restricted pairwise (~0.79).

**Details not on the slide**

- The x-axis labels are (confounder-fraction, positive-rate) cells; the pattern holds across all of them.
- This is exactly what you'd predict from slide 7: trapezoid's positives-only gradient can't push a contesting negative down.

---

## Slide 12 — Why it wins: the band *is* a hard-negative miner

Now the mechanism. The pairwise band contrasts each positive against the negatives at the top of the ranking — and by construction those are the decoys, the only negatives that reach the top. So almost all the gradient is spent on the positives-vs-decoy contrast, which is exactly the comparison the cue is needed to resolve. Two diagnostics on identical data make it concrete: the band is ~73% decoys against a 1.2% base rate (~60× enrichment), selected with no decoy labels; and PAUC puts ~96% of its negative-gradient mass on decoys vs ~58% for CE.

**Details not on the slide**

- The figure's arrows show in-band decoys being pushed down-rank (toward lower scores); bulk negatives below the band are inert; a few decoys "escape" above t_α (setup for the next-but-one slide).
- "Label-free" is the key adjective — the band finds the hard negatives without anyone telling it which negatives are decoys.

---

## Slide 13 — It's allocation, not capacity — and CE can copy it

The deflationary punchline, delivered honestly. If PAUC merely *represented* the cue better, giving CE the same gradient concentration wouldn't help — but it does. An oracle decoy up-weight (0.767) and a label-free top-score hard-negative-mining CE (0.765) both recover ~90% of the gap toward PAUC (0.792). So the win is a gradient-allocation effect, not something unique to the partial-AUC objective. PAUC's distinct contribution is delivering that concentration adaptively and without labels.

**Details not on the slide**

- A linear probe on penultimate activations scores 0.895 (CE) vs 0.910 (PAUC) — nearly equal, so both models *encode* the cue; they differ only in whether the objective acts on it at the budget. That directly rules out the capacity story.
- Concentration must be *bounded*: crude over-concentration hurts a pointwise loss (oracle ×30 < ×10; ×50→0.702; ×200→0.632), while PAUC's bounded contrast over an adaptive band does not.
- The effect transfers across cue form and budget; label-free HNM-CE matches or beats PAUC in 3 of 4 cells. Be upfront about this — it's the intellectually honest framing.

---

## Slide 14 — One knob that matters: don't let decoys escape the band

The same mechanism explains a failure of the *old* default band. Decoys pile up at the very top, but gradient only reaches [α, β]. The old [budget/2, 1.5·budget] band left 21–41% of decoys escaping *above* t_α with no gradient. Lowering α toward 0 widens the band up to t_α = max(neg), covering those escaped top decoys — worth +0.017 to +0.056 per cell. A full sweep found the robust optimum at α=0, β=budget, which is now the recommended default.

**Details not on the slide**

- Old band captured only 40–48% of decoys; a top-2% miner captures 92–94%.
- The gain concentrates at pos_rate ≪ budget and vanishes once pos_rate ≥ budget, where coverage@budget is mechanically capped at budget/pos_rate.
- Practical takeaway: a label-free HNM-CE and PAUC with a wide-enough band are roughly equivalent implementations of the same gradient-concentration idea.

---

## Slide 15 — When to reach for it — and when not

The decision slide. Reach for it when: your metric is recall at a fixed low budget; the top is contested by hard negatives (use pairwise); and your pool holds far more than 1/β iid negatives. Prefer something else when: the top is cleanly separable (well-weighted CE or SmoothAP is hard to beat); you care about the whole ranking (SmoothAP) or a single threshold (Recall-at-Quantile); or the pool can't supply a stable band-edge quantile.

**Details not on the slide**

- On cleanly separable tops (e.g. Kaggle credit-card fraud, AUROC≈0.97), weighted CE already catches ~84% at 50 bps and all losses land within noise. That's the honest common case for easy data. (Informal observation, not a repo artifact.)
- This slide is where a skeptical practitioner decides — don't oversell.

---

## Slide 16 — Configuration cheat-sheet

A reference slide to photograph. Start with band α=0, β=budget. Contested-negative top → pairwise + pos_numerator="pool". Non-contested top with few live positives and a big queue → trapezoid + pos_numerator="live". Stability → tau_scale="iqr", temperature≈0.1; wide/volatile band → tau_scale="band", temperature≈1.0. Train with CE warmup → blend → PAUC via LossWarmupWrapper. Watch the diagnostics.

**Details not on the slide**

- pos_numerator is surrogate-dependent: "live" rescues trapezoid from gradient dilution; pairwise wants "pool" because restricting its contrast to a few live positives starves it.
- return_diagnostics=True exposes band_neg_count, grad_pos_count, t_alpha, t_beta, tau_eff, pauc_var per class.

---

## Slide 17 — Run it in one command

Make it concrete and runnable. Show the constructor with the recommended settings and the one-command demo. The demo reproduces the favorable regime: weighted CE 0.573 → PAUC pairwise 0.772, a +35% coverage@50bps lift that is seed-stable.

**Details not on the slide**

- Binary mode is num_classes=1, logits [N,1], targets in {0,1}. Multi-class is one-vs-rest.
- The demo (`examples/coverage_at_budget_demo.py`) also runs SmoothAP and both pos_numerator settings so the audience can see the full comparison.
- queue_size stabilizes the tail quantile; at low FPR you need many negatives for a meaningful band edge.

---

## Slide 18 — Limitations — read before adopting

Close the technical content with candor. Weighted CE is a strong baseline and the wins are regime-specific and, on synthetic data, modest and seed-noisy. The surrogate and pos_numerator are both regime-dependent — a real configuration burden. The band edge is tail-quantile sensitive at small pools. And there's no large-scale benchmark validation. Gate adoption on a per-deployment diagnostic and real-data validation.

**Details not on the slide**

- pos_numerator="live" + max_pool_size subsampling are not jointly exercised; the live numerator is intended for the minibatch-with-queue regime.
- Be explicit that "no math errors were found" (a technical review verified the quantile→FPR mapping, the trapezoid normalization, and the detached-gradient structure) — correctness is not the caveat; *regime-specificity* is.

---

## Slide 19 — Takeaways

Three lines to leave them with: optimize the band you deploy on; it wins when the top is contested by a cue CE under-learns (pairwise, wide band, ~+0.21 in the favorable regime); the mechanism is gradient allocation, so a label-free HNM-CE gets you most of the way and PAUC delivers it adaptively without labels. Point to the deep-dive and the reproducible study for anyone who wants the full evidence.

**Details not on the slide**

- If pressed for a one-sentence recommendation: try a hard-negative-mined cross-entropy first; reach for pairwise PAUC when you want that concentration delivered adaptively, without a tuned up-weight and without decoy labels.
