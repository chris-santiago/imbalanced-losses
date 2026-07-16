# Changelog

All notable changes to this project are documented here. Prior releases (≤ 0.3.1)
are available on the [GitHub releases page](https://github.com/chris-santiago/imbalanced-losses/releases).

## Unreleased

*No unreleased changes.*

## 0.5.0 — 2026-07-16

### Added

- **`PAUCAtBudgetLoss` gains a `budget_basis` parameter** (`{"fpr", "population"}`,
  default `"fpr"`). It selects what the band edges `t_alpha`/`t_beta` (and the
  scale-aware temperature) are quantiles *of*. `"fpr"` (the default) uses the iid
  negatives only, so `beta` is a false-positive rate — this path is byte-identical
  to prior behavior. `"population"` uses the whole pooled population (positives +
  negatives), so `beta` is a top-k fraction over *all* scores — the deployment
  "alert budget" interpretation that matches a coverage@budget metric ranked over
  the whole population. Only the quantile reference set moves; the pairwise band
  still selects only negatives for the contrast. With `surrogate="trapezoid"`,
  `budget_basis="population"` is approximately `RecallAtQuantileLoss`.

  An 8-seed A/B on synthetic contested-top data found the two bases within seed
  noise at the recommended `alpha=0` band — with `alpha=0` the upper edge already
  spans every top negative, so which population defines the quantile is immaterial —
  and `alpha=0` beats `"population"` at any other band. `"fpr"` therefore remains
  the recommended default; `"population"` is a documented alternative for the
  alert-budget reading, not a coverage win. See the `PAUCAtBudgetLoss` deep dive
  (§5.4) for the ablation.

## 0.4.2 — 2026-06-25

### Fixed

- **`LossWarmupWrapper` no longer fails silently when `on_train_batch_start` is
  omitted.** In epoch mode (and the no-warmup fast path), forgetting to wire the
  per-step hook left temperature scheduling silently disabled: `temp_start`,
  `temp_end`, and `temp_decay_steps` were ignored and `main_loss.temperature`
  never decayed, with no error and a plausible-looking constant in the logs.
  Phase switching still worked, so the failure was invisible. `forward` now emits
  a one-time `UserWarning` when the main phase is active, the main loss exposes a
  `temperature`, and the batch hook has never been called. The batch hook is
  required in epoch mode too, not only step mode.

### Other

- Added a public `lab/pauc_vs_ce_regimes/` study (PAUC-vs-CE operating-point
  regimes) and linked it from the `PAUCAtBudgetLoss` deep-dive. Research material
  only; not part of the shipped package.

## 0.4.1 — 2026-06-06

### Changed

- **`PAUCAtBudgetLoss` default band** is now `alpha=0.0, beta=0.005` (was
  `alpha=0.0025, beta=0.0075`). The new default sets the upper threshold to
  `max(neg_iid)` and the lower edge to the budget quantile, so positives are
  contrasted against every false-positive above the operating point. A band sweep
  (8 seeds, synthetic contested-top data) placed the previous `[budget/2, 1.5·budget]`
  band in a poorly-performing high-`alpha` region: it excludes the highest-scoring
  negatives and extends below the operating threshold. **This changes default
  behavior** — pass `alpha=0.0025, beta=0.0075` explicitly to retain the previous band.

### Fixed

- `PAUCAtBudgetLoss`: avoid a latent `ZeroDivisionError` in the degenerate-dispersion
  warning path when `alpha=0` (the warning message no longer divides by `alpha`).

### Other

- Expanded the `PAUCAtBudgetLoss` deep-dive with the win mechanism (adaptive
  hard-negative mining at the operating point), a CI-backed cue-linearity ablation,
  and the band-escape / `alpha`-lever analysis. Updated the reference, how-to,
  README, and demo to the recommended `alpha=0, beta=budget` band.

## 0.4.0 — 2026-06-05

### Added

- `PAUCAtBudgetLoss`: differentiable partial AUC over a false-positive-rate band
  `[alpha, beta]` around a target operating point (e.g. 50 bps). Trapezoid (default)
  and pairwise surrogates, scale-aware temperature, memory queue, and DDP all-gather.
- `return_diagnostics=True` for `PAUCAtBudgetLoss` — per-class `band_neg_count`,
  `grad_pos_count`, `pauc_var`, `t_alpha`, `t_beta`, `tau_eff` with no extra passes.
- `pos_numerator="live"` for `PAUCAtBudgetLoss` — computes the soft-TPR numerator
  over live-batch positives only, removing the memory queue's gradient dilution at
  extreme imbalance.
- `iid_mask` support in the queued-ranking forward template (backed by a `_q_iid`
  queue buffer), so FPR band edges stay anchored to iid negatives under caller-side
  negative densification.
- `examples/coverage_at_budget_demo.py` — coverage-at-budget comparison demo.

### Fixed

- `PAUCAtBudgetLoss` now skips (marks invalid) a class whose iid-negative score
  dispersion is degenerate (≈ 0), with a one-time warning, instead of producing a
  signal-free or exploding gradient.

### Other

- Documentation for `PAUCAtBudgetLoss` and `pos_numerator` across the reference,
  how-to, and explanation pages, plus README and demo-reference updates.
- Scale-invariance, DDP iid-gather, degenerate-dispersion, and `pos_numerator`
  test coverage for `PAUCAtBudgetLoss`.

## 0.3.2 — 2026-05-20

### Changed

- Extracted the shared memory queue and forward flow into `_MemoryQueue` and the
  `_QueuedRankingLoss` base class (`SmoothAPLoss` and `RecallAtQuantileLoss` now
  build on it).

### Fixed

- Corrected misleading "stratified" terminology for minimum-quota pool subsampling.

### Other

- README: added missing params and demos, documented DDP variable-size support,
  fixed required annotations.
- Linked demo references to GitHub source; added "See also" / "Next steps" demo
  references across the how-to and tutorial pages.
