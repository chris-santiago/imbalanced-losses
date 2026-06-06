# Changelog

All notable changes to this project are documented here. Prior releases (≤ 0.3.1)
are available on the [GitHub releases page](https://github.com/chris-santiago/imbalanced-losses/releases).

## Unreleased

*No unreleased changes.*

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
- Corrected misleading "stratified" terminology for minimum-quota pool subsampling.

### Changed

- Extracted the shared memory queue and forward flow into `_MemoryQueue` and the
  `_QueuedRankingLoss` base class (`SmoothAPLoss` and `RecallAtQuantileLoss` now
  build on it).

### Other

- Documentation for `PAUCAtBudgetLoss` and `pos_numerator` across the reference,
  how-to, and explanation pages, plus README and demo-reference updates.
- Scale-invariance, DDP iid-gather, degenerate-dispersion, and `pos_numerator`
  test coverage for `PAUCAtBudgetLoss`.
