# Changelog

All notable changes to this project are documented here. Prior releases (≤ 0.3.1)
are available on the [GitHub releases page](https://github.com/chris-santiago/imbalanced-losses/releases).

## Unreleased

### Added

- **Every loss accepts an optional `sample_weight` argument on `forward()`, moving the objective from an item-level count to a value-level sum.** `SigmoidFocalLoss`, `SoftmaxFocalLoss`, `SmoothAPLoss`, `RecallAtQuantileLoss`, and `PAUCAtBudgetLoss` all take a trailing keyword-only `sample_weight`: a non-negative per-observation weight (shape `[N]` for the ranking losses, `targets.shape` for `SoftmaxFocalLoss`, broadcastable to `inputs` for `SigmoidFocalLoss`). The motivating case is a dollar-weighted objective, for example optimizing recall at a fixed alert budget by review value rather than review count, but the mechanism is general to any per-observation weight. See the new [Weight Samples by Value](https://chris-santiago.github.io/imbalanced-losses/how-to/weight-samples/) how-to for a worked example.

  **Semantics:** for the focal losses, count-based reduction denominators become weight-mass denominators (`mean` divides by weight mass over valid elements instead of the valid count; `mean_positive` keeps its RetinaNet numerator-over-all-valid-elements asymmetry and normalizes only by positive weight mass). For the ranking losses, only positives' weights enter the objective (a queue row's weight is the weight it was enqueued with, default `1`), while thresholds, ranks, and the pAUC band stay unweighted count-based order statistics by design. A zero-weight positive therefore contributes nothing to the loss but still occupies its place in ranks and band membership, so the loss is not in general equal to the loss with that row removed. The weight survives every rail it needs to: DDP all-gather (a fourth no-grad collective, issued only when a weight is supplied; an unweighted step issues no extra collective, and all ranks must agree on whether one is present on a given step), the memory queue (persisted across steps and checkpoints; a checkpoint saved before this change loads with `strict=True` and behaves as weight `1`), `ignore_index` filtering, and `max_pool_size` subsampling (which remains weight-*blind* in its row selection; see follow-ups below). `LossWarmupWrapper` forwards `sample_weight` to `main_loss` in every phase, and to `warmup_loss` only when its `forward` declares that parameter.

  **Absent `sample_weight` is unchanged.** When an instance never receives `sample_weight` (and its queue has never held a weighted row), every loss value and gradient is bitwise identical to a release without this feature: no `* weight`, no materialized ones tensor, no extra DDP collective. This is verified by a golden-fixture bitwise-identity sweep across all five losses, both focal reductions, both pAUC surrogates, `pos_numerator` and `budget_basis` values, `queue_size` 0 and > 0 across multiple steps, float32/float64, and a single-process `gloo` replay of the same grid with the real DDP gather call sites forced active.

  **Follow-ups (logged, not built):** weighted budgets/quantiles (`RecallAtQuantileLoss`'s threshold and `PAUCAtBudgetLoss`'s band edges remain unweighted order statistics), weight-aware `subsample_pool` selection (currently uniform-random regardless of weight), and a weighted `pauc_var` diagnostic.

### Fixed

- **`LossWarmupWrapper` no longer silently drops keyword arguments (including `sample_weight`) on the blend path.** `**kwargs` passed to the wrapper's `forward()` (`return_per_class`, `sample_weight`, or any other forwarded argument) previously reached `main_loss` only once `main_weight == 1.0` (the pure main-loss phase); during warmup-ended and blend-phase steps they were silently discarded. They now reach `main_loss` in all three phases. `warmup_loss` is unaffected by this fix: it only ever receives `sample_weight`, and only when its own `forward` declares a parameter of that name (checked once at construction), so third-party warmup losses that don't accept it keep training unweighted with no error.
- **`PAUCAtBudgetLoss`: the trapezoid endpoint knots now resolve to exactly the
  band-edge thresholds.** The first/last knot levels and the band edges are the
  same nominal quantiles (`1 - alpha`, `1 - beta`), but were computed by two
  arithmetic routes: float32 tensor subtraction from a `linspace` for the
  knots, Python-double subtraction then a cast for the edges. For non-dyadic
  `alpha`/`beta` (e.g. `0.9`, `0.85`, `1/3`) the two float32 results differ by
  an ULP-scale amount, and under the non-interpolating
  `quantile_interpolation` modes (including the default `"higher"`) that is
  enough to select an adjacent order statistic. The threshold the trapezoid
  integrated at its endpoint could therefore disagree with the `t_alpha` /
  `t_beta` that define the band (at the defaults with `beta=0.9` on an
  11-sample pool, the last knot resolved a different sample than `t_beta`).
  The endpoint knot levels are now pinned to the band edges' exact bits, which
  makes the disagreement unrepresentable for every pool, pool size, device and
  interpolation mode. **Scope of the numerical change:** only
  float32 + `surrogate="trapezoid"` + a non-dyadic band edge is affected.
  `t_alpha`/`t_beta` (including the `return_diagnostics` values), the default
  band (`alpha=0.0, beta=0.005`), dyadic bands (`0.125`/`0.25`/`0.5`), float64
  scores, and the pairwise surrogate are all byte-identical to 0.5.2. Where a
  config is affected, each endpoint knot threshold moves by at most one
  adjacent order statistic of the reference pool: the level itself moves by
  at most `2^-24` (about 6e-8), so the sampled rank shifts by less than one
  for any pool `torch.quantile` accepts (it refuses inputs above `2^24`
  elements). The resulting loss/gradient change is bounded by that
  inter-sample gap, which can be material where adjacent reference scores
  are far apart (see "Thresholds are order statistics" in the failure-modes
  guide).

## 0.5.2 — 2026-08-10

### Other

- **`PAUCAtBudgetLoss` resolves all of a class's quantiles in one sort.** The band
  edges, the trapezoid knots and the IQR dispersion quantiles were issued as four
  or five separate `torch.quantile` calls, each sorting the same reference
  population independently. They are now packed into a single call, and the
  trapezoid surrogate reuses the resolved vector as its knot thresholds instead of
  recomputing them. Measured on CPU at batch 4096, `queue_size=32768`,
  `num_classes=1`, positive rate 0.5%: trapezoid 12.33 ms → 3.99 ms (3.1×),
  pairwise 11.56 ms → 5.50 ms (2.1×). An independent reproduction on different
  hardware measured 3.7× and 2.9× at the same settings, so treat these as
  conservative; the pairwise ratio in particular falls as the positive rate rises,
  since its `O(|P| × |band|)` term does not shrink. The gain grows with
  `queue_size` and with `num_classes`, so it is largest exactly where the loss was
  previously most expensive. **No API change and no numerical change:** every
  resolved threshold is bitwise unchanged, verified across 79 200 threshold-level
  and 2 400 end-to-end configurations, including index-tie-dense pool sizes where
  the non-interpolating `quantile_interpolation` modes are most sensitive to a
  level change.
- **README parameter table corrections.** `quantile_interpolation` was marked as
  `RecallAtQuantileLoss`-only although `PAUCAtBudgetLoss` has accepted it since
  0.5.0, and `budget_basis` was missing from the table entirely. Documentation
  only.
- **Documented what dominates `PAUCAtBudgetLoss` step time, and how thresholds
  move.** The deep-dive quoted only the surrogate costs and never mentioned the
  threshold quantile, which sorts the reference pool at `O(M log M)` per class and
  dominates both at realistic queue sizes — so the speedup above was
  unexplainable from the docs. Also added a failure mode covering the fact that
  thresholds are order statistics, not smoothly-varying values: the sampled
  position is `level × (n_ref − 1)`, so a threshold depends on the pool size (a
  partially-filled queue resolves a different index than a full one), and near a
  rounding boundary it jumps a whole gap to the adjacent sample.

## 0.5.1 — 2026-07-29

### Fixed

- **`LossWarmupWrapper` now persists its phase and temperature-decay state in
  `state_dict()`.** The wrapper's schedule state (`_epoch`, `_global_step`,
  `_switch_step`, `_batch_hook_seen`) was plain Python, so `nn.Module.state_dict()`
  silently dropped it. On resume the wrapper found no recorded phase-switch step,
  treated the first post-resume batch as the switch, reset `main_loss.temperature`
  back to `temp_start`, and called `reset_queue()` — wiping the memory queue the
  checkpoint had just restored correctly. A mid-blend resume also restarted the
  blend ramp, since `main_weight` reads the lost epoch/step counters. Any run
  resumed from a checkpoint was therefore training with a restarted temperature
  schedule and an empty queue, silently. State is now saved via
  `get_extra_state`/`set_extra_state` under the standard `_extra_state` key.
  Checkpoints written by earlier versions still load under `strict=True` (they
  restart the schedule, exactly as they did before), so no action is required.

### Other

- **Documented that `LossWarmupWrapper` is the only class in the library that
  persists training-progress state.** The `temperature` attribute on
  `SmoothAPLoss`, `RecallAtQuantileLoss` and `PAUCAtBudgetLoss` is a plain float,
  not a registered buffer, so a hand-rolled annealing schedule applied directly to
  `loss.temperature` silently reverts to the constructor value on resume. Warning
  admonitions added to the `LossWarmupWrapper` reference and the failure-modes
  guide, with a matching row in the diagnostic summary table. Tracked in
  [#16](https://github.com/chris-santiago/imbalanced-losses/issues/16).

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
