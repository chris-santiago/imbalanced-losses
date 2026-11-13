# Changelog

All notable changes to this project are documented here. Prior releases (≤ 0.3.1)
are available on the [GitHub releases page](https://github.com/chris-santiago/imbalanced-losses/releases).

## Unreleased

### Changed

- **`SoftmaxFocalLoss` raises `ValueError` when `alpha`'s length differs from the number of classes.** Previously a short `alpha` was silently clamped (every class index past its end received the last entry's weight) and a long one had its extra entries ignored, so a misconfigured class weighting trained with no signal that anything was wrong. The check runs in `forward`, since the class count is not known at construction. A correct-length `alpha` and `alpha=None` are unchanged.
- **`SmoothAPLoss` and `PAUCAtBudgetLoss` raise `ValueError` on multiclass targets outside `[0, num_classes)`.** `RecallAtQuantileLoss` already did; the other two accepted such a row and treated it as a negative for every class, which distorts every class's pool. The check now lives in the shared base class, keeps the same message, and runs on the full pool before `max_pool_size` subsampling. **This is an observable behavior change for callers that passed out-of-range class ids to those two losses:** mark such rows with `ignore_index` to exclude them. Rows already held in the memory queue are checked too, so a queue restored from a checkpoint that contains such rows raises on every call until `reset_queue()` is called; the error message names `reset_queue()` when a queued row is the offender, and `ignore_index` when a row in the current batch is. **Private API note:** the hook `_QueuedRankingLoss._validate_filtered_targets` now carries this rule by default (it was a no-op that only `RecallAtQuantileLoss` overrode) and is called with a keyword argument `is_live`. An out-of-library subclass that overrides it must accept that keyword, and should call `super()` to keep the range check. Binary mode (`num_classes=1`) is unchanged: targets outside `{0, 1}` still warn and count as positive.
- **`SigmoidFocalLoss` and `SoftmaxFocalLoss` reject a negative or non-finite `gamma` at construction.** `gamma` must be finite and `>= 0`. A negative value produced an infinite or NaN loss at saturated predictions, and `nan` or `inf` produced meaningless ones. `gamma=0` (plain BCE/CE) and fractional values remain valid.

- **Under DDP, gathered losses now apply the full-batch gradient: `all_gather_with_grad` scales the local slice's gradient by `world_size`.** Each rank differentiates the gathered global loss through its own slice only, and `DistributedDataParallel` averages parameter gradients across ranks, so every update was `1/world_size` of the gradient a single process would compute on the combined batch (measured in a two-process run: a gradient-norm ratio of 0.5 at two ranks). The documentation said DDP summed the per-slice gradients; it averages them. `all_gather_with_grad` now takes `scale_grad=True` and scales the gradient flowing into the local slice with an identity-forward autograd function. Loss values are bitwise unchanged, and nothing changes at `world_size == 1`. This affects all five losses whenever they gather, which is the default under an initialized process group. **This is an observable behavior change: effective gradients under DDP are `world_size` times larger.** If you compensated for the old scale (a learning rate multiplied by the number of GPUs for this reason, or a loss multiplied by `world_size`), remove the compensation or you will overshoot. To retain the previous behavior, construct the loss with `gather_distributed=False` and gather manually with `all_gather_with_grad(logits, scale_grad=False)`; use the same switch if your setup sums gradients across ranks instead of averaging them.
- **`LossWarmupWrapper` schedules temperature only when asked: `temp_start` and `temp_end` now default to `None`.** They used to default to `0.05` and `0.005`, so the wrapper overwrote `main_loss.temperature` at the phase switch and decayed it on every step whether or not a schedule was wanted. A `PAUCAtBudgetLoss(temperature=0.1)`, whose temperature is a dimensionless multiplier, was silently trained at 0.05 decaying to 0.005. With neither argument given, the wrapper never writes `main_loss.temperature`: not at construction, not at the switch, not afterwards, and not when a checkpoint is loaded (even one written by a scheduling wrapper). With both given, behavior is identical to before. Passing exactly one raises `ValueError`. Two warnings change condition to match: the constructor's "no `temperature` attribute" warning is emitted only when a schedule was requested, and the forward-time "`on_train_batch_start()` was never called" warning is emitted when the missing hook disables a requested schedule or the switch-time queue reset, and its text now names which. **This is an observable behavior change for every caller that relied on the implicit schedule.** To retain the previous behavior, pass `temp_start=0.05, temp_end=0.005`.
- **With `max_pool_size` active, `RecallAtQuantileLoss` and `PAUCAtBudgetLoss` take their thresholds from the full pool before subsampling.** The subsampler reserves a quota per class and drops mostly majority rows, so the subsample's score distribution is not the pool's, and thresholds estimated on it sat at a different operating point from the one configured. In a 4,000-row binary pool with 20 positives, `quantile=0.05` and a cap of 512, the capped recall loss differed from the uncapped one by 0.06. The recall threshold, and pAUC's band edges, trapezoid knots and dispersion scale (hence `tau_eff` and the reported diagnostics), are now computed on the full `ignore_index`-filtered pool, and the loss is computed over the subsample against them. The cost is one sort of the full pool per class per forward. `SmoothAPLoss` has no thresholds and is unchanged, as is every call where the cap does not trigger. Three consequences: under DDP the thresholds are now identical on every rank (each rank still draws its own subsample); a pAUC class whose subsample happens to hold no iid negatives is valid as long as the full pool has them; and with `surrogate="pairwise"` the band negatives are the subsample's negatives that fall inside the full-pool band, so a tight cap can leave a class with none and skip it for that step (prefer the trapezoid surrogate or a larger cap). **Loss values change whenever the cap triggers, and there is no option to restore the previous behavior.**
- **The memory queue stores only rows whose target is not `ignore_index`.** The whole batch used to be enqueued and padding rows were filtered out again when the pool was built, so on padded sequence batches most queue slots held rows that could never be used: with 256-row batches of which 230 are padding, a 1,024-slot queue held about 104 usable rows where it now holds 1,024. Rows are stored in their original order with their iid flags and weights, the filter is applied to the gathered batch so DDP ranks stay in sync, and wholesale replacement now triggers on the number of valid rows. No buffer is added or renamed, so checkpoints load unchanged; padding rows already in a saved queue are filtered at merge as before. A batch with no ignored rows is enqueued exactly as before. **Loss values change after the first steps of any run that has `ignore_index` rows and `queue_size > 0`, and there is no option to restore the previous behavior.** The bitwise golden fixture was regenerated for this change; only the third-step entries of its ten `queue_size=64` configurations differ, which `tests/_golden_sample_weight.py --compare` checks.

  **Private API note:** `_QueuedRankingLoss` gained an `_estimate_thresholds(pool)` hook (default `None`), called only when subsampling is about to occur, and `_compute_per_class` gained an optional `thresholds` keyword that is passed only when the hook returned a value, so a subclass that overrides neither keeps working. `_validate_filtered_targets` now also owns the binary-mode target warning, the per-class computes no longer check targets, and the reduction lives in one `_reduce` method.

### Fixed

- **The binary-mode target warning points at your code and no longer depends on the subsample.** With `num_classes=1`, a non-ignored target outside `{0, 1}` counts as positive and emits a `UserWarning`. The check ran after `max_pool_size` subsampling, so a lone bad target was reported only when the subsample happened to keep it (roughly two calls in three with a cap of 2 in a 400-row pool), and its fixed stack level attributed it to a frame inside the library or torch on some call paths. The check now sits in the shared base class beside the multiclass range check: it sees the full pool before subsampling, fires once per forward, and is attributed to the first frame outside the library and torch whether you call the loss, call `.forward()` directly, or go through `LossWarmupWrapper`. The message text is unchanged, and such targets still count as positive.
- **`RecallAtQuantileLoss` and `PAUCAtBudgetLoss` accept pools above 16,777,216 rows.** `torch.quantile` refuses inputs above `2**24` elements, which a large flattened pool can exceed (and which the full-pool threshold estimate above would otherwise hit even with `max_pool_size` set). Above that size the quantiles are read from one sort with the same conventions; at or below it `torch.quantile` is called exactly as before, so existing results do not move. The sort route computes the rank in float64, so for float32 scores it can select the order statistic next to the one `torch.quantile`'s float32 rank arithmetic would have picked, a difference of one rank in more than sixteen million.
- **Documentation corrections.** `n_knots=2` was described as "accurate for narrow bands"; on the default `alpha=0` band it under-estimates the pAUC by about 2% at a 1,280-row pool and by 25 to 30% at 20,000 to 200,000 rows, because the first knot is the maximum negative (five knots are within about 4%, nine within about 1%; the default is unchanged). The pages now also state that a queue frozen in eval mode is still merged into the pool and give one instruction for a validation-only value (a separate instance with `queue_size=0`), where two pages previously disagreed about resetting before validation. The manual-gather examples now construct the loss with `gather_distributed=False`; as written they gathered every row twice under DDP.
- **`SigmoidFocalLoss` and `SoftmaxFocalLoss` no longer return NaN gradients when `0 < gamma < 1`.** The modulator `(1 - p_t) ** gamma` has an infinite derivative at `p_t = 1`, which a float32 prediction reaches at a logit margin of roughly 17, and autograd turned that into NaN for the element (sigmoid) or the whole row (softmax). One confidently classified easy example was enough to poison every parameter on the next optimizer step. For that range of `gamma` the modulator is now guarded at exact saturation, where its gradient is set to the analytic limit of zero. **Scope of the numerical change:** only `0 < gamma < 1` is affected, and only at elements whose `1 - p_t` is exactly zero. Every loss value, and every gradient that was finite before, is bitwise unchanged, including the default `gamma=2`.
- **`SmoothAPLoss`, `RecallAtQuantileLoss`, and `PAUCAtBudgetLoss` accept float16 and bfloat16 logits.** `RecallAtQuantileLoss` and `PAUCAtBudgetLoss` raised `RuntimeError: quantile() input tensor must be either float or double dtype` on half-precision logits, because `torch.quantile` has no half-precision kernel. That made them unusable in a mixed-precision region wherever the backend does not promote `quantile` itself (reproduced with explicit half-precision inputs and under MPS autocast; CPU autocast already promoted it). Half-precision logits are now computed in float32 and the loss is returned as float32, with gradient still reaching the caller's tensor in its own dtype. **This is an observable change wherever a loss already ran on half-precision logits.** `SmoothAPLoss` always did, computing its rank sums in half precision, and under CPU autocast (where `quantile` was already promoted) `RecallAtQuantileLoss` ran too and returned a bfloat16 loss. Both now return a float32 loss whose value is slightly different and more accurate (for `SmoothAPLoss` at a pool of 4,096 the old result was off by about `1.6e-4` in float16 and `6.5e-4` in bfloat16). `PAUCAtBudgetLoss` under CPU autocast already returned float32 and its value is unchanged there. float32 and float64 inputs are bitwise unchanged, and float64 is never downcast.
- **An empty pool under `reduction="none"` returns `nan` per class, not zeros.** When every row is `ignore_index` (or the batch has zero rows and the queue is empty), the ranking losses returned a `[C]` vector of zeros, while a non-empty pool whose classes were all degenerate returned `nan`, which is what the README and reference pages document. Zeros read as a perfect per-class loss in logging. Both cases now return `nan`. **This is an observable behavior change for callers that reduce the `"none"` output themselves:** on an all-padding batch the vector is now `nan` with no gradient path, exactly as it already was for an all-degenerate pool, where it used to be zeros that supported `.backward()`. `reduction="mean"` and `"sum"` are unchanged (`0.0` with a gradient path), as are the per-class vector and validity mask from `return_per_class=True`; use those if a training loop must tolerate all-padding batches.
- **README no longer lists `ignore_index` masking under `SigmoidFocalLoss`.** The focal feature list attributed `ignore_index` masking and a valid-count `mean` to both focal losses. Only `SoftmaxFocalLoss` has them; `SigmoidFocalLoss` takes no `ignore_index` and averages over every element.

## 0.6.0 — 2026-09-22

### Added

- **Every loss accepts an optional `sample_weight` argument on `forward()`, moving the objective from an item-level count to a value-level sum.** `SigmoidFocalLoss`, `SoftmaxFocalLoss`, `SmoothAPLoss`, `RecallAtQuantileLoss`, and `PAUCAtBudgetLoss` all take a trailing keyword-only `sample_weight`: a non-negative, finite per-observation weight (shape `[N]` for the ranking losses, `targets.shape` for `SoftmaxFocalLoss`, and for `SigmoidFocalLoss` a shape whose dim 0 equals `inputs.size(0)` and whose trailing dims broadcast to `inputs`). A negative, `NaN`, or `inf` entry raises `ValueError`; the weight is detached and cast to the loss dtype, so a `float64` weight never promotes a `float32` loss. The motivating case is a dollar-weighted objective, for example optimizing recall at a fixed alert budget by review value rather than review count, but the mechanism is general to any per-observation weight. See the new [Weight Samples by Value](https://chris-santiago.github.io/imbalanced-losses/how-to/weight-samples/) how-to for a worked example.

  **Semantics:** for the focal losses, count-based reduction denominators become weight-mass denominators (`mean` divides by weight mass over valid elements instead of the valid count; `mean_positive` keeps its RetinaNet numerator-over-all-valid-elements asymmetry and normalizes only by positive weight mass). For the ranking losses, only positives' weights enter the objective (a queue row's weight is the weight it was enqueued with, default `1`), while thresholds, ranks, and the pAUC band stay unweighted count-based order statistics by design. A zero-weight positive therefore contributes nothing to the loss but still occupies its place in ranks and band membership, so the loss is not in general equal to the loss with that row removed. The weight survives every rail it needs to: DDP all-gather (a fourth no-grad collective, issued only when a weight is supplied; an unweighted step issues no extra collective, and all ranks must agree on whether one is present on a given step), the memory queue (persisted across steps and checkpoints; a checkpoint saved before this change loads with `strict=True` and behaves as weight `1`), `ignore_index` filtering, and `max_pool_size` subsampling (which remains weight-*blind* in its row selection; see follow-ups below). `LossWarmupWrapper` forwards every keyword argument to `main_loss` in every phase. `warmup_loss` receives only `sample_weight`, and only when its `forward` declares that parameter by name or declares `**kwargs`; no other keyword argument reaches it in any phase, because the blend step does scalar arithmetic on its return value and must not be handed a tuple. The wrapper's return shape therefore tracks `main_loss` whenever `main_loss` is called (blend and main phases); during pure warmup the wrapper returns `warmup_loss`'s scalar and return-shape arguments have no effect. One shape to be aware of when logging per-class metrics: mid blend, `return_per_class=True` returns a blended scalar alongside `main_loss`'s *unblended* `per_class`/`valid` elements, since there is no warmup-side per-class counterpart, so the scalar is not the reduction of the returned vector during the blend phase.

  **Absent `sample_weight` is unchanged.** When an instance never receives `sample_weight` (and its queue has never held a weighted row), every loss value and gradient is bitwise identical to a release without this feature: no `* weight`, no materialized ones tensor, no extra DDP collective. This is verified by a golden-fixture bitwise-identity sweep across all five losses, both focal reductions, both pAUC surrogates, `pos_numerator` and `budget_basis` values, `queue_size` 0 and > 0 across multiple steps, float32/float64, and a single-process `gloo` replay of the same grid with the real DDP gather call sites forced active.

  **Private API note:** the private `_compute_per_class` hook on `_QueuedRankingLoss` gained a `sample_weight` argument, and the private `_MemoryQueue.merge` / `subsample_pool` helpers now exchange a `PooledBatch` named tuple instead of variable-arity tuples. Out-of-library subclasses that override `_compute_per_class` (unsupported, per the spec) need the new parameter.

  **Follow-ups (logged, not built):** weighted budgets/quantiles (`RecallAtQuantileLoss`'s threshold and `PAUCAtBudgetLoss`'s band edges remain unweighted order statistics), weight-aware `subsample_pool` selection (currently uniform-random regardless of weight), and a weighted `pauc_var` diagnostic. Weighted budgets are tracked in [#18](https://github.com/chris-santiago/imbalanced-losses/issues/18).

### Fixed

- **`LossWarmupWrapper` no longer silently drops keyword arguments (including `sample_weight`) on the blend path.** `**kwargs` passed to the wrapper's `forward()` (`return_per_class`, `sample_weight`, or any other forwarded argument) previously reached `main_loss` only once `main_weight == 1.0` (the pure main-loss phase); during warmup-ended and blend-phase steps they were silently discarded. They now reach `main_loss` in all three phases. **This is an observable behavior change for callers that passed keyword arguments during blend:** `iid_mask` now shapes the blend-phase thresholds it was silently excluded from, and a call with `return_per_class=True` (or `return_diagnostics=True`) during blend now returns the tuple it returns in the main phase instead of a bare scalar; if you relied on the old silent drop, stop passing those arguments during blend. Calls that pass no keyword arguments are unchanged. `warmup_loss` is unaffected by this fix: it only ever receives `sample_weight`, and only when its own `forward` declares a parameter of that name or declares `**kwargs` (checked once at construction), so third-party warmup losses that don't accept it keep training unweighted with no error.
- **`PAUCAtBudgetLoss`: the trapezoid endpoint knots now resolve to exactly the band-edge thresholds.** The first/last knot levels and the band edges are the same nominal quantiles (`1 - alpha`, `1 - beta`), but were computed by two arithmetic routes: float32 tensor subtraction from a `linspace` for the knots, Python-double subtraction then a cast for the edges. For non-dyadic `alpha`/`beta` (e.g. `0.9`, `0.85`, `1/3`) the two float32 results differ by an ULP-scale amount, and under the non-interpolating `quantile_interpolation` modes (including the default `"higher"`) that is enough to select an adjacent order statistic. The threshold the trapezoid integrated at its endpoint could therefore disagree with the `t_alpha` / `t_beta` that define the band (at the defaults with `beta=0.9` on an 11-sample pool, the last knot resolved a different sample than `t_beta`). The endpoint knot levels are now pinned to the band edges' exact bits, which makes the disagreement unrepresentable for every pool, pool size, device and interpolation mode. **Scope of the numerical change:** only float32 + `surrogate="trapezoid"` + a non-dyadic band edge is affected. `t_alpha`/`t_beta` (including the `return_diagnostics` values), the default band (`alpha=0.0, beta=0.005`), dyadic bands (`0.125`/`0.25`/`0.5`), float64 scores, and the pairwise surrogate are all byte-identical to 0.5.2. Where a config is affected, each endpoint knot threshold moves by at most one adjacent order statistic of the reference pool: the level itself moves by at most `2^-24` (about 6e-8), so the sampled rank shifts by less than one for any pool `torch.quantile` accepts (it refuses inputs above `2^24` elements). The resulting loss/gradient change is bounded by that inter-sample gap, which can be material where adjacent reference scores are far apart (see "Thresholds are order statistics" in the failure-modes guide).

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
