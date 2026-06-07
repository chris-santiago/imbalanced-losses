# When does `PAUCAtBudgetLoss` beat well-tuned cross-entropy?

**One-line answer.** PAUC-pairwise beats well-tuned cross-entropy on **coverage@budget** (recall at a
fixed low false-positive budget) **when the operating point is contested by a cue CE under-learns —
specifically a nonlinear, rarely-relevant cue.** There the win is large (**+0.21 coverage ≈ +38%** at
50 bps, bootstrap-CI-separated); when the cue is linear (CE captures it), the top is uncontested, the
budget is too tight, or the positive rate is high, the advantage **vanishes**.

> **Where to start.** The headline is the **nonlinear-cue** regime, established by a controlled
> ablation. See `reports/TECHNICAL_REPORT.md` for the full results, mechanism, and limitations.

> **Reproducibility.** The canonical reproducer is the Metaflow pipeline at `flow/pauc_flow.py`
> (config in `conf/`, validator at `flow/validate.py`). Run a slice with:
> `uv run flow/pauc_flow.py --config-value cfg "hydra_overrides: [experiment=<name>]" run`
> (capacity also needs `geometry=hard_bulk,training=capacity`; confounder needs `geometry=confounder`).
> Results are Metaflow run artifacts — query via `Run('PaucFlow/<id>')` or validate with
> `flow/validate.py <experiment>`. The 8 reproduction slices are: `cue_ablation`, `mechanism_probe`,
> `mechanism_transfer`, `band_vs_hnm`, `alpha_widen`, `band_default_sweep`, `capacity_warmup`,
> `confounder_sweep`. The pipeline reproduced 7/8 slices' published numbers within CI/seed-noise;
> capacity's forensic sub-studies are intentionally not reproduced (see `reports/TECHNICAL_REPORT.md` §2.5).

## The decisive result (controlled cue ablation, 8 seeds, bootstrap CIs)

The `cue_ablation` slice (`conf/experiment/cue_ablation.yaml`), pos 0.15%, MLP, confounder = 0. All cues use the **same
two discriminative features**; only the cue's functional form varies:

| cue (50 bps) | CE cov | PAUC-pairwise | lift [95% CI] |
|---|---:|---:|---|
| **linear** | 0.864 | 0.880 | **+0.015** [+0.005, +0.024] |
| **nonlinear, product (`f5·f6`)** | 0.575 | 0.791 | **+0.216** [+0.183, +0.257] |
| **nonlinear, radial (distinct nonlinearity)** | 0.426 | 0.638 | **+0.212** [+0.177, +0.247] |

A **linear** cue (+0.015, CE captures it) versus two **nonlinear** cues (~+0.21) — same two features
throughout — shows the cue's **linearity** is the determinant, and the effect replicates across two
distinct nonlinear forms. See `figures/fig_headline_cue_nonlinearity.png`.

**Mechanism.** A *linear* cue is cheap for CE to capture, so CE already covers the top (little to
recover). A *nonlinear* cue relevant only to the rare decoys at the operating point is one CE's
bulk-dominated gradient never invests in; PAUC's band-focused gradient drives the model to learn it.
The advantage is operating-point-specific (AUROC stays ~0.99), carried only by the **pairwise**
surrogate (trapezoid collapses), requires `pos_numerator="pool"` and a model with enough capacity to
represent the cue, and is **budget-dependent** (falls to +0.045 at 100 bps; see `figures/fig_headline_budget.png`).

## Quickstart

```bash
# Run the decisive ablation via the Metaflow pipeline (results stored as run artifacts)
uv run flow/pauc_flow.py --config-value cfg "hydra_overrides: [experiment=cue_ablation]" run
# Validate against published numbers
uv run flow/validate.py cue_ablation

# Run the Cycle-3 confounder sweep
uv run flow/pauc_flow.py --config-value cfg "hydra_overrides: [experiment=confounder_sweep,geometry=confounder]" run
```

## Where PAUC does NOT help
Linear / cheap cue (CE captures it) · uncontested top · tight budget (10 bps — band too sparse to
anchor) · high positive rate · whole-curve metrics (AUROC unaffected; use CE or Smooth-AP there).

## Artifacts
- **Pipeline (canonical reproducer):** `flow/pauc_flow.py` (self-contained FlowSpec + components),
  `conf/` (Hydra config groups: `experiment/`, `geometry/`, `arm/`, `training/`), `flow/validate.py`.
  Results live in the Metaflow datastore — query via `Run('PaucFlow/<id>')` or `flow/validate.py`.
- **Figures:** `figures/fig_headline_*`, `figures/fig_confounder_sweep.png`, etc.
- **Report:** `reports/TECHNICAL_REPORT.md` — self-contained writeup covering methods, results,
  mechanism, deployment, and limitations.

**Caveat:** all evidence is synthetic. Adoption should be gated on a real-data diagnostic — does
well-tuned CE actually leave coverage@budget unrealized at your operating point? See `reports/TECHNICAL_REPORT.md`
§4 (Deployment) and §5 (Limitations).
