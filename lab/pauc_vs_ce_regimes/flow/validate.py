# /// script
# requires-python = ">=3.10"
# dependencies = ["metaflow>=2.19"]
# ///
"""Pull the aggregated lift CIs + analysis results from the latest PaucFlow run and
compare against the published reproduction-gate numbers (spec §9). CI-overlap acceptance,
not bit-exact. Run from the repo root: `uv run lab/pauc_vs_ce_regimes/flow/validate.py <experiment>`.
"""
import sys

from metaflow import Flow, Run, namespace

namespace(None)  # see runs regardless of user namespace

# Published targets: experiment -> list of (cue/cell-match dict, budget, lift_mean, lift_lo, lift_hi)
# Acceptance: observed CI overlaps the published CI (or observed mean within published CI).
TARGETS = {
    "cue_ablation": [
        ({"cue": "linear", "budget": 0.005}, 0.015, 0.005, 0.024),
        ({"cue": "nonlinear_prod", "budget": 0.005}, 0.216, 0.183, 0.257),
        ({"cue": "nonlinear_radial", "budget": 0.005}, 0.212, 0.177, 0.247),
        ({"cue": "nonlinear_prod", "budget": 0.010}, 0.045, 0.035, 0.055),
    ],
}


# Mechanism point targets (value, abs-tolerance) — "within seed noise" acceptance.
MECH_TARGETS = {
    "grad_mass_ce": (0.58, 0.08),
    "grad_mass_pauc": (0.96, 0.05),
    "band_decoyfrac_pauc": (0.73, 0.10),
    "probe_auc_ce": (0.895, 0.05),
    "probe_auc_pauc": (0.910, 0.05),
}
MECH_COV_TARGETS = {  # coverage[arm] point targets
    "weighted_ce": (0.576, 0.05),
    "pauc_pairwise": (0.792, 0.05),
}


def check_transfer(run):
    """Reproduction: HNM/oracle (concentrated CE) matches or beats PAUC in 3/4 cells."""
    ar = run["an_transfer"].task.data.an_result
    rows = (ar or {}).get("rows") or []
    print("\n=== Reproduction gate: mechanism_transfer ===")
    matches_or_beats = 0
    for r in rows:
        bvp = r.get("best_vs_pauc")
        mb = bvp is not None and bvp >= -0.01  # matches or beats PAUC
        matches_or_beats += int(mb)
        print(f"  {r.get('cue')} @{r.get('budget')}: CE={r.get('weighted_ce'):.3f} "
              f"HNM={r.get('ce_hnm'):.3f} PAUC={r.get('pauc_pairwise'):.3f}  "
              f"best_vs_PAUC={bvp:+.3f} {'(HNM≥PAUC)' if mb else '(PAUC ahead)'}")
    ok = matches_or_beats >= 3
    print(f"  {'PASS' if ok else 'FAIL'}  HNM matches/beats PAUC in {matches_or_beats}/{len(rows)} cells (target 3/4)")


def check_band_escape(run):
    """Reproduction: decoys in-band 40-48%, top-2% 92-94%, above-band 21-41%."""
    ar = run["an_band_escape"].task.data.an_result
    rows = (ar or {}).get("rows") or []
    print("\n=== Reproduction gate: band_vs_hnm (band escape) ===")
    # (key, lo_ok, hi_ok) acceptance windows spanning the report's 50/100bps ranges
    win = {"in_band": (0.34, 0.54), "in_top2": (0.88, 0.97), "above_band": (0.15, 0.46)}
    allpass = True
    for r in rows:
        line = f"  {r.get('cue')} @{r.get('budget')}: "
        for k, (lo, hi) in win.items():
            m = r[k]["mean"]
            ok = lo <= m <= hi
            allpass = allpass and ok
            line += f"{k}={m:.2f}{'' if ok else '!'}  "
        print(line)
    print(f"  {'ALL PASS' if allpass else 'SOME OUT OF WINDOW'} (in-band 40-48 / top2 92-94 / above 21-41)")


def check_alpha_lever(run):
    """Reproduction: widening alpha->0 improves PAUC every cell (+0.017..+0.056);
    wide-band matches/beats HNM in 3/4 cells."""
    rows = (run["an_alpha_lever"].task.data.an_result or {}).get("rows") or []
    print("\n=== Reproduction gate: alpha_widen ===")
    improved_all = True
    wide_ge_hnm = 0
    for r in rows:
        wms = r.get("wide_minus_std") or {}
        m = wms.get("mean")
        imp = m is not None and m > 0
        improved_all = improved_all and imp
        ge = r.get("cov_pauc_wide") is not None and r.get("cov_hnm") is not None and \
            r["cov_pauc_wide"] >= r["cov_hnm"] - 0.01
        wide_ge_hnm += int(ge)
        print(f"  {r.get('cue')} @{r.get('budget')}: std={r.get('cov_pauc_std'):.3f} "
              f"wide={r.get('cov_pauc_wide'):.3f} hnm={r.get('cov_hnm'):.3f}  "
              f"wide-std={m:+.3f}{'' if imp else '!'}")
    print(f"  {'PASS' if improved_all else 'FAIL'} widening improves PAUC in all cells; "
          f"wide>=HNM in {wide_ge_hnm}/{len(rows)} (target 3/4)")


def check_surrogate(run):
    """Reproduction: at confounder_frac=0, pos 0.15%: trivial .37/CE .58/trap .38/smoothap .72/pairwise .79."""
    rows = (run["an_surrogate"].task.data.an_result or {}).get("rows") or []
    cell = next((r for r in rows if abs(float(r["confounder_frac"]) - 0.0) < 1e-9
                 and abs(float(r["pos_rate"]) - 0.0015) < 1e-6), None)
    print("\n=== Reproduction gate: confounder/surrogate (frac=0, pos 0.15%) ===")
    if not cell:
        print("  headline cell missing"); return
    targets = {"trivial": (0.37, 0.05), "weighted_ce": (0.58, 0.05),
               "pauc_trapezoid": (0.38, 0.05), "smoothap": (0.72, 0.06),
               "pauc_pairwise": (0.79, 0.05)}
    conds = cell["conditions"]
    allpass = True
    for arm, (tgt, tol) in targets.items():
        v = (conds.get(arm) or {}).get("coverage")
        ok = v is not None and abs(v - tgt) <= tol
        allpass = allpass and ok
        print(f"  {'PASS' if ok else 'FAIL'}  {arm:<16} obs={v if v is None else round(v,3)} target={tgt}±{tol}")
    print(f"  {'ALL PASS' if allpass else 'SOME FAILED'}")


def check_default_sweep(run):
    """Reproduction: robust optimum band alpha=0, beta=budget (alpha_m=0, beta_m=1.0)."""
    ar = run["an_default_sweep"].task.data.an_result or {}
    ar = ar.get("result", ar)
    robust = ar.get("robust_band") or ar.get("robust") or {}
    print("\n=== Reproduction gate: band_default_sweep ===")
    am, bm = robust.get("alpha_m"), robust.get("beta_m")
    # Accept: beta=budget (beta_m=1.0) unanimous and alpha near 0 is the per-pos optimum.
    per_pos = ar.get("per_pos") or []
    beta_budget = all(p.get("best", {}).get("beta_m") == 1.0 for p in per_pos) if per_pos else False
    alpha_low = sum(p.get("best", {}).get("alpha_m", 1) <= 0.1 for p in per_pos)
    ok = (am == 0.0 and bm == 1.0) or (beta_budget and alpha_low >= len(per_pos) - 1)
    print(f"  robust optimum band: alpha_m={am} beta_m={bm}; per-pos beta=budget unanimous={beta_budget}, "
          f"alpha<=0.1 in {alpha_low}/{len(per_pos)}  {'PASS' if ok else 'FAIL'} (target alpha=0, beta=budget)")
    for p in per_pos:
        b = p.get("best", {})
        print(f"  pos={p.get('pos_rate')}: best a{b.get('alpha_m')}_b{b.get('beta_m')} cov={b.get('cov'):.3f} "
              f"hnm={p.get('cov_hnm'):.3f}")


def overlaps(lo1, hi1, lo2, hi2):
    return not (hi1 < lo2 or hi2 < lo1)


def check_mechanism(run):
    ar = run["an_mechanism"].task.data.an_result
    if not ar:
        print("  no mechanism result"); return
    ar = ar.get("result", ar)  # metrics nested under "result"
    print("\n=== Reproduction gate: mechanism_probe (point ± tol) ===")
    allpass = True
    for k, (tgt, tol) in MECH_TARGETS.items():
        v = ar.get(k)
        ok = v is not None and abs(v - tgt) <= tol
        allpass = allpass and ok
        print(f"  {'PASS' if ok else 'FAIL'}  {k:<22} obs={v if v is None else round(v,3)}  target={tgt}±{tol}")
    cov = ar.get("coverage", {})
    for arm, (tgt, tol) in MECH_COV_TARGETS.items():
        v = cov.get(arm)
        ok = v is not None and abs(v - tgt) <= tol
        allpass = allpass and ok
        print(f"  {'PASS' if ok else 'FAIL'}  coverage[{arm}]      obs={v if v is None else round(v,3)}  target={tgt}±{tol}")
    print(f"  coverage(all arms): { {a: round(c,3) for a,c in cov.items()} }")
    print(f"  gap_vs_pauc: { {a: round(g,3) for a,g in (ar.get('gap_vs_pauc') or {}).items()} } (oracle/HNM ~ -0.025 expected)")
    print(f"\n  {'ALL PASS' if allpass else 'SOME FAILED'}")


def cell_matches(cell, want):
    return all(str(cell.get(k)) == str(v) for k, v in want.items())


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    run = Run(sys.argv[2]) if len(sys.argv) > 2 else Flow("PaucFlow").latest_run
    print(f"Run: {run.pathspec}  finished={run.finished}  successful={run.successful}")

    agg = run["aggregate"].task.data
    lift_results = agg.lift_results
    aggregate_results = agg.aggregate_results

    def hkey(cell):
        return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                            for k, v in cell.items()))

    # coverage means per (cell, arm)
    cov = {}
    for r in aggregate_results:
        cov[(hkey(r["cell"]), r["arm"])] = r

    print("\n=== Paired lifts (pauc_pairwise - weighted_ce) ===")
    for lr in sorted(lift_results, key=lambda x: str(x["cell"])):
        c = lr["cell"]
        ckey = hkey(c)
        ce = cov.get((ckey, "weighted_ce"), {}).get("coverage_mean")
        pa = cov.get((ckey, "pauc_pairwise"), {}).get("coverage_mean")
        ce_s = f"{ce:.3f}" if ce is not None else "?"
        pa_s = f"{pa:.3f}" if pa is not None else "?"
        star = "*" if lr["ci_excludes_zero"] else " "
        print(f"  {str(c):<48} CE={ce_s} PAUC={pa_s}  lift={lr['lift_mean']:+.3f} "
              f"[{lr['lift_lo']:+.3f},{lr['lift_hi']:+.3f}]{star} (n={lr['n_seeds']})")

    if exp == "mechanism_probe":
        check_mechanism(run)
    if exp == "mechanism_transfer":
        check_transfer(run)
    if exp == "band_vs_hnm":
        check_band_escape(run)
    if exp == "alpha_widen":
        check_alpha_lever(run)
    if exp == "confounder_sweep":
        check_surrogate(run)
    if exp == "band_default_sweep":
        check_default_sweep(run)

    if exp and exp in TARGETS:
        print(f"\n=== Reproduction gate: {exp} (CI-overlap acceptance) ===")
        allpass = True
        for want, tm, tlo, thi in TARGETS[exp]:
            match = [lr for lr in lift_results if cell_matches(lr["cell"], want)]
            if not match:
                print(f"  MISSING  {want}  (target {tm:+.3f} [{tlo:+.3f},{thi:+.3f}])")
                allpass = False
                continue
            lr = match[0]
            ok_overlap = overlaps(lr["lift_lo"], lr["lift_hi"], tlo, thi)
            ok_point = tlo <= lr["lift_mean"] <= thi
            ok = ok_overlap or ok_point
            allpass = allpass and ok
            print(f"  {'PASS' if ok else 'FAIL'}  {want}  "
                  f"obs {lr['lift_mean']:+.3f} [{lr['lift_lo']:+.3f},{lr['lift_hi']:+.3f}]  "
                  f"vs target {tm:+.3f} [{tlo:+.3f},{thi:+.3f}]")
        print(f"\n  {'ALL PASS' if allpass else 'SOME FAILED'}")

    # dump the experiment's analysis result if present
    for stepname in [s.id for s in run if s.id.startswith("an_")]:
        try:
            ar = run[stepname].task.data.an_result
        except Exception:
            continue
        if ar and (not exp or ar.get("experiment") == exp):
            print(f"\n=== analysis [{stepname}] {ar.get('experiment')} ===")
            for row in (ar.get("rows") or [])[:20]:
                print("  ", row)


if __name__ == "__main__":
    main()
