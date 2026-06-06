"""
Coverage at a budget: recovering the operating point weighted CE leaves behind.

The metric you deploy on for an alerting/review workload is *coverage at a budget*:
if you can review the top 0.5 % of scores (50 bps), what fraction of positives do
you catch? That is recall in the top 0.5 % of scores -- decided at the very top of
the ranking. ``PAUCAtBudgetLoss`` optimizes the partial AUC over a band around that
operating point, instead of the whole curve (AUCPR/SmoothAP) or a single threshold.

The data has a *contested top*: hard "decoy" negatives sit at the operating point,
separable from true positives only by a weak non-linear cue, plus "confounder"
negatives that carry that cue without the easy signal (so over-using the cue costs
AUCPR). Weighted CE spreads its gradient across the whole distribution, protects
the bulk, and under-resolves the contested top -- it leaves coverage@50bps on the
table (cov ~0.57 here). The band loss that contrasts positives directly against the
near-threshold decoys recovers it.

What the 6-row table shows
--------------------------
* **PAUC pairwise** wins coverage@50bps by a wide margin (~+35% over weighted CE,
  seed-stable). The pairwise surrogate *pushes the band negatives down*, which is
  exactly what a decoy-contested top needs.
* **PAUC trapezoid collapses** on this data: it only lifts positives toward a
  frozen threshold and never suppresses band negatives, so it is the wrong
  surrogate when the top is contested by hard *negatives*. (Trapezoid is the right
  choice when the top is contested by hard *positives* -- see PAUCAtBudgetLoss docs.)
* **pos_numerator is surrogate-dependent.** ``"pool"`` keeps many positives in the
  pairwise positive x band-negative contrast and stabilizes it; ``"live"`` starves
  it here. Conversely ``"live"`` de-dilutes the *trapezoid* numerator when the queue
  swamps the few live positives (see the easy-top regime in the docs). The table
  prints both so the interaction is explicit.
* **SmoothAP** is a strong whole-curve baseline.

Metrics are averaged over several seeds because coverage at 50 bps is estimated from
few validation positives and is noisy per-seed.

Usage
-----
    python examples/coverage_at_budget_demo.py
    python examples/coverage_at_budget_demo.py --n-seeds 8 --epochs 20
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from imbalanced_losses import LossWarmupWrapper, PAUCAtBudgetLoss, SmoothAPLoss


# ---- data -------------------------------------------------------------------


def make_imbalanced_data(
    pos_rate: float = 0.0015,
    n_samples: int = 200_000,
    n_features: int = 20,
    decoy_frac: float = 0.012,
    confounder_frac: float = 0.03,
    easy: float = 2.5,
    sep: float = 1.7,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Synthetic extreme-imbalance data with a top contested by hard negatives.

    * **Easy signal** (features ``[0:3]``) lifts positives *and* decoy negatives to
      the top; it drives AUCPR but cannot separate them.
    * **Non-linear separator** (product of features ``[5]`` and ``[6]``) is the only
      cue that distinguishes positives from decoys at the top.
    * **Decoy negatives** (``decoy_frac``) share the easy signal, so they crowd the
      operating point; only the separator pushes them below positives.
    * **Confounder negatives** (``confounder_frac``) carry the separator signal
      *without* the easy signal, so leaning on the separator lifts them in the
      mid-range and costs AUROC/AUCPR. This is what makes weighted CE hold back
      from the separator (protecting the bulk) and lose coverage at the top.
    """
    rng = np.random.default_rng(seed)
    N, F = n_samples, n_features
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = np.zeros(N, dtype=np.float32)

    n_pos = max(8, round(pos_rate * N))
    pos = rng.choice(N, size=n_pos, replace=False)
    y[pos] = 1.0

    neg = np.where(y == 0.0)[0]
    decoy = rng.choice(neg, size=round(decoy_frac * len(neg)), replace=False)
    rest = np.setdiff1d(neg, decoy, assume_unique=True)
    conf = rng.choice(rest, size=round(confounder_frac * len(neg)), replace=False)

    # Easy signal: positives and decoys both rise to the top.
    X[np.ix_(pos, [0, 1, 2])] += easy
    X[np.ix_(decoy, [0, 1, 2])] += easy

    # Non-linear separator (f5*f6, same-sign) marks true positives...
    sp = rng.choice([-1.0, 1.0], size=n_pos).astype(np.float32)
    X[pos, 5] += sep * sp
    X[pos, 6] += sep * sp
    # ...and confounder negatives carry the SAME separator without the easy signal.
    sc = rng.choice([-1.0, 1.0], size=len(conf)).astype(np.float32)
    X[conf, 5] += sep * sc
    X[conf, 6] += sep * sc
    # Decoys keep f5,f6 ~ noise (product ~ 0): only the separator beats them.

    perm = rng.permutation(N)
    n_va = N // 4
    va, tr = perm[:n_va], perm[n_va:]
    return (
        torch.tensor(X[tr]),
        torch.tensor(y[tr]).unsqueeze(1),  # [N, 1]
        torch.tensor(X[va]),
        y[va],  # numpy [N] for sklearn metrics
    )


# ---- model ------------------------------------------------------------------


def make_model(in_dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


# ---- metrics ----------------------------------------------------------------


def coverage_at_budget(y_true: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """Recall among the top-``budget`` fraction of scores (the alert budget)."""
    total_pos = float(y_true.sum())
    if total_pos == 0:
        return float("nan")
    k = max(1, int(np.ceil(budget * len(scores))))
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum() / total_pos)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y_np: np.ndarray, budget: float) -> dict:
    model.eval()
    scores = model(X).squeeze(1).numpy()
    model.train()
    return {
        "aucpr": float(average_precision_score(y_np, scores)),
        "auroc": float(roc_auc_score(y_np, scores)),
        "cov50": coverage_at_budget(y_np, scores, budget),
    }


# ---- training ---------------------------------------------------------------


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    n_pos = float(y.sum())
    n_neg = float(y.numel()) - n_pos
    return torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)


def _loader(X: torch.Tensor, y: torch.Tensor, bs: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True, generator=g)


def train_weighted_ce(
    X: torch.Tensor, y: torch.Tensor, hidden: int, epochs: int, bs: int, seed: int = 0
) -> nn.Module:
    torch.manual_seed(seed)
    model = make_model(X.size(1), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y))
    for _ in range(epochs):
        model.train()
        for xb, yb in _loader(X, y, bs, seed):
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def train_ranking(
    X: torch.Tensor,
    y: torch.Tensor,
    main_loss: nn.Module,
    hidden: int,
    epochs: int,
    bs: int,
    temp_start: float,
    temp_end: float,
    seed: int = 0,
    warmup_frac: float = 0.30,
    blend_frac: float = 0.15,
) -> nn.Module:
    """Weighted-BCE warmup -> ranking loss, step-scheduled via LossWarmupWrapper."""
    torch.manual_seed(seed)
    model = make_model(X.size(1), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = _loader(X, y, bs, seed)
    total = epochs * len(loader)
    warmup_steps = int(warmup_frac * total)
    blend_steps = int(blend_frac * total)
    wrapper = LossWarmupWrapper(
        warmup_loss=nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y)),
        main_loss=main_loss,
        warmup_steps=warmup_steps,
        blend_steps=blend_steps,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=max(1, total - warmup_steps),
    )

    gstep = 0
    for epoch in range(epochs):
        wrapper.on_train_epoch_start(epoch)
        model.train()
        for xb, yb in loader:
            wrapper.on_train_batch_start(gstep)
            opt.zero_grad()
            wrapper(model(xb), yb).backward()
            opt.step()
            gstep += 1
    return model


# ---- pos_numerator pool-vs-live comparison ----------------------------------


def run_compare(
    pos_rate: float, budget: float, hidden: int, epochs: int,
    bs: int, queue: int, n_samples: int, seed: int, n_seeds: int,
):
    a, b = budget / 2, budget * 1.5  # band brackets the operating point

    def pauc(surrogate, pos_numerator):
        return lambda X, y, s: train_ranking(
            X, y,
            PAUCAtBudgetLoss(num_classes=1, alpha=a, beta=b, surrogate=surrogate,
                             pos_numerator=pos_numerator, queue_size=queue),
            hidden, epochs, bs, temp_start=0.5, temp_end=0.1, seed=s,
        )

    # Each builder constructs a FRESH loss per seed (ranking losses are stateful).
    builders = {
        "Weighted CE": lambda X, y, s: train_weighted_ce(X, y, hidden, epochs, bs, s),
        "SmoothAP": lambda X, y, s: train_ranking(
            X, y, SmoothAPLoss(num_classes=1, queue_size=queue),
            hidden, epochs, bs, temp_start=0.1, temp_end=0.01, seed=s,
        ),
        "PAUC trap (pool)": pauc("trapezoid", "pool"),
        "PAUC trap (live)": pauc("trapezoid", "live"),
        "PAUC pair (pool)": pauc("pairwise", "pool"),
        "PAUC pair (live)": pauc("pairwise", "live"),
    }

    seeds = [seed + i for i in range(n_seeds)]
    agg: dict[str, dict[str, list]] = {
        name: {"aucpr": [], "auroc": [], "cov50": []} for name in builders
    }
    n_pos = n_va = 0
    for s in seeds:
        X_tr, y_tr, X_va, y_va = make_imbalanced_data(pos_rate, n_samples, seed=s)
        n_pos, n_va = int(y_tr.sum()), len(X_va)
        for name, build in builders.items():
            m = evaluate(build(X_tr, y_tr, s), X_va, y_va, budget)
            for k in agg[name]:
                agg[name][k].append(m[k])

    live_pos_per_batch = n_pos * bs / (n_va * 4)
    print(
        f"Synthetic imbalance  |  train pos ~{n_pos} / {n_va * 4}  "
        f"|  hidden={hidden}  bs={bs} (~{live_pos_per_batch:.1f} live pos/batch)  "
        f"queue={queue}  budget={budget * 1e4:.0f} bps  |  mean over {n_seeds} seeds\n"
    )
    col = 18
    hdr = f"{'loss':>{col}}  {'AUCPR':>8}  {'AUROC':>8}  {'cov@50bps':>16}"
    print(hdr)
    print("-" * len(hdr))
    means = {}
    for name, d in agg.items():
        cov_mean, cov_std = float(np.mean(d["cov50"])), float(np.std(d["cov50"]))
        means[name] = cov_mean
        print(
            f"{name:>{col}}  {np.mean(d['aucpr']):>8.4f}  {np.mean(d['auroc']):>8.4f}  "
            f"{cov_mean:>8.4f} ± {cov_std:.4f}"
        )

    ce = means["Weighted CE"]
    best = max(means, key=lambda k: means[k])
    pair_pool = means["PAUC pair (pool)"]
    lift = (pair_pool - ce) / ce * 100 if ce > 0 else float("nan")
    print(
        f"\nBest mean coverage@50bps: {best}.  "
        f"PAUC pair (pool) vs Weighted CE: {pair_pool:.4f} vs {ce:.4f} ({lift:+.0f}%)."
    )
    print(
        "Weighted CE protects the bulk (AUCPR) and under-resolves the contested top. "
        "PAUC pairwise pushes the near-threshold decoys down and recovers coverage@50bps. "
        "Trapezoid collapses here (it can't suppress band negatives -- wrong surrogate "
        "for a hard-negative top). pos_numerator is surrogate-dependent: 'pool' keeps "
        "enough positives in the pairwise contrast; 'live' de-dilutes the trapezoid "
        "numerator in the easy-top regime (see docs)."
    )


# ---- CLI --------------------------------------------------------------------


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pos-rate", type=float, default=0.0015, help="Positive fraction (default 0.0015; a rare-positive, <1% setting).")
    p.add_argument("--budget", type=float, default=0.005, help="Operating-point budget (default 50 bps).")
    p.add_argument("--hidden", type=int, default=64, help="Encoder width.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Smaller batch => fewer live positives => more queue dilution under 'pool'.")
    p.add_argument("--queue-size", type=int, default=8192,
                   help="Larger queue => more detached positives diluting the 'pool' numerator.")
    p.add_argument("--n-samples", type=int, default=200_000)
    p.add_argument("--n-seeds", type=int, default=5, help="Seeds to average over.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_compare(
        pos_rate=args.pos_rate, budget=args.budget, hidden=args.hidden,
        epochs=args.epochs, bs=args.batch_size, queue=args.queue_size,
        n_samples=args.n_samples, seed=args.seed, n_seeds=args.n_seeds,
    )
