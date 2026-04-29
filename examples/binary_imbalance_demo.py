"""
When does SmoothAPLoss beat BCE and Focal Loss?

Sweeps positive-class frequency from 25 % down to 0.5 % on a synthetic binary
dataset and prints AUCPR for three loss strategies.  The results show that
SmoothAPLoss earns its largest gains when positives are rare -- exactly the
fraud-detection / rare-event regime where standard losses fail.

Why it wins
-----------
BCE and Focal Loss compute per-sample cross-entropy.  When positives are rare,
95-99 % of every gradient step comes from easy-to-classify negatives.  Those
gradients push the model to over-predict "negative", suppressing sensitivity.

SmoothAPLoss directly optimizes Average Precision: it scores each positive
*relative to all negatives* in the batch + queue.  Easy negatives are already
ranked below the positive, so they contribute almost nothing.  The loss signal
is concentrated on the hard, informative examples.

Usage
-----
    python examples/binary_imbalance_demo.py
    python examples/binary_imbalance_demo.py --positive-rate 0.05 --epochs 40
    python examples/binary_imbalance_demo.py --sweep
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from imbalanced_losses import SigmoidFocalLoss, SmoothAPLoss, LossWarmupWrapper


# ---- data -------------------------------------------------------------------


def make_data(
    positive_rate: float,
    n_samples: int = 10_000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        weights=[1 - positive_rate, positive_rate],
        flip_y=0,
        random_state=seed,
    )
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)  # [N, 1]
    return X_tr_t, y_tr_t, X_va_t, y_va


# ---- model ------------------------------------------------------------------


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))


# ---- metric -----------------------------------------------------------------


@torch.no_grad()
def aucpr(model: nn.Module, X: torch.Tensor, y_np) -> float:
    model.train(False)
    score = average_precision_score(y_np, model(X).squeeze().numpy())
    model.train(True)
    return float(score)


# ---- single-run: one positive rate, per-epoch output -----------------------


def run_single(
    positive_rate: float,
    epochs: int = 30,
    queue_size: int = 512,
    warmup_epochs: int = 5,
    blend_epochs: int = 2,
    n_samples: int = 10_000,
    seed: int = 42,
):
    X_tr, y_tr, X_va, y_va = make_data(positive_rate, n_samples, seed)
    n_pos = int(y_tr.sum().item())
    print(f"Positive rate: {positive_rate:.1%}  |  train positives: {n_pos} / {len(X_tr)}\n")

    results: dict[str, list[float]] = {"BCE": [], "Focal": [], "SmoothAP": []}

    # BCE
    torch.manual_seed(0)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(X_tr), y_tr).backward()
        opt.step()
        results["BCE"].append(aucpr(model, X_va, y_va))

    # Focal
    torch.manual_seed(0)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(X_tr), y_tr).backward()
        opt.step()
        results["Focal"].append(aucpr(model, X_va, y_va))

    # SmoothAP
    torch.manual_seed(0)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = LossWarmupWrapper(
        warmup_loss=nn.BCEWithLogitsLoss(),
        main_loss=SmoothAPLoss(num_classes=1, queue_size=queue_size),
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        temp_start=0.1,
        temp_end=0.01,
        temp_decay_steps=epochs,
    )
    for epoch in range(epochs):
        loss_fn.on_train_epoch_start(epoch)
        model.train()
        loss_fn.on_train_batch_start(epoch)
        opt.zero_grad()
        loss_fn(model(X_tr), y_tr).backward()
        opt.step()
        results["SmoothAP"].append(aucpr(model, X_va, y_va))

    # per-epoch table
    col = 10
    hdr = f"{'epoch':>5}  " + "  ".join(f"{k:>{col}}" for k in results)
    print(hdr)
    print("-" * len(hdr))
    for ep in range(epochs):
        row = f"{ep:5d}  " + "  ".join(f"{results[k][ep]:{col}.4f}" for k in results)
        print(row)

    bests = {k: max(v) for k, v in results.items()}
    base = max(bests["BCE"], bests["Focal"])
    lift = (bests["SmoothAP"] - base) / base * 100 if base > 0 else 0
    print(f"\nBest AUCPR  --  BCE: {bests['BCE']:.4f}  |  Focal: {bests['Focal']:.4f}"
          f"  |  SmoothAP: {bests['SmoothAP']:.4f}  ({lift:+.0f}% vs best baseline)")


# ---- sweep: multiple positive rates ----------------------------------------


def run_sweep(
    rates: list[float] | None = None,
    epochs: int = 30,
    queue_size: int = 512,
    warmup_epochs: int = 5,
    blend_epochs: int = 2,
    n_samples: int = 10_000,
    seed: int = 42,
):
    if rates is None:
        rates = [0.25, 0.15, 0.10, 0.05, 0.02, 0.01, 0.005]

    def _best_standard(loss_fn, X_tr, y_tr, X_va, y_va):
        torch.manual_seed(0)
        m = make_model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            loss_fn(m(X_tr), y_tr).backward(); opt.step()
        return aucpr(m, X_va, y_va)

    def _best_ap(X_tr, y_tr, X_va, y_va):
        torch.manual_seed(0)
        m = make_model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        fn = LossWarmupWrapper(
            warmup_loss=nn.BCEWithLogitsLoss(),
            main_loss=SmoothAPLoss(num_classes=1, queue_size=queue_size),
            warmup_epochs=warmup_epochs,
            blend_epochs=blend_epochs,
            temp_start=0.1, temp_end=0.01,
            temp_decay_steps=epochs,
        )
        for epoch in range(epochs):
            fn.on_train_epoch_start(epoch)
            m.train(); fn.on_train_batch_start(epoch)
            opt.zero_grad(); fn(m(X_tr), y_tr).backward(); opt.step()
        return aucpr(m, X_va, y_va)

    print(f"{'rate':>6}  {'n_pos':>6}  {'BCE':>8}  {'Focal':>8}  {'SmoothAP':>10}  {'lift':>8}")
    print("-" * 62)
    for r in rates:
        X_tr, y_tr, X_va, y_va = make_data(r, n_samples, seed)
        n_pos = int(y_tr.sum().item())
        bce   = _best_standard(nn.BCEWithLogitsLoss(), X_tr, y_tr, X_va, y_va)
        focal = _best_standard(
            SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean"),
            X_tr, y_tr, X_va, y_va,
        )
        ap    = _best_ap(X_tr, y_tr, X_va, y_va)
        base  = max(bce, focal)
        lift  = (ap - base) / base * 100 if base > 0 else 0
        flag  = " *" if ap > base else "  "
        print(f"{r:>6.3f}  {n_pos:>6d}  {bce:>8.4f}  {focal:>8.4f}  {ap:>10.4f}  {lift:>+7.0f}%{flag}")


# ---- CLI --------------------------------------------------------------------


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sweep", action="store_true",
        help="Run all positive rates; print summary table.",
    )
    p.add_argument(
        "--positive-rate", type=float, default=0.05,
        help="Positive-class fraction for single-rate mode (default: 0.05).",
    )
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--queue-size",    type=int,   default=512)
    p.add_argument("--warmup-epochs", type=int,   default=5)
    p.add_argument("--blend-epochs",  type=int,   default=2)
    p.add_argument("--n-samples",     type=int,   default=10_000)
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    shared = dict(
        epochs=args.epochs,
        queue_size=args.queue_size,
        warmup_epochs=args.warmup_epochs,
        blend_epochs=args.blend_epochs,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    if args.sweep:
        run_sweep(**shared)
    else:
        run_single(positive_rate=args.positive_rate, **shared)
