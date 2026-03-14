"""
Side-by-side AUCPR comparison: BCE vs BCE+weight vs SigmoidFocalLoss.

Trains four models on the same imbalanced dataset and seed, then prints
per-epoch AUCPR so you can see how focal loss compares to vanilla BCE and
class-reweighted BCE.

Strategies
----------
BCE           : vanilla BCEWithLogitsLoss; easy negatives dominate
BCE+weight    : BCEWithLogitsLoss with pos_weight = neg_count / pos_count
focal-alpha   : SigmoidFocalLoss(alpha=0.25, gamma=2); RetinaNet defaults —
                gamma down-weights easy examples, alpha re-balances pos/neg
focal-gamma   : SigmoidFocalLoss(alpha=-1,   gamma=2); modulating factor only,
                no alpha reweighting — isolates the hard-example focusing effect

Usage:
    python examples/focal_demo.py
    python examples/focal_demo.py --pos-rate 0.02   # easier problem
    python examples/focal_demo.py --gamma 5 --alpha 0.5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

from proxy_losses import SigmoidFocalLoss


# ── synthetic data ───────────────────────────────────────────────────────────


def make_data(
    n: int = 60_000,
    pos_rate: float = 0.005,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - pos_rate, pos_rate],
        flip_y=0.02,
        class_sep=0.8,
        random_state=seed,
    )
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    split = int(0.75 * n)
    return X_t[:split], y_t[:split], X_t[split:], y_t[split:]


# ── loss wrappers ────────────────────────────────────────────────────────────


class BCELoss(nn.Module):
    """Vanilla BCEWithLogitsLoss; accepts (N,1) logits and long targets."""

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.view(-1), targets.float())


class FocalLossWrapper(nn.Module):
    """SigmoidFocalLoss; accepts (N,1) logits and long targets."""

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.focal = SigmoidFocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits.view(-1), targets.float())


# ── model ────────────────────────────────────────────────────────────────────


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 10, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    scores = model(X).view(-1).sigmoid().cpu().numpy()
    ap = average_precision_score(y.cpu().numpy(), scores)
    model.train()
    return float(ap)


# ── training loop ────────────────────────────────────────────────────────────


def run_one(
    loss_fn: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    total_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[float]:
    """Train a TinyMLP with loss_fn; return per-epoch AUCPR."""
    torch.manual_seed(seed)
    n = X_train.shape[0]
    model = TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    aucpr_history = []

    for _ in range(total_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        aucpr_history.append(evaluate(model, X_val, y_val))

    return aucpr_history


# ── compare ──────────────────────────────────────────────────────────────────


def compare(
    total_epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    pos_rate: float = 0.005,
    alpha: float = 0.25,
    gamma: float = 2.0,
    seed: int = 42,
):
    X_train, y_train, X_val, y_val = make_data(pos_rate=pos_rate, seed=seed)

    n_pos = int(y_train.sum().item())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    shared = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        total_epochs=total_epochs, batch_size=batch_size,
        lr=lr, seed=seed,
    )

    strategies = [
        ("BCE",         BCELoss()),
        ("BCE+weight",  BCELoss(pos_weight=pos_weight)),
        (f"focal α={alpha} γ={gamma}", FocalLossWrapper(alpha=alpha, gamma=gamma)),
        (f"focal α=-1  γ={gamma}", FocalLossWrapper(alpha=-1, gamma=gamma)),
    ]

    print(
        f"pos_rate={pos_rate}  n_pos={n_pos}  n_neg={n_neg}  "
        f"pos_weight={pos_weight:.1f}  alpha={alpha}  gamma={gamma}\n"
    )

    results: list[tuple[str, list[float]]] = []
    for label, fn in strategies:
        print(f"Running {label}...")
        results.append((label, run_one(fn, **shared)))

    # ── side-by-side table ───────────────────────────────────────────────────
    col = 14
    header = f"{'epoch':>5}  " + "  ".join(f"{lbl:>{col}}" for lbl, _ in results)
    print(f"\nVal AUCPR per epoch")
    print(f"{header}")
    print("-" * len(header))

    for epoch in range(total_epochs):
        row = f"{epoch:5d}  "
        row += "  ".join(f"{hist[epoch]:{col}.4f}" for _, hist in results)
        print(row)

    bests = [(lbl, max(hist)) for lbl, hist in results]
    print()
    print("Best val AUCPR  " + "  ".join(f"{lbl}: {best:.4f}" for lbl, best in bests))


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--total-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos-rate", type=float, default=0.005,
                   help="positive class rate (default: 0.005 → 200:1 imbalance)")
    p.add_argument("--alpha", type=float, default=0.25,
                   help="focal alpha for the alpha+gamma strategy (default: 0.25)")
    p.add_argument("--gamma", type=float, default=2.0,
                   help="focusing exponent for both focal strategies (default: 2.0)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    compare(**vars(args))
