"""
Side-by-side AUCPR comparison: warmup-only vs AP-only vs warmup+blend+AP.

Trains three models on the same data and seed, then prints per-epoch AUCPR
for each strategy so you can see the effect of the warmup and blend phases.

Strategies
----------
warmup-only   : BCE for all epochs; never switches to AP
AP-only       : SmoothAPLoss from epoch 0, no warmup
warmup+blend  : BCE warmup → linear blend → pure SmoothAPLoss (default)

Usage:
    python examples/compare_demo.py
    python examples/compare_demo.py --pos-rate 0.05   # easier problem
    python examples/compare_demo.py --warmup-epochs 5 --blend-epochs 3
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

from proxy_losses import SmoothAPLoss
from proxy_losses import LossWarmupWrapper

# ── synthetic data ──────────────────────────────────────────────────────────


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


class BCEWarmupLoss(nn.Module):
    """BCEWithLogitsLoss that accepts (N,1) logits and long targets."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.view(-1), targets.float())


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


# ── evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    scores = model(X).squeeze(-1).sigmoid().cpu().numpy()
    ap = average_precision_score(y.cpu().numpy(), scores)
    model.train()
    return float(ap)


# ── generic training loop ───────────────────────────────────────────────────


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
    global_step = 0

    for epoch in range(total_epochs):
        if hasattr(loss_fn, "on_train_epoch_start"):
            loss_fn.on_train_epoch_start(epoch)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            if hasattr(loss_fn, "on_train_batch_start"):
                loss_fn.on_train_batch_start(global_step)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
        aucpr_history.append(evaluate(model, X_val, y_val))

    return aucpr_history


# ── compare ─────────────────────────────────────────────────────────────────


def compare(
    warmup_epochs: int = 3,
    blend_epochs: int = 2,
    total_epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    queue_size: int = 2048,
    temp_start: float = 0.35,
    temp_end: float = 0.01,
    pos_rate: float = 0.005,
    seed: int = 42,
):
    X_train, y_train, X_val, y_val = make_data(pos_rate=pos_rate, seed=seed)
    n = X_train.shape[0]
    steps_per_epoch = math.ceil(n / batch_size)
    shared = dict(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        total_epochs=total_epochs, batch_size=batch_size, lr=lr, seed=seed,
    )

    # warmup-only: BCE for all epochs, no AP loss
    warmup_only_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=SmoothAPLoss(num_classes=1, queue_size=queue_size, temperature=temp_start),
        warmup_epochs=total_epochs,  # warmup never ends
        blend_epochs=0,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=total_epochs * steps_per_epoch,
    )

    # AP-only: SmoothAPLoss from epoch 0, no warmup
    ap_only_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=SmoothAPLoss(num_classes=1, queue_size=queue_size, temperature=temp_start),
        warmup_epochs=0,
        blend_epochs=0,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=total_epochs * steps_per_epoch,
    )

    # warmup + blend + AP
    warmup_blend_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=SmoothAPLoss(num_classes=1, queue_size=queue_size, temperature=temp_start),
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=(total_epochs - warmup_epochs) * steps_per_epoch,
    )

    print(f"warmup_epochs={warmup_epochs}  blend_epochs={blend_epochs}  "
          f"total_epochs={total_epochs}  pos_rate={pos_rate}\n")
    print("Running warmup-only...")
    r_warmup = run_one(warmup_only_fn, **shared)
    print("Running AP-only...")
    r_ap = run_one(ap_only_fn, **shared)
    print("Running warmup+blend+AP...")
    r_blend = run_one(warmup_blend_fn, **shared)

    # ── phase label helper ───────────────────────────────────────────────────
    def phase_label(epoch: int) -> str:
        if epoch < warmup_epochs:
            return "warmup"
        if epoch < warmup_epochs + blend_epochs:
            return "blend"
        return "AP"

    # ── side-by-side table ───────────────────────────────────────────────────
    col = 12
    print(
        f"\n{'epoch':>5}  {'phase':>6}  "
        f"{'warmup-only':>{col}}  {'AP-only':>{col}}  {'warmup+blend':>{col}}"
    )
    print("-" * (5 + 2 + 6 + 2 + col + 2 + col + 2 + col))
    for epoch, (wu, ap, wb) in enumerate(zip(r_warmup, r_ap, r_blend)):
        print(
            f"{epoch:5d}  {phase_label(epoch):>6}  "
            f"{wu:{col}.4f}  {ap:{col}.4f}  {wb:{col}.4f}"
        )

    print()
    print(f"Final AUCPR  warmup-only: {r_warmup[-1]:.4f}  "
          f"AP-only: {r_ap[-1]:.4f}  warmup+blend: {r_blend[-1]:.4f}")


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--blend-epochs", type=int, default=2)
    p.add_argument("--total-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--queue-size", type=int, default=2048)
    p.add_argument("--temp-start", type=float, default=0.35)
    p.add_argument("--temp-end", type=float, default=0.01)
    p.add_argument("--pos-rate", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    compare(**vars(args))
