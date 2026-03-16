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

from imbalanced_losses import LossWarmupWrapper, RecallAtQuantileLoss, SmoothAPLoss

# ── synthetic data ──────────────────────────────────────────────────────────


def make_data(
    n: int = 60_000,
    pos_rate: float = 0.005,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        n_clusters_per_class=3,
        weights=[1 - pos_rate, pos_rate],
        flip_y=0.05,
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
    def __init__(self, d_in: int = 20, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
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


def make_main_loss(
    loss: str,
    queue_size: int,
    temp_start: float,
    quantile: float,
) -> nn.Module:
    if loss == "recall":
        return RecallAtQuantileLoss(
            num_classes=1,
            queue_size=queue_size,
            quantile=quantile,
            temperature=temp_start,
        )
    return SmoothAPLoss(num_classes=1, queue_size=queue_size, temperature=temp_start)


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
    loss: str = "ap",
    quantile: float | None = None,
    decay_steps: int | None = None,
    seed: int = 42,
):
    X_train, y_train, X_val, y_val = make_data(pos_rate=pos_rate, seed=seed)
    n = X_train.shape[0]
    steps_per_epoch = math.ceil(n / batch_size)
    q = quantile if quantile is not None else pos_rate
    shared = dict(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        total_epochs=total_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )

    # warmup-only: BCE for all epochs, never switches to main loss
    warmup_only_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=make_main_loss(loss, queue_size, temp_start, q),
        warmup_epochs=total_epochs,  # warmup never ends
        blend_epochs=0,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=decay_steps
        if decay_steps is not None
        else total_epochs * steps_per_epoch,
    )

    # main loss only: from epoch 0, no warmup
    main_only_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=make_main_loss(loss, queue_size, temp_start, q),
        warmup_epochs=0,
        blend_epochs=0,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=decay_steps
        if decay_steps is not None
        else total_epochs * steps_per_epoch,
    )

    # warmup + blend + main loss
    warmup_blend_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=make_main_loss(loss, queue_size, temp_start, q),
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=decay_steps
        if decay_steps is not None
        else (total_epochs - warmup_epochs) * steps_per_epoch,
    )

    print(
        f"warmup_epochs={warmup_epochs}  blend_epochs={blend_epochs}  "
        f"total_epochs={total_epochs}  pos_rate={pos_rate}\n"
    )
    loss_label = loss.upper()
    print("Running warmup-only...")
    r_warmup = run_one(warmup_only_fn, **shared)
    print(f"Running {loss_label}-only...")
    r_main = run_one(main_only_fn, **shared)
    print(f"Running warmup+blend+{loss_label}...")
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
    main_col_label = f"{loss_label}-only"
    blend_col_label = f"warmup+blend"
    print(
        f"\n{'epoch':>5}  {'phase':>6}  "
        f"{'warmup-only':>{col}}  {main_col_label:>{col}}  {blend_col_label:>{col}}"
    )
    print("-" * (5 + 2 + 6 + 2 + col + 2 + col + 2 + col))
    for epoch, (wu, mn, wb) in enumerate(zip(r_warmup, r_main, r_blend)):
        print(
            f"{epoch:5d}  {phase_label(epoch):>6}  "
            f"{wu:{col}.4f}  {mn:{col}.4f}  {wb:{col}.4f}"
        )

    print()
    print(
        f"Best AUCPR  warmup-only: {max(r_warmup):.4f}  "
        f"{main_col_label}: {max(r_main):.4f}  {blend_col_label}: {max(r_blend):.4f}"
    )


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--blend-epochs", type=int, default=2)
    p.add_argument("--total-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--queue-size", type=int, default=2048)
    p.add_argument("--temp-start", type=float, default=0.35)
    p.add_argument("--temp-end", type=float, default=0.01)
    p.add_argument("--pos-rate", type=float, default=0.01)
    p.add_argument(
        "--loss",
        choices=["ap", "recall"],
        default="ap",
        help="main loss: SmoothAPLoss (ap) or RecallAtQuantileLoss (recall)",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="quantile for RecallAtQuantileLoss (default: pos-rate)",
    )
    p.add_argument(
        "--decay-steps",
        type=int,
        default=1000,
        help="temperature decay steps (default: AP-phase steps per strategy)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    compare(**vars(args))
