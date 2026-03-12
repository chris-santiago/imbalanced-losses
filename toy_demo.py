"""
Toy binary classification demo: BCE warmup → blend → SmoothAP.

Generates an imbalanced 2D dataset (5% positive rate), trains a small MLP,
and prints AUCPR + loss after every epoch so you can see whether the
warmup→AP transition helps or hurts ranking quality.

Usage:
    python toy_demo.py
    python toy_demo.py --blend-epochs 0   # hard switch (no blend) for comparison
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

from ap_loss import SmoothAPLoss
from warmup_wrapper import LossWarmupWrapper

# ── synthetic data ──────────────────────────────────────────────────────────


def make_data(
    n: int = 4000,
    pos_rate: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X_train, y_train, X_val, y_val) using sklearn make_classification."""
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


# ── model ───────────────────────────────────────────────────────────────────


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
        return self.net(x)  # [N, 1]


# ── evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    scores = model(X).squeeze(-1).sigmoid().cpu().numpy()
    ap = average_precision_score(y.cpu().numpy(), scores)
    model.train()
    return float(ap)


# ── training loop ───────────────────────────────────────────────────────────


def train(
    warmup_epochs: int = 3,
    blend_epochs: int = 2,
    total_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    queue_size: int = 256,
    temp_start: float = 0.5,
    temp_end: float = 0.01,
    seed: int = 42,
):
    torch.manual_seed(seed)
    X_train, y_train, X_val, y_val = make_data(seed=seed)
    n = X_train.shape[0]
    steps_per_epoch = math.ceil(n / batch_size)
    temp_decay_steps = (total_epochs - warmup_epochs) * steps_per_epoch

    model = TinyMLP()
    loss_fn = LossWarmupWrapper(
        warmup_loss=BCEWarmupLoss(),
        main_loss=SmoothAPLoss(
            num_classes=1, queue_size=queue_size, temperature=temp_start
        ),
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=temp_decay_steps,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    header = f"{'epoch':>5}  {'phase':>10}  {'ap_wt':>5}  {'temp':>8}  {'loss':>8}  {'AUCPR':>7}"
    print(header)
    print("-" * len(header))

    global_step = 0
    for epoch in range(total_epochs):
        loss_fn.on_train_epoch_start(epoch)
        model.train()

        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            loss_fn.on_train_batch_start(global_step)
            logits = model(xb)  # [B, 1]
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        # Eval
        aucpr = evaluate(model, X_val, y_val)
        avg_loss = epoch_loss / n_batches
        temp = loss_fn.current_temperature

        if loss_fn.in_warmup:
            phase = "warmup"
        elif loss_fn.in_blend:
            phase = "blend"
        else:
            phase = "AP"

        print(
            f"{epoch:5d}  {phase:>10}  {loss_fn.ap_weight:5.2f}  "
            f"{temp:8.5f}  {avg_loss:8.4f}  {aucpr:7.4f}"
        )


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--blend-epochs", type=int, default=2)
    p.add_argument("--total-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--queue-size", type=int, default=256)
    p.add_argument("--temp-start", type=float, default=0.35)
    p.add_argument("--temp-end", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(**vars(args))
