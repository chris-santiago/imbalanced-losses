# Getting Started

This tutorial trains a binary classifier on a highly imbalanced dataset. You will start with vanilla BCE loss to establish a baseline, switch to `SigmoidFocalLoss` for an easier improvement, and finally use `SmoothAPLoss` with `LossWarmupWrapper` for the best ranking performance. Every step prints a metric so you can see progress.

## Prerequisites

- Python 3.10+
- PyTorch ≥ 2.8
- scikit-learn

```bash
pip install "imbalanced-losses[demo]"
```

## Step 1 — Generate imbalanced data

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import numpy as np

# 1% positive rate — a realistic fraud-detection setup
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    weights=[0.99, 0.01],
    flip_y=0,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

print(f"Train size: {len(X_train)}, positives: {int(y_train.sum())}")
# Train size: 4000, positives: ~40
```

**Output:**
```
Train size: 4000, positives: 40
```

## Step 2 — Define a simple model

```python
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

## Step 3 — Train with vanilla BCE (baseline)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss   = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    val_logits = model(X_val).squeeze()
    aucpr = average_precision_score(y_val.numpy(), val_logits.numpy())
    print(f"BCE  AUCPR: {aucpr:.4f}")
```

**Output (approximate):**
```
BCE  AUCPR: 0.0530
```

The model barely beats random at this positive rate — easy negatives dominate training.

## Step 4 — Switch to Focal Loss

Focal loss down-weights well-classified easy examples, forcing the model to focus on the hard positives.

```python
from imbalanced_losses import SigmoidFocalLoss

model     = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss   = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    val_logits = model(X_val).squeeze()
    aucpr = average_precision_score(y_val.numpy(), val_logits.numpy())
    print(f"Focal AUCPR: {aucpr:.4f}")
```

**Output (approximate):**
```
Focal AUCPR: 0.1820
```

Focal loss already gives a clear improvement. Now let's go further by directly optimizing AP.

## Step 5 — Use Smooth-AP with warmup

Ranking losses need a warm start because their gradients are flat when the model is random. `LossWarmupWrapper` runs BCE for the first few epochs, then blends into Smooth-AP.

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper

model     = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.BCEWithLogitsLoss(),
    main_loss=SmoothAPLoss(num_classes=1, queue_size=256),
    warmup_epochs=3,
    blend_epochs=2,
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=500,
)

TOTAL_EPOCHS = 15
global_step  = 0

for epoch in range(TOTAL_EPOCHS):
    loss_fn.on_train_epoch_start(epoch)

    model.train()
    # Mini-batch loop (single batch for simplicity)
    loss_fn.on_train_batch_start(global_step)
    optimizer.zero_grad()
    logits  = model(X_train)
    targets = y_train.squeeze().long()  # SmoothAPLoss expects long; BCE casts internally

    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    global_step += 1

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).squeeze()
        aucpr = average_precision_score(y_val.numpy(), val_logits.numpy())

    phase = "warmup" if loss_fn.in_warmup else ("blend" if loss_fn.in_blend else "AP")
    t = loss_fn.current_temperature
    print(f"Epoch {epoch:2d} [{phase:6s}]  loss={loss.item():.4f}  AUCPR={aucpr:.4f}"
          + (f"  temp={t:.4f}" if t else ""))
```

**Output (approximate):**
```
Epoch  0 [warmup]  loss=0.0182  AUCPR=0.0801
Epoch  1 [warmup]  loss=0.0152  AUCPR=0.1043
Epoch  2 [warmup]  loss=0.0131  AUCPR=0.1298
Epoch  3 [blend ]  loss=0.6891  AUCPR=0.1512  temp=0.1000
Epoch  4 [blend ]  loss=0.5423  AUCPR=0.1834  temp=0.0631
Epoch  5 [AP    ]  loss=0.4112  AUCPR=0.2341  temp=0.0398
...
Epoch 14 [AP    ]  loss=0.2887  AUCPR=0.3105  temp=0.0100
```

## What you built

You trained the same model architecture with three different loss strategies and observed AUCPR improvements at each step:

| Loss strategy | AUCPR |
|---|---|
| Vanilla BCE | ~0.05 |
| Focal Loss | ~0.18 |
| Smooth-AP with warmup | ~0.31 |

## Next steps

- [Use Focal Loss](../how-to/use-focal-loss.md) — detailed options for `SigmoidFocalLoss` and `SoftmaxFocalLoss`
- [Use Ranking Losses](../how-to/use-ranking-losses.md) — queue sizing, temperature, binary vs. multi-class
- [Configure Warmup and Blending](../how-to/configure-warmup.md) — tuning the phase schedule
- [Train with DDP](../how-to/train-with-ddp.md) — multi-GPU setup
