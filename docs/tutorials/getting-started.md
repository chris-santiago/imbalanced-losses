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

torch.manual_seed(0)

# 5% positive rate — a realistic fraud-detection setup
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    weights=[0.95, 0.05],
    flip_y=0,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train  = torch.tensor(X_train, dtype=torch.float32)
X_val    = torch.tensor(X_val,   dtype=torch.float32)
y_train  = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # [N, 1]
y_val_np = y_val  # keep as numpy for sklearn metrics

print(f"Train size: {len(X_train)}, positives: {int(y_train.sum())}")
```

**Output:**
```
Train size: 8000, positives: 391
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
torch.manual_seed(0)
model     = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.BCEWithLogitsLoss()

for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    val_logits = model(X_val).squeeze()
    aucpr = average_precision_score(y_val_np, val_logits.numpy())
    print(f"BCE  AUCPR: {aucpr:.4f}")
```

**Output:**
```
BCE  AUCPR: 0.1822
```

The model learns but is dominated by the majority class — easy negatives suppress gradient signal to positives.

## Step 4 — Switch to Focal Loss

Focal loss down-weights well-classified easy examples, forcing the model to focus on the hard positives.

```python
from imbalanced_losses import SigmoidFocalLoss

torch.manual_seed(0)
model     = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    val_logits = model(X_val).squeeze()
    aucpr = average_precision_score(y_val_np, val_logits.numpy())
    print(f"Focal AUCPR: {aucpr:.4f}")
```

**Output:**
```
Focal AUCPR: 0.1874
```

A modest improvement. Now let's go further by directly optimizing AP.

## Step 5 — Use Smooth-AP with warmup

Ranking losses need a warm start because their gradients are flat when the model is random. `LossWarmupWrapper` runs BCE for the first few epochs, then blends into Smooth-AP.

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper

torch.manual_seed(0)
model     = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.BCEWithLogitsLoss(),
    main_loss=SmoothAPLoss(num_classes=1, queue_size=512),
    warmup_epochs=5,
    blend_epochs=2,
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=1000,
)

TOTAL_EPOCHS = 25
global_step  = 0

for epoch in range(TOTAL_EPOCHS):
    loss_fn.on_train_epoch_start(epoch)

    model.train()
    loss_fn.on_train_batch_start(global_step)
    optimizer.zero_grad()
    # Pass y_train ([N, 1] float) directly — BCE and SmoothAPLoss both handle this shape
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    global_step += 1

    model.eval()
    with torch.no_grad():
        aucpr = average_precision_score(y_val_np, model(X_val).squeeze().numpy())

    phase = "warmup" if loss_fn.in_warmup else ("blend" if loss_fn.in_blend else "AP")
    t = loss_fn.current_temperature
    temp_str = f"  temp={t:.4f}" if (t is not None and not loss_fn.in_warmup) else ""
    print(f"Epoch {epoch:2d} [{phase:6s}]  loss={loss.item():.4f}  AUCPR={aucpr:.4f}{temp_str}")
```

**Output:**
```
Epoch  0 [warmup]  loss=0.8829  AUCPR=0.1145
Epoch  1 [warmup]  loss=0.8533  AUCPR=0.1182
Epoch  2 [warmup]  loss=0.8246  AUCPR=0.1216
Epoch  3 [warmup]  loss=0.7969  AUCPR=0.1247
Epoch  4 [warmup]  loss=0.7701  AUCPR=0.1281
Epoch  5 [blend ]  loss=0.7974  AUCPR=0.1320  temp=0.1000
Epoch  6 [blend ]  loss=0.8399  AUCPR=0.1364  temp=0.0998
Epoch  7 [AP    ]  loss=0.8963  AUCPR=0.1418  temp=0.0995
Epoch  8 [AP    ]  loss=0.8921  AUCPR=0.1482  temp=0.0993
Epoch  9 [AP    ]  loss=0.8874  AUCPR=0.1558  temp=0.0991
Epoch 10 [AP    ]  loss=0.8821  AUCPR=0.1639  temp=0.0989
Epoch 11 [AP    ]  loss=0.8762  AUCPR=0.1733  temp=0.0986
Epoch 12 [AP    ]  loss=0.8696  AUCPR=0.1827  temp=0.0984
Epoch 13 [AP    ]  loss=0.8622  AUCPR=0.1953  temp=0.0982
Epoch 14 [AP    ]  loss=0.8539  AUCPR=0.2091  temp=0.0979
Epoch 15 [AP    ]  loss=0.8446  AUCPR=0.2242  temp=0.0977
Epoch 16 [AP    ]  loss=0.8339  AUCPR=0.2405  temp=0.0975
Epoch 17 [AP    ]  loss=0.8218  AUCPR=0.2587  temp=0.0973
Epoch 18 [AP    ]  loss=0.8080  AUCPR=0.2802  temp=0.0971
Epoch 19 [AP    ]  loss=0.7922  AUCPR=0.3065  temp=0.0968
Epoch 20 [AP    ]  loss=0.7743  AUCPR=0.3364  temp=0.0966
Epoch 21 [AP    ]  loss=0.7540  AUCPR=0.3601  temp=0.0964
Epoch 22 [AP    ]  loss=0.7313  AUCPR=0.3793  temp=0.0962
Epoch 23 [AP    ]  loss=0.7067  AUCPR=0.4048  temp=0.0959
Epoch 24 [AP    ]  loss=0.6812  AUCPR=0.4248  temp=0.0957
```

## What you built

You trained the same model architecture with three different loss strategies and observed AUCPR improvements at each step:

| Loss strategy | AUCPR |
|---|---|
| Vanilla BCE | 0.1822 |
| Focal Loss | 0.1874 |
| Smooth-AP with warmup | 0.4248 |

## Next steps

- [Use Focal Loss](../how-to/use-focal-loss.md) — detailed options for `SigmoidFocalLoss` and `SoftmaxFocalLoss`
- [Use Ranking Losses](../how-to/use-ranking-losses.md) — queue sizing, temperature, binary vs. multi-class
- [Configure Warmup and Blending](../how-to/configure-warmup.md) — tuning the phase schedule
- [Train with DDP](../how-to/train-with-ddp.md) — multi-GPU setup
- [`examples/binary_imbalance_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/binary_imbalance_demo.py) — sweeps positive rates from 25 % down to 0.5 % to show where `SmoothAPLoss` earns its largest gains
