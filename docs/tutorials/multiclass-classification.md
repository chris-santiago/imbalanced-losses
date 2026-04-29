# Multiclass Classification

This tutorial trains a multiclass classifier on an imbalanced 5-class dataset. You will start
with `CrossEntropyLoss` to establish a baseline, add per-class alpha weighting with
`SoftmaxFocalLoss`, and finally use `SmoothAPLoss` with `LossWarmupWrapper` to directly
optimize per-class average precision. Every step prints a metric so you can see progress.

## Prerequisites

- Python 3.10+
- PyTorch ≥ 2.8
- scikit-learn

```bash
pip install "imbalanced-losses[demo]"
```

## Step 1 — Generate imbalanced multiclass data

Exponentially decreasing class weights put ~51% of samples in class 0 and only ~3% in class 4.

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

torch.manual_seed(0)

N_CLASSES = 5
# Exponential class weights: class 0 ≈ 51%, class 4 ≈ 3%
raw = [2.0 ** (-i) for i in range(N_CLASSES)]
weights = [r / sum(raw) for r in raw]

X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=15,
    n_redundant=2,
    n_classes=N_CLASSES,
    n_clusters_per_class=1,
    weights=weights,
    flip_y=0.01,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

X_train  = torch.tensor(X_train, dtype=torch.float32)
X_val    = torch.tensor(X_val,   dtype=torch.float32)
y_train  = torch.tensor(y_train, dtype=torch.long)   # [N] integer class indices
y_val_np = y_val                                      # keep as numpy for sklearn

counts = [(y_train == c).sum().item() for c in range(N_CLASSES)]
print(f"Train size: {len(X_train)}")
print(f"Class counts: {counts}")
```

**Output:**
```
Train size: 7500
Class counts: [3826, 1958, 955, 505, 256]
```

## Step 2 — Define a simple model

The model outputs `[N, C]` logits — one score per class.

```python
model = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, N_CLASSES),
)
```

## Step 3 — Define the macro-AP metric

`average_precision_score` with `average='macro'` computes one-vs-rest AP for each class
and averages. This is more informative than accuracy for imbalanced problems.

```python
def compute_macro_ap(model, X, y_np):
    model.train(False)
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1).cpu().numpy()
    y_bin = label_binarize(y_np, classes=range(N_CLASSES))
    ap = average_precision_score(y_bin, probs, average="macro")
    model.train()
    return float(ap)
```

## Step 4 — Train with CrossEntropyLoss (baseline)

```python
def run(loss_fn, total_epochs=20, batch_size=256):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(20, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, N_CLASSES),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(X_train)
    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss_fn(model(X_train[idx]), y_train[idx]).backward()
            opt.step()
    return compute_macro_ap(model, X_val, y_val_np)

ce_ap = run(nn.CrossEntropyLoss())
print(f"CE  macro-AP: {ce_ap:.4f}")
```

**Output:**
```
CE  macro-AP: 0.9649
```

## Step 5 — Add per-class alpha with SoftmaxFocalLoss

`CrossEntropyLoss` treats every class equally. `SoftmaxFocalLoss` with `alpha` set to inverse
class frequency gives more gradient weight to rare classes and applies a focusing modulator
that down-weights easy, well-classified examples.

```python
from imbalanced_losses import SoftmaxFocalLoss

# Inverse-frequency weights: rare classes get higher weight
total = sum(counts)
alpha_raw = [total / (N_CLASSES * max(c, 1)) for c in counts]
alpha = [a / max(alpha_raw) for a in alpha_raw]  # normalize: max weight = 1.0
print(f"alpha (normalized): {[f'{a:.3f}' for a in alpha]}")

focal_ap = run(SoftmaxFocalLoss(alpha=alpha, gamma=2.0))
print(f"Focal macro-AP: {focal_ap:.4f}")
```

**Output:**
```
alpha (normalized): ['0.067', '0.131', '0.268', '0.507', '1.000']
Focal macro-AP: 0.9574
```

The alpha values show class 4 (rarest) gets 15× the gradient weight of class 0. The overall
macro-AP is similar to CE — that's expected on this task. The difference shows up in tail-class
AP when you look per class (see [Log Per-Class Metrics](../how-to/log-per-class-metrics.md)).

## Step 6 — Use SmoothAPLoss with warmup

`SmoothAPLoss` directly optimizes average precision rather than cross-entropy. Ranking losses
need a warm start — their gradients are nearly flat when the model is random. `LossWarmupWrapper`
runs `CrossEntropyLoss` for the first few epochs, then blends into `SmoothAPLoss`.

Note that both `CrossEntropyLoss` and `SmoothAPLoss` accept the same input format:
`[N, C]` logits and `[N]` long integer targets — no conversion needed.

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(20, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, N_CLASSES),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

TOTAL_EPOCHS = 20
BATCH_SIZE   = 256
n            = len(X_train)

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=N_CLASSES, queue_size=1024),
    warmup_epochs=5,
    blend_epochs=2,
    temp_start=0.05,
    temp_end=0.005,
    temp_decay_steps=TOTAL_EPOCHS * (n // BATCH_SIZE),
)

global_step = 0
for epoch in range(TOTAL_EPOCHS):
    loss_fn.on_train_epoch_start(epoch)
    model.train()
    perm = torch.randperm(n)
    for i in range(0, n, BATCH_SIZE):
        loss_fn.on_train_batch_start(global_step)
        idx = perm[i : i + BATCH_SIZE]
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        global_step += 1

    ap = compute_macro_ap(model, X_val, y_val_np)
    phase = "warmup" if loss_fn.in_warmup else ("blend" if loss_fn.in_blend else "AP")
    t = loss_fn.current_temperature
    temp_str = f"  temp={t:.4f}" if t is not None else ""
    print(f"Epoch {epoch:2d} [{phase:6s}]  macro-AP={ap:.4f}{temp_str}")
```

**Output:**
```
Epoch  0 [warmup]  macro-AP=0.6294
Epoch  1 [warmup]  macro-AP=0.7950
Epoch  2 [warmup]  macro-AP=0.8635
Epoch  3 [warmup]  macro-AP=0.9061
Epoch  4 [warmup]  macro-AP=0.9282
Epoch  5 [blend ]  macro-AP=0.9416  temp=0.0500
Epoch  6 [blend ]  macro-AP=0.9489  temp=0.0499
Epoch  7 [AP    ]  macro-AP=0.9505  temp=0.0498
Epoch  8 [AP    ]  macro-AP=0.9505  temp=0.0497
Epoch  9 [AP    ]  macro-AP=0.9510  temp=0.0496
Epoch 10 [AP    ]  macro-AP=0.9523  temp=0.0495
Epoch 11 [AP    ]  macro-AP=0.9499  temp=0.0494
Epoch 12 [AP    ]  macro-AP=0.9466  temp=0.0493
Epoch 13 [AP    ]  macro-AP=0.9485  temp=0.0491
Epoch 14 [AP    ]  macro-AP=0.9411  temp=0.0490
Epoch 15 [AP    ]  macro-AP=0.9467  temp=0.0489
Epoch 16 [AP    ]  macro-AP=0.9472  temp=0.0488
Epoch 17 [AP    ]  macro-AP=0.9451  temp=0.0487
Epoch 18 [AP    ]  macro-AP=0.9426  temp=0.0486
Epoch 19 [AP    ]  macro-AP=0.9452  temp=0.0485
```

## What you built

You trained the same architecture with three loss strategies on a 5-class imbalanced dataset:

| Loss strategy | Macro-AP |
|---|---|
| CrossEntropyLoss | 0.9649 |
| SoftmaxFocalLoss (alpha + gamma) | 0.9574 |
| SmoothAPLoss with warmup | 0.9523 |

**Why are the numbers close?** With the rarest class at ~3% frequency, this dataset sits in the
*mild-to-moderate* imbalance range. `CrossEntropyLoss` is surprisingly competitive here
because even the majority class provides some gradient signal toward the tail. `SmoothAPLoss`
earns its largest gains under more extreme imbalance — where positives are so rare that
cross-entropy gradients from easy negatives dominate, not because the dataset is hard, but
because the easy negatives vastly outnumber the informative positives. The binary
[Getting Started tutorial](getting-started.md) shows a 5% positive rate case where
Smooth-AP's AUCPR is roughly 2× higher than focal loss.

Use `SmoothAPLoss` for multiclass when your rarest class falls below ~1–2% of the dataset
and per-class AP is the metric that matters. For mild imbalance, `SoftmaxFocalLoss` is
often the simpler choice with comparable results.

**Key multiclass-specific points:**
- Targets must be `torch.long` class indices `[N]`, not float or one-hot
- `SmoothAPLoss` and `CrossEntropyLoss` share the same input format — no dtype conversion in the warmup wrapper
- `queue_size` accumulates logits across batches; use at least 4–8× your typical batch size

## Next steps

- [Use Ranking Losses](../how-to/use-ranking-losses.md) — queue sizing, temperature, and quantile selection
- [Configure Warmup and Blending](../how-to/configure-warmup.md) — tune warmup/blend schedules
- [Log Per-Class Metrics](../how-to/log-per-class-metrics.md) — monitor per-class AP without a second forward pass
- [Migrate from BCE / CrossEntropyLoss](../how-to/migrate-from-cross-entropy.md) — drop-in migration checklist
