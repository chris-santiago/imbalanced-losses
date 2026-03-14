# proxy-losses

**proxy-losses** is a PyTorch library of differentiable proxy losses for ranking metrics — intended as drop-in replacements for cross-entropy when the real objective is Average Precision or recall at a specific operating point.

**What's in it:**

- **`SmoothAPLoss`** — Differentiable approximation of AP (Brown et al., ECCV 2020). Uses sigmoid-based soft rank estimation; O(M²) in pool size. Supports multi-class, binary, and seq2seq settings.
- **`RecallAtQuantileLoss`** — Optimizes recall above a score threshold set at the *q*-th quantile of the pooled distribution. Useful for alert/detection workloads (e.g. top 0.5% of scores).
- **`LossWarmupWrapper`** — Training utility that runs a standard loss (BCE/CE) during warmup, linearly blends into the ranking loss over a configurable transition window, then applies geometric temperature decay. Automatically resets the memory queue at the phase switch to prevent queue poisoning from warmup-era logits.

**Design points:**
- Circular memory queue stabilizes gradient estimates across small batches — critical at low positive rates (e.g. 0.5%)
- Compatible with PyTorch Lightning via `on_train_epoch_start` / `on_train_batch_start` hooks
- `toy_demo.py` demonstrates the full warmup→blend→AP pipeline on a highly imbalanced binary classification task using sklearn's `make_classification

## Losses

### `SmoothAPLoss` — Smooth Average Precision (Brown et al., 2020)

Approximates AP using sigmoid-based soft rank estimation. For each positive *i* in the pool:

```
ŝ_i   = 1 + Σ_{j≠i}       σ((s_j − s_i) / τ)   # soft overall rank
ŝ_i^+ = 1 + Σ_{j≠i, j∈P} σ((s_j − s_i) / τ)   # soft rank among positives
AP ≈ (1/|P|) · Σ_{i∈P}  ŝ_i^+ / ŝ_i
loss = 1 − AP
```

**Complexity:** O(M²) in memory and compute where M = batch + queue size. Keep M ≤ ~4096.

### `RecallAtQuantileLoss` — Recall at Quantile

Optimizes recall above a score threshold set at the *q*-th quantile of the pooled score distribution. The threshold is treated as a stop-gradient constant each forward pass:

```
θ = quantile(scores, 1 − q)          [detached — no grad]
soft_recall = (1/|P|) · Σ_{i∈P} σ((s_i − θ) / τ)
loss = 1 − soft_recall
```

Gradient flows only through positive scores, pushing them above the cutoff. Useful for alert/detection settings (e.g. `quantile=0.005` = top 50 bps).

## Features

Both losses share the same interface and design:

- **Memory queue** — circular buffer accumulates past batches to stabilize estimates over small batch sizes; set `queue_size=0` to disable
- **Multi-class** — one-vs-rest per class using `logits[:, c]`
- **Binary** — set `num_classes=1` with targets in `{0, 1}`
- **Seq2seq** — flatten `[B, T, C]` → `[B*T, C]` upstream before passing
- **Padding** — `ignore_index` rows are excluded from ranking and the positive set
- **Reductions** — `'mean'` (default), `'sum'`, or `'none'` (per-class tensor; degenerate classes are `nan`)
- **Per-class logging** — `return_per_class=True` returns `(loss, per_class, valid_mask)` without a second forward pass

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.8.

```bash
# from PyPI
pip install proxy-losses

# from GitHub (latest dev)
pip install git+https://github.com/chris-santiago/proxy-losses.git

# with uv (for development / contributing)
git clone https://github.com/chris-santiago/proxy-losses.git
cd proxy-losses
uv sync
```

To run the example scripts, install the optional demo dependencies:

```bash
pip install "proxy-losses[demo]"
# or with uv:
uv sync --extra demo
```

## Usage

```python
from proxy_losses import SmoothAPLoss
from proxy_losses import RecallAtQuantileLoss

# Multi-class AP loss
loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024, temperature=0.01)
logits  = torch.randn(32, 4)   # [N, C] raw logits
targets = torch.randint(0, 4, (32,))  # [N] integer class labels
loss = loss_fn(logits, targets)
loss.backward()

# Recall at top-0.5%
loss_fn = RecallAtQuantileLoss(num_classes=4, quantile=0.005, queue_size=1024)
loss = loss_fn(logits, targets)
loss.backward()

# Binary classification
loss_fn = SmoothAPLoss(num_classes=1, queue_size=256)
logits  = torch.randn(32, 1)
targets = torch.randint(0, 2, (32,))  # {0, 1}
loss = loss_fn(logits, targets)

# Per-class logging (e.g. PyTorch Lightning)
loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
for c in valid.nonzero(as_tuple=True)[0].tolist():
    self.log(f"train/ap_loss_class_{c}", per_class[c])

# Seq2seq: flatten upstream
logits  = logits.view(-1, C)
targets = targets.view(-1)
loss = loss_fn(logits, targets)

# Reset queue between training and validation
loss_fn.reset_queue()
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_classes` | required | Number of output classes; use `1` for binary |
| `queue_size` | `1024` | Circular buffer size (rows); `0` to disable |
| `temperature` | `0.01` | Sigmoid sharpness τ; smaller = sharper gradients |
| `reduction` | `'mean'` | `'mean'`, `'sum'`, or `'none'` |
| `ignore_index` | `-100` | Target value for padding positions |
| `update_queue_in_eval` | `False` | Allow queue updates during `model.eval()` |
| `quantile` | `0.005` | *(RecallAtQuantileLoss only)* Top fraction to target |
| `quantile_interpolation` | `'higher'` | *(RecallAtQuantileLoss only)* `torch.quantile` interpolation method |

**Temperature guidance:** `0.005–0.05` is the practical range. Lower values approximate the true discontinuous rank more closely but produce harder gradients.

**Queue size guidance:** For `quantile=0.005` (top 50 bps) you need at least ~200 samples in the pool for a meaningful 99.5th percentile estimate.

## `LossWarmupWrapper` — BCE/CE warmup + loss blending + geometric temperature decay

A wrapper that trains with a standard loss (e.g. `CrossEntropyLoss`) for a warmup period, optionally blends both losses over a transition period, then switches to the ranking loss with a geometrically decaying temperature schedule.

```
temp(t) = temp_start × (temp_end / temp_start) ^ (elapsed_steps / temp_decay_steps)
```

The schedule clock starts at the moment of phase switch, not at training start.

**Queue poisoning fix:** At the switch point the wrapper automatically calls `main_loss.reset_queue()` (if available), ensuring the ranking loss never sees stale warmup-era logits.

### Blending

`blend_epochs` adds a linear ramp between warmup and pure AP:

```
Epoch 0–W-1:  warmup_loss only          (ap_weight = 0)
Epoch W:      (1−w)×warmup + w×AP       w = 1/(blend_epochs+1)
Epoch W+1:    (1−w)×warmup + w×AP       w = 2/(blend_epochs+1)
...
Epoch W+B+:   main_loss only            (ap_weight = 1)
```

With `warmup_epochs=2, blend_epochs=2`: epochs 2→`1/3 AP`, 3→`2/3 AP`, 4+→pure AP.

### Usage (PyTorch Lightning)

```python
from proxy_losses import SmoothAPLoss
from proxy_losses import LossWarmupWrapper

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
            warmup_epochs=5,
            blend_epochs=2,        # gradual transition
            temp_start=0.5,        # soft at switch — stable gradients
            temp_end=0.01,         # sharp after schedule — closer to true rank
            temp_decay_steps=50_000,
        )

    def on_train_epoch_start(self):
        self.loss_fn.on_train_epoch_start(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        self.loss_fn.on_train_batch_start(self.global_step)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss)
        self.log("train/ap_weight", self.loss_fn.ap_weight)
        if (t := self.loss_fn.current_temperature) is not None:
            self.log("train/temperature", t)
        return loss
```

`**kwargs` (e.g. `return_per_class=True`) are forwarded to `main_loss` only when `ap_weight == 1.0`; silently ignored during warmup and blend phases.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `warmup_loss` | required | Loss used during warmup; must accept `(logits, targets)` |
| `main_loss` | required | Loss used after warmup; must accept `(logits, targets, **kwargs)` |
| `warmup_epochs` | required | Epochs to use `warmup_loss`; `0` to skip warmup entirely |
| `temp_start` | required | Temperature at phase switch |
| `temp_end` | required | Temperature after `temp_decay_steps` steps |
| `temp_decay_steps` | required | Steps over which to decay temperature |
| `blend_epochs` | `0` | Epochs to linearly ramp from warmup to main loss; `0` = hard switch |
| `reset_queue_each_epoch` | `False` | Call `main_loss.reset_queue()` at the start of each main-phase epoch |

### Properties / methods

| | Description |
|---|---|
| `in_warmup` | `True` while `epoch < warmup_epochs` |
| `in_blend` | `True` during the `blend_epochs` transition period |
| `ap_weight` | Current AP loss weight: `0.0` during warmup, linear ramp during blend, `1.0` after |
| `current_temperature` | Current `main_loss.temperature`, or `None` if unavailable |
| `on_train_epoch_start(epoch)` | Advance epoch counter; detect phase switch; optionally reset queue |
| `on_train_batch_start(global_step)` | Latch `switch_step` on first main-phase batch; reset queue; update temperature |

## Examples

Require the `demo` extras:

```bash
uv sync --extra demo
# or: pip install scikit-learn
```

### `toy_demo.py` — single-run trace

Trains one model (warmup → blend → AP) and prints epoch-by-epoch phase, ap_weight, temperature, loss, and AUCPR.

```bash
python examples/toy_demo.py                    # default: 3 warmup + 2 blend epochs
python examples/toy_demo.py --blend-epochs 0   # hard switch (no blend)
python examples/toy_demo.py --pos-rate 0.05    # easier problem
```

### `compare_demo.py` — side-by-side comparison

Trains three models on the same data and seed and prints a per-epoch AUCPR table:

| Strategy | Description |
|---|---|
| warmup-only | BCE for all epochs; never switches to AP |
| AP-only | SmoothAPLoss from epoch 0, no warmup |
| warmup+blend | BCE warmup → linear blend → pure SmoothAPLoss |

```bash
python examples/compare_demo.py
python examples/compare_demo.py --pos-rate 0.05
python examples/compare_demo.py --warmup-epochs 5 --blend-epochs 3
```

Key flags (both scripts): `--pos-rate`, `--warmup-epochs`, `--blend-epochs`, `--total-epochs`, `--batch-size`, `--queue-size`, `--temp-start`, `--temp-end`, `--lr`, `--seed`.

## Tests

```bash
pytest tests/ -v
```

## References

Brown, A., Xie, W., Kalogeiton, V., & Zisserman, A. (2020). [Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163). *ECCV 2020*.
