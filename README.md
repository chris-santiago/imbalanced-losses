# imbalanced-losses

**imbalanced-losses** is a PyTorch library of training losses for class-imbalanced classification — including Focal Loss, Smooth-AP, and Recall-at-Quantile — with built-in DDP all-gather support for globally-correct rank estimation and normalization across multi-GPU training.

**What's in it:**

- **`SigmoidFocalLoss`** — Binary/multi-label focal loss (Lin et al., ICCV 2017). Sigmoid activation; `alpha` re-balances pos/neg, `gamma` down-weights easy examples. Drop-in replacement for `BCEWithLogitsLoss`.
- **`SoftmaxFocalLoss`** — Multiclass focal loss with softmax. Supports `mean_positive` reduction (RetinaNet convention: normalize by positive count), per-class `alpha` weighting, label smoothing, and arbitrary spatial/sequence input shapes.
- **`SmoothAPLoss`** — Differentiable approximation of AP (Brown et al., ECCV 2020). Uses sigmoid-based soft rank estimation; O(|P|×M) where |P| is the positive count and M = batch + queue size. Supports multi-class, binary, and seq2seq settings.
- **`RecallAtQuantileLoss`** — Optimizes recall above a score threshold set at the *q*-th quantile of the pooled distribution. Useful for alert/detection workloads (e.g. top 0.5% of scores).
- **`LossWarmupWrapper`** — Training utility that runs a standard loss (BCE/CE) during warmup, linearly blends into the ranking loss over a configurable transition window, then applies geometric temperature decay. Automatically resets the memory queue at the phase switch to prevent queue poisoning from warmup-era logits.

**Design points:**
- Circular memory queue stabilizes gradient estimates across small batches — critical at low positive rates (e.g. 0.5%)
- Compatible with PyTorch Lightning via `on_train_epoch_start` / `on_train_batch_start` hooks
- `max_pool_size` caps the pairwise matrix for seq2seq / large-pool settings without reducing batch size
- Variable dim-0 DDP all-gather — no `drop_last=True` required; unequal last-batch sizes across ranks are handled automatically
- Six runnable example scripts covering binary, multiclass, focal, per-class logging, and warmup workflows

## Losses

### `SigmoidFocalLoss` — Focal Loss, binary / multi-label (Lin et al., 2017)

Replaces `BCEWithLogitsLoss` for imbalanced binary or multi-label classification. `gamma` suppresses the contribution of easy (well-classified) examples so training focuses on hard ones; `alpha` re-weights the positive class:

```
p_t  = sigmoid(logit) · y  +  (1 − sigmoid(logit)) · (1 − y)
loss = −α_t · (1 − p_t)^γ · log(p_t)
```

```python
from imbalanced_losses import SigmoidFocalLoss

loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
logits  = torch.randn(32, 1)          # arbitrary shape
targets = torch.randint(0, 2, (32, 1)).float()  # float 0/1
loss = loss_fn(logits, targets)
loss.backward()
```

### `SoftmaxFocalLoss` — Focal Loss, multiclass (Lin et al., 2017)

Extends focal loss to mutually-exclusive classification via softmax. Supports all standard input shapes `(N, C)`, `(N, C, L)`, `(N, C, H, W)`, etc.

```python
from imbalanced_losses import SoftmaxFocalLoss

# Standard multiclass
loss_fn = SoftmaxFocalLoss(gamma=2.0, reduction="mean")
logits  = torch.randn(32, 10)         # [N, C]
targets = torch.randint(0, 10, (32,)) # [N] integer labels
loss = loss_fn(logits, targets)

# RetinaNet-style: normalize by positive count, not total
loss_fn = SoftmaxFocalLoss(
    gamma=2.0,
    alpha=[0.25] * 10,        # per-class weights
    reduction="mean_positive", # denominator = #positives only
    background_class=0,
    ignore_index=-100,
)
loss = loss_fn(logits, targets)
```

**`mean_positive` reduction:** The numerator sums loss over *all* valid (non-ignored) positions including background. The denominator counts only non-background valid positions. This matches the RetinaNet convention and stabilizes the loss scale when the vast majority of samples are background.

### `SmoothAPLoss` — Smooth Average Precision (Brown et al., 2020)

Approximates AP using sigmoid-based soft rank estimation. For each positive *i* in the pool:

```
ŝ_i   = 1 + Σ_{j≠i}       σ((s_j − s_i) / τ)   # soft overall rank
ŝ_i^+ = 1 + Σ_{j≠i, j∈P} σ((s_j − s_i) / τ)   # soft rank among positives
AP ≈ (1/|P|) · Σ_{i∈P}  ŝ_i^+ / ŝ_i
loss = 1 − AP
```

**Complexity:** O(|P|×M) where |P| is the number of positives and M = batch + queue size. At a 0.5% positive rate this is ~200× cheaper than O(M²). Use `max_pool_size` to cap M for seq2seq or other large-pool settings without reducing batch size.

### `RecallAtQuantileLoss` — Recall at Quantile

Optimizes recall above a score threshold set at the *q*-th quantile of the pooled score distribution. The threshold is treated as a stop-gradient constant each forward pass:

```
θ = quantile(scores, 1 − q)          [detached — no grad]
soft_recall = (1/|P|) · Σ_{i∈P} σ((s_i − θ) / τ)
loss = 1 − soft_recall
```

Gradient flows only through positive scores, pushing them above the cutoff. Useful for alert/detection settings (e.g. `quantile=0.005` = top 50 bps).

## Features

**All losses** support DDP all-gather via `gather_distributed` (auto-detected by default).

**Focal losses** (`SigmoidFocalLoss`, `SoftmaxFocalLoss`):
- Arbitrary input shapes — `(N, C)`, `(N, C, L)`, `(N, C, H, W)`, …
- `ignore_index` masking — padded positions contribute zero loss and zero gradient
- `mean` reduction divides by valid (non-ignored) count, not total tensor size
- `mean_positive` reduction (softmax only) — normalizes by positive count for detection tasks
- `alpha` — scalar (sigmoid) or per-class tensor (softmax) class reweighting
- `label_smoothing` (softmax only) — forwarded directly to `F.cross_entropy`

**Ranking losses** (`SmoothAPLoss`, `RecallAtQuantileLoss`):
- **Memory queue** — circular buffer accumulates past batches to stabilize estimates over small batch sizes; set `queue_size=0` to disable
- **Multi-class** — one-vs-rest per class using `logits[:, c]`
- **Binary** — set `num_classes=1` with targets in `{0, 1}`
- **Seq2seq** — flatten `[B, T, C]` → `[B*T, C]` upstream before passing
- **Pool size cap** — `max_pool_size` applies minimum-quota subsampling after the gather+queue merge, bounding pairwise matrix memory for large M (e.g. seq2seq with long sequences)
- **Padding** — `ignore_index` rows are excluded from ranking and the positive set
- **Eval queue freeze** — `update_queue_in_eval=False` (default) prevents validation-phase logits from contaminating the training queue
- **Reductions** — `'mean'` (default), `'sum'`, or `'none'` (per-class tensor; degenerate classes are `nan`)
- **Per-class logging** — `return_per_class=True` returns `(loss, per_class, valid_mask)` without a second forward pass

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.8.

```bash
# from PyPI
pip install imbalanced-losses

# from GitHub (latest dev)
pip install git+https://github.com/chris-santiago/imbalanced-losses.git

# with uv (for development / contributing)
git clone https://github.com/chris-santiago/imbalanced-losses.git
cd imbalanced-losses
uv sync
```

To run the example scripts, install the optional demo dependencies:

```bash
pip install "imbalanced-losses[demo]"
# or with uv:
uv sync --extra demo
```

## Usage

```python
from imbalanced_losses import SmoothAPLoss
from imbalanced_losses import RecallAtQuantileLoss

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

### Focal losses

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.25` / `None` | Pos/neg balance weight in `[0,1]` or `-1` to disable (sigmoid); per-class tensor or `None` (softmax) |
| `gamma` | `2.0` | Focusing exponent; `0` recovers vanilla BCE/CE |
| `reduction` | `'none'` / `'mean'` | `'none'`, `'mean'`, `'sum'`, or `'mean_positive'` (softmax only) |
| `ignore_index` | `-100` | *(SoftmaxFocalLoss only)* Target value for padding positions |
| `background_class` | `0` | *(SoftmaxFocalLoss only)* Class excluded from `mean_positive` denominator |
| `label_smoothing` | `0.0` | *(SoftmaxFocalLoss only)* Forwarded to `F.cross_entropy` |
| `gather_distributed` | `None` | `None` = auto-detect DDP; `False` = always local; `True` = always gather |

### Ranking losses

| Parameter | Default | Description |
|---|---|---|
| `num_classes` | required | Number of output classes; use `1` for binary |
| `queue_size` | `1024` | Circular buffer size (rows); `0` to disable |
| `temperature` | `0.01` | Sigmoid sharpness τ; smaller = sharper gradients |
| `reduction` | `'mean'` | `'mean'`, `'sum'`, or `'none'` |
| `ignore_index` | `-100` | Target value for padding positions |
| `update_queue_in_eval` | `False` | Allow queue updates during `model.eval()` |
| `gather_distributed` | `None` | `None` = auto-detect DDP; `False` = always local; `True` = always gather |
| `max_pool_size` | `None` | Cap on pool rows after gather+queue merge; minimum-quota subsampling applied when exceeded. Size as `target_|P_c| × 2 × n_classes`. `None` disables. |
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
Epoch 0–W-1:  warmup_loss only          (main_weight = 0)
Epoch W:      (1−w)×warmup + w×AP       w = 1/(blend_epochs+1)
Epoch W+1:    (1−w)×warmup + w×AP       w = 2/(blend_epochs+1)
...
Epoch W+B+:   main_loss only            (main_weight = 1)
```

With `warmup_epochs=2, blend_epochs=2`: epochs 2→`1/3 AP`, 3→`2/3 AP`, 4+→pure AP.

### Usage (PyTorch Lightning)

```python
from imbalanced_losses import SmoothAPLoss
from imbalanced_losses import LossWarmupWrapper

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
        self.log("train/main_weight", self.loss_fn.main_weight)
        if (t := self.loss_fn.current_temperature) is not None:
            self.log("train/temperature", t)
        return loss
```

`**kwargs` (e.g. `return_per_class=True`) are forwarded to `main_loss` only when `main_weight == 1.0`; silently ignored during warmup and blend phases.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `warmup_loss` | required | Loss used during warmup; must accept `(logits, targets)` |
| `main_loss` | required | Loss used after warmup; must accept `(logits, targets, **kwargs)` |
| `warmup_epochs` | `0` | Epochs to use `warmup_loss`; `0` skips warmup entirely |
| `temp_start` | `0.05` | Temperature at phase switch |
| `temp_end` | `0.005` | Temperature after `temp_decay_steps` steps |
| `temp_decay_steps` | `10_000` | Steps over which to decay temperature |
| `blend_epochs` | `0` | Epochs to linearly ramp from warmup to main loss; `0` = hard switch |
| `warmup_steps` | `None` | Step-based warmup (mutually exclusive with `warmup_epochs > 0`) |
| `blend_steps` | `None` | Step-based blend (requires `warmup_steps`; mutually exclusive with `blend_epochs > 0`) |
| `final_main_weight` | `1.0` | Main loss weight after blend completes; `< 1.0` keeps a permanent warmup/main mix |
| `reset_queue_each_epoch` | `False` | Call `main_loss.reset_queue()` at the start of each main-phase epoch |
| `gather_distributed` | `None` | Forwarded to `main_loss.gather_distributed` if the attribute exists |

### Properties / methods

| | Description |
|---|---|
| `in_warmup` | `True` while `epoch < warmup_epochs` |
| `in_blend` | `True` during the `blend_epochs` transition period |
| `main_weight` | Current main loss weight: `0.0` during warmup, linear ramp during blend, `1.0` after |
| `current_temperature` | Current `main_loss.temperature`, or `None` if unavailable |
| `on_train_epoch_start(epoch)` | Advance epoch counter; detect phase switch; optionally reset queue |
| `on_train_batch_start(global_step)` | Latch `switch_step` on first main-phase batch; reset queue; update temperature |

## Distributed Training (DDP)

All losses support DDP via built-in all-gather, but globally-correct computation is especially critical for rank-based losses. The `imbalanced_losses.distributed` module provides two all-gather helpers that handle this correctly.

### Why this matters

In DDP each GPU sees only `N/world_size` samples. The soft-rank computation in `SmoothAPLoss` and the quantile threshold in `RecallAtQuantileLoss` become noisy or biased when computed on a shard. For `SoftmaxFocalLoss` with `mean_positive` reduction, the positive count in the denominator is similarly unreliable when positives are rare and unevenly distributed across ranks. Gathering logits and targets across all workers before passing them to the loss fixes this for all three cases.

### Helpers

| Function | Description |
|---|---|
| `all_gather_with_grad(tensor)` | Gathers tensors across all workers; **preserves gradients for the local rank's slice** so autograd works correctly. Variable dim-0 sizes across ranks are supported. |
| `all_gather_no_grad(tensor)` | Gathers tensors without gradient tracking; use for integer targets/labels. Variable dim-0 sizes across ranks are supported. |

`all_gather_with_grad` replaces the local rank's slice in the output with the original tensor (restoring the gradient connection), while other workers' slices remain detached — matching standard DDP semantics where each worker optimizes its own parameters via all-reduced gradients.

**Variable batch sizes:** Both helpers handle unequal dim-0 sizes across ranks (e.g. the last batch without `drop_last=True`). Tensors are zero-padded to the maximum size for the collective, then trimmed before concatenation. An equal-size fast path skips the overhead when all ranks contribute the same number of rows.

**Queue synchronization:** Because every worker calls `all_gather` before passing to the loss, every worker enqueues the same global-batch data. No extra synchronization of the memory queue is needed.

### Usage

```python
from imbalanced_losses import SmoothAPLoss
from imbalanced_losses.distributed import all_gather_with_grad, all_gather_no_grad

loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024)

# Inside training_step on each GPU:
logits_global  = all_gather_with_grad(logits)   # [sum(N_i), C] — grad flows for local rank
targets_global = all_gather_no_grad(targets)    # [sum(N_i)]    — no grad
loss = loss_fn(logits_global, targets_global)
loss.backward()
```

Both helpers raise `RuntimeError` if `torch.distributed` is not available or not initialized. They are no-ops (return the input unchanged) when `world_size == 1`.

### PyTorch Lightning (DDP)

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper
from imbalanced_losses.distributed import all_gather_with_grad, all_gather_no_grad

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        logits_g  = all_gather_with_grad(logits)
        targets_g = all_gather_no_grad(targets)
        loss = self.loss_fn(logits_g, targets_g)
        return loss
```

## Examples

Require the `demo` extras:

```bash
uv sync --extra demo
# or: pip install scikit-learn
```

### [`toy_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/toy_demo.py) — single-run trace

Trains one model (warmup → blend → AP) and prints epoch-by-epoch phase, main_weight, temperature, loss, and AUCPR.

```bash
python examples/toy_demo.py                    # default: 3 warmup + 2 blend epochs
python examples/toy_demo.py --blend-epochs 0   # hard switch (no blend)
python examples/toy_demo.py --pos-rate 0.05    # easier problem
```

### [`focal_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/focal_demo.py) — BCE vs focal loss comparison

Trains four models on the same imbalanced data and prints per-epoch AUCPR:

| Strategy | Description |
|---|---|
| BCE | Vanilla `BCEWithLogitsLoss`; easy negatives dominate |
| BCE+weight | `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos` |
| focal α γ | `SigmoidFocalLoss(alpha=0.25, gamma=2)` — RetinaNet defaults |
| focal γ only | `SigmoidFocalLoss(alpha=-1, gamma=2)` — focusing only, no alpha |

```bash
python examples/focal_demo.py
python examples/focal_demo.py --pos-rate 0.02   # easier problem
python examples/focal_demo.py --gamma 5 --alpha 0.5
```

### [`compare_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/compare_demo.py) — side-by-side comparison

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

### [`binary_imbalance_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/binary_imbalance_demo.py) — positive-rate sweep

Sweeps positive rates from 25% down to 0.5% and compares `SmoothAPLoss`, `BCEWithLogitsLoss`, and `SigmoidFocalLoss`. Shows where SmoothAP's ranking advantage over BCE becomes meaningful.

```bash
python examples/binary_imbalance_demo.py --sweep          # summary table across all rates
python examples/binary_imbalance_demo.py --positive-rate 0.005  # per-epoch curve at 0.5%
```

### [`multiclass_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/multiclass_demo.py) — CE vs Focal vs SmoothAP (multiclass)

Trains three models on the same imbalanced multiclass data and prints a per-epoch macro-AP table: `CrossEntropyLoss`, `SoftmaxFocalLoss`, and `SmoothAPLoss` with warmup.

```bash
python examples/multiclass_demo.py
python examples/multiclass_demo.py --n-classes 8 --pos-rate 0.05
```

### [`per_class_metrics_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/per_class_metrics_demo.py) — per-class AP logging

Demonstrates `return_per_class=True` for both `SmoothAPLoss` and `RecallAtQuantileLoss`, including the `valid_mask` guard pattern for degenerate classes.

```bash
python examples/per_class_metrics_demo.py
```

## Tests

```bash
pytest tests/ -v
```

## References

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). *ICCV 2017*.

Brown, A., Xie, W., Kalogeiton, V., & Zisserman, A. (2020). [Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163). *ECCV 2020*.
