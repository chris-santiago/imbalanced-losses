# proxy-losses

Differentiable proxy losses for ranking metrics in PyTorch — drop-in `nn.Module` replacements for cross-entropy when you care about AP or recall at a specific operating point.

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

Requires Python ≥ 3.10 and PyTorch ≥ 2.10.

```bash
# with uv
uv sync

# or pip
pip install torch>=2.10
```

## Usage

```python
from ap_loss import SmoothAPLoss
from recall_loss import RecallAtQuantileLoss

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

## Tests

```bash
pytest test_smooth_ap_loss.py test_recall_at_quantile_loss.py -v
```

## References

Brown, A., Xie, W., Kalogeiton, V., & Zisserman, A. (2020). [Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163). *ECCV 2020*.
