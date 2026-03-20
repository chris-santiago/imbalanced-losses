# imbalanced-losses

**imbalanced-losses** is a PyTorch library of training losses for class-imbalanced classification. It provides Focal Loss, Smooth Average Precision (Smooth-AP), and Recall-at-Quantile, all with built-in DDP all-gather support for globally-correct rank estimation and normalization across multi-GPU training.

## When to use it

Use imbalanced-losses when:

- Your dataset has significant class imbalance (e.g. fraud detection, rare event classification, object detection with many background anchors)
- Standard cross-entropy or BCE loss produces degenerate models that predict the majority class
- You are optimizing for ranking-based metrics (Average Precision, Recall at a fixed operating point) rather than accuracy
- You are running distributed multi-GPU training and need losses that are globally correct across all workers

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.8.

```bash
pip install imbalanced-losses
```

For development or contributing:

```bash
git clone https://github.com/chris-santiago/imbalanced-losses.git
cd imbalanced-losses
uv sync
```

## Losses at a glance

| Loss | Use case |
|---|---|
| `SigmoidFocalLoss` | Binary / multi-label (sigmoid per logit, classes are independent); drop-in for `BCEWithLogitsLoss` |
| `SoftmaxFocalLoss` | Mutually-exclusive multiclass (softmax couples all class logits); drop-in for `CrossEntropyLoss` |
| `SmoothAPLoss` | Directly optimizes Average Precision |
| `RecallAtQuantileLoss` | Optimizes recall above a fixed score threshold |
| `LossWarmupWrapper` | Warmup on CE/BCE, then blend/anneal into a ranking loss |

## Documentation sections

- [**Tutorials**](tutorials/index.md) — hands-on walkthroughs that take you from zero to a working training loop
- [**How-To Guides**](how-to/index.md) — goal-oriented recipes for common tasks
- [**Reference**](reference/index.md) — full API documentation for every public class and function
- [**Explanation**](explanation/index.md) — background on design decisions, trade-offs, and non-obvious behavior
