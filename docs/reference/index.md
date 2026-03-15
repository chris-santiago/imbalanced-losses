# API Reference

Full documentation for every public class and function in `imbalanced_losses`.

## Losses

| Class | Module | Description |
|---|---|---|
| [`SigmoidFocalLoss`](sigmoid-focal-loss.md) | `imbalanced_losses` | Binary / multi-label focal loss |
| [`SoftmaxFocalLoss`](softmax-focal-loss.md) | `imbalanced_losses` | Multiclass focal loss with softmax |
| [`SmoothAPLoss`](smooth-ap-loss.md) | `imbalanced_losses` | Differentiable Average Precision loss |
| [`RecallAtQuantileLoss`](recall-at-quantile-loss.md) | `imbalanced_losses` | Differentiable Recall-at-Quantile loss |

## Utilities

| Class / Function | Module | Description |
|---|---|---|
| [`LossWarmupWrapper`](loss-warmup-wrapper.md) | `imbalanced_losses` | Phase-switching warmup + temperature decay |
| [`all_gather_with_grad`](distributed.md) | `imbalanced_losses.distributed` | All-gather preserving gradient flow |
| [`all_gather_no_grad`](distributed.md) | `imbalanced_losses.distributed` | All-gather without gradient tracking |
