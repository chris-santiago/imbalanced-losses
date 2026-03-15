# How-To Guides

How-to guides are goal-oriented recipes. Each one solves a specific task — assume you already know the basics from the [Getting Started tutorial](../tutorials/getting-started.md).

## Available guides

- [**Use Focal Loss**](use-focal-loss.md) — `SigmoidFocalLoss` for binary/multi-label, `SoftmaxFocalLoss` for multiclass, with `alpha`, `gamma`, and `mean_positive` reduction
- [**Use Ranking Losses**](use-ranking-losses.md) — `SmoothAPLoss` and `RecallAtQuantileLoss` with queue sizing and temperature guidance
- [**Configure Warmup and Blending**](configure-warmup.md) — tune phase schedules, blend epochs, and temperature decay in `LossWarmupWrapper`
- [**Train with DDP**](train-with-ddp.md) — multi-GPU all-gather setup for all losses
- [**Log Per-Class Metrics**](log-per-class-metrics.md) — retrieve per-class loss tensors without a second forward pass
