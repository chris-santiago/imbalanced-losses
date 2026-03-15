# Explanation

Explanation answers *why* rather than *how*. These articles cover design decisions, trade-offs, and the non-obvious behavior behind the library.

## Articles

- [**Why Imbalanced Losses**](why-imbalanced-losses.md) — why standard cross-entropy fails under class imbalance, and what each loss in this library actually optimizes
- [**Memory Queue Design**](memory-queue.md) — why a circular buffer is necessary for ranking losses at low positive rates, and how it interacts with DDP
- [**Temperature and Soft Ranking**](temperature-soft-ranking.md) — how the sigmoid approximation replaces the discrete rank function and what temperature controls
- [**DDP All-Gather and Gradients**](ddp-all-gather.md) — why rank-based losses are biased under standard DDP sharding and how all-gather fixes this without breaking autograd
