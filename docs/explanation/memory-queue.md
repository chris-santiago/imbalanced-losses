# Memory Queue Design

## The problem: sparse positives per batch

`SmoothAPLoss` estimates Average Precision by comparing every positive in the pool against every other sample. At a 1% positive rate with a batch size of 32, you expect about 0 or 1 positives per batch. A pool of 1–2 positives produces a near-zero AP estimate with essentially no gradient signal.

The memory queue solves this by accumulating past batches. With a `queue_size=1024` and a batch size of 32, the total pool is ~1056 samples, yielding ~10 positives at a 1% rate — enough for a stable AP estimate.

## How the circular buffer works

The queue is a fixed-size circular buffer of `(logits, targets)` rows. On every training forward pass:

1. The live batch is appended to the queue contents to form the full pool
2. AP or recall is computed on the full pool
3. The live batch is written into the queue, overwriting the oldest entries

Queue entries are stored *detached* — no gradient flows through queued logits. Gradients only flow through the live batch's portion of the pool. This is important: you cannot backpropagate through historical logits that were computed by a previous version of the model.

## Why detaching queue logits is correct

At first this seems like it would bias the soft-rank computation: the rank of a live positive is estimated relative to a pool that includes stale, detached logits. In practice this bias is small because:

1. The queue rotates, so entries are never more than `queue_size / batch_size` steps old
2. Soft ranks are a sum over the pool, so the live-batch contribution is fully differentiable
3. The queue mainly provides a *reference distribution* for the rank, not a gradient signal

This is the same reasoning used in MoCo-style contrastive learning, where negative key embeddings are also kept in a queue with detached gradients.

## Queue poisoning and the phase switch

When `LossWarmupWrapper` switches from BCE warmup to AP loss, the queue contains logits from a model trained with BCE. These "warmup-era" logits may have a very different score distribution than the AP-phase model — the ranking statistics are meaningless.

If these stale entries remain in the queue, the AP loss computes ranks relative to a corrupted reference distribution for the first `queue_size / batch_size` batches of the AP phase.

`LossWarmupWrapper` prevents this by automatically calling `main_loss.reset_queue()` at the exact step of the phase switch. After reset, the queue fills with AP-phase logits over the next few batches before the full pool is used.

## When to reset manually

- Between training and validation: `loss_fn.reset_queue()` before the val loop prevents training logits from appearing in val-phase AP estimates
- After changing model architecture or checkpoint
- When `reset_queue_each_epoch=True` in `LossWarmupWrapper` is set — useful when the model changes significantly epoch-to-epoch and stale logits would bias ranking

## Queue size vs. pool size limits

The core AP computation is O(|P| × M) where M = batch + queue. At low positive rates this is closer to O(|P| × M) than O(M²), but M still has a practical upper limit of ~4096 for reasonable training step times on a single GPU. At a 0.5% positive rate with M=4096, you get ~20 positives — a comfortable signal.

## DDP queue synchronization

In distributed training, every worker calls `all_gather` before passing to the loss. This means every worker sees the same global batch and enqueues the same data. No explicit queue synchronization across workers is needed — they are identical by construction.
