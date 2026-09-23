# Memory Queue Design

## The problem: sparse positives per batch

`SmoothAPLoss` estimates Average Precision by comparing every positive in the pool against every other sample. At a 1% positive rate with a batch size of 32, you expect about 0 or 1 positives per batch. A pool of 1–2 positives produces a near-zero AP estimate with essentially no gradient signal.

The memory queue solves this by accumulating past batches. With a `queue_size=1024` and a batch size of 32, the total pool is ~1056 samples, yielding ~10 positives at a 1% rate, enough for a stable AP estimate.

## How the circular buffer works

The queue is a fixed-size circular buffer of `(logits, targets)` rows. On every training forward pass:

1. The live batch is appended to the queue contents to form the full pool
2. AP or recall is computed on the full pool
3. The live batch is written into the queue, overwriting the oldest entries

Queue entries are stored *detached*: no gradient flows through queued logits. Gradients only flow through the live batch's portion of the pool. This is important: you cannot backpropagate through historical logits that were computed by a previous version of the model.

## Why detaching queue logits is correct

At first this seems like it would bias the soft-rank computation: the rank of a live positive is estimated relative to a pool that includes stale, detached logits. In practice this bias is small because:

1. The queue rotates, so entries are never more than `queue_size / batch_size` steps old
2. Soft ranks are a sum over the pool, so the live-batch contribution is fully differentiable
3. The queue mainly provides a *reference distribution* for the rank, not a gradient signal

This is the same reasoning used in MoCo-style contrastive learning, where negative key embeddings are also kept in a queue with detached gradients.

## Queue poisoning and the phase switch

When `LossWarmupWrapper` switches from BCE warmup to AP loss, the queue contains logits from a model trained with BCE. These "warmup-era" logits may have a very different score distribution than the AP-phase model; the ranking statistics are meaningless.

If these stale entries remain in the queue, the AP loss computes ranks relative to a corrupted reference distribution for the first `queue_size / batch_size` batches of the AP phase.

`LossWarmupWrapper` prevents this by automatically calling `main_loss.reset_queue()` at the exact step of the phase switch. After reset, the queue fills with AP-phase logits over the next few batches before the full pool is used.

## When to reset manually

- Between training and validation: `loss_fn.reset_queue()` before the val loop prevents training logits from appearing in val-phase AP estimates
- After changing model architecture or checkpoint
- When `reset_queue_each_epoch=True` in `LossWarmupWrapper` is set, which is useful when the model changes significantly epoch-to-epoch and stale logits would bias ranking

## Stored `sample_weight`

Each queue row stores a fourth value alongside `logits`, `targets`, and the iid flag: the weight it was enqueued with. Rows enqueued from a call that supplied `sample_weight` store that weight; rows enqueued from a call that did not (including every row enqueued before this feature existed) store `1.0`, so an unweighted step following a weighted one contributes weight-`1` rows to the pool while the pool's older weighted rows keep their stored weights. Mixed weighted/unweighted training is well-defined. `reset_queue()` restores every stored weight to `1.0` along with the rest of the buffer.

The weighted arithmetic in `_compute_per_class` only activates when this call supplied a weight or the queue holds at least one row with a stored weight other than `1`; the queue exposes this as `has_weights`, a plain Python `bool` re-evaluated from the stored weights every time the buffer changes (enqueue, `reset_queue()`, checkpoint load). It is a property of the rows currently held, not of the call history: enqueuing an explicitly all-ones weight leaves it `False`, and once the last non-unit row has been overwritten by later unweighted steps it returns to `False`. This is what keeps the unweighted code path bitwise unchanged when `sample_weight` is never used: no weight tensor is ever materialized, and `merge()` performs exactly the operations it did before this feature existed.

**Checkpoint compatibility:** a checkpoint saved before `sample_weight` existed has no `_q_weight` buffer in its state dict. Loading it with `strict=True` still succeeds: the missing buffer is injected as `torch.ones(queue_size)` before `load_state_dict` runs its strict-key check, mirroring the existing `_q_iid` shim. A checkpoint saved *with* weighted queue rows restores `has_weights=True` on load by inspecting the loaded `_q_weight` buffer (`(w != 1).any()`), since `has_weights` itself is not persisted.

## Queue size vs. pool size limits

The core AP computation is O(|P| × M) by construction (only the positive rows of the pairwise comparison matrix are formed), where M = batch + queue. The positive rate does not change that complexity; it determines the savings factor relative to a naive O(M²) implementation (about 200× at a 0.5% positive rate). M still has a practical upper limit of ~4096 for reasonable training step times on a single GPU. At a 0.5% positive rate with M=4096, you get ~20 positives, a comfortable signal.

## DDP queue synchronization

In distributed training, every worker calls `all_gather` before passing to the loss. This means every worker sees the same global batch and enqueues the same data. No explicit queue synchronization across workers is needed: they are identical by construction.

One caveat on loss values (not queues): if `max_pool_size` subsampling triggers, each rank draws its own random subset (unseeded `torch.randperm` per rank), so per-rank loss values can differ slightly on those steps. The queues remain identical, because the full post-gather batch is enqueued, not the subsampled pool.
