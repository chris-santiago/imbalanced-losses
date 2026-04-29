# DDP All-Gather and Gradients

## Why DDP sharding breaks ranking losses

In standard DDP training, each GPU processes `N / world_size` samples. Each GPU computes the loss independently on its own shard, and gradients are all-reduced across workers before the optimizer step.

This works for losses that decompose across samples — cross-entropy, BCE, MSE. Each sample's loss is independent, so sharding is equivalent to computing on the full batch.

Ranking losses do not decompose. `SmoothAPLoss` estimates the rank of each positive within a pool. A positive on GPU 0 is ranked relative to the `N / world_size` samples on GPU 0, not the full `N` samples. With 4 GPUs and a 1% positive rate, each GPU may see 0 or 1 positives per batch. The rank estimate is not just noisy — it's qualitatively wrong.

`RecallAtQuantileLoss` has the same issue: the quantile threshold is estimated from the local score distribution, which is a biased sample of the global distribution when positives and negatives are unevenly distributed.

`SoftmaxFocalLoss` with `mean_positive` reduction is also affected: if positives happen to land on one GPU, the local positive count is much larger than the global average, inflating the loss normalization.

## The all-gather solution

All-gather collects tensors from all workers and concatenates them before the loss computation. Every worker sees the full global batch, computes the same ranks and thresholds, and produces the same loss value. Gradient all-reduction then works correctly because the loss is the same function of each worker's local parameters.

## Why gradient flow is non-trivial

Standard `dist.all_gather` returns detached tensors — the gathered slices have no connection to the computation graph. If you simply all-gathered logits and passed them to the loss, only 1/world_size of the gradient would flow back to the model parameters (through the local slice), and the other 3/4 would be silently dropped.

`all_gather_with_grad` fixes this by replacing the local rank's slice in the gathered output with the original tensor (which is connected to the computation graph), while keeping other workers' slices detached:

```python
gathered[rank] = tensor   # restores gradient connection
```

This matches DDP semantics: each worker's optimizer step uses gradients from the local model parameters only, but the loss is computed globally.

## Variable batch sizes across ranks

By default, `dist.all_gather` requires every rank to contribute a tensor with exactly the same shape. This breaks when ranks have different batch sizes — for example, the last batch of an epoch when the dataset size is not evenly divisible by the number of GPUs, or when using variable-length data without padding.

Both gather helpers handle this automatically using a **pad-to-max-then-trim** strategy:

1. Each rank broadcasts its local dim-0 size in a cheap scalar collective.
2. Each rank pads its tensor with zeros to the maximum size across all ranks.
3. `dist.all_gather` runs on the padded tensors (now equal shape).
4. Each gathered slice is trimmed back to its true length before concatenation.

Gradient flow is unaffected: the local rank's slice is still replaced with the original (unpadded) tensor, which carries the autograd graph.

When all ranks contribute the same number of rows, an **equal-size fast path** detects this and skips padding and trimming entirely — no overhead compared to the previous behavior.

This means `drop_last=True` in `DistributedSampler` is no longer required for correctness, though it remains a useful optimization to minimize wasted computation from padding.

## All-gather is a no-op for world_size == 1

Both `all_gather_with_grad` and `all_gather_no_grad` check `world_size` and return the input unchanged when running on a single GPU. There is no overhead in single-GPU training.

## Queue synchronization under DDP

Because every worker calls `all_gather` before passing to the loss, every worker enqueues the same full global batch. After any number of training steps, all workers have identical queues — no additional synchronization is needed. This would not hold if each worker enqueued only its local shard.

## Auto-detection vs. explicit control

The default `gather_distributed=None` resolves gathering on the first forward call after the process group is initialized. This is safe because:

1. `dist.is_initialized()` returns `False` before `init_process_group` is called
2. The result is cached — the check happens once, not on every forward

Set `gather_distributed=False` to opt out entirely (e.g. if you are manually gathering before calling the loss, or if you are using a custom distributed backend that does not support `dist.all_gather`).
