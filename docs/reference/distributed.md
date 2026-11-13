# Distributed Utilities

Helper functions for DDP all-gather with correct gradient handling. Located in `imbalanced_losses.distributed`.

::: imbalanced_losses.distributed.all_gather_with_grad

---

::: imbalanced_losses.distributed.all_gather_no_grad

---

## Usage pattern

```python
from imbalanced_losses.distributed import all_gather_with_grad, all_gather_no_grad

# The loss must not gather again: construct it with gather_distributed=False.
loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024, gather_distributed=False)

# In a DDP training step:
logits_global  = all_gather_with_grad(logits)    # [sum(N_i), C] — grad flows, scaled by world_size
targets_global = all_gather_no_grad(targets)     # [sum(N_i)]   — no grad

loss = loss_fn(logits_global, targets_global)
loss.backward()
```

`all_gather_with_grad` scales the local slice's gradient by `world_size` so that DDP's gradient averaging yields the full-batch gradient. Pass `scale_grad=False` when gradients are summed across ranks instead of averaged; the returned values are identical either way.

## Behavior summary

| Function | Gradient | Use for |
|---|---|---|
| `all_gather_with_grad` | Flows through local rank's slice, scaled by `world_size` (disable with `scale_grad=False`) | Logits, embeddings |
| `all_gather_no_grad` | None | Integer targets, labels |

Both functions:

- Raise `RuntimeError` if `torch.distributed` is not available or not initialized
- Are no-ops (return input unchanged) when `world_size == 1`
- Support **variable dim-0 sizes** across ranks (e.g. unequal last-batch without `drop_last=True`). Tensors are padded to the max size for the collective, then trimmed. An equal-size fast path skips this overhead when all ranks have the same batch size.
