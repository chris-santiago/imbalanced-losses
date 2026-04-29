# Distributed Utilities

Helper functions for DDP all-gather with correct gradient handling. Located in `imbalanced_losses.distributed`.

::: imbalanced_losses.distributed.all_gather_with_grad

---

::: imbalanced_losses.distributed.all_gather_no_grad

---

## Usage pattern

```python
from imbalanced_losses.distributed import all_gather_with_grad, all_gather_no_grad

# In a DDP training step:
logits_global  = all_gather_with_grad(logits)    # [sum(N_i), C] — grad flows
targets_global = all_gather_no_grad(targets)     # [sum(N_i)]   — no grad

loss = loss_fn(logits_global, targets_global)
loss.backward()
```

## Behavior summary

| Function | Gradient | Use for |
|---|---|---|
| `all_gather_with_grad` | Flows through local rank's slice | Logits, embeddings |
| `all_gather_no_grad` | None | Integer targets, labels |

Both functions:

- Raise `RuntimeError` if `torch.distributed` is not available or not initialized
- Are no-ops (return input unchanged) when `world_size == 1`
- Support **variable dim-0 sizes** across ranks (e.g. unequal last-batch without `drop_last=True`). Tensors are padded to the max size for the collective, then trimmed. An equal-size fast path skips this overhead when all ranks have the same batch size.
