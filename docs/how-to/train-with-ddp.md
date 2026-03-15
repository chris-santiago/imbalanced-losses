# Train with DDP

All losses in imbalanced-losses support multi-GPU training via PyTorch's DistributedDataParallel. By default, losses auto-detect when a distributed process group is initialized and all-gather inputs before computing.

## Auto-detection (default behavior)

No extra configuration is needed. The first forward call after `dist.init_process_group` will detect the distributed context and begin gathering:

```python
from imbalanced_losses import SmoothAPLoss

loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024)
# gather_distributed=None (default): auto-detects DDP on first forward
```

## Explicit manual gather

For full control, disable auto-detection and gather inputs yourself before calling the loss. This is the recommended pattern when you need to verify shapes or log global batch statistics:

```python
from imbalanced_losses import SmoothAPLoss
from imbalanced_losses.distributed import all_gather_with_grad, all_gather_no_grad

loss_fn = SmoothAPLoss(num_classes=4, queue_size=1024, gather_distributed=False)

# In your training step (each GPU):
logits_global  = all_gather_with_grad(logits)    # [world*N, C], grad flows
targets_global = all_gather_no_grad(targets)     # [world*N],    no grad
loss = loss_fn(logits_global, targets_global)
loss.backward()
```

**Important:** Use `all_gather_with_grad` for logits (gradients must flow back to model parameters) and `all_gather_no_grad` for integer targets (no gradient required).

## PyTorch Lightning with DDP

```python
import pytorch_lightning as pl
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper
from imbalanced_losses.distributed import all_gather_with_grad, all_gather_no_grad

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = ...
        self.loss_fn  = SmoothAPLoss(num_classes=4, queue_size=1024)
        # gather_distributed=None: loss will auto-gather when dist is active

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.backbone(x)

        # Option A: let the loss handle gathering automatically
        loss = self.loss_fn(logits, targets)

        # Option B: gather manually for visibility
        # (construct with gather_distributed=False first, then gather yourself)
        # logits_g  = all_gather_with_grad(logits)
        # targets_g = all_gather_no_grad(targets)
        # loss = self.loss_fn(logits_g, targets_g)

        return loss
```

## Opt out of gathering for a single loss

If you explicitly do not want gathering (e.g. debugging on a single GPU while a distributed group is initialized), set `gather_distributed=False`:

```python
loss_fn = SmoothAPLoss(num_classes=4, gather_distributed=False)
```

## Confirm distributed setup

Both helpers raise `RuntimeError` if called before `dist.init_process_group`. They are no-ops (return the input unchanged) when `world_size == 1`:

```python
# Single GPU: no-op, returns tensor unchanged
logits_global = all_gather_with_grad(logits)
assert logits_global is logits   # True on single GPU
```
