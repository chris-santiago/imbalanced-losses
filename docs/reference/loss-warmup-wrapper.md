# LossWarmupWrapper

A training utility that wraps a warmup loss and a main ranking loss. Manages three phases: warmup (standard loss only), optional linear blend, and main phase with geometric temperature decay. Exposes PyTorch Lightning hooks.

::: imbalanced_losses.warmup_wrapper.LossWarmupWrapper

## Quick example

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper
import torch.nn as nn

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_epochs=5,
    blend_epochs=2,
    temp_start=0.5,
    temp_end=0.01,
    temp_decay_steps=50_000,
)
```

## PyTorch Lightning integration

```python
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = LossWarmupWrapper(...)

    def on_train_epoch_start(self):
        self.loss_fn.on_train_epoch_start(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        self.loss_fn.on_train_batch_start(self.global_step)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss)
        self.log("train/ap_weight", self.loss_fn.ap_weight)
        if (t := self.loss_fn.current_temperature) is not None:
            self.log("train/temperature", t)
        return loss
```

## Phase schedule

With `warmup_epochs=5, blend_epochs=2`:

| Epoch range | Phase | `in_warmup` | `in_blend` | `ap_weight` |
|---|---|---|---|---|
| 0–4 | warmup | `True` | `False` | `0.0` |
| 5 | blend | `False` | `True` | `0.333` |
| 6 | blend | `False` | `True` | `0.667` |
| 7+ | main | `False` | `False` | `1.0` |

## Temperature schedule

Temperature decays geometrically from `temp_start` to `temp_end` over `temp_decay_steps` steps, measured from the moment of phase switch:

```
temp(t) = temp_start * (temp_end / temp_start) ^ (elapsed / temp_decay_steps)
```

The clock starts at the first main-phase batch, not at training epoch 0.

## Parameter reference

| Parameter | Default | Description |
|---|---|---|
| `warmup_loss` | required | Loss used during warmup (e.g. `CrossEntropyLoss`) |
| `main_loss` | required | Loss used after warmup (e.g. `SmoothAPLoss`) |
| `warmup_epochs` | required | Epochs before switching; `0` skips warmup |
| `temp_start` | required | Temperature at phase switch |
| `temp_end` | required | Temperature after `temp_decay_steps` steps |
| `temp_decay_steps` | required | Steps over which to decay temperature |
| `blend_epochs` | `0` | Linear blend epochs; `0` = hard switch |
| `reset_queue_each_epoch` | `False` | Reset `main_loss` queue each main-phase epoch |
| `gather_distributed` | `None` | Forwarded to `main_loss.gather_distributed`; `None` auto-detects DDP |

## Properties

| Property | Type | Description |
|---|---|---|
| `in_warmup` | `bool` | `True` while `epoch < warmup_epochs` |
| `in_blend` | `bool` | `True` during the blend transition |
| `ap_weight` | `float` | Current AP weight: `0.0` → ramp → `1.0` |
| `current_temperature` | `float or None` | Current `main_loss.temperature`; `None` if unavailable |
