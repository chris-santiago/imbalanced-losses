# Configure Warmup and Blending

`LossWarmupWrapper` manages a three-phase training schedule: warmup, optional blend, and main (ranking) phase with geometric temperature decay.

## Minimal setup

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper
import torch.nn as nn

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_epochs=5,
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=10_000,
)
```

## Wire up the training-loop hooks

Call the hooks on every epoch and every batch. In plain PyTorch:

```python
global_step = 0

for epoch in range(total_epochs):
    loss_fn.on_train_epoch_start(epoch)

    for batch in dataloader:
        loss_fn.on_train_batch_start(global_step)
        logits, targets = batch
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
```

In PyTorch Lightning:

```python
class MyModel(pl.LightningModule):
    def on_train_epoch_start(self):
        self.loss_fn.on_train_epoch_start(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        self.loss_fn.on_train_batch_start(self.global_step)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        return self.loss_fn(logits, targets)
```

**Confirm:** `loss_fn.in_warmup` is `True` for the first `warmup_epochs` epochs and `False` after.

## Add a blend period

`blend_epochs` adds a linear ramp between warmup and pure AP loss, avoiding an abrupt gradient change:

```python
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_epochs=5,
    blend_epochs=3,          # 3-epoch ramp
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=10_000,
)
```

During the blend, `loss_fn.ap_weight` increases from `1/4` to `3/4` in equal steps:

| Epoch | Phase | `ap_weight` |
|---|---|---|
| 0–4 | warmup | 0.0 |
| 5 | blend | 0.25 |
| 6 | blend | 0.50 |
| 7 | blend | 0.75 |
| 8+ | main | 1.0 |

## Tune temperature decay

Temperature controls how sharply the sigmoid approximates a hard rank. Start high (soft, stable gradients) and decay to a low value (sharp, closer to true AP):

```python
loss_fn = LossWarmupWrapper(
    ...,
    temp_start=0.5,          # soft at phase switch — stable early gradients
    temp_end=0.005,          # sharp after schedule — approaches true AP
    temp_decay_steps=50_000, # slow decay for large datasets
)
```

Use `loss_fn.current_temperature` to log the current temperature:

```python
if (t := loss_fn.current_temperature) is not None:
    self.log("train/temperature", t)
```

## Reset queue each epoch

If positive samples are rare enough that queue contamination matters across epochs, reset the queue at the start of each main-phase epoch:

```python
loss_fn = LossWarmupWrapper(
    ...,
    reset_queue_each_epoch=True,
)
```

**Note:** The wrapper always resets the queue automatically at the warmup-to-main phase switch, regardless of this setting.

## Skip warmup entirely

Set `warmup_epochs=0` to start directly with the ranking loss. Temperature is set to `temp_start` immediately:

```python
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),   # unused but required
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_epochs=0,
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=10_000,
)
```

**Confirm:** `loss_fn.in_warmup` is immediately `False`.
