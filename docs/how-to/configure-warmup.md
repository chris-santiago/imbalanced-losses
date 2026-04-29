# Configure Warmup and Blending

`LossWarmupWrapper` manages a three-phase training schedule: warmup, optional linear blend, and main (ranking) phase with geometric temperature decay.

Warmup and blend can be specified in **epochs** or **steps** — use whichever maps more naturally to your training setup. The two styles are mutually exclusive per axis (`warmup_epochs` vs `warmup_steps`, `blend_epochs` vs `blend_steps`).

## Epoch-based warmup

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

Call both hooks in your training loop:

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

## Step-based warmup

Prefer steps when your effective epoch length varies (e.g. multi-dataset sampling, streaming data) or when you want fine-grained control:

```python
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_steps=5_000,
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=10_000,
)
```

In step mode **only `on_train_batch_start` is required** — the epoch hook is optional (only needed for `reset_queue_each_epoch`):

```python
for step, batch in enumerate(dataloader):
    loss_fn.on_train_batch_start(step)
    logits, targets = batch
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

In PyTorch Lightning:

```python
class MyModel(pl.LightningModule):
    def on_train_batch_start(self, batch, batch_idx):
        self.loss_fn.on_train_batch_start(self.global_step)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        return self.loss_fn(logits, targets)
```

**Confirm:** `loss_fn.in_warmup` is `True` for the first `warmup_steps` steps and `False` after.

## Add a blend period

`blend_epochs` / `blend_steps` adds a linear ramp between warmup and the main loss, avoiding an abrupt gradient change.

**Epoch blend:**

```python
loss_fn = LossWarmupWrapper(
    ...,
    warmup_epochs=5,
    blend_epochs=3,   # 3-epoch ramp
)
```

| Epoch | Phase | `main_weight` |
|---|---|---|
| 0–4 | warmup | 0.0 |
| 5 | blend | 0.25 |
| 6 | blend | 0.50 |
| 7 | blend | 0.75 |
| 8+ | main | 1.0 |

**Step blend:**

```python
loss_fn = LossWarmupWrapper(
    ...,
    warmup_steps=5_000,
    blend_steps=3_000,  # 3000-step ramp
)
```

`main_weight` follows the same `(k + 1) / (blend_steps + 1)` formula per step, scaled to `final_main_weight`.

## Hold a permanent mix with `final_main_weight`

By default the blend ramp ends at `main_weight = 1.0` (pure main loss). Set `final_main_weight` to hold a permanent split instead — the ramp scales to that target and stays there.

```python
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_epochs=5,
    blend_epochs=3,
    final_main_weight=0.75,  # hold 75% main / 25% CE forever
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=10_000,
)
```

| Epoch | Phase | `main_weight` |
|---|---|---|
| 0–4 | warmup | 0.0 |
| 5 | blend | 0.1875 |
| 6 | blend | 0.375 |
| 7 | blend | 0.5625 |
| 8+ | main | 0.75 |

Works without a blend too — the hard switch lands at `final_main_weight` instead of `1.0`:

```python
loss_fn = LossWarmupWrapper(
    ...,
    warmup_epochs=5,
    final_main_weight=0.75,  # no blend, hard switch to 75%
)
```

`final_main_weight` must be in `(0, 1]`. The default is `1.0`.

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

In step mode, also call `on_train_epoch_start` so the wrapper knows when each epoch begins:

```python
for epoch in range(total_epochs):
    loss_fn.on_train_epoch_start(epoch)
    for step, batch in enumerate(dataloader):
        loss_fn.on_train_batch_start(global_step)
        ...
```

**Note:** The wrapper always resets the queue automatically at the warmup-to-main phase switch, regardless of this setting.

## Skip warmup entirely

Pass `warmup_epochs=0` (epoch mode) or `warmup_steps=0` (step mode) to start directly with the ranking loss:

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

## See also

- [`examples/compare_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/compare_demo.py) — side-by-side AUCPR for warmup-only, AP-only (no warmup), and warmup+blend+AP so you can see the effect of each phase
- [`examples/toy_demo.py`](https://github.com/chris-santiago/imbalanced-losses/blob/main/examples/toy_demo.py) — minimal end-to-end binary training loop using `LossWarmupWrapper` with per-epoch loss and AUCPR output
