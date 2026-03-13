"""
LossWarmupWrapper — phase-switching loss with geometric temperature decay.

Usage in a LightningModule
---------------------------
    class MyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.loss_fn = LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
                warmup_epochs=5,
                temp_start=0.05,
                temp_end=0.005,
                temp_decay_steps=10_000,
            )

        def on_train_epoch_start(self):
            self.loss_fn.on_train_epoch_start(self.current_epoch)

        def on_train_batch_start(self, batch, batch_idx):
            self.loss_fn.on_train_batch_start(self.global_step)

        def training_step(self, batch, batch_idx):
            logits, targets = batch
            loss = self.loss_fn(logits, targets)
            self.log("train/loss", loss)
            self.log("train/in_warmup", float(self.loss_fn.in_warmup))
            if (t := self.loss_fn.current_temperature) is not None:
                self.log("train/temperature", t)
            return loss
"""

from __future__ import annotations

import math
import warnings

import torch.nn as nn


class LossWarmupWrapper(nn.Module):
    """
    Wraps a warmup loss and a main ranking loss with two features:

    1. **Phase switching** — ``warmup_loss`` is active for epochs
       ``< warmup_epochs``; ``main_loss`` is active thereafter.

    2. **Geometric temperature decay** — ``main_loss.temperature`` decays
       from ``temp_start`` to ``temp_end`` over ``temp_decay_steps``
       global training steps, starting from the moment of phase switch::

           temp(t) = temp_start * (temp_end / temp_start) ** (t / temp_decay_steps)

       After ``temp_decay_steps`` steps the temperature is held at
       ``temp_end``.

    Call :meth:`on_train_epoch_start` and :meth:`on_train_batch_start`
    from the corresponding PyTorch Lightning hooks (or your training loop).

    Parameters
    ----------
    warmup_loss : nn.Module
        Loss used during warmup.  Must accept ``(logits, targets)``.
        Typical choice: ``nn.CrossEntropyLoss()``.
    main_loss : nn.Module
        Loss used after warmup.  Must accept ``(logits, targets, **kwargs)``.
        Typical choice: ``SmoothAPLoss``, ``RecallAtQuantileLoss``.
    warmup_epochs : int
        Number of epochs to use ``warmup_loss`` (0 = skip warmup entirely).
    temp_start : float
        Temperature at the start of the main phase.
    temp_end : float
        Temperature after ``temp_decay_steps`` steps.
    temp_decay_steps : int
        Number of global training steps over which to decay temperature.
    blend_epochs : int, optional
        Number of epochs after warmup to linearly blend from ``warmup_loss``
        to ``main_loss``.  During blend epoch ``k`` (0-indexed),
        ``ap_weight = (k + 1) / (blend_epochs + 1)``.  After the blend
        period, ``ap_weight = 1.0`` (pure ``main_loss``).  Default: 0
        (hard switch, backward-compatible).
    reset_queue_each_epoch : bool, optional
        Call ``main_loss.reset_queue()`` at the start of each epoch in
        the main phase (if the method exists).  Default: False.
    gather_distributed : bool or None, optional
        Forwarded to ``main_loss.gather_distributed`` if the attribute
        exists.  ``None`` (default) auto-detects DDP at first forward;
        ``False`` explicitly disables gathering.  No-op if ``main_loss``
        does not have a ``gather_distributed`` attribute.  Default: None.
    """

    def __init__(
        self,
        warmup_loss: nn.Module,
        main_loss: nn.Module,
        warmup_epochs: int,
        temp_start: float,
        temp_end: float,
        temp_decay_steps: int,
        *,
        blend_epochs: int = 0,
        reset_queue_each_epoch: bool = False,
        gather_distributed: bool | None = None,
    ) -> None:
        super().__init__()

        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if temp_start <= 0 or temp_end <= 0:
            raise ValueError("temp_start and temp_end must be positive")
        if temp_decay_steps <= 0:
            raise ValueError(
                f"temp_decay_steps must be positive, got {temp_decay_steps}"
            )
        if blend_epochs < 0:
            raise ValueError(f"blend_epochs must be >= 0, got {blend_epochs}")

        self.warmup_loss = warmup_loss
        self.main_loss = main_loss
        self.warmup_epochs = warmup_epochs
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.temp_decay_steps = temp_decay_steps
        self.blend_epochs = blend_epochs
        self.reset_queue_each_epoch = reset_queue_each_epoch

        if hasattr(main_loss, "gather_distributed"):
            main_loss.gather_distributed = gather_distributed  # type: ignore[union-attr]

        self._has_temperature: bool = hasattr(main_loss, "temperature")
        self._has_reset_queue: bool = hasattr(main_loss, "reset_queue")

        if not self._has_temperature:
            warnings.warn(
                f"{type(main_loss).__name__} has no 'temperature' attribute; "
                "temperature scheduling will be skipped.",
                UserWarning,
                stacklevel=2,
            )
        if reset_queue_each_epoch and not self._has_reset_queue:
            warnings.warn(
                f"{type(main_loss).__name__} has no 'reset_queue' method; "
                "reset_queue_each_epoch will have no effect.",
                UserWarning,
                stacklevel=2,
            )

        self._epoch: int = 0
        self._switch_step: int | None = None  # global step when main phase began

        if warmup_epochs == 0:
            self._switch_step = 0
            self._apply_temperature(self.temp_start)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def in_blend(self) -> bool:
        """Whether the wrapper is currently in the blend phase."""
        return (
            not self.in_warmup
            and self.blend_epochs > 0
            and self._epoch < self.warmup_epochs + self.blend_epochs
        )

    @property
    def ap_weight(self) -> float:
        """Current AP loss weight (0.0 during warmup, ramp during blend, 1.0 after)."""
        if self.in_warmup:
            return 0.0
        if self.blend_epochs == 0 or self._epoch >= self.warmup_epochs + self.blend_epochs:
            return 1.0
        blend_epoch_index = self._epoch - self.warmup_epochs
        return (blend_epoch_index + 1) / (self.blend_epochs + 1)

    @property
    def in_warmup(self) -> bool:
        """
        Whether the wrapper is currently in the warmup phase.

        Returns
        -------
        bool
            True while ``_epoch < warmup_epochs``; False once the main
            loss is active.
        """
        return self._epoch < self.warmup_epochs

    @property
    def current_temperature(self) -> float | None:
        """
        The temperature currently set on ``main_loss``.

        Returns
        -------
        float or None
            ``float(main_loss.temperature)`` if ``main_loss`` has a
            ``temperature`` attribute, ``None`` otherwise.  During
            warmup the value reflects whatever was last written to
            ``main_loss.temperature`` (typically ``temp_start``).
        """
        if not self._has_temperature:
            return None
        return float(self.main_loss.temperature)  # type: ignore[union-attr]

    # ── Lightning / training-loop hooks ─────────────────────────────────────

    def on_train_epoch_start(self, epoch: int) -> None:
        """
        Advance the epoch counter and handle phase transition bookkeeping.

        Call this from ``LightningModule.on_train_epoch_start`` passing
        ``self.current_epoch``.  Responsibilities:

        - Updates the internal epoch counter.
        - On the first epoch of the main phase, sets the ``_switch_step``
          sentinel so that :meth:`on_train_batch_start` can latch the
          exact global step.
        - Calls ``main_loss.reset_queue()`` at the start of each main-phase
          epoch when ``reset_queue_each_epoch=True`` and the method exists.

        Parameters
        ----------
        epoch : int
            Zero-indexed current epoch number, as provided by
            ``self.current_epoch`` in a LightningModule.
        """
        self._epoch = epoch

        if not self.in_warmup and self._switch_step is None:
            # First epoch in the main phase — record switch point.
            # _step is not tracked; we derive temperature from global_step
            # passed to on_train_batch_start, so initialise switch_step lazily.
            self._switch_step = -1  # sentinel; overwritten on first batch hook

        if not self.in_warmup and self.reset_queue_each_epoch and self._has_reset_queue:
            self.main_loss.reset_queue()  # type: ignore[union-attr]

    def on_train_batch_start(self, global_step: int) -> None:
        """
        Update the temperature schedule for the current training step.

        Call this from ``LightningModule.on_train_batch_start`` passing
        ``self.global_step``.  Responsibilities:

        - On the first main-phase batch, latches ``_switch_step`` to
          ``global_step`` and sets temperature to ``temp_start``.
        - On all subsequent main-phase batches, applies the geometric
          decay formula and writes the result to ``main_loss.temperature``.
        - Is a no-op during warmup or before the phase sentinel is set.

        Parameters
        ----------
        global_step : int
            Monotonically increasing global step counter, as provided by
            ``self.global_step`` in a LightningModule.
        """
        if self.in_warmup or self._switch_step is None:
            return

        # Latch the exact step at which the main phase began.
        if self._switch_step == -1:
            self._switch_step = global_step
            self._apply_temperature(self.temp_start)
            if self._has_reset_queue:
                self.main_loss.reset_queue()  # type: ignore[union-attr]
            return

        elapsed = global_step - self._switch_step
        frac = min(1.0, elapsed / self.temp_decay_steps)
        temp = self.temp_start * math.exp(
            frac * math.log(self.temp_end / self.temp_start)
        )
        self._apply_temperature(temp)

    # ─�� helpers ─────────────────────────────────────────────────────────────

    def _apply_temperature(self, temp: float) -> None:
        """
        Write a temperature value to ``main_loss.temperature``.

        Parameters
        ----------
        temp : float
            Temperature value to assign.

        Notes
        -----
        No-op if ``main_loss`` has no ``temperature`` attribute
        (``_has_temperature`` is False).
        """
        if self._has_temperature:
            self.main_loss.temperature = temp  # type: ignore[union-attr]

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, logits, targets, **kwargs):
        """
        Compute loss using the currently active loss module.

        Parameters
        ----------
        logits : torch.Tensor
            Raw class scores, shape as expected by the active loss.
        targets : torch.Tensor
            Integer class labels or binary targets, shape as expected by
            the active loss.
        **kwargs
            Additional keyword arguments forwarded to ``main_loss`` only
            (e.g. ``return_per_class=True``).  Silently ignored during
            the warmup phase.

        Returns
        -------
        torch.Tensor or tuple
            During warmup or blend: scalar tensor.  After blend: output of
            ``main_loss`` (scalar or tuple when ``return_per_class=True``).
            ``**kwargs`` are forwarded to ``main_loss`` only when
            ``ap_weight == 1.0``; they are silently ignored during warmup
            and blend phases.
        """
        if self.in_warmup:
            return self.warmup_loss(logits, targets)
        w = self.ap_weight
        if w >= 1.0:
            return self.main_loss(logits, targets, **kwargs)
        return (1 - w) * self.warmup_loss(logits, targets) + w * self.main_loss(logits, targets)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    from proxy_losses.ap_loss import SmoothAPLoss

    torch.manual_seed(0)
    C, B = 4, 16
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))

    wrapper = LossWarmupWrapper(
        warmup_loss=nn.CrossEntropyLoss(),
        main_loss=SmoothAPLoss(num_classes=C, queue_size=64),
        warmup_epochs=2,
        temp_start=0.5,
        temp_end=0.01,
        temp_decay_steps=100,
    )

    for epoch in range(4):
        wrapper.on_train_epoch_start(epoch)
        for step in range(3):
            global_step = epoch * 3 + step
            wrapper.on_train_batch_start(global_step)
            with torch.no_grad():
                loss = wrapper(logits, targets)
            t = wrapper.current_temperature
            print(
                f"epoch={epoch} step={global_step:3d}  "
                f"in_warmup={wrapper.in_warmup}  "
                f"temp={f'{t:.5f}' if t is not None else 'N/A'}  "
                f"loss={loss.item():.4f}"
            )
