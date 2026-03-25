"""
Differentiable Recall-at-Quantile loss with a memory queue.

For a given quantile q (e.g. 0.005 = top 50bps), the threshold θ is the
(1-q) quantile of ALL scores in the pool (live batch + queue). Recall@q is
then approximated as the fraction of positives scoring above θ:

    θ = quantile(scores, 1 - q)                     [detached — no grad]
    soft_recall = (1/|P|) · Σ_{i∈P} σ((s_i − θ) / τ)
    loss = 1 − soft_recall

Gradient flows only through the positive scores, pushing them above the
cutoff. The threshold is treated as a fixed constant each forward pass,
analogous to a stop-gradient in contrastive losses.

Multi-class: one-vs-rest per class, same convention as SmoothAPLoss.
Binary:      num_classes=1, logits [N,1], targets 0/1.
Seq2seq:     flatten to [N, C] / [N] upstream.
Padding:     ignore_index=-100 rows are dropped before threshold estimation
             and recall computation.

Note: This is an original loss design, not from a published paper. It
combines quantile-based threshold estimation (stop-gradient) with sigmoid
soft recall in a way that, to our knowledge, has not appeared in prior
literature.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn
from imbalanced_losses.distributed import all_gather_no_grad, all_gather_with_grad


class RecallAtQuantileLoss(nn.Module):
    """
    Differentiable Recall-at-Quantile loss with an optional memory queue.

    For a given quantile q, a threshold θ is estimated from the pooled score
    distribution (live batch + queue) without gradient, then soft recall over
    positives is computed per class:

        θ = quantile(scores, 1 - q)                 [stop-gradient]
        soft_recall = mean_{i∈P} σ((s_i − θ) / τ)
        loss = 1 − soft_recall

    Multi-class: one-vs-rest per class using logits[:, c], then reduce.
    Binary:      logits[:, 0] with targets in {0, 1}.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Use 1 for binary mode.
    quantile : float, optional
        Fraction of the score distribution treated as the alert region.
        E.g. 0.005 = top 50 bps, 0.01 = top 100 bps. Must be in (0, 1).
        Default: 0.005.
    queue_size : int, optional
        Circular buffer size (rows). Larger queues stabilise the quantile
        estimate — at 50 bps you need at least ~200 samples for a
        meaningful 99.5th percentile. Set to 0 to disable. Default: 1024.

        **DDP note:** when ``gather_distributed=True``, the all-gather runs
        *before* the enqueue, so each rank stores global-batch rows. The
        effective pool per forward pass is already
        ``global_batch_size + queue_size``. At large global batches the
        quantile is already well-estimated from the live batch alone;
        consider setting ``queue_size=0`` to reduce memory overhead.
    temperature : float, optional
        Sigmoid sharpness τ around the threshold. Larger values give
        smoother gradients but less precise recall estimates. Default: 0.01.
    reduction : {'mean', 'sum', 'none'}, optional
        How to aggregate per-class losses.
        - 'mean': scalar average over valid classes.
        - 'sum':  scalar sum over valid classes.
        - 'none': tensor of shape [C]; classes with no positives are nan.
        Default: 'mean'.
    ignore_index : int, optional
        Target value marking padded positions. Excluded from threshold
        estimation and recall. Default: -100.
    update_queue_in_eval : bool, optional
        If False (default), the queue is frozen during eval mode. Default: False.
    gather_distributed : bool or None, optional
        Whether to all-gather logits and targets across DDP workers before
        computing the loss. ``None`` (default) auto-detects: gathers when
        ``torch.distributed`` is initialized with world_size > 1. Set
        ``False`` to explicitly disable. Resolved once on first forward call,
        so safe to construct before ``dist.init_process_group``. Default: None.
    quantile_interpolation : str, optional
        Interpolation method passed to torch.quantile. 'higher' is the
        conservative default — the threshold never undershoots the true
        cutoff. One of ('linear', 'lower', 'higher', 'nearest', 'midpoint').
        Default: 'higher'.

    Examples
    --------
    >>> loss_fn = RecallAtQuantileLoss(num_classes=4, quantile=0.005, queue_size=512)
    >>> logits  = torch.randn(32, 4)
    >>> targets = torch.randint(0, 4, (32,))
    >>> loss = loss_fn(logits, targets)
    >>> loss.backward()

    Notes
    -----
    The quantile must exceed the positive class fraction for the threshold
    to fall in the negative region under perfect classification. With C=4
    balanced classes (25% positives), use quantile > 0.25 for sanity tests.

    In DDP, the all-gather runs *before* the enqueue, so every rank stores
    identical global-batch rows and queues stay in sync automatically. The
    effective pool per step is ``global_batch_size + queue_size``. At large
    global batch sizes the queue contribution may be negligible; prefer
    ``queue_size=0`` when the global batch already provides a stable quantile
    estimate.
    """

    _VALID_INTERPOLATIONS = ("linear", "lower", "higher", "nearest", "midpoint")

    def __init__(
        self,
        num_classes: int,
        quantile: float = 0.005,
        queue_size: int = 1024,
        temperature: float = 0.01,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
        update_queue_in_eval: bool = False,
        gather_distributed: bool | None = None,
        quantile_interpolation: str = "higher",
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if queue_size < 0:
            raise ValueError(f"queue_size must be >= 0, got {queue_size}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction '{reduction}'")
        if quantile_interpolation not in self._VALID_INTERPOLATIONS:
            raise ValueError(
                f"quantile_interpolation must be one of {self._VALID_INTERPOLATIONS}, "
                f"got '{quantile_interpolation}'"
            )

        self.num_classes = num_classes
        self.quantile = float(quantile)
        self.queue_size = queue_size
        self.temperature = float(temperature)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.update_queue_in_eval = update_queue_in_eval
        self.gather_distributed = gather_distributed
        self._gather_resolved: bool | None = None
        self.quantile_interpolation = quantile_interpolation

        if queue_size > 0:
            # Unfilled slots carry ignore_index targets and are stripped naturally.
            self.register_buffer("_q_logits", torch.zeros(queue_size, num_classes))
            self.register_buffer(
                "_q_targets", torch.full((queue_size,), ignore_index, dtype=torch.long)
            )
            self.register_buffer("_q_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Write a detached batch into the circular buffer.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
        targets : torch.Tensor, shape [N]

        Notes
        -----
        If N >= queue_size the buffer is replaced wholesale with the last
        queue_size rows and the pointer is reset to 0.
        """
        if self.queue_size == 0:
            return

        n = logits.size(0)

        if n >= self.queue_size:
            self._q_logits.copy_(logits.detach()[-self.queue_size :])
            self._q_targets.copy_(targets.detach()[-self.queue_size :])
            self._q_ptr.zero_()
            return

        ptr = int(self._q_ptr)
        end = ptr + n

        if end <= self.queue_size:
            self._q_logits[ptr:end] = logits.detach()
            self._q_targets[ptr:end] = targets.detach()
        else:
            first, second = self.queue_size - ptr, n - (self.queue_size - ptr)
            self._q_logits[ptr:] = logits.detach()[:first]
            self._q_targets[ptr:] = targets.detach()[:first]
            self._q_logits[:second] = logits.detach()[first:]
            self._q_targets[:second] = targets.detach()[first:]

        self._q_ptr.fill_((ptr + n) % self.queue_size)

    def _merge_with_queue(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate the live batch with current queue contents.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
        targets : torch.Tensor, shape [N]

        Returns
        -------
        all_logits : torch.Tensor, shape [N + Q, C]
        all_targets : torch.Tensor, shape [N + Q]
        """
        if self.queue_size == 0:
            return logits, targets
        q_logits = self._q_logits.to(device=logits.device, dtype=logits.dtype)
        q_targets = self._q_targets.to(device=targets.device)
        return torch.cat([logits, q_logits], dim=0), torch.cat(
            [targets, q_targets], dim=0
        )

    @torch.no_grad()
    def reset_queue(self) -> None:
        """
        Clear the circular buffer.

        Resets all stored logits to zero, targets to ignore_index, and
        the write pointer to 0.
        """
        if self.queue_size > 0:
            self._q_logits.zero_()
            self._q_targets.fill_(self.ignore_index)
            self._q_ptr.zero_()

    def _should_gather(self) -> bool:
        """
        Return True if logits/targets should be all-gathered before this forward.

        Resolved once on first call and cached. Safe to call before
        ``dist.init_process_group`` — will simply return False until dist
        is initialized.

        Returns
        -------
        bool
        """
        if self._gather_resolved is None:
            import torch.distributed as dist
            self._gather_resolved = (
                self.gather_distributed is not False
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
        return self._gather_resolved

    def _should_update_queue(self) -> bool:
        """
        Return True if the queue should be updated on this forward pass.

        Returns
        -------
        bool
        """
        return self.queue_size > 0 and (self.training or self.update_queue_in_eval)

    def _soft_recall_at_quantile(
        self,
        scores: torch.Tensor,
        is_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """
        Compute soft recall above the score quantile for one class.

        Parameters
        ----------
        scores : torch.Tensor, shape [M]
            Pooled scores for one class (live + queue, padding stripped).
            The threshold is computed from all scores (positives and
            negatives), then applied only to positives.
        is_pos : torch.Tensor, shape [M], dtype=bool
            Positive mask for this class.

        Returns
        -------
        recall : torch.Tensor, scalar
            Soft recall estimate in [0, 1].
        valid : bool
            False if there are no positives in the pool. Classes with
            no positives are excluded from the reduction rather than
            contributing a misleading 0.

        Notes
        -----
        The threshold θ is detached before use. Gradient flows only
        through the positive scores, pushing them above the cutoff.
        """
        n_pos = int(is_pos.sum())
        if n_pos == 0:
            return scores.new_zeros(()), False

        theta = torch.quantile(
            scores.detach(),
            1.0 - self.quantile,
            interpolation=self.quantile_interpolation,
        )
        soft_above = torch.sigmoid((scores[is_pos] - theta) / self.temperature)
        return soft_above.mean(), True

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_per_class: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Recall-at-Quantile loss.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Raw class scores. Flatten seq2seq inputs upstream:
                logits  = logits.view(-1, C)
                targets = targets.view(-1)
        targets : torch.Tensor, shape [N]
            Integer class labels in [0, C). ignore_index positions are
            excluded from the threshold and recall computation.
        return_per_class : bool, optional
            If True, also return per-class losses and a validity mask.
            Default: False.

        Returns
        -------
        loss : torch.Tensor
            Scalar (reduction 'mean'/'sum') or shape [C] (reduction 'none').
            Classes with no positives are nan when reduction='none'.
        per_class_loss : torch.Tensor, shape [C]
            Per-class loss; nan for classes with no positives.
            Only returned when return_per_class=True.
        valid_classes : torch.Tensor, shape [C], dtype=bool
            True for classes with at least one positive in the pool.
            Only returned when return_per_class=True.

        Raises
        ------
        ValueError
            If logits or targets have unexpected shapes, or targets contain
            class ids outside [0, num_classes).

        Examples
        --------
        Per-class logging in a Lightning training_step:

        >>> loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
        >>> for c in valid.nonzero(as_tuple=True)[0].tolist():
        ...     self.log(f"train/recall_loss_class_{c}", per_class[c])
        >>> return loss
        """
        if targets.ndim == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if logits.ndim != 2 or logits.size(1) != self.num_classes:
            raise ValueError(
                f"Expected logits [N, {self.num_classes}], got {tuple(logits.shape)}"
            )
        if targets.ndim != 1 or targets.size(0) != logits.size(0):
            raise ValueError(
                f"targets must be [N] matching logits, got {tuple(targets.shape)}"
            )

        if self._should_gather():
            logits  = all_gather_with_grad(logits)
            targets = all_gather_no_grad(targets)

        all_logits, all_targets = self._merge_with_queue(logits, targets)

        valid_rows = all_targets != self.ignore_index
        all_logits, all_targets = all_logits[valid_rows], all_targets[valid_rows]

        if all_logits.size(0) == 0:
            out = logits.sum() * 0.0
            if self.reduction == "none":
                out = out.expand(self.num_classes)
            if return_per_class:
                return (
                    out,
                    logits.new_full((self.num_classes,), float("nan")),
                    torch.zeros(
                        self.num_classes, dtype=torch.bool, device=logits.device
                    ),
                )
            return out

        # Validate target range (cheap, catches real bugs early).
        if self.num_classes > 1:
            bad = all_targets[(all_targets < 0) | (all_targets >= self.num_classes)]
            if bad.numel() > 0:
                raise ValueError(
                    f"targets contain class ids outside [0, {self.num_classes}); "
                    f"examples: {bad[:8].tolist()}"
                )

        # --- compute per-class recall and validity -----------------------
        if self.num_classes == 1:
            bad = all_targets[(all_targets != 0) & (all_targets != 1)]
            if bad.numel() > 0:
                warnings.warn(
                    f"Binary mode (num_classes=1) expects targets in {{0, 1}}, "
                    f"but found values: {bad[:8].tolist()}. "
                    "Non-zero values are treated as positive.",
                    UserWarning,
                    stacklevel=2,
                )
            recall, is_valid = self._soft_recall_at_quantile(
                all_logits[:, 0], all_targets.bool()
            )
            loss_vals = [1.0 - recall]
            valid_mask = [is_valid]
        else:
            loss_vals, valid_mask = [], []
            for c in range(self.num_classes):
                recall, is_valid = self._soft_recall_at_quantile(
                    all_logits[:, c], all_targets == c
                )
                loss_vals.append(1.0 - recall)
                valid_mask.append(is_valid)

        loss_vec = torch.stack(loss_vals)  # [C] or [1]
        valid_vec = torch.tensor(valid_mask, device=logits.device)  # [C] or [1]

        if self._should_update_queue():
            self._enqueue(logits, targets)

        # --- reduction ---------------------------------------------------
        if self.reduction == "none":
            out = loss_vec.masked_fill(~valid_vec, float("nan"))
        else:
            valid_losses = loss_vec[valid_vec]
            if valid_losses.numel() == 0:
                out = logits.sum() * 0.0
            elif self.reduction == "sum":
                out = valid_losses.sum()
            else:
                out = valid_losses.mean()

        if return_per_class:
            per_class = loss_vec.masked_fill(~valid_vec, float("nan"))
            return out, per_class, valid_vec
        return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    C, B = 4, 64

    # For perfect/worst we need the threshold to fall between negatives and
    # positives.  Positives are 1/C = 25% of scores, so quantile > 0.25.
    SANITY_Q = 0.30

    targets_p = torch.arange(B) % C  # balanced: 16 per class

    # --- perfect ---------------------------------------------------------
    logits_p = torch.full((B, C), -10.0)
    for i, t in enumerate(targets_p):
        logits_p[i, t] = 10.0
    logits_p = logits_p.requires_grad_(True)
    fn_perfect = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
    loss_p = fn_perfect(logits_p, targets_p)
    loss_p.backward()
    print(f"[perfect] loss={loss_p.item():.4f}  (expected ≈ 0.0)")

    # --- worst -----------------------------------------------------------
    logits_w = torch.full((B, C), 10.0)
    for i, t in enumerate(targets_p):
        logits_w[i, t] = -10.0
    logits_w = logits_w.requires_grad_(True)
    fn_worst = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=0)
    loss_w = fn_worst(logits_w, targets_p)
    loss_w.backward()
    print(f"[worst]   loss={loss_w.item():.4f}  (expected ≈ 1.0)")

    # --- random with queue -----------------------------------------------
    fn = RecallAtQuantileLoss(num_classes=C, quantile=0.005, queue_size=256)
    logits_r = torch.randn(B, C, requires_grad=True)
    targets_r = torch.randint(0, C, (B,))
    targets_r[:4] = -100
    loss_r = fn(logits_r, targets_r)
    loss_r.backward()
    print(f"[random]  loss={loss_r.item():.4f}  grad_norm={logits_r.grad.norm():.4f}")

    # --- reduction='none' ------------------------------------------------
    fn_none = RecallAtQuantileLoss(
        num_classes=C, quantile=SANITY_Q, queue_size=0, reduction="none"
    )
    per_class = fn_none(
        torch.randn(B, C, requires_grad=True), torch.randint(0, C, (B,))
    )
    print(f"[none]    per-class: {[f'{v:.4f}' for v in per_class.tolist()]}")

    # --- eval: queue should not update -----------------------------------
    fn_eval = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=64)
    fn_eval.eval()
    ptr_before = int(fn_eval._q_ptr)
    fn_eval(torch.randn(B, C), torch.randint(0, C, (B,)))
    ptr_after = int(fn_eval._q_ptr)
    assert ptr_before == ptr_after, "queue should not update during eval"
    print(f"[eval]    queue ptr unchanged={ptr_after == ptr_before}  (expected True)")

    # --- binary ----------------------------------------------------------
    fn_bin = RecallAtQuantileLoss(num_classes=1, quantile=SANITY_Q, queue_size=64)
    logits_b = torch.randn(B, 1, requires_grad=True)
    targets_b = torch.randint(0, 2, (B,))
    targets_b[0] = -100
    loss_b = fn_bin(logits_b, targets_b)
    loss_b.backward()
    print(f"[binary]  loss={loss_b.item():.4f}  grad_norm={logits_b.grad.norm():.4f}")

    # --- batch larger than queue -----------------------------------------
    fn_small_q = RecallAtQuantileLoss(num_classes=C, quantile=SANITY_Q, queue_size=8)
    fn_small_q(torch.randn(32, C), torch.randint(0, C, (32,)))
    print(f"[big batch] queue ptr={int(fn_small_q._q_ptr)}  (expected 0, wrapped)")

    # --- quantile_interpolation='higher' (default) -----------------------
    fn_interp = RecallAtQuantileLoss(
        num_classes=C, quantile=SANITY_Q, queue_size=0, quantile_interpolation="higher"
    )
    loss_i = fn_interp(torch.randn(B, C, requires_grad=True), torch.randint(0, C, (B,)))
    loss_i.backward()
    print(f"[interp]  loss={loss_i.item():.4f}  (higher interpolation)")
