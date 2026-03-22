"""
Smooth-AP loss (Brown et al., 2020) with a memory queue.

For each positive i, soft ranks are estimated via:
    ŝ_i   = 1 + Σ_{j≠i}       σ((s_j − s_i) / τ)   # overall
    ŝ_i^+ = 1 + Σ_{j≠i, j∈P} σ((s_j − s_i) / τ)   # among positives
    AP ≈ (1/|P|) · Σ_{i∈P}  ŝ_i^+ / ŝ_i

Multi-class: one-vs-rest per class (scores = logits[:, c]).
Binary:      num_classes=1, logits [N, 1], targets in {0, 1}.
Seq2seq:     flatten to [N, C] / [N] upstream.
Padding:     ignore_index=-100 rows dropped before ranking.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn
from imbalanced_losses.distributed import all_gather_no_grad, all_gather_with_grad


class SmoothAPLoss(nn.Module):
    """
    Differentiable Average Precision loss with an optional memory queue.

    Approximates AP using soft sigmoid-based rank estimation (Smooth-AP,
    Brown et al. 2020). Supports multi-class (one-vs-rest) and binary
    (num_classes=1) classification. Expects logits [N, C] and targets [N];
    this class is agnostic to sequence structure — flatten upstream.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Use 1 for binary mode.
    queue_size : int, optional
        Number of (logits, targets) rows stored in the circular buffer.
        Larger queues give more stable AP estimates at the cost of O(|P|×M)
        memory in _compute_smooth_ap, where |P| is the number of positives.
        Set to 0 to disable. Default: 1024.
    temperature : float, optional
        Sigmoid sharpness τ. Smaller values approximate the true
        discontinuous rank more closely but produce harder gradients.
        Typical range: 0.005–0.05. Default: 0.01.
    reduction : {'mean', 'sum', 'none'}, optional
        How to aggregate per-class losses.
        - 'mean': scalar average over valid classes.
        - 'sum':  scalar sum over valid classes.
        - 'none': tensor of shape [C]; degenerate classes are nan.
        Default: 'mean'.
    ignore_index : int, optional
        Target value marking padded positions. Matching rows are excluded
        from ranking and the positive set. Default: -100.
    update_queue_in_eval : bool, optional
        If False (default), the queue is frozen during eval mode. Set to
        True to allow queue updates during validation. Default: False.
    gather_distributed : bool or None, optional
        Whether to all-gather logits and targets across DDP workers before
        computing the loss. ``None`` (default) auto-detects: gathers when
        ``torch.distributed`` is initialized with world_size > 1. Set
        ``False`` to explicitly disable. Resolved once on first forward call,
        so safe to construct before ``dist.init_process_group``. Default: None.

    Examples
    --------
    >>> loss_fn = SmoothAPLoss(num_classes=4, queue_size=512)
    >>> logits  = torch.randn(32, 4)
    >>> targets = torch.randint(0, 4, (32,))
    >>> loss = loss_fn(logits, targets)
    >>> loss.backward()

    Notes
    -----
    Complexity of _compute_smooth_ap is O(|P| × M), where |P| is the number
    of positives in the pool and M = batch_size + queue_size. At low positive
    rates this is much cheaper than the naive O(M²) formulation.

    In DDP, set ``gather_distributed=False`` to opt out; otherwise the loss
    auto-detects and all-gathers on first forward when world_size > 1.
    """

    def __init__(
        self,
        num_classes: int,
        queue_size: int = 1024,
        temperature: float = 0.01,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
        update_queue_in_eval: bool = False,
        gather_distributed: bool | None = None,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if queue_size < 0:
            raise ValueError(f"queue_size must be >= 0, got {queue_size}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction '{reduction}'")

        self.num_classes = num_classes
        self.queue_size = queue_size
        self.temperature = float(temperature)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.update_queue_in_eval = update_queue_in_eval
        self.gather_distributed = gather_distributed
        self._gather_resolved: bool | None = None

        if queue_size > 0:
            # Unfilled slots carry ignore_index targets and are stripped naturally.
            self.register_buffer("_q_logits",  torch.zeros(queue_size, num_classes))
            self.register_buffer("_q_targets", torch.full((queue_size,), ignore_index, dtype=torch.long))
            self.register_buffer("_q_ptr",     torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Write a detached batch into the circular buffer.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Live-batch logits to store (detached internally).
        targets : torch.Tensor, shape [N]
            Corresponding integer targets.

        Notes
        -----
        If N >= queue_size the buffer is replaced wholesale with the last
        queue_size rows of the batch and the pointer is reset to 0.
        Wrap-around writes are handled with explicit head/tail slicing.
        """
        if self.queue_size == 0:
            return

        n = logits.size(0)

        if n >= self.queue_size:
            self._q_logits.copy_(logits.detach()[-self.queue_size:])
            self._q_targets.copy_(targets.detach()[-self.queue_size:])
            self._q_ptr.zero_()
            return

        ptr = int(self._q_ptr)
        end = ptr + n

        if end <= self.queue_size:
            self._q_logits[ptr:end]  = logits.detach()
            self._q_targets[ptr:end] = targets.detach()
        else:
            first, second = self.queue_size - ptr, n - (self.queue_size - ptr)
            self._q_logits[ptr:]     = logits.detach()[:first]
            self._q_targets[ptr:]    = targets.detach()[:first]
            self._q_logits[:second]  = logits.detach()[first:]
            self._q_targets[:second] = targets.detach()[first:]

        self._q_ptr.fill_((ptr + n) % self.queue_size)

    def _merge_with_queue(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate the live batch with the current queue contents.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
        targets : torch.Tensor, shape [N]

        Returns
        -------
        all_logits : torch.Tensor, shape [N + Q, C]
            Live logits followed by queue logits (cast to matching
            device/dtype). Q = queue_size; unfilled slots have
            ignore_index targets and are filtered downstream.
        all_targets : torch.Tensor, shape [N + Q]
        """
        if self.queue_size == 0:
            return logits, targets
        q_logits  = self._q_logits.to(device=logits.device, dtype=logits.dtype)
        q_targets = self._q_targets.to(device=targets.device)
        return torch.cat([logits, q_logits], dim=0), torch.cat([targets, q_targets], dim=0)

    @torch.no_grad()
    def reset_queue(self) -> None:
        """
        Clear the circular buffer.

        Resets all stored logits to zero, all stored targets to
        ignore_index, and the write pointer to 0. Typically called
        between training and evaluation epochs.
        """
        if self.queue_size > 0:
            self._q_logits.zero_()
            self._q_targets.fill_(self.ignore_index)
            self._q_ptr.zero_()

    @staticmethod
    def _compute_smooth_ap(
        scores: torch.Tensor,
        is_pos: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, bool]:
        """
        Compute Smooth-AP for a single binary partition of the pool.

        Parameters
        ----------
        scores : torch.Tensor, shape [M]
            Raw scores for one class across the full pool (live + queue).
            Gradients flow through live-batch scores only; queue scores
            are detached before being passed in.
        is_pos : torch.Tensor, shape [M], dtype=bool
            True for positive samples (target == c for class c).
        tau : float
            Sigmoid temperature. See class docstring.

        Returns
        -------
        ap : torch.Tensor, scalar
            Smooth-AP estimate in [0, 1]. Zero (no gradient) for
            degenerate cases.
        valid : bool
            False if the class is degenerate (all-positive or all-negative
            in the pool). Degenerate classes are excluded from the
            mean/sum reduction rather than contributing a misleading 0.

        Notes
        -----
        Pairwise soft rank (computed only for positive rows):
            diff[k, j]    = s_j - s_pos_k            k ∈ P, j ∈ [M]
            soft_gt[k, j] ≈ P(s_j > s_pos_k) = σ(diff[k,j] / τ)
            rank_all[k]   = 1 + Σ_j soft_gt[k, j]   (self zeroed)
            rank_pos[k]   = 1 + Σ_{j∈P} soft_gt[k, j]
            AP            = mean_{k∈P} rank_pos[k] / rank_all[k]

        Complexity is O(|P| × M) rather than O(M²), reducing memory and
        compute by roughly 1/pos_rate (e.g. ~200× at 0.5% positives).
        """
        m     = scores.size(0)
        n_pos = int(is_pos.sum())

        if n_pos == 0 or n_pos == m:
            return scores.new_zeros(()), False

        # Only compute rows for positives: [|P|, M] instead of [M, M].
        # Reduces memory/compute by ~1/pos_rate (e.g. 200× at 0.5% positives).
        pos_idx  = is_pos.nonzero(as_tuple=False).squeeze(1)           # [P]
        diff_pos = scores.unsqueeze(0) - scores[pos_idx].unsqueeze(1)  # [P, M]; diff[k,j] = s_j - s_pos_k
        soft_gt  = torch.sigmoid(diff_pos / tau)                        # [P, M]
        # Zero self-comparisons without in-place ops (would break autograd).
        self_mask = torch.zeros(n_pos, m, device=scores.device, dtype=torch.bool)
        self_mask[torch.arange(n_pos, device=scores.device), pos_idx] = True
        soft_gt   = soft_gt.masked_fill(self_mask, 0.0)

        rank_all = 1.0 + soft_gt.sum(dim=1)            # [P]
        rank_pos = 1.0 + soft_gt[:, is_pos].sum(dim=1) # [P]

        ap = (rank_pos / rank_all).mean()
        return ap, True

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

        The queue is updated when the module is in training mode, or when
        update_queue_in_eval=True is set explicitly.

        Returns
        -------
        bool
        """
        return self.queue_size > 0 and (self.training or self.update_queue_in_eval)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_per_class: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Smooth-AP loss.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Raw (un-normalised) class scores. For seq2seq models, flatten
            to 2-D upstream before passing:
                logits  = logits.view(-1, C)
                targets = targets.view(-1)
        targets : torch.Tensor, shape [N]
            Integer class labels in [0, C). Positions equal to
            ignore_index are excluded from ranking.
        return_per_class : bool, optional
            If True, also return per-class losses and a validity mask.
            Useful for per-class metric logging in a training loop without
            a second forward pass. Default: False.

        Returns
        -------
        loss : torch.Tensor
            Scalar (reduction='mean' or 'sum') or shape [C]
            (reduction='none'). Values in [0, 1]; degenerate classes are
            nan when reduction='none'.
        per_class_loss : torch.Tensor, shape [C]
            Per-class loss values; nan for degenerate classes.
            Only returned when return_per_class=True.
        valid_classes : torch.Tensor, shape [C], dtype=bool
            True for classes with at least one positive and one negative
            in the pool. Only returned when return_per_class=True.

        Raises
        ------
        ValueError
            If logits or targets have unexpected shapes.

        Examples
        --------
        Basic usage:

        >>> loss = loss_fn(logits, targets)
        >>> loss.backward()

        Per-class logging in a Lightning training_step:

        >>> loss, per_class, valid = loss_fn(logits, targets, return_per_class=True)
        >>> for c in valid.nonzero(as_tuple=True)[0].tolist():
        ...     self.log(f"train/ap_loss_class_{c}", per_class[c])
        >>> return loss
        """
        if targets.ndim == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if logits.ndim != 2 or logits.size(1) != self.num_classes:
            raise ValueError(f"Expected logits [N, {self.num_classes}], got {tuple(logits.shape)}")
        if targets.ndim != 1 or targets.size(0) != logits.size(0):
            raise ValueError(f"targets must be [N] matching logits, got {tuple(targets.shape)}")

        if self._should_gather():
            logits  = all_gather_with_grad(logits)
            targets = all_gather_no_grad(targets)

        all_logits, all_targets = self._merge_with_queue(logits, targets)

        valid = all_targets != self.ignore_index
        all_logits, all_targets = all_logits[valid], all_targets[valid]

        if all_logits.size(0) == 0:
            out = logits.sum() * 0.0
            if self.reduction == "none":
                out = out.expand(self.num_classes)
            if return_per_class:
                return (
                    out,
                    logits.new_full((self.num_classes,), float("nan")),
                    torch.zeros(self.num_classes, dtype=torch.bool, device=logits.device),
                )
            return out

        # --- compute per-class AP and validity ---------------------------
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
            ap, is_valid = self._compute_smooth_ap(all_logits[:, 0], all_targets.bool(), self.temperature)
            loss_vals  = [1.0 - ap]
            valid_mask = [is_valid]
        else:
            loss_vals, valid_mask = [], []
            for c in range(self.num_classes):
                ap, is_valid = self._compute_smooth_ap(all_logits[:, c], all_targets == c, self.temperature)
                loss_vals.append(1.0 - ap)
                valid_mask.append(is_valid)

        loss_vec  = torch.stack(loss_vals)                          # [C] or [1]
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
    C, B = 4, 16

    targets_p = torch.arange(B) % C

    # --- perfect ---------------------------------------------------------
    logits_p = torch.full((B, C), -10.0)
    for i, t in enumerate(targets_p):
        logits_p[i, t] = 10.0
    logits_p = logits_p.requires_grad_(True)
    loss_p = SmoothAPLoss(num_classes=C, queue_size=0)(logits_p, targets_p)
    loss_p.backward()
    print(f"[perfect] loss={loss_p.item():.4f}  (expected ≈ 0.0)")

    # --- worst -----------------------------------------------------------
    logits_w = torch.full((B, C), 10.0)
    for i, t in enumerate(targets_p):
        logits_w[i, t] = -10.0
    logits_w = logits_w.requires_grad_(True)
    loss_w = SmoothAPLoss(num_classes=C, queue_size=0)(logits_w, targets_p)
    loss_w.backward()
    print(f"[worst]   loss={loss_w.item():.4f}  (expected ≈ 1.0)")

    # --- random, seq2one + seq2seq ---------------------------------------
    loss_fn = SmoothAPLoss(num_classes=C, queue_size=64)

    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,)); targets[0] = -100
    loss = loss_fn(logits, targets)
    loss.backward()
    print(f"[seq2one] loss={loss.item():.4f}  grad_norm={logits.grad.norm():.4f}")

    logits2 = torch.randn(B * 8, C, requires_grad=True)
    targets2 = torch.randint(0, C, (B * 8,)); targets2[:4] = -100
    loss2 = loss_fn(logits2, targets2)
    loss2.backward()
    print(f"[seq2seq] loss={loss2.item():.4f}  grad_norm={logits2.grad.norm():.4f}")

    # --- reduction='none' (degenerate classes show nan) ------------------
    loss_fn_none = SmoothAPLoss(num_classes=C, queue_size=0, reduction="none")
    per_class = loss_fn_none(torch.randn(B, C, requires_grad=True), torch.randint(0, C, (B,)))
    print(f"[none]    per-class: {[f'{v:.4f}' for v in per_class.tolist()]}")

    # --- update_queue_in_eval=False (default) ----------------------------
    loss_fn_eval = SmoothAPLoss(num_classes=C, queue_size=64)
    loss_fn_eval.eval()
    ptr_before = int(loss_fn_eval._q_ptr)
    loss_fn_eval(torch.randn(B, C), torch.randint(0, C, (B,)))
    ptr_after = int(loss_fn_eval._q_ptr)
    assert ptr_before == ptr_after, "queue should not update during eval"
    print(f"[eval]    queue ptr unchanged={ptr_after == ptr_before}  (expected True)")

    # --- binary ----------------------------------------------------------
    loss_fn_bin = SmoothAPLoss(num_classes=1, queue_size=32)
    logits4 = torch.randn(B, 1, requires_grad=True)
    targets4 = torch.randint(0, 2, (B,)); targets4[0] = -100
    loss4 = loss_fn_bin(logits4, targets4)
    loss4.backward()
    print(f"[binary]  loss={loss4.item():.4f}  grad_norm={logits4.grad.norm():.4f}")

    # --- batch larger than queue -----------------------------------------
    loss_fn_small_q = SmoothAPLoss(num_classes=C, queue_size=8)
    loss_fn_small_q(torch.randn(32, C), torch.randint(0, C, (32,)))
    print(f"[big batch] queue ptr={int(loss_fn_small_q._q_ptr)}  (expected 0, wrapped)")
