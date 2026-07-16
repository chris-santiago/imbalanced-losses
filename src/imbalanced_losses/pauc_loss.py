"""
Partial-AUC-over-an-FPR-band loss with a memory queue.

Optimizes the normalized partial AUC over a false-positive-rate band
``[alpha, beta]`` that brackets a target operating point (e.g. FPR ~ 0.005),
rather than the full AUC or a single-threshold recall.

Band edges are estimated as score quantiles of a reference population under
stop-gradient; the ``budget_basis`` parameter selects that reference --
*iid negatives* only (``'fpr'``, default) or the whole pooled iid population
(positives+negatives, ``'population'``) -- so they **approximate** true
population FPR (or a population-quantile alert budget).  The
approximation is reliable when the pooled reference count substantially
exceeds the reciprocal of the band's smaller nonzero edge -- ``1/beta`` when
``alpha=0`` (the default), ``1/alpha`` when ``alpha > 0``; at small counts
the tail quantile is biased toward the maximum.  Check the
``band_neg_count`` diagnostic (and ``queue_size``) as an empirical quality
indicator.  Classes whose iid-negative scores show
near-zero dispersion are automatically skipped (marked INVALID) because the
sigmoid temperature cannot be calibrated; see the ``_degenerate_warned`` note
in the class docstring.

    t_alpha = quantile(neg_iid, 1 - alpha)          [detached]
    t_beta  = quantile(neg_iid, 1 - beta)           [detached]   (t_beta <= t_alpha)

The sigmoid temperature is scale-aware: ``tau_eff = temperature * scale`` where
``scale`` is a detached robust dispersion of the iid negatives (IQR by default,
or the band width ``t_alpha - t_beta``). This keeps the kernel sharpness
constant in FPR units as the model's score scale drifts during training.

Two surrogates:

* ``surrogate="trapezoid"`` (default) -- normalized pAUC by composite trapezoid
  over ``n_knots`` equally-spaced FPR knots in ``[alpha, beta]``:

      TPR_k = mean_{i in P} sigmoid((s_i - t_k) / tau_eff)
      pauc  = trapezoid(TPR_k) over the uniform FPR grid
      loss  = 1 - pauc

  Gradient flows only through positive scores; negatives enter solely via the
  detached thresholds/scale. Cost is O(|P| x n_knots).

* ``surrogate="pairwise"`` -- band-restricted Smooth-pAUC. Band negatives
  (``t_beta <= s <= t_alpha``) are taken from the gradient pool:

      pauc = mean_{i in P, j in band} sigmoid((s_i - s_j) / tau_eff)
      loss = 1 - pauc

  Band negatives carry gradient (intended). Cost is O(|P| x |band|). Intended
  for wide/volatile bands where band-negatives are plentiful.

Multi-class: one-vs-rest per class, same convention as SmoothAPLoss.
Binary:      num_classes=1, logits [N, 1], targets 0/1.
Seq2seq:     flatten to [N, C] / [N] upstream.
Padding:     ignore_index=-100 rows are dropped before threshold estimation.

Note: This is an original loss design, not from a published paper. It combines
iid-anchored stop-gradient FPR-band thresholds with a scale-aware soft-TPR
trapezoid (or band-restricted pairwise) surrogate.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import torch

from imbalanced_losses._base import _QueuedRankingLoss


class PAUCAtBudgetLoss(_QueuedRankingLoss):
    """
    Differentiable partial-AUC-over-an-FPR-band loss with an optional memory queue.

    For each class, FPR-band edges ``t_alpha``/``t_beta`` are estimated from the
    pooled *iid negatives* (stop-gradient) as score quantiles, a scale-aware
    sigmoid temperature ``tau_eff`` is derived from a detached robust dispersion
    of those negatives, and the normalized partial AUC over ``[alpha, beta]`` is
    optimized via a trapezoid (default) or band-restricted pairwise surrogate.
    Loss is ``1 - pauc``.

    Multi-class: one-vs-rest per class using logits[:, c], then reduce.
    Binary:      logits[:, 0] with targets in {0, 1}.

    Inherits queue management, DDP gather, ignore-index filtering, subsampling,
    and reduction logic from ``_QueuedRankingLoss``.

    **Degenerate-dispersion guard:** if the robust dispersion of iid-negative
    scores for a class is at or below ``_SCALE_EPS`` (all-equal scores, a
    collapsed band with ``tau_scale='band'``, or too few iid negatives to
    resolve the tail quantile), that class is marked INVALID and excluded from
    reduction rather than silently computing with a near-zero temperature.  A
    one-time ``UserWarning`` is emitted on the first such occurrence.  To avoid
    degenerate classes, increase ``queue_size`` or ensure iid negatives cover a
    meaningful score range.

    The band-edge approximation of population FPR is reliable when the pooled
    iid-negative count substantially exceeds the reciprocal of the band's
    smaller nonzero edge: ``1/beta`` when ``alpha=0`` (the default),
    ``1/alpha`` when ``alpha > 0``.  Monitor ``band_neg_count`` in
    diagnostics and set ``queue_size`` accordingly.

    The recommended band convention is ``alpha ≈ 0, beta ≈ budget`` (where
    ``budget`` is the target FPR, e.g. ``beta=0.005`` for a 50 bps operating
    point).  This sets the upper threshold ``t_alpha = quantile(neg, 1.0) =
    max(neg)`` and the lower threshold ``t_beta = quantile(neg, 1 - budget)``,
    so the band covers every false-positive that falls above the budget
    threshold, i.e. all negatives scoring at or above the operating point.

    The older convention ``[budget/2, 1.5·budget]`` — e.g. ``[0.0025, 0.0075]``
    for a 50 bps point — excludes the highest-scoring (worst) negatives via its
    lower edge ``alpha = budget/2`` and extends below the operating threshold via
    its upper edge ``beta = 1.5·budget``.  A band sweep (8 seeds, synthetic
    contested-top extreme-imbalance data, 50 bps budget) found coverage@budget
    to be monotone in both edges: smaller ``alpha`` and smaller ``beta`` are
    better in every cell, and the old convention sits in the poorly-performing
    high-``alpha`` region (the worst cell being ``alpha=budget/2, beta=2.5*budget``).
    The recommended ``alpha=0, beta=budget`` band fixes both defects by
    contrasting positives against all false-positives at the budget.

    **Caveats:** the sweep evidence is synthetic, at a single budget (50 bps),
    and in the contested-top regime.  The improvement is concentrated at
    ``pos_rate ≪ budget``; once ``pos_rate ≥ budget``, coverage@budget is
    mechanically capped at ``budget/pos_rate`` and band choice is irrelevant.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Use 1 for binary mode.
    alpha : float, optional
        Lower FPR band edge. Must satisfy ``0 <= alpha < beta <= 1``.
        ``alpha=0`` sets ``t_alpha = max(neg_iid)``, contrasting positives
        against all negatives above the budget threshold.
        Default: 0.0 (recommended for contested-top extreme-imbalance).
    beta : float, optional
        Upper FPR band edge. Must satisfy ``0 <= alpha < beta <= 1``.
        Set to your target operating-point FPR (e.g. ``0.005`` for 50 bps).
        Default: 0.005.
    surrogate : {'trapezoid', 'pairwise'}, optional
        pAUC estimator. ``'trapezoid'`` (default) integrates soft-TPR over
        ``n_knots`` FPR knots; gradient flows through positives only.
        ``'pairwise'`` compares positives against band negatives drawn from the
        gradient pool (band negatives carry gradient). Default: 'trapezoid'.
    n_knots : int, optional
        Number of equally-spaced FPR knots in ``[alpha, beta]`` for the
        trapezoid surrogate (knot 0 = alpha, knot n_knots-1 = beta). Must be
        >= 2. Ignored when ``surrogate='pairwise'``. Default: 2.

        The default of 2 (trapezoid rule) is accurate for narrow bands where
        TPR(FPR) is approximately linear over ``[alpha, beta]``; the
        integration error scales as ``(beta - alpha)^3 * TPR''``.  For wide
        bands where TPR curvature is non-negligible, ``n_knots >= 3`` is
        recommended.
    tau_scale : {'iqr', 'band'}, optional
        Robust dispersion used to make the temperature scale-aware.
        ``'iqr'`` (default) uses ``IQR(neg_iid)`` -- a stable bulk statistic
        (pair with small ``temperature``, e.g. 0.1). ``'band'`` uses
        ``t_alpha - t_beta`` -- sized directly to the operating region (pair
        with ``temperature`` near 1.0; recommended for wide/volatile bands).
        Default: 'iqr'.
    pos_numerator : {'pool', 'live'}, optional
        Which positives form the soft-TPR numerator (and the pairwise positive
        set). ``'pool'`` (default) uses all pooled positives (live batch + queue),
        matching the queue's stabilising role but diluting the live-positive
        gradient by ``1 / |P_pool|`` when the queue holds many detached positives.
        ``'live'`` uses only the live-batch positives, giving an undiluted
        gradient -- useful at extreme imbalance where the queue swamps the few
        live positives -- at the cost of a higher-variance TPR estimate (mean over
        the ~few live positives). Thresholds and ``tau_eff`` always use the full
        pooled iid negatives regardless of this setting. A class with no live
        positives in a step is skipped (invalid) under ``'live'``. Default: 'pool'.
    budget_basis : {'fpr', 'population'}, optional
        Reference population for the band-edge quantiles ``t_alpha``/``t_beta``
        (and the ``tau_scale`` dispersion computed from that same reference).
        ``'fpr'`` (default) uses the iid negatives only, so ``beta`` is the
        fraction of negatives above the band -- true FPR semantics. ``'population'``
        uses the whole pooled iid population (positives + negatives), so
        ``beta`` is the fraction of the *population* above the band -- matching
        a top-k alert budget spent over all scores rather than over negatives
        alone. Two consequences of ``'population'``: with ``alpha=0``,
        ``t_alpha = max`` over the whole population, which can be a positive's
        own score rather than a negative's; and the pool-size reliability
        heuristic (pool count should substantially exceed ``1/beta`` --
        see ``band_neg_count`` above) is over the pooled *population* count,
        not the negative count. ``surrogate='trapezoid'`` combined with
        ``budget_basis='population'`` is approximately equivalent to
        ``RecallAtQuantileLoss`` (band-averaged soft recall over population
        score quantiles); it is supported here for completeness and documented
        as a near-equivalent rather than a distinct recommended configuration.
        Default: 'fpr' (backward-compatible).
    queue_size : int, optional
        Circular buffer size (rows). Larger queues stabilise the quantile-based
        band edges -- at low FPR you need many negatives for a meaningful tail
        quantile. Set to 0 to disable. Default: 1024.

        **DDP note:** when ``gather_distributed=True``, the all-gather runs
        *before* the enqueue, so each rank stores global-batch rows. The
        effective pool per forward pass is already
        ``global_batch_size + queue_size``.
    temperature : float, optional
        Dimensionless multiplier on ``tau_eff = temperature * scale``. Larger
        values give smoother gradients but bias soft-TPR toward 0.5; smaller
        values approximate true TPR but risk sigmoid saturation. Default: 0.1.
    reduction : {'mean', 'sum', 'none'}, optional
        How to aggregate per-class losses.
        - 'mean': scalar average over valid classes.
        - 'sum':  scalar sum over valid classes.
        - 'none': tensor of shape [C]; invalid classes are nan.
        Default: 'mean'.
    ignore_index : int, optional
        Target value marking padded positions. Excluded from threshold
        estimation and the positive set. Default: -100.
    update_queue_in_eval : bool, optional
        If False (default), the queue is frozen during eval mode. Default: False.
    gather_distributed : bool or None, optional
        Whether to all-gather logits, targets, and the iid mask across DDP
        workers before computing the loss. ``None`` (default) auto-detects:
        gathers when ``torch.distributed`` is initialized with world_size > 1.
        Set ``False`` to explicitly disable. Resolved once on first forward
        call. Default: None.
    quantile_interpolation : str, optional
        Interpolation method passed to torch.quantile for the band edges.
        'higher' is the conservative default. One of ('linear', 'lower',
        'higher', 'nearest', 'midpoint'). Default: 'higher'.
    max_pool_size : int or None, optional
        Maximum number of rows in the ranking pool (live batch + queue after
        ignore_index filtering). When exceeded, minimum-quota subsampling caps
        it. See ``RecallAtQuantileLoss`` for details. ``None`` (default)
        disables the cap.

    Examples
    --------
    >>> loss_fn = PAUCAtBudgetLoss(num_classes=4, alpha=0.0, beta=0.005)
    >>> logits  = torch.randn(256, 4)
    >>> targets = torch.randint(0, 4, (256,))
    >>> loss = loss_fn(logits, targets)
    >>> loss.backward()

    Notes
    -----
    The iid-negative band edges depend only on rows flagged
    ``iid_mask=True``; appending non-iid negatives (caller-side densification)
    does not shift ``t_alpha``/``t_beta``. ``iid_mask=None`` treats all rows as
    iid (the common case when negatives are never densified by class).

    Trapezoid cost is O(|P| x n_knots); pairwise cost is O(|P| x |band|).
    No O(M^2) path.

    References
    ----------
    This loss is an original design, not a published method, but the
    partial-AUC-over-a-band objective and its estimators build on prior work:

    .. [1] D. K. McClish (1989). "Analyzing a Portion of the ROC Curve."
       Medical Decision Making 9(3), 190-195. -- partial AUC over an ROC region.
    .. [2] L. E. Dodd and M. S. Pepe (2003). "Partial AUC Estimation and
       Regression." Biometrics 59(3), 614-623. -- nonparametric pAUC estimator
       (the consistent plug-in the trapezoid surrogate relates to).
    .. [3] H. Narasimhan and S. Agarwal (2013). "A Structural SVM Based Approach
       for Optimizing Partial AUC." ICML 2013. -- optimizing pAUC over an FPR
       band [alpha, beta] as a learning objective (the KDD 2013 "tight" variant
       gives the boundary-corrected estimator).
    .. [4] D. Zhu, G. Li, B. Wang, X. Wu, and T. Yang (2022). "When AUC meets
       DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence
       Guarantee." ICML 2022. -- deep one-/two-way pAUC surrogate optimization.
    """

    _VALID_INTERPOLATIONS = ("linear", "lower", "higher", "nearest", "midpoint")
    _VALID_SURROGATES = ("trapezoid", "pairwise")
    _VALID_TAU_SCALES = ("iqr", "band")
    _VALID_POS_NUMERATORS = ("pool", "live")
    _VALID_BUDGET_BASES = ("fpr", "population")

    # Floor on the detached dispersion to avoid div-by-zero when all iid
    # negative scores are (near) equal.
    _SCALE_EPS = 1e-12

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.0,
        beta: float = 0.005,
        surrogate: Literal["trapezoid", "pairwise"] = "trapezoid",
        n_knots: int = 2,
        tau_scale: Literal["iqr", "band"] = "iqr",
        pos_numerator: Literal["pool", "live"] = "pool",
        budget_basis: Literal["fpr", "population"] = "fpr",
        queue_size: int = 1024,
        temperature: float = 0.1,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
        update_queue_in_eval: bool = False,
        gather_distributed: bool | None = None,
        quantile_interpolation: str = "higher",
        max_pool_size: int | None = None,
    ) -> None:
        if not (0.0 <= alpha < beta <= 1.0):
            raise ValueError(
                f"alpha and beta must satisfy 0 <= alpha < beta <= 1, "
                f"got alpha={alpha}, beta={beta}"
            )
        if not isinstance(n_knots, int) or n_knots < 2:
            raise ValueError(f"n_knots must be an int >= 2, got {n_knots}")
        if surrogate not in self._VALID_SURROGATES:
            raise ValueError(
                f"surrogate must be one of {self._VALID_SURROGATES}, got '{surrogate}'"
            )
        if tau_scale not in self._VALID_TAU_SCALES:
            raise ValueError(
                f"tau_scale must be one of {self._VALID_TAU_SCALES}, got '{tau_scale}'"
            )
        if pos_numerator not in self._VALID_POS_NUMERATORS:
            raise ValueError(
                f"pos_numerator must be one of {self._VALID_POS_NUMERATORS}, "
                f"got '{pos_numerator}'"
            )
        if budget_basis not in self._VALID_BUDGET_BASES:
            raise ValueError(
                f"budget_basis must be one of {self._VALID_BUDGET_BASES}, "
                f"got '{budget_basis}'"
            )
        if quantile_interpolation not in self._VALID_INTERPOLATIONS:
            raise ValueError(
                f"quantile_interpolation must be one of {self._VALID_INTERPOLATIONS}, "
                f"got '{quantile_interpolation}'"
            )

        super().__init__(
            num_classes=num_classes,
            queue_size=queue_size,
            temperature=temperature,
            reduction=reduction,
            ignore_index=ignore_index,
            update_queue_in_eval=update_queue_in_eval,
            gather_distributed=gather_distributed,
            max_pool_size=max_pool_size,
        )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.surrogate = surrogate
        self.n_knots = n_knots
        self.tau_scale = tau_scale
        self.pos_numerator = pos_numerator
        self.budget_basis = budget_basis
        self.quantile_interpolation = quantile_interpolation

        # Warn once per instance when a class is skipped due to near-zero
        # iid-negative dispersion (mirrors the _subsample_warned pattern).
        self._degenerate_warned = False

    # ------------------------------------------------------------------
    # Backward-compatible access to queue internals
    # ------------------------------------------------------------------
    # Tests and external code may access _q_logits, _q_targets, _q_ptr
    # directly on the loss instance. These properties delegate to the
    # nested _MemoryQueue submodule.

    @property
    def _q_logits(self):
        return self._queue._q_logits

    @_q_logits.setter
    def _q_logits(self, value):
        self._queue._q_logits = value

    @property
    def _q_targets(self):
        return self._queue._q_targets

    @_q_targets.setter
    def _q_targets(self, value):
        self._queue._q_targets = value

    @property
    def _q_ptr(self):
        return self._queue._q_ptr

    @_q_ptr.setter
    def _q_ptr(self, value):
        self._queue._q_ptr = value

    # ------------------------------------------------------------------
    # Backward-compatible queue methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _enqueue(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Delegate to the internal ``_MemoryQueue``."""
        self._queue.enqueue(logits, targets)

    def _merge_with_queue(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate to the internal ``_MemoryQueue``."""
        return self._queue.merge(logits, targets)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _band_thresholds_and_scale(
        self, neg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute detached band edges and the raw robust dispersion.

        Parameters
        ----------
        neg : torch.Tensor, shape [n_iid_neg]
            Detached iid-negative scores for one class.

        Returns
        -------
        t_alpha : torch.Tensor, scalar
            Lower-FPR band edge ``quantile(neg, 1 - alpha)`` (detached).
        t_beta : torch.Tensor, scalar
            Upper-FPR band edge ``quantile(neg, 1 - beta)`` (detached);
            always ``t_beta <= t_alpha`` since ``alpha < beta``.
        scale : torch.Tensor, scalar
            Raw (unclamped) robust dispersion of ``neg`` -- IQR or band
            width depending on ``tau_scale``.  The caller must test this
            against ``_SCALE_EPS`` before computing ``tau_eff``.
        """
        t_alpha = torch.quantile(
            neg, 1.0 - self.alpha, interpolation=self.quantile_interpolation
        )
        t_beta = torch.quantile(
            neg, 1.0 - self.beta, interpolation=self.quantile_interpolation
        )

        if self.tau_scale == "iqr":
            q75 = torch.quantile(
                neg, 0.75, interpolation=self.quantile_interpolation
            )
            q25 = torch.quantile(
                neg, 0.25, interpolation=self.quantile_interpolation
            )
            scale = q75 - q25
        else:  # "band"
            scale = t_alpha - t_beta

        return t_alpha, t_beta, scale

    def _compute_pauc(
        self,
        scores: torch.Tensor,
        is_pos: torch.Tensor,
        is_neg: torch.Tensor,
        is_iid: torch.Tensor,
        is_live: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        """
        Compute normalized partial AUC over ``[alpha, beta]`` for one class.

        Parameters
        ----------
        scores : torch.Tensor, shape [M]
            Pooled scores for one class (live + queue, padding stripped).
            Gradient flows through live-batch scores; queue scores are
            already detached upstream.
        is_pos : torch.Tensor, shape [M], dtype=bool
            Positive mask for this class.
        is_neg : torch.Tensor, shape [M], dtype=bool
            Negative mask for this class (``~is_pos``).
        is_iid : torch.Tensor, shape [M], dtype=bool
            Per-row iid-eligibility flag.
        is_live : torch.Tensor, shape [M], dtype=bool
            Per-row live-batch flag (True = live-batch row, False = queue).
            Consulted only when ``self.pos_numerator == "live"``.

        Returns
        -------
        pauc : torch.Tensor, scalar
            Normalized partial AUC estimate in [0, 1]. Zero (no gradient) for
            invalid classes.
        valid : bool
            False if there are no positives, no reference-population rows
            (iid negatives under ``budget_basis="fpr"``, or the whole iid
            population under ``budget_basis="population"``), the reference
            dispersion is near-zero (degenerate), or (pairwise) no band
            negatives. When ``pos_numerator="live"``, also False if there are
            no live positives for this class (no gradient signal this step).
            Invalid classes are excluded from reduction.
        diag : dict
            Diagnostic scalars for this class (all detached) when
            ``self._want_diag`` is True. Keys: ``t_alpha``, ``t_beta``,
            ``tau_eff``, ``band_neg_count``, ``pauc_var``. Returns an empty
            dict ``{}`` for invalid/degenerate classes or when
            ``self._want_diag`` is False; the caller (``_compute_per_class``)
            tolerates empty dicts via ``if diag:`` and leaves ``self._last_diag``
            at its nan/0 sentinel default for those classes.

        Notes
        -----
        Band edges ``t_alpha``/``t_beta`` and ``tau_eff`` are computed from a
        DETACHED reference population -- selected by ``self.budget_basis``
        (iid negatives under ``"fpr"``, the whole iid population under
        ``"population"``) -- regardless of ``pos_numerator``: the queue still
        stabilizes thresholds even when the numerator is restricted to live
        positives. The ``band_neg_count`` diagnostic always counts iid
        negatives in the band, independent of ``budget_basis``, since its
        semantics are population-level FPR. In trapezoid mode gradient
        reaches the numerator positives only; in pairwise mode it reaches
        numerator positives and band negatives (the band itself always
        selects negatives, only the quantile reference population moves).
        """
        iid_neg = is_neg & is_iid
        n_pos = int(is_pos.sum())
        n_iid_neg = int(iid_neg.sum())
        # Reference population for the band-edge quantiles: iid negatives only
        # ("fpr", true FPR semantics) or the whole pooled iid population
        # ("population", a top-k budget over all scores).  Either basis still
        # needs iid negatives to anchor the operating point, so the guard is on
        # them: in "fpr" mode this is exactly the pre-change n_iid_neg guard; in
        # "population" mode it additionally rejects a positives-only reference.
        ref_mask = is_iid if self.budget_basis == "population" else iid_neg
        if n_pos == 0 or n_iid_neg == 0:
            return scores.new_zeros(()), False, {}

        # Determine the numerator positive set.
        if self.pos_numerator == "live":
            pos_num = is_pos & is_live
            if int(pos_num.sum()) == 0:
                # No live positives this step: no gradient signal, mark invalid.
                return scores.new_zeros(()), False, {}
        else:
            # "pool": use all pooled positives (pre-change behavior).
            pos_num = is_pos

        ref = scores[ref_mask].detach()
        t_alpha, t_beta, scale = self._band_thresholds_and_scale(ref)

        # Degeneracy guard: if the robust dispersion is ~zero, the sigmoid
        # temperature cannot be calibrated.  Mark as invalid rather than
        # computing with tau_eff ≈ 1e-13 (which yields a signal-free loss=1
        # or an exploding gradient).  Emit a one-time warning.
        if scale <= self._SCALE_EPS:
            if not self._degenerate_warned:
                # The band edges + scale are computed over this reference set,
                # which depends on budget_basis.
                _ref_desc = (
                    "the pooled iid population (positives + negatives)"
                    if self.budget_basis == "population" else "the iid negatives"
                )
                _alpha_note = (
                    f"alpha=0 means t_alpha=max of {_ref_desc}; dispersion is "
                    f"near-zero because those scores are equal (or near-equal)."
                    if self.alpha == 0.0 else
                    f"fewer than ~{1.0 / self.alpha:.4g} reference rows are needed "
                    f"to resolve the tail quantile at alpha={self.alpha:.4g}."
                )
                warnings.warn(
                    f"{type(self).__name__}: reference-population score dispersion "
                    f"is near-zero (scale={scale.item():.2e} <= _SCALE_EPS={self._SCALE_EPS:.2e}) "
                    f"for at least one class. The band edges and scale are computed "
                    f"over {_ref_desc}; this typically means those scores are equal "
                    f"(or near-equal), or the band [{self.alpha}, {self.beta}] is too "
                    f"narrow relative to the available reference count ({_alpha_note}) "
                    f"The affected class is skipped (marked INVALID). "
                    f"To fix: increase queue_size or ensure {_ref_desc} cover a "
                    f"meaningful score range. "
                    f"(This warning is shown once per instance.)",
                    UserWarning,
                    stacklevel=5,
                )
                self._degenerate_warned = True
            return scores.new_zeros(()), False, {}

        # Apply the div-by-zero floor only AFTER the degeneracy check passes.
        tau_eff = self.temperature * scale.clamp_min(self._SCALE_EPS)

        if self.surrogate == "trapezoid":
            # FPR knots equally spaced over [alpha, beta]; threshold per knot.
            # dtype must match scores so torch.quantile doesn't raise on float64.
            f_k = torch.linspace(
                self.alpha, self.beta, self.n_knots,
                device=scores.device, dtype=scores.dtype
            )
            t_k = torch.quantile(
                ref, 1.0 - f_k, interpolation=self.quantile_interpolation
            )  # [n_knots], detached
            p = scores[pos_num]  # gradient flows here (numerator positive set)
            # [n_pos_num, n_knots]; each row is the contribution vector for one positive.
            contrib_mat = torch.sigmoid(
                (p.unsqueeze(1) - t_k.unsqueeze(0)) / tau_eff
            )
            # [n_knots] -- mean over numerator positives.
            tpr = contrib_mat.mean(dim=0)
            # Composite trapezoid on a uniform grid, normalized to [alpha, beta].
            pauc = (
                0.5 * tpr[0] + tpr[1:-1].sum() + 0.5 * tpr[-1]
            ) / (self.n_knots - 1)
            if not self._want_diag:
                return pauc, True, {}
            # Per-positive pAUC contribution: apply the same trapezoid weights
            # per row so that mean(v_i) == pauc.
            with torch.no_grad():
                weights = contrib_mat.new_ones(self.n_knots)
                weights[0] = 0.5
                weights[-1] = 0.5
                # [n_pos_num]
                v = (contrib_mat.detach() * weights.unsqueeze(0)).sum(dim=1) / (self.n_knots - 1)
                pauc_var = v.var(unbiased=False)
            # band_neg_count: iid negatives in the band [t_beta, t_alpha].
            # Always counted over iid negatives (population-FPR semantics),
            # regardless of budget_basis / ref.
            neg_iid_scores = scores[iid_neg].detach()
            band_neg_count = int(
                ((neg_iid_scores >= t_beta) & (neg_iid_scores <= t_alpha)).sum()
            )
            diag = {
                "t_alpha": t_alpha.detach(),
                "t_beta": t_beta.detach(),
                "tau_eff": tau_eff.detach(),
                "band_neg_count": band_neg_count,
                "pauc_var": pauc_var.detach(),
            }
            return pauc, True, diag

        # surrogate == "pairwise": band negatives from the GRADIENT POOL.
        band = is_neg & (scores >= t_beta) & (scores <= t_alpha)
        if int(band.sum()) == 0:
            return scores.new_zeros(()), False, {}
        p = scores[pos_num]   # numerator positive set (gradient flows here)
        b = scores[band]      # band negatives carry gradient (intended)
        # [n_pos_num, n_band]; each row is the per-positive contribution vector.
        contrib_mat = torch.sigmoid(
            (p.unsqueeze(1) - b.unsqueeze(0)) / tau_eff
        )
        pauc = contrib_mat.mean()
        if not self._want_diag:
            return pauc, True, {}
        # Per-positive contribution: mean over band negatives for each positive.
        with torch.no_grad():
            v = contrib_mat.detach().mean(dim=1)  # [n_pos_num]
            pauc_var = v.var(unbiased=False)
        # band_neg_count counts IID negatives in the band (consistent with
        # trapezoid), since the diagnostic semantics are population-level FPR.
        # Always counted over iid negatives regardless of budget_basis / ref.
        neg_iid_scores = scores[iid_neg].detach()
        band_neg_count = int(
            ((neg_iid_scores >= t_beta) & (neg_iid_scores <= t_alpha)).sum()
        )
        diag = {
            "t_alpha": t_alpha.detach(),
            "t_beta": t_beta.detach(),
            "tau_eff": tau_eff.detach(),
            "band_neg_count": band_neg_count,
            "pauc_var": pauc_var.detach(),
        }
        return pauc, True, diag

    # ------------------------------------------------------------------
    # Diagnostics-aware forward override
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        iid_mask: torch.Tensor | None = None,
        return_per_class: bool = False,
        return_diagnostics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
    ):
        """
        Compute the pAUC loss, optionally returning per-class diagnostics.

        Parameters
        ----------
        logits : torch.Tensor, shape [N, C]
            Raw (un-normalised) class scores.
        targets : torch.Tensor, shape [N]
            Integer class labels.  Positions equal to ``ignore_index``
            are excluded.
        iid_mask : torch.Tensor, shape [N], dtype=bool, optional
            Per-row iid-eligibility flag.  ``None`` treats all rows as iid.
        return_per_class : bool, optional
            If True, also return per-class losses and a validity mask.
        return_diagnostics : bool, optional
            If True, also return a ``stats`` dict with per-class diagnostic
            tensors of shape ``[C]``.  Invalid or degenerate classes yield
            ``nan`` for float fields and ``0`` for count fields.

            Keys: ``t_alpha``, ``t_beta``, ``tau_eff``, ``band_neg_count``,
            ``pauc_var``, ``grad_pos_count``.

            ``grad_pos_count`` is rank-local (computed from the live pre-gather
            batch); under DDP the true gradient-carrying positive population is
            the sum across all ranks.

            When ``False`` (default), behavior is bit-identical to the
            base-class forward.

        Returns
        -------
        loss : torch.Tensor
            Scalar or shape ``[C]`` (``reduction='none'``).
        per_class_loss : torch.Tensor, shape [C]
            Only when ``return_per_class=True``.
        valid_classes : torch.Tensor, shape [C], dtype=bool
            Only when ``return_per_class=True``.
        stats : dict[str, Tensor[C]]
            Only when ``return_diagnostics=True``.  Order in the tuple
            is ``(loss, per_class, valid, stats)`` or ``(loss, stats)``
            depending on ``return_per_class``.

        Notes
        -----
        ``self._last_diag`` is transient per-call internal state; it is
        reset at the top of every forward call.  Statefulness here is a
        deliberate tradeoff to avoid changing the shared ``_QueuedRankingLoss``
        base-class contract (which cannot accept extra return values from
        ``_compute_per_class``).
        """
        # --- squeeze [N,1] targets early so grad_pos_count sees the right shape --
        if targets.ndim == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # --- reset transient diagnostic state before every call ----------------
        # Always reset so _compute_per_class (called by super().forward) can
        # safely write to self._last_diag regardless of return_diagnostics.
        # Sentinel structure: float fields default to nan, count fields to 0.
        # The empty-pool early-return path in the base class skips
        # _compute_per_class, so _last_diag stays at this nan/0 default,
        # which is exactly the right diagnostic output for an empty pool.
        _nan = float("nan")
        self._last_diag: list[dict] = [
            {
                "t_alpha": _nan,
                "t_beta": _nan,
                "tau_eff": _nan,
                "band_neg_count": 0,
                "pauc_var": _nan,
            }
            for _ in range(self.num_classes)
        ]

        # --- grad_pos_count: live-batch positives per class (after ignore_index) -
        # Computed here because _compute_per_class sees the merged pool and
        # cannot distinguish live rows from queue rows.
        # Gate all diagnostic tensor ops in _compute_pauc behind this flag.
        # Must be set BEFORE super().forward() calls _compute_per_class.
        self._want_diag = bool(return_diagnostics)

        if not return_diagnostics:
            # Fast path: no diagnostics needed — bit-identical to base forward.
            return super().forward(
                logits, targets, iid_mask=iid_mask, return_per_class=return_per_class
            )

        valid_mask = targets != self.ignore_index
        filtered_targets = targets[valid_mask]
        grad_pos_count = logits.new_zeros(self.num_classes, dtype=torch.long)
        if self.num_classes == 1:
            grad_pos_count[0] = int(filtered_targets.bool().sum())
        else:
            for c in range(self.num_classes):
                grad_pos_count[c] = int((filtered_targets == c).sum())

        # --- delegate to base forward (runs _compute_per_class as side effect) --
        base_out = super().forward(
            logits, targets, iid_mask=iid_mask, return_per_class=return_per_class
        )

        # --- assemble stats dict from _last_diag --------------------------------
        dev = logits.device
        dtype = logits.dtype

        def _scalar_or_nan(val, is_nan_sentinel):
            """Return a float tensor from val, or nan if is_nan_sentinel."""
            if is_nan_sentinel:
                return torch.tensor(float("nan"), device=dev, dtype=dtype)
            if isinstance(val, torch.Tensor):
                return val.to(device=dev, dtype=dtype)
            return torch.tensor(float(val), device=dev, dtype=dtype)

        t_alpha_vals, t_beta_vals, tau_eff_vals = [], [], []
        band_neg_counts, pauc_var_vals = [], []

        for c in range(self.num_classes):
            d = self._last_diag[c]
            is_invalid = isinstance(d.get("t_alpha"), float) and (
                d["t_alpha"] != d["t_alpha"]  # nan check
            )
            t_alpha_vals.append(_scalar_or_nan(d["t_alpha"], is_invalid))
            t_beta_vals.append(_scalar_or_nan(d["t_beta"], is_invalid))
            tau_eff_vals.append(_scalar_or_nan(d["tau_eff"], is_invalid))
            band_neg_counts.append(
                torch.tensor(0 if is_invalid else d["band_neg_count"],
                             device=dev, dtype=torch.long)
            )
            pauc_var_vals.append(_scalar_or_nan(d["pauc_var"], is_invalid))

        stats: dict[str, torch.Tensor] = {
            "t_alpha":        torch.stack(t_alpha_vals),
            "t_beta":         torch.stack(t_beta_vals),
            "tau_eff":        torch.stack(tau_eff_vals),
            "band_neg_count": torch.stack(band_neg_counts),
            "pauc_var":       torch.stack(pauc_var_vals),
            "grad_pos_count": grad_pos_count.to(device=dev),
        }

        # --- build return value -------------------------------------------------
        if return_per_class:
            # base_out is (loss, per_class, valid)
            loss, per_class, valid = base_out
            return loss, per_class, valid, stats
        else:
            return base_out, stats

    # ------------------------------------------------------------------
    # Per-class dispatch (required by _QueuedRankingLoss)
    # ------------------------------------------------------------------

    def _compute_per_class(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_iid: torch.Tensor,
        is_live: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 1 - pAUC for each class via one-vs-rest decomposition.

        Parameters
        ----------
        logits : torch.Tensor, shape [M, C]
            Pooled logits (live batch + queue, ignore-index rows removed,
            subsampling applied).
        targets : torch.Tensor, shape [M]
            Corresponding integer targets.
        is_iid : torch.Tensor, shape [M], dtype=bool
            Per-row iid-eligibility flag for FPR-band threshold estimation.
        is_live : torch.Tensor, shape [M], dtype=bool
            Per-row live-batch flag; threaded into ``_compute_pauc`` so
            ``pos_numerator="live"`` can restrict the numerator positive set.

        Returns
        -------
        loss_vec : torch.Tensor, shape [C]
            Per-class loss values (1 - pAUC).
        valid_vec : torch.Tensor, shape [C], dtype=bool
            True for classes with positives, iid negatives, and (pairwise)
            band negatives.
        """
        if self.num_classes == 1:
            # Binary mode: warn on out-of-range targets
            bad = targets[(targets != 0) & (targets != 1)]
            if bad.numel() > 0:
                warnings.warn(
                    f"Binary mode (num_classes=1) expects targets in {{0, 1}}, "
                    f"but found values: {bad[:8].tolist()}. "
                    "Non-zero values are treated as positive.",
                    UserWarning,
                    stacklevel=4,
                )
            is_pos = targets.bool()
            pauc, is_valid, diag = self._compute_pauc(
                logits[:, 0], is_pos, ~is_pos, is_iid, is_live
            )
            loss_vals = [1.0 - pauc]
            valid_mask = [is_valid]
            if diag:
                self._last_diag[0] = diag
        else:
            loss_vals, valid_mask = [], []
            for c in range(self.num_classes):
                is_pos = targets == c
                pauc, is_valid, diag = self._compute_pauc(
                    logits[:, c], is_pos, ~is_pos, is_iid, is_live
                )
                loss_vals.append(1.0 - pauc)
                valid_mask.append(is_valid)
                if diag:
                    self._last_diag[c] = diag

        loss_vec = torch.stack(loss_vals)
        valid_vec = torch.tensor(valid_mask, device=logits.device)
        return loss_vec, valid_vec
