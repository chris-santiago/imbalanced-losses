# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8",
#   "numpy",
#   "scikit-learn",
#   "matplotlib",
#   "metaflow>=2.19",
#   "hydra-core",
#   "omegaconf",
#   "pyyaml",
# ]
# ///
"""
Component functions for the consolidated PAUC-vs-CE Metaflow pipeline.

All component functions are defined at module level.  Top-level imports cover
only torch, numpy, and sklearn so that ``import pauc_flow`` for testing pulls
only those packages.  Orchestration libraries (metaflow, hydra, omegaconf,
matplotlib) are imported lazily inside the functions that use them.

Test suite command (no pyproject.toml changes needed):
    uv run --with torch --with numpy --with scikit-learn --with pytest \\
        pytest lab/pauc_vs_ce_regimes/flow/test_pauc_flow.py -q

Run from the repo root or any directory — the command above is self-contained.

grp codes
---------
    0  easy-negative
    1  positive
    2  decoy
    3  confounder
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

# Make imbalanced_losses importable when this file is run as a PEP-723 script
# (uv run flow/pauc_flow.py).  In project mode (pytest) the package is already
# on sys.path via the installed editable install, so this insert is a no-op.
# Resolves to .../imbalanced-losses/src, which contains imbalanced_losses/.
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


# ===========================================================================
# Geometry primitives
# ===========================================================================

def inject_easy(X: np.ndarray, idx: np.ndarray, easy: float = 2.5, easy_feats: "list[int] | None" = None) -> None:
    """Add easy-signal shift on the given easy feature columns in-place.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    idx : np.ndarray
        Row indices to shift.
    easy : float
        Shift magnitude.  Default 2.5 matches conf/geometry/headline.yaml.
    easy_feats : list[int], optional
        Column indices for the easy features.  Default [0, 1, 2].
    """
    if easy_feats is None:
        easy_feats = [0, 1, 2]
    X[np.ix_(idx, easy_feats)] += easy


def inject_cue(
    X: np.ndarray,
    pos_idx: np.ndarray,
    cue: str,
    rng: np.random.Generator,
    sep: float = 1.7,
    cue_feats: "list[int] | None" = None,
) -> None:
    """Apply operating-point cue to positive rows in-place.

    Parameters
    ----------
    X : np.ndarray
    pos_idx : np.ndarray
    cue : str
        'linear', 'nonlinear_prod' / 'product', or 'nonlinear_radial' / 'radial'.
    rng : np.random.Generator
    sep : float
        Cue separation magnitude.  Default 1.7 matches conf/geometry/headline.yaml.
    cue_feats : list[int], optional
        Column indices for the cue features.  Default [5, 6].

    cue values:
        'linear'           -- additive shift on cue_feats; linear boundary.
        'nonlinear_prod'   -- same-sign ±sep on cue_feats; product cue.
        'nonlinear_radial' -- independent-sign ±sep on cue_feats; radial cue.
    These names match 01_cue_ablation conventions.  04 and 05 use 'product' /
    'radial' as aliases for the same logic.
    """
    if cue_feats is None:
        cue_feats = [5, 6]
    f0, f1 = cue_feats[0], cue_feats[1]
    n = len(pos_idx)
    if cue in ("linear",):
        X[pos_idx, f0] += sep
        X[pos_idx, f1] += sep
    elif cue in ("nonlinear_prod", "product"):
        s = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        X[pos_idx, f0] += sep * s
        X[pos_idx, f1] += sep * s
    elif cue in ("nonlinear_radial", "radial"):
        s5 = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        s6 = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        X[pos_idx, f0] += sep * s5
        X[pos_idx, f1] += sep * s6
    else:
        raise ValueError(f"Unknown cue: {cue!r}")


def inject_decoys(
    X: np.ndarray,
    neg_idx: np.ndarray,
    rng: np.random.Generator,
    decoy_frac: float = 0.012,
    easy: float = 2.5,
    easy_feats: "list[int] | None" = None,
) -> np.ndarray:
    """Choose decoy indices from neg_idx, apply easy-signal shift in-place.

    Returns the decoy indices (subset of neg_idx).

    Parameters
    ----------
    decoy_frac : float
        Fraction of negatives to make decoys.  Default 0.012 matches headline.yaml.
    easy : float
        Easy-signal magnitude.  Default 2.5 matches headline.yaml.
    easy_feats : list[int], optional
        Column indices for the easy features.  Default [0, 1, 2].
    """
    if easy_feats is None:
        easy_feats = [0, 1, 2]
    n_decoy = round(decoy_frac * len(neg_idx))
    decoy = rng.choice(neg_idx, size=n_decoy, replace=False)
    X[np.ix_(decoy, easy_feats)] += easy
    return decoy


def inject_confounders(
    X: np.ndarray,
    neg_idx: np.ndarray,
    decoy_idx: np.ndarray,
    rng: np.random.Generator,
    confounder_frac: float,
    sep: float = 1.7,
    cue_feats: "list[int] | None" = None,
) -> np.ndarray:
    """Choose confounder indices from neg_idx (excluding decoys), apply nonlinear
    product cue in-place.  Confounders carry the cue but NOT the easy signal.

    Returns the confounder indices.

    Parameters
    ----------
    sep : float
        Cue separation magnitude.  Default 1.7 matches headline.yaml.
    cue_feats : list[int], optional
        Column indices for the cue features.  Default [5, 6].
    """
    if cue_feats is None:
        cue_feats = [5, 6]
    f0, f1 = cue_feats[0], cue_feats[1]
    rest = np.setdiff1d(neg_idx, decoy_idx, assume_unique=True)
    n_conf = round(confounder_frac * len(neg_idx))
    if n_conf == 0 or len(rest) == 0:
        return np.array([], dtype=np.int64)
    conf = rng.choice(rest, size=min(n_conf, len(rest)), replace=False)
    if len(conf):
        sc = rng.choice([-1.0, 1.0], size=len(conf)).astype(np.float32)
        X[conf, f0] += sep * sc
        X[conf, f1] += sep * sc
    return conf


def make_data(
    geometry_spec: dict,
    seed: int,
    rng: "np.random.Generator | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate synthetic data deterministically from geometry_spec and seed.

    Parameters
    ----------
    geometry_spec : dict
        Must contain 'kind' in {'headline', 'confounder', 'hard_bulk'}.

        headline / confounder keys (all optional, have defaults):
            n          int   total samples           (default: 150000)
            n_feat     int   feature dimension        (default: 20)
            easy       float easy-signal magnitude    (default: 2.5)
            sep        float cue separation           (default: 1.7)
            pos_rate   float positive rate            (default: 0.0015)
            decoy_frac float decoy fraction of negs   (default: 0.012)
            cue        str   cue form for headline    (default: 'nonlinear_prod')
            confounder_frac float  (confounder only)  (default: 0.0)

        hard_bulk keys (all optional):
            n          int   total samples         (default: 100000)
            pos_rate   float                       (default: 0.005)
            decoy_frac float                       (default: 0.0 — no decoy by default)
            d_bulk     int   bulk feature dims     (default: 4)
            d_contest  int   contest feature dims  (default: 3)
            d_noise    int   noise feature dims    (default: 24)
            mu_bulk    float bulk shift magnitude  (default: 0.30)
            mu_contest float contest shift mag     (default: 0.40)

        reproducibility key (all kinds):
            seed_base  int   added to seed before passing to np.random.default_rng.
                             Default 7000, matching experiments 01–06/08.
                             hard_bulk/confounder cells do NOT reproduce 07/08's
                             compound seed formulas under the default base; CI
                             overlap is the acceptance criterion.  For exact
                             replication of a specific experiment's per-seed
                             numbers, pass the appropriate seed_base.

    seed : int
        Passed to np.random.default_rng(seed_base + seed) when rng is None.
        Ignored when an external rng is supplied.  Default seed_base=7000
        matches the convention in experiments 01–06.

    rng : np.random.Generator, optional
        If supplied, this generator is used directly (the seed and seed_base
        fields in geometry_spec are ignored for RNG construction).  The caller
        is responsible for seeding it.  This is the entry point for the
        sequential split convention (experiments 01–06), where a single RNG
        is advanced across the train call and then the test call to reproduce
        the original two-call stream from 01_cue_ablation.py:
            rng = np.random.default_rng(7000 + seed)
            make_data(train_spec, seed, rng)   # advances rng
            make_data(test_spec,  seed, rng)   # continues from same stream

    Returns
    -------
    X : torch.Tensor, shape [N, D], dtype=float32
    y : torch.Tensor, shape [N, 1], dtype=float32
        1 for positives, 0 for negatives.
    grp : np.ndarray, shape [N], dtype=int64
        Group codes: 0=easy-negative, 1=positive, 2=decoy, 3=confounder.
    """
    kind = geometry_spec["kind"]
    if rng is None:
        seed_base = geometry_spec.get("seed_base", 7000)
        rng = np.random.default_rng(seed_base + seed)

    if kind == "headline":
        return _make_headline(geometry_spec, rng)
    elif kind == "confounder":
        return _make_confounder(geometry_spec, rng)
    elif kind == "hard_bulk":
        return _make_hard_bulk(geometry_spec, rng)
    else:
        raise ValueError(f"Unknown geometry kind: {kind!r}")


def _make_headline(spec: dict, rng: np.random.Generator) -> tuple:
    """Headline geometry used in experiments 01-06.

    All geometry parameters are read from spec; no module-level constants used.
    Defaults match conf/geometry/headline.yaml.
    """
    n = spec.get("n", 150_000)
    n_feat = spec.get("n_feat", 20)
    easy = spec.get("easy", 2.5)
    sep = spec.get("sep", 1.7)
    pos_rate = spec.get("pos_rate", 0.0015)
    decoy_frac = spec.get("decoy_frac", 0.012)
    cue = spec.get("cue", "nonlinear_prod")
    easy_feats = list(spec.get("easy_feats", [0, 1, 2]))
    cue_feats = list(spec.get("cue_feats", [5, 6]))

    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    grp = np.zeros(n, dtype=np.int64)

    n_pos = max(8, round(pos_rate * n))
    pos = rng.choice(n, size=n_pos, replace=False)
    y[pos] = 1.0
    grp[pos] = 1

    neg_idx = np.where(y == 0)[0]
    decoy = inject_decoys(X, neg_idx, rng, decoy_frac, easy=easy, easy_feats=easy_feats)
    grp[decoy] = 2

    inject_easy(X, pos, easy=easy, easy_feats=easy_feats)
    inject_cue(X, pos, cue, rng, sep=sep, cue_feats=cue_feats)

    return torch.tensor(X), torch.tensor(y).unsqueeze(1), grp


def _make_confounder(spec: dict, rng: np.random.Generator) -> tuple:
    """Confounder geometry used in experiment 08.

    Positives and decoys get the easy signal; confounders carry the nonlinear
    product cue WITHOUT the easy signal.

    All geometry parameters are read from spec; no module-level constants used.
    Defaults match conf/geometry/confounder.yaml.
    """
    n = spec.get("n", 150_000)
    n_feat = spec.get("n_feat", 20)
    easy = spec.get("easy", 2.5)
    sep = spec.get("sep", 1.7)
    pos_rate = spec.get("pos_rate", 0.0015)
    decoy_frac = spec.get("decoy_frac", 0.012)
    confounder_frac = spec.get("confounder_frac", 0.0)
    easy_feats = list(spec.get("easy_feats", [0, 1, 2]))
    cue_feats = list(spec.get("cue_feats", [5, 6]))

    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    grp = np.zeros(n, dtype=np.int64)

    n_pos = max(8, round(pos_rate * n))
    pos = rng.choice(n, size=n_pos, replace=False)
    y[pos] = 1.0
    grp[pos] = 1

    neg_idx = np.where(y == 0.0)[0]
    decoy = inject_decoys(X, neg_idx, rng, decoy_frac, easy=easy, easy_feats=easy_feats)
    grp[decoy] = 2

    conf = inject_confounders(X, neg_idx, decoy, rng, confounder_frac, sep=sep, cue_feats=cue_feats)
    grp[conf] = 3

    # inject_decoys already applied easy signal to decoys; apply separately to positives.
    inject_easy(X, pos, easy=easy, easy_feats=easy_feats)

    # Nonlinear product cue on positives
    inject_cue(X, pos, "nonlinear_prod", rng, sep=sep, cue_feats=cue_feats)

    return torch.tensor(X), torch.tensor(y).unsqueeze(1), grp


def _make_hard_bulk(spec: dict, rng: np.random.Generator) -> tuple:
    """Hard-bulk geometry used in experiment 07.

    Layout: 31 dims = D_BULK(4) + D_CONTEST(3) + D_NOISE(24).
    Positives: bulk and contest features shifted up.
    Easy negatives: bulk down, contest down.
    Decoys: bulk up, contest down (compete in bulk, lose on contest).
    No decoys by default (decoy_frac=0.0).

    grp: 0=easy-negative, 1=positive, 2=decoy.
    (No confounders in hard_bulk geometry.)

    All geometry parameters are read from spec; no module-level constants used.
    Defaults match conf/geometry/hard_bulk.yaml.
    """
    n = spec.get("n", 100_000)
    pos_rate = spec.get("pos_rate", 0.005)
    decoy_frac = spec.get("decoy_frac", 0.0)
    d_bulk = spec.get("d_bulk", 4)
    d_contest = spec.get("d_contest", 3)
    d_noise = spec.get("d_noise", 24)
    mu_bulk = spec.get("mu_bulk", 0.30)
    mu_contest = spec.get("mu_contest", 0.40)

    d = d_bulk + d_contest + d_noise
    b = slice(0, d_bulk)
    c = slice(d_bulk, d_bulk + d_contest)

    n_pos = max(1, int(round(n * pos_rate)))
    n_neg = n - n_pos
    n_decoy = int(round(n_neg * decoy_frac))
    n_easy = n_neg - n_decoy

    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)

    # Positives occupy the first n_pos rows before permutation
    X[:n_pos, b] += mu_bulk
    X[:n_pos, c] += mu_contest
    y[:n_pos] = 1

    e0, e1 = n_pos, n_pos + n_easy
    X[e0:e1, b] -= mu_bulk
    X[e0:e1, c] -= mu_contest

    # Decoys: bulk up, contest down
    X[e1:, b] += mu_bulk
    X[e1:, c] -= mu_contest

    perm = rng.permutation(n)
    X = X[perm]
    y_perm = y[perm]

    # Build grp after permutation: map original row ranges to permuted positions.
    grp = np.zeros(n, dtype=np.int64)
    orig_pos = np.where(y == 1)[0]
    orig_easy = np.arange(n_pos, n_pos + n_easy)
    orig_decoy = np.arange(n_pos + n_easy, n)

    grp[perm[orig_pos]] = 1
    grp[perm[orig_easy]] = 0
    grp[perm[orig_decoy]] = 2

    y_float = y_perm.astype(np.float32)
    return (
        torch.tensor(X),
        torch.tensor(y_float).unsqueeze(1),
        grp,
    )


# ===========================================================================
# Model builder
# ===========================================================================

def build_model(capacity: str, d_in: int = 20, hid: int = 64) -> nn.Module:
    """Build a model for a given capacity string.

    Parameters
    ----------
    capacity : {'linear', 'mlp_1x16', 'mlp_2x64'}
        Architecture variant.  'linear' is a single linear layer.
        'mlp_1x16' is a 1-hidden-layer MLP with 16 units.
        'mlp_2x64' is the 2-hidden-layer hid-unit MLP used in 01-08
        as the default headline/confounder model.
    d_in : int
        Input dimension.  Defaults to 20 (headline/confounder).
        Pass 31 for hard_bulk geometry.
    hid : int
        Hidden layer width for mlp_2x64.  Default 64 matches
        conf/training/default.yaml.  Sourced from cfg.training.hid in the flow.

    Notes
    -----
    The headline/confounder default in experiments 01-06/08 is equivalent to
    'mlp_2x64' with d_in=20, hid=64.  Calling build_model('mlp_2x64') reproduces that.
    """
    if capacity == "linear":
        return nn.Linear(d_in, 1)
    elif capacity == "mlp_1x16":
        return nn.Sequential(
            nn.Linear(d_in, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    elif capacity == "mlp_2x64":
        return nn.Sequential(
            nn.Linear(d_in, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1),
        )
    else:
        raise ValueError(f"Unknown capacity: {capacity!r}")


# ===========================================================================
# Training utilities
# ===========================================================================

def pos_weight(y: torch.Tensor) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from label tensor [N,1] or [N]."""
    n_pos = float(y.sum())
    n_neg = float(y.numel()) - n_pos
    return torch.tensor([n_neg / max(1.0, n_pos)])


def idx_batches(n: int, seed: int, batch_size: int = 4096) -> list[torch.Tensor]:
    """Return a deterministic per-epoch shuffle of indices as a list of index tensors.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Generator seed.  Call with ``seed + epoch`` to get per-epoch reshuffles.
    batch_size : int
        Batch size.  Default 4096 matches conf/training/default.yaml.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return [perm[i:i + batch_size] for i in range(0, n, batch_size)]


@torch.no_grad()
def scores_of(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Run model in eval mode and return squeezed numpy scores.

    Restores the original training/eval mode on exit.  Gradient is disabled.
    """
    was_training = model.training
    model.eval()
    s = model(X).squeeze(1).numpy()
    model.train(was_training)
    return s


# ===========================================================================
# train_arm dispatcher
# ===========================================================================

def train_arm(
    arm_spec: dict,
    data: dict,
    seed: int,
) -> tuple[nn.Module, dict]:
    """Train a model according to arm_spec and return (model, info).

    Parameters
    ----------
    arm_spec : dict
        Must contain 'kind' in {'trivial', 'ce', 'pauc', 'smoothap',
        'ce_warmup_only', 'pauc_cold'}.

        Common optional keys:
            capacity   str    'linear'|'mlp_1x16'|'mlp_2x64'  (default 'mlp_2x64')
            lr         float  learning rate                      (default 1e-3)
            wd         float  weight decay                       (default 0.0)
            epochs     int    training epochs                    (default 15)
            batch_size int                                       (default 4096)
            early_stop bool   if True, use val coverage for early stopping
            eval_every int    early-stop check cadence           (default 25)
            budget     float  FPR budget for coverage@budget     (default 0.005)

        ce-specific optional keys:
            hnm        dict   {'top_q': float, 'factor': float}
            oracle     dict   {'factor': float}

        pauc-specific optional keys:
            surrogate  str    'pairwise'|'trapezoid'             (default 'pairwise')
            band       tuple  (alpha_mult, beta_mult)            (default (0.5, 1.5))
            queue_size int                                       (default 8192)
            temp       float  temperature (start, for warmup anneal)  (default 0.5)
            temp_end   float  annealed-to temperature            (default 0.1)
            warmup_frac float fraction of total steps for CE warmup  (default 0.30)
            blend_frac float fraction of steps for blend        (default 0.15)

        smoothap-specific optional keys:
            queue_size int
            temp       float  (default 0.1)
            temp_end   float  (default 0.01)
            warmup_frac float

    data : dict
        Expected keys: 'X_train', 'y_train', 'grp_train'.
        If arm_spec['early_stop'] is True, also expects 'X_val', 'y_val', 'grp_val'.

    seed : int

    Returns
    -------
    model : nn.Module
    info : dict
        Contains 'best_val_cov' when early_stop is used; otherwise empty or
        contains the curve when requested.
    """
    kind = arm_spec["kind"]
    X = data["X_train"]
    y = data["y_train"]
    grp = data.get("grp_train")

    capacity = arm_spec.get("capacity", "mlp_2x64")
    d_in = X.shape[1]
    lr = arm_spec.get("lr", 1e-3)
    wd = arm_spec.get("wd", 0.0)
    epochs = arm_spec.get("epochs", 15)
    batch_size = arm_spec.get("batch_size", 4096)
    budget = arm_spec.get("budget", 0.005)
    early_stop = arm_spec.get("early_stop", False)
    eval_every = arm_spec.get("eval_every", 25)

    if kind == "trivial":
        return _train_trivial(arm_spec, data, seed)

    hid = arm_spec.get("hid", 64)
    torch.manual_seed(seed)
    model = build_model(capacity, d_in, hid=hid)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if kind == "ce":
        return _train_ce(
            model, opt, X, y, grp, seed, epochs, batch_size, budget,
            arm_spec, early_stop, eval_every, data,
        )
    elif kind in ("pauc", "pauc_cold"):
        return _train_pauc(
            model, opt, X, y, seed, epochs, batch_size, budget,
            arm_spec, early_stop, eval_every, data, cold=(kind == "pauc_cold"),
        )
    elif kind == "ce_warmup_only":
        return _train_ce_warmup_only(
            model, opt, X, y, grp, seed, epochs, batch_size, budget,
            arm_spec, early_stop, eval_every, data,
        )
    elif kind == "smoothap":
        return _train_smoothap(
            model, opt, X, y, seed, epochs, batch_size, budget,
            arm_spec, early_stop, eval_every, data,
        )
    else:
        raise ValueError(f"Unknown arm kind: {kind!r}")


def _train_trivial(
    arm_spec: dict, data: dict, seed: int
) -> tuple[nn.Module, dict]:
    """Trivial baseline: mean of easy-signal features [0,1,2].

    Returns a dummy nn.Module whose forward returns the feature mean.
    Works for both headline/confounder (features 0-1-2 are easy) and
    hard_bulk (features 0..D_BULK-1 are the bulk signal).
    """
    bulk_feats = arm_spec.get("bulk_feats", [0, 1, 2])

    class _TrivialModel(nn.Module):
        def __init__(self, feats, X_mean, X_std):
            super().__init__()
            self.feats = feats
            # Register as buffers so they move with the model
            self.register_buffer("_mean", X_mean)
            self.register_buffer("_std", X_std)

        def forward(self, X):
            normed = (X - self._mean) / self._std
            return normed[:, self.feats].mean(dim=1, keepdim=True)

    X = data["X_train"]
    mu = X.mean(0)
    sd = X.std(0).clamp_min(1e-8)
    return _TrivialModel(bulk_feats, mu, sd), {}


def _train_ce(
    model, opt, X, y, grp, seed, epochs, batch_size, budget,
    arm_spec, early_stop, eval_every, data,
) -> tuple[nn.Module, dict]:
    """Plain CE training with optional HNM or oracle up-weighting."""
    pw = pos_weight(y)
    base_w = pw.item()
    grp_t = torch.tensor(grp) if grp is not None else None

    hnm_cfg = arm_spec.get("hnm")
    oracle_cfg = arm_spec.get("oracle")

    # Static per-sample weights (oracle mode) or dynamic (hnm mode)
    w = torch.ones(X.size(0))
    if grp_t is not None:
        w[grp_t == 1] = base_w
        if oracle_cfg is not None:
            w[grp_t == 2] = oracle_cfg["factor"]

    best_val_cov = -1.0
    best_state = None

    for epoch in range(epochs):
        # Dynamic HNM: recompute hard-negative weights each epoch
        if hnm_cfg is not None and grp_t is not None:
            sc = scores_of(model, X)
            negmask = grp != 1
            thr = np.quantile(sc[negmask], 1.0 - hnm_cfg["top_q"])
            w = torch.ones(X.size(0))
            w[grp_t == 1] = base_w
            hot = torch.tensor((sc >= thr) & negmask)
            w[hot] = hnm_cfg["factor"]

        for idx in idx_batches(X.size(0), seed + epoch, batch_size):
            xb, yb, wb = X[idx], y[idx], w[idx].unsqueeze(1)
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(xb), yb, weight=wb).backward()
            opt.step()

        if early_stop and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            X_val = data["X_val"]
            grp_val = data.get("grp_val")
            val_y = (grp_val == 1).astype(np.float32) if grp_val is not None else data["y_val"].squeeze(1).numpy()
            vc = coverage(val_y, scores_of(model, X_val), budget)
            if vc > best_val_cov:
                best_val_cov = vc
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_cov": best_val_cov if early_stop else None}


def _train_pauc(
    model, opt, X, y, seed, epochs, batch_size, budget,
    arm_spec, early_stop, eval_every, data, cold: bool = False,
) -> tuple[nn.Module, dict]:
    """PAUC training via LossWarmupWrapper (or cold-start without warmup).

    cold=True skips the CE warmup (pauc_cold variant from 07).

    Per-epoch reshuffle convention
    --------------------------------
    Batches are reshuffled every epoch via seed + epoch, matching the CE arm
    and the explicit contract in experiment 03 (03_mechanism_transfer.py):
    "Reshuffle per epoch (seed + e) to match the CE/HNM arm — keeps the
    optimizer trajectory symmetric across arms; only the loss differs."

    Warmup overridability for 07-style capacity cells
    --------------------------------------------------
    Experiment 07 uses a fixed-temperature warmup (temp_start == temp_end, no
    anneal) with a longer warmup_frac.  Express this by passing explicit
    ``temp``, ``temp_end``, and ``warmup_frac`` keys in arm_spec.  The
    defaults (0.5->0.1 anneal, warmup_frac=0.30) are designed for the headline
    regime (01-06/08) and are unsafe for capacity/07 cells; those cells MUST
    override all three keys.
    """
    from imbalanced_losses import LossWarmupWrapper, PAUCAtBudgetLoss

    alpha_mult, beta_mult = arm_spec.get("band", (0.5, 1.5))
    alpha = alpha_mult * budget
    beta = beta_mult * budget
    surrogate = arm_spec.get("surrogate", "pairwise")
    queue_size = arm_spec.get("queue_size", 8192)
    temp_start = arm_spec.get("temp", 0.5)
    temp_end = arm_spec.get("temp_end", 0.1)
    warmup_frac = arm_spec.get("warmup_frac", 0.30)
    blend_frac = arm_spec.get("blend_frac", 0.15)

    # Compute total steps using a representative epoch to size warmup/blend.
    # Batches are regenerated per epoch (see convention note in docstring).
    n_batches_per_epoch = len(idx_batches(X.size(0), seed, batch_size))
    total = epochs * n_batches_per_epoch

    pw = pos_weight(y)
    main_loss = PAUCAtBudgetLoss(
        num_classes=1,
        alpha=alpha,
        beta=beta,
        surrogate=surrogate,
        pos_numerator="pool",
        queue_size=queue_size,
    )

    if cold:
        loss_fn = main_loss
        mode = "pauc_cold"
    else:
        loss_fn = LossWarmupWrapper(
            warmup_loss=nn.BCEWithLogitsLoss(pos_weight=pw),
            main_loss=main_loss,
            warmup_steps=int(warmup_frac * total),
            blend_steps=int(blend_frac * total),
            temp_start=temp_start,
            temp_end=temp_end,
            temp_decay_steps=max(1, total - int(warmup_frac * total)),
        )
        mode = "wrapper"

    best_val_cov = -1.0
    best_state = None
    gstep = 0

    for epoch in range(epochs):
        if mode == "wrapper":
            loss_fn.on_train_epoch_start(epoch)
        # Reshuffle per epoch (seed + epoch) — symmetric with the CE arm.
        # Source: 03_mechanism_transfer.py, train_pauc, with inline comment.
        for idx in idx_batches(X.size(0), seed + epoch, batch_size):
            if mode == "wrapper":
                loss_fn.on_train_batch_start(gstep)
            xb, yb = X[idx], y[idx]
            logits = model(xb)
            if mode == "pauc_cold":
                loss = loss_fn(logits, yb.squeeze(1).long())
            else:
                loss = loss_fn(logits, yb)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                opt.step()
            gstep += 1

        if early_stop and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            X_val = data["X_val"]
            grp_val = data.get("grp_val")
            val_y = (grp_val == 1).astype(np.float32) if grp_val is not None else data["y_val"].squeeze(1).numpy()
            vc = coverage(val_y, scores_of(model, X_val), budget)
            if vc > best_val_cov:
                best_val_cov = vc
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_cov": best_val_cov if early_stop else None}


def _train_ce_warmup_only(
    model, opt, X, y, grp, seed, epochs, batch_size, budget,
    arm_spec, early_stop, eval_every, data,
) -> tuple[nn.Module, dict]:
    """CE-only training for the warmup fraction of total epochs (from 07 F3).

    Trains for warmup_frac * epochs epochs using plain weighted CE.
    """
    warmup_frac = arm_spec.get("warmup_frac", 0.30)
    warm_ep = max(1, int(round(epochs * warmup_frac)))
    warm_spec = dict(arm_spec)
    warm_spec["kind"] = "ce"
    warm_spec["epochs"] = warm_ep
    return _train_ce(
        model, opt, X, y, grp, seed, warm_ep, batch_size, budget,
        warm_spec, early_stop, eval_every, data,
    )


def _train_smoothap(
    model, opt, X, y, seed, epochs, batch_size, budget,
    arm_spec, early_stop, eval_every, data,
) -> tuple[nn.Module, dict]:
    """SmoothAP training via LossWarmupWrapper (matches 08 convention).

    Per-epoch reshuffle (seed + epoch) is used for symmetry with the CE arm;
    see _train_pauc docstring for the source citation.
    """
    from imbalanced_losses import LossWarmupWrapper, SmoothAPLoss

    queue_size = arm_spec.get("queue_size", 8192)
    temp_start = arm_spec.get("temp", 0.1)
    temp_end = arm_spec.get("temp_end", 0.01)
    warmup_frac = arm_spec.get("warmup_frac", 0.30)
    blend_frac = arm_spec.get("blend_frac", 0.15)

    # Compute total steps using a representative epoch to size warmup/blend.
    n_batches_per_epoch = len(idx_batches(X.size(0), seed, batch_size))
    total = epochs * n_batches_per_epoch

    pw = pos_weight(y)
    loss_fn = LossWarmupWrapper(
        warmup_loss=nn.BCEWithLogitsLoss(pos_weight=pw),
        main_loss=SmoothAPLoss(num_classes=1, queue_size=queue_size),
        warmup_steps=int(warmup_frac * total),
        blend_steps=int(blend_frac * total),
        temp_start=temp_start,
        temp_end=temp_end,
        temp_decay_steps=max(1, total - int(warmup_frac * total)),
    )

    best_val_cov = -1.0
    best_state = None
    gstep = 0

    for epoch in range(epochs):
        loss_fn.on_train_epoch_start(epoch)
        # Reshuffle per epoch (seed + epoch) — symmetric with the CE arm.
        # Source: 03_mechanism_transfer.py, train_pauc, with inline comment.
        for idx in idx_batches(X.size(0), seed + epoch, batch_size):
            loss_fn.on_train_batch_start(gstep)
            xb, yb = X[idx], y[idx]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                opt.step()
            gstep += 1

        if early_stop and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            X_val = data["X_val"]
            grp_val = data.get("grp_val")
            val_y = (grp_val == 1).astype(np.float32) if grp_val is not None else data["y_val"].squeeze(1).numpy()
            vc = coverage(val_y, scores_of(model, X_val), budget)
            if vc > best_val_cov:
                best_val_cov = vc
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_cov": best_val_cov if early_stop else None}


# ===========================================================================
# Metric functions
# ===========================================================================

def coverage(y: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """Top-K recall estimator: fraction of positives in the top ceil(budget*N) scores.

    Parameters
    ----------
    y : np.ndarray, shape [N]
        Binary label array (1=positive, 0=negative).  Any positive > 0.5 counts.
    scores : np.ndarray, shape [N]
        Model scores (higher = more positive).
    budget : float
        FPR budget, e.g. 0.005 for 50 bps.

    Returns
    -------
    float in [0, 1].
    """
    k = max(1, int(np.ceil(budget * len(scores))))
    top_idx = np.argsort(-scores)[:k]
    n_pos = float(y.sum())
    if n_pos == 0:
        return 0.0
    return float(y[top_idx].sum() / n_pos)


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Wrapper around sklearn roc_auc_score."""
    return float(roc_auc_score(y, scores))


def aucpr(y: np.ndarray, scores: np.ndarray) -> float:
    """Wrapper around sklearn average_precision_score."""
    return float(average_precision_score(y, scores))


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI.

    Parameters
    ----------
    values : array-like, shape [N]
    n_resamples : int
    seed : int
        RNG seed for determinism.

    Returns
    -------
    (mean, lo, hi) : tuple of floats
        lo = 2.5th percentile, hi = 97.5th percentile of bootstrap means.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(v, size=len(v), replace=True).mean()
        for _ in range(n_resamples)
    ])
    return float(v.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ===========================================================================
# Diagnostic functions
# ===========================================================================

def grad_mass_on_decoys(
    model: nn.Module,
    data: dict,
) -> dict[str, float]:
    """Fraction of negative-gradient L1 mass that lands on decoys.

    Measures on the training set using fixed logits (CE-warmed model's scores).
    Computes the fraction under both the CE loss and the PAUC pairwise loss.

    Parameters
    ----------
    model : nn.Module
        Trained model (used for its current scores).
    data : dict
        Must contain 'X_train', 'y_train', 'grp_train'.
        Budget is read from data.get('budget', 0.005).

    Returns
    -------
    dict with keys 'ce' and 'pauc', each a float fraction in [0, 1].
    """
    from imbalanced_losses import PAUCAtBudgetLoss

    X = data["X_train"]
    y = data["y_train"]
    grp = data["grp_train"]
    budget = data.get("budget", 0.005)
    alpha = budget / 2
    beta = budget * 1.5

    z_np = scores_of(model, X)
    y_np = y.squeeze(1).numpy() if y.dim() == 2 else y.numpy()
    grp_np = grp

    neg = grp_np != 1
    out: dict[str, float] = {}

    # CE gradient mass on decoys
    z_ce = torch.tensor(z_np, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    yt = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    pw = pos_weight(yt)
    F.binary_cross_entropy_with_logits(z_ce, yt, pos_weight=pw).backward()
    g_ce = z_ce.grad.squeeze(1).abs().numpy()
    out["ce"] = float(g_ce[grp_np == 2].sum() / max(g_ce[neg].sum(), 1e-12))

    # PAUC pairwise gradient mass on decoys (queue_size=0 uses only current batch)
    z_pauc = torch.tensor(z_np, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    yl = torch.tensor((grp_np == 1).astype(np.int64))
    lf = PAUCAtBudgetLoss(
        num_classes=1,
        alpha=alpha,
        beta=beta,
        surrogate="pairwise",
        pos_numerator="pool",
        queue_size=0,
    )
    lf(z_pauc, yl).backward()
    g_pauc = z_pauc.grad.squeeze(1).abs().numpy()
    out["pauc"] = float(g_pauc[grp_np == 2].sum() / max(g_pauc[neg].sum(), 1e-12))

    return out


def repr_probe_auc(model: nn.Module, data: dict) -> float:
    """In-sample logistic-probe AUC separating positives from decoys on
    penultimate activations.

    The probe is fit and scored on the same rows (in-sample), so the absolute
    value is optimistically biased.  That bias is shared by CE and PAUC arms,
    so the CE-vs-PAUC comparison is unaffected; only the absolute number is
    inflated.

    This function is designed for nonlinear (Sequential) models that support
    ``model[:-1]`` to extract penultimate features (e.g. the mlp_2x64 used in
    experiments 01-06/08).  Passing a bare ``nn.Linear`` (capacity='linear')
    raises TypeError because indexing is not defined on non-Sequential modules;
    this is intentional — the function has no meaning for linear models because
    there is no hidden representation to probe.

    Parameters
    ----------
    model : nn.Sequential
        Must support model[:-1] to extract penultimate features.
    data : dict
        Must contain 'X_test' and 'grp_test', or 'X_train' and 'grp_train'.
        Raises KeyError if neither pair is present.

    Returns
    -------
    float AUC (in-sample).

    Raises
    ------
    TypeError
        If ``model`` is not an ``nn.Sequential`` (e.g. bare ``nn.Linear``).
    KeyError
        If no feature tensor (X_test / X_train) or group array (grp_test /
        grp_train) is found in ``data``.
    """
    if not isinstance(model, nn.Sequential):
        raise TypeError(
            f"repr_probe_auc requires an nn.Sequential model (got {type(model).__name__}). "
            "Probing penultimate activations is not defined for bare nn.Linear models."
        )

    # Prefer test data for the probe; fall back to train if test not available.
    # NOTE: use explicit None checks, not ``a or b`` — X/grp are tensors/arrays
    # and ``tensor or ...`` raises "Boolean value of Tensor is ambiguous".
    X = data.get("X_test")
    if X is None:
        X = data.get("X_train")
    grp = data.get("grp_test")
    if grp is None:
        grp = data.get("grp_train")

    if X is None:
        raise KeyError("repr_probe_auc: data must contain 'X_test' or 'X_train'")
    if grp is None:
        raise KeyError("repr_probe_auc: data must contain 'grp_test' or 'grp_train'")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        # Extract penultimate layer (output of last ReLU before final linear)
        h = model[:-1](X).numpy()
    model.train(was_training)

    mask = (grp == 1) | (grp == 2)
    lab = (grp[mask] == 1).astype(int)
    Xp = h[mask]

    clf = LogisticRegression(max_iter=200).fit(Xp, lab)
    return float(roc_auc_score(lab, clf.decision_function(Xp)))


def band_decoy_geometry(
    scores: np.ndarray,
    grp: np.ndarray,
    budget: float,
    band: tuple[float, float],
) -> dict[str, float]:
    """Decoy geometry inside/above the PAUC band vs the HNM top-2% of negatives.

    Used by an_band_escape (experiment 04).  All three quantities are DECOY
    RECALL fractions (denominator = total decoys).  Do NOT change this to
    precision; band_decoy_precision is the separate function for experiment 02.

    Parameters
    ----------
    scores : np.ndarray, shape [N]
        Model scores on the evaluation set.
    grp : np.ndarray, shape [N], dtype int64
        Group codes (0=easy-neg, 1=pos, 2=decoy, 3=confounder).
    budget : float
        FPR budget (used only for display / documentation; band overrides it).
    band : (alpha_mult, beta_mult)
        Band multipliers relative to budget.  The actual FPR band edges are
        (alpha_mult * budget, beta_mult * budget).  e.g. (0.5, 1.5) gives the
        default [budget/2, 1.5*budget] band.  (0.0, 1.0) gives the recommended
        wide band.

    Returns
    -------
    dict with keys:
        'in_band'    -- fraction of decoys whose score falls in [t_beta, t_alpha]
                        (decoy RECALL in-band; used by an_band_escape / exp 04)
        'in_top2'    -- fraction of decoys in HNM top-2% of negatives
        'above_band' -- fraction of decoys scoring above t_alpha (escape above band)
    """
    alpha_mult, beta_mult = band
    alpha = alpha_mult * budget
    beta = beta_mult * budget

    neg_mask = grp != 1
    sn = scores[neg_mask]
    sd = scores[grp == 2]

    if len(sn) == 0:
        return {"in_band": 0.0, "in_top2": 0.0, "above_band": 0.0}

    # t_alpha > t_beta (lower FPR = higher score threshold)
    t_alpha = np.quantile(sn, 1.0 - alpha) if alpha > 0.0 else sn.max()
    t_beta = np.quantile(sn, 1.0 - beta)
    t_top2 = np.quantile(sn, 1.0 - 0.02)

    if len(sd) == 0:
        return {"in_band": 0.0, "in_top2": 0.0, "above_band": 0.0}

    in_band = float(np.mean((sd >= t_beta) & (sd <= t_alpha)))
    in_top2 = float(np.mean(sd >= t_top2))
    above_band = float(np.mean(sd > t_alpha))

    return {"in_band": in_band, "in_top2": in_top2, "above_band": above_band}


def band_decoy_precision(
    scores: np.ndarray,
    grp: np.ndarray,
    budget: float,
    band: tuple[float, float],
) -> float:
    """Fraction of the PAUC band that is decoys (band precision).

    Used by an_mechanism (experiment 02) to reproduce the §3 claim: "The PAUC
    band is 73% decoys against a 1.2% base rate."  This matches
    experiments/02_mechanism_probe.py's ``band_decoy_fraction``:

        neg = grp != 1
        lo, hi = quantile(scores[neg], 1-beta), quantile(scores[neg], 1-alpha)
        band_mask = neg & (scores >= lo) & (scores <= hi)
        return mean(grp[band_mask] == 2)

    Denominator = all samples in the band (negatives whose score lies in
    [t_beta, t_alpha]).  Numerator = those that are decoys (grp == 2).

    This is DISTINCT from band_decoy_geometry's 'in_band', which is decoy
    RECALL (denominator = total decoys).  Both are computed from the same band
    threshold pair; only the denominator differs.

    Parameters
    ----------
    scores : np.ndarray, shape [N]
    grp : np.ndarray, shape [N], dtype int64
    budget : float
    band : (alpha_mult, beta_mult)

    Returns
    -------
    float in [0, 1]: fraction of band members that are decoys.
    """
    alpha_mult, beta_mult = band
    alpha = alpha_mult * budget
    beta = beta_mult * budget

    neg_mask = grp != 1
    sn = scores[neg_mask]

    if len(sn) == 0:
        return 0.0

    t_alpha = np.quantile(sn, 1.0 - alpha) if alpha > 0.0 else sn.max()
    t_beta = np.quantile(sn, 1.0 - beta)

    # Band members: negatives whose score falls in [t_beta, t_alpha]
    band_mask = neg_mask & (scores >= t_beta) & (scores <= t_alpha)
    if band_mask.sum() == 0:
        return 0.0
    return float((grp[band_mask] == 2).mean())


# ===========================================================================
# DAG helpers — imported only when running as a Metaflow flow
# ===========================================================================

import itertools as _itertools
import pathlib as _pathlib

_CONF_DIR = _pathlib.Path(__file__).parent.parent / "conf"


def _hydra_parser(text: str) -> dict:
    """Metaflow Config parser: Hydra compose to plain nested dict.

    The caller may embed a ``hydra_overrides`` list in the YAML text to select
    config groups without replacing the entire file:

        --config-value cfg "hydra_overrides: [experiment=cue_ablation,seeds=1]"

    The ``hydra_overrides`` key is stripped before compose so it never appears
    in the resolved config.
    """
    import yaml
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    raw = yaml.safe_load(text) or {}
    overrides = raw.pop("hydra_overrides", [])

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONF_DIR.resolve()), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=overrides)

    return OmegaConf.to_container(cfg, resolve=True)


def _load_arm_cfg(arm_name: str) -> dict:
    """Load an arm config YAML as a plain dict (no Hydra composition needed)."""
    import yaml

    path = _CONF_DIR / "arm" / f"{arm_name}.yaml"
    return yaml.safe_load(path.read_text())


def _expand_arm_configs(
    arm_name: str,
    arm_cfg: dict,
    cell: dict,
    arm_overrides: "dict | None" = None,
) -> list[dict]:
    """Expand sweep axes in arm_cfg into a list of concrete arm dicts.

    Sweep keys recognised:
        wd_sweep    list[float]  -> expands 'wd' axis
        temp_sweep  list[float]  -> expands 'temp' axis (PAUC/smoothap only)

    Non-sweep keys pass through unchanged.  The cell dict is NOT merged here;
    callers merge cell-level overrides (budget, band, capacity) into each
    concrete spec after expansion.

    arm_overrides : dict, optional
        Per-experiment override map keyed by arm name.  If the current
        arm_name appears here, the override dict is merged into arm_cfg
        BEFORE sweep expansion, allowing an experiment to collapse a sweep
        to a singleton or substitute specific sweep values without touching
        the arm YAML.

        Example (cue_ablation — no sweep, single config per arm):
            arm_overrides:
              weighted_ce:   {wd_sweep: [0.0]}
              pauc_pairwise: {wd_sweep: [0.0], temp_sweep: [0.5]}

        Example (capacity_warmup — custom temp sweep for pauc):
            arm_overrides:
              pauc_pairwise: {temp_sweep: [0.2, 0.4]}

        Any key in the override replaces the corresponding key in arm_cfg
        (including *_sweep keys), so a singleton list collapses the sweep
        to exactly one config.

    Returns
    -------
    list of dicts, each a self-contained arm_spec ready for train_arm().
    The dict has an extra '_arm_name' key for record provenance.
    """
    # Merge per-experiment arm overrides before expansion
    effective_cfg = dict(arm_cfg)
    if arm_overrides and arm_name in arm_overrides:
        effective_cfg.update(arm_overrides[arm_name])

    base = {k: v for k, v in effective_cfg.items() if not k.endswith("_sweep")}
    base["_arm_name"] = arm_name

    sweep_axes: dict[str, list] = {}
    if "wd_sweep" in effective_cfg:
        sweep_axes["wd"] = list(effective_cfg["wd_sweep"])
    if "temp_sweep" in effective_cfg:
        sweep_axes["temp"] = list(effective_cfg["temp_sweep"])

    if not sweep_axes:
        return [dict(base)]

    keys = list(sweep_axes.keys())
    combos = list(_itertools.product(*[sweep_axes[k] for k in keys]))
    specs = []
    for combo in combos:
        spec = dict(base)
        for k, v in zip(keys, combo):
            spec[k] = v
        specs.append(spec)
    return specs


def _build_data_dict(geometry_cfg: dict, cell: dict, seed: int) -> dict:
    """Build the data dict (train/val/test) from geometry config and cell overrides.

    Cell axes that modify geometry:
        cue            str    -> geometry_spec['cue']
        pos_rate       float  -> geometry_spec['pos_rate']
        decoy_frac     float  -> geometry_spec['decoy_frac']
        confounder_frac float -> geometry_spec['confounder_frac']

    Cell axes that only affect arms (budget, band, capacity) are NOT applied here.

    Split conventions
    -----------------
    The ``split_convention`` key in geometry_cfg selects the split strategy.

    ``sequential`` (default for experiments 01–06):
        Train and test are drawn from a SINGLE RNG advanced sequentially:
            rng = np.random.default_rng(seed_base + seed)
            train = make_data(train_spec, seed, rng)   # advances rng
            test  = make_data(test_spec,  seed, rng)   # continues stream

        This exactly reproduces the original 01_cue_ablation.py convention:
            rng = np.random.default_rng(7000+seed)
            make_data(N_TR, cue, rng)
            make_data(N_TE, cue, rng)

        No validation set is produced — ``X_val`` / ``y_val`` / ``grp_val``
        are absent (None).  Arms with ``early_stop=True`` must not be used
        under ``sequential`` (their val-based selection is meaningless).

    ``independent`` (experiments 07/08 — val-based model selection):
        Train, val, and test are drawn from independent RNGs with seed offsets:
            train: seed_base + seed
            val:   seed_base + seed + 1000
            test:  seed_base + seed + 2000

        This supports val-based model selection (early_stop, sweep selection).
    """
    gspec_base: dict = {
        "kind": geometry_cfg["kind"],
        "seed_base": geometry_cfg.get("seed_base", 7000),
    }
    # Geometry-level parameters (may be overridden by cell axes)
    for key in ("pos_rate", "decoy_frac", "confounder_frac"):
        if key in geometry_cfg:
            gspec_base[key] = geometry_cfg[key]
    # cue defaults to 'nonlinear_prod' for headline/confounder unless the cell
    # specifies it; hard_bulk ignores cue.
    if "cue" in geometry_cfg:
        gspec_base["cue"] = geometry_cfg["cue"]

    # Apply cell overrides for geometry axes
    for key in ("cue", "pos_rate", "decoy_frac", "confounder_frac"):
        if key in cell:
            gspec_base[key] = cell[key]

    # Capacity is an arm-level axis; also used by hard_bulk branches in exp 07
    # to select model architecture.  It does NOT affect data generation.

    seed_base = gspec_base["seed_base"]
    convention = geometry_cfg.get("split_convention", "sequential")

    n_train = geometry_cfg.get("n_train", 150_000)
    n_test = geometry_cfg.get("n_test", 300_000)

    if convention == "sequential":
        # Single RNG advanced train-then-test, matching 01_cue_ablation.py.
        # No val set produced.
        shared_rng = np.random.default_rng(seed_base + seed)

        train_spec = dict(gspec_base)
        train_spec["n"] = n_train
        X_tr, y_tr, grp_tr = make_data(train_spec, seed, rng=shared_rng)

        test_spec = dict(gspec_base)
        test_spec["n"] = n_test
        X_te, y_te, grp_te = make_data(test_spec, seed, rng=shared_rng)

        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "grp_train": grp_tr,
            "X_val": None,
            "y_val": None,
            "grp_val": None,
            "X_test": X_te,
            "y_test": y_te,
            "grp_test": grp_te,
            "_y_train_np": y_tr.squeeze(1).numpy(),
            "_y_val_np": None,
            "_y_test_np": y_te.squeeze(1).numpy(),
        }

    elif convention == "independent":
        # Independent seeds per split — supports val-based model selection (07/08).
        n_val = geometry_cfg.get("n_val", 50_000)

        def make_split(n: int, seed_offset: int) -> tuple:
            spec = dict(gspec_base)
            spec["n"] = n
            spec["seed_base"] = seed_base + seed_offset
            return make_data(spec, seed)

        X_tr, y_tr, grp_tr = make_split(n_train, 0)
        X_val, y_val, grp_val = make_split(n_val, 1000)
        X_te, y_te, grp_te = make_split(n_test, 2000)

        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "grp_train": grp_tr,
            "X_val": X_val,
            "y_val": y_val,
            "grp_val": grp_val,
            "X_test": X_te,
            "y_test": y_te,
            "grp_test": grp_te,
            "_y_train_np": y_tr.squeeze(1).numpy(),
            "_y_val_np": y_val.squeeze(1).numpy(),
            "_y_test_np": y_te.squeeze(1).numpy(),
        }

    else:
        raise ValueError(
            f"Unknown split_convention: {convention!r}. "
            "Expected 'sequential' or 'independent'."
        )


def _apply_cell_to_arm(arm_spec: dict, cell: dict, experiment_cfg: dict) -> dict:
    """Merge cell-level and experiment-level overrides into arm_spec.

    Axes applied here:
        budget    -> arm_spec['budget']
        capacity  -> arm_spec['capacity']
        band      (from cell or experiment_cfg) -> arm_spec['band'] for PAUC arms

    The 'band' priority is: cell['band'] > experiment_cfg['band'] > arm_spec['band'].
    """
    spec = dict(arm_spec)

    if "budget" in cell:
        spec["budget"] = cell["budget"]

    if "capacity" in cell:
        spec["capacity"] = cell["capacity"]

    # Band resolution (only meaningful for PAUC/smoothap arms)
    if "band" in cell:
        spec["band"] = list(cell["band"])
    elif "band" in experiment_cfg:
        spec["band"] = list(experiment_cfg["band"])
    # else: arm's own default remains

    return spec


def _resolve_training_cfg(cfg: dict, cell: "dict | None" = None) -> dict:
    """Extract the resolved training config block from the top-level config.

    The training block at ``cfg['training']`` is authoritative.  When
    ``cfg['training']['epochs']`` is a dict (capacity experiment), the
    caller must resolve the per-capacity epoch count using:
        epochs_val = training_cfg['epochs'][capacity]

    Parameters
    ----------
    cfg : dict
        Top-level Hydra-resolved config dict from ``self.cfg``.
    cell : dict, optional
        Current cell dict.  Not used for resolution here; passed through
        for callers that need it alongside the training block.

    Returns
    -------
    dict
        A copy of ``cfg['training']``, guaranteed to contain at minimum:
        hid, batch, lr, queue, warmup_frac, blend_frac,
        temp_start, temp_end, eval_every.  The 'epochs' key may be an
        int (default) or a dict keyed by capacity (capacity experiment).
    """
    training = cfg.get("training", {})
    return dict(training)


def _merge_training_into_arm(arm_spec: dict, training_cfg: dict, cell: "dict | None" = None) -> dict:
    """Merge training config defaults into arm_spec without overriding arm values.

    The arm YAML is authoritative for keys it explicitly sets; training_cfg
    provides fallback defaults for keys absent from the arm YAML.

    For capacity experiments where training_cfg['epochs'] is a dict, the
    per-capacity epoch count is resolved here from cell['capacity'] and
    merged as a plain int.

    Parameters
    ----------
    arm_spec : dict
        Concrete arm spec (post sweep-expansion and cell-override).
    training_cfg : dict
        Resolved training block from ``_resolve_training_cfg``.
    cell : dict, optional
        Current cell dict.  Used to look up capacity for epoch resolution.

    Returns
    -------
    dict
        arm_spec with training defaults filled in for missing keys.
    """
    spec = dict(arm_spec)

    # Map training key -> arm_spec key (training uses shorter names)
    key_map = {
        "hid": "hid",
        "batch": "batch_size",
        "lr": "lr",
        "queue": "queue_size",
        "warmup_frac": "warmup_frac",
        "blend_frac": "blend_frac",
        "temp_start": "temp",
        "temp_end": "temp_end",
        "eval_every": "eval_every",
    }

    for tr_key, arm_key in key_map.items():
        if tr_key in training_cfg and arm_key not in spec:
            spec[arm_key] = training_cfg[tr_key]

    # Epoch resolution: training may carry a per-capacity dict
    if "epochs" not in spec and "epochs" in training_cfg:
        epochs_val = training_cfg["epochs"]
        if isinstance(epochs_val, dict):
            # Capacity experiment: look up by the arm's capacity value
            capacity = spec.get("capacity", (cell or {}).get("capacity", "mlp_2x64"))
            spec["epochs"] = epochs_val[capacity]
        else:
            spec["epochs"] = epochs_val

    return spec


def _run_diagnostics(
    diag_names: list[str],
    model,
    data: dict,
    arm_spec: dict,
    cell: dict,
) -> dict:
    """Dispatch named diagnostics and return a flat dict of results.

    Supported diagnostic names:
        'metric_specificity'  -- coverage and auroc on test set (for trend analysis)
        'grad_mass'           -- grad_mass_on_decoys on training set
        'repr_probe'          -- repr_probe_auc on test set (Sequential models only)
        'band_decoy_geometry' -- band_decoy_geometry on test set scores

    Unknown names are logged as a warning and skipped (not raised), so stubs
    for future experiments don't break the current DAG.
    """
    result: dict = {}
    budget = arm_spec.get("budget", 0.005)
    y_test_np = data["_y_test_np"]
    test_scores = scores_of(model, data["X_test"])

    for name in diag_names:
        if name == "metric_specificity":
            result["test_coverage"] = coverage(y_test_np, test_scores, budget)
            result["test_auroc"] = auroc(y_test_np, test_scores)
            result["test_aucpr"] = aucpr(y_test_np, test_scores)

        elif name == "grad_mass":
            gd = data.copy()
            gd["budget"] = budget
            try:
                gm = grad_mass_on_decoys(model, gd)
                result["grad_mass_ce"] = gm["ce"]
                result["grad_mass_pauc"] = gm["pauc"]
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                # Model may lack grp_train decoys (KeyError), have no positives
                # (ValueError from sklearn), or encounter a runtime issue.
                result["grad_mass_ce"] = None
                result["grad_mass_pauc"] = None
                result["grad_mass_error"] = f"{type(exc).__name__}: {exc}"

        elif name == "repr_probe":
            try:
                result["repr_probe_auc"] = repr_probe_auc(model, data)
            except (TypeError, KeyError, ValueError, RuntimeError) as exc:
                # TypeError: linear model (no penultimate layer — not applicable).
                # KeyError: data missing X_test/X_train or grp arrays.
                # ValueError/RuntimeError: sklearn or torch failure.
                result["repr_probe_auc"] = None
                result["repr_probe_error"] = f"{type(exc).__name__}: {exc}"

        elif name == "band_decoy_geometry":
            band = arm_spec.get("band", [0.5, 1.5])
            grp_te = data.get("grp_test")
            if grp_te is not None:
                # Recall quantities (in_band / in_top2 / above_band): denominator is
                # total decoys.  Used by an_band_escape (exp 04).
                bg = band_decoy_geometry(test_scores, grp_te, budget, tuple(band))
                result.update({f"bdg_{k}": v for k, v in bg.items()})
                # Precision quantity: fraction of the band that IS decoys.
                # Used by an_mechanism (exp 02) to reproduce the §3 "73% decoys"
                # claim.  Stored separately because recall and precision differ and
                # must not be conflated.
                result["bdg_decoyfrac"] = band_decoy_precision(
                    test_scores, grp_te, budget, tuple(band)
                )
        else:
            # Unknown diagnostic: skip; will surface in the record as a missing key
            print(f"[diagnostics] unknown diagnostic {name!r}, skipping", flush=True)

    return result


def _select_by_val_pure(recs: list) -> dict:
    """Select the best record from a (cell, arm, seed) group by val_coverage.

    This is the pure selection logic extracted from the select_by_val step so
    it can be unit-tested without Metaflow.

    Rules:
      - If val_coverage is None (sequential split, no val set): keep recs[0].
        If len(recs) > 1 in this situation, emit a UserWarning — this means a
        sweep was paired with a split convention that has no val set, which
        causes silent record loss.
      - If len(recs) == 1: return recs[0] regardless of val_coverage.
      - Otherwise: return the record with the highest val_coverage.

    Parameters
    ----------
    recs : list[dict]
        Non-empty list of records for the same (cell, arm, seed) group.

    Returns
    -------
    dict  The selected record.
    """
    import warnings as _warnings

    if recs[0]["val_coverage"] is None:
        if len(recs) > 1:
            _warnings.warn(
                f"select_by_val: a group of {len(recs)} records has val_coverage=None "
                "(sequential split); keeping only recs[0].  This is unexpected for a "
                "sweep with val selection — check that the split_convention and "
                "early_stop settings are consistent.",
                stacklevel=2,
            )
        return recs[0]
    if len(recs) == 1:
        return recs[0]
    return max(recs, key=lambda r: r["val_coverage"])


def _cell_product(axes: dict) -> list[dict]:
    """Return the Cartesian product of axis values as a list of cell dicts."""
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in _itertools.product(*values)]


def _task_key(experiment: str, cell: dict, seed: int) -> str:
    """Stable string key for a (experiment, cell, seed) task — used for logging."""
    cell_str = ",".join(f"{k}={v}" for k, v in sorted(cell.items()))
    return f"{experiment}|{cell_str}|seed={seed}"


# ===========================================================================
# Dataset-keyed foreach expansion
# ===========================================================================
#
# Most experiment axes do not change the dataset; only a declared subset
# (data_axes) does.  The flow generates each dataset ONCE and trains every model
# that uses it on the shared in-memory tensors.  These pure helpers build the
# dataset-key list (the foreach grain) and the per-dataset training-combo lists.

# Budget-agnostic arm kinds: the trained model does NOT depend on the FPR budget,
# so the arm trains ONCE and is evaluated at every budget in its combo list.
#   trivial         — feature mean, no budget
#   ce              — weighted CE / HNM / oracle; loss has no budget term
#   ce_warmup_only  — CE for a warmup fraction; no budget term
#   smoothap        — SmoothAP ranking loss; budget-free
# Budget-dependent arm kinds: the loss band is a function of budget (alpha/beta
# = f(budget)), so the model must be retrained per budget.
#   pauc, pauc_cold — PAUCAtBudgetLoss with alpha=alpha_mult*budget, beta=...
_BUDGET_AGNOSTIC_KINDS = frozenset({"trivial", "ce", "ce_warmup_only", "smoothap"})
_BUDGET_DEPENDENT_KINDS = frozenset({"pauc", "pauc_cold"})


def is_budget_agnostic_arm(arm_kind: str) -> bool:
    """Return True if an arm of this kind trains independently of the budget.

    Budget-agnostic arms (trivial / ce / ce_warmup_only / smoothap) train once
    and are evaluated at every budget.  Budget-dependent arms (pauc / pauc_cold)
    bake the budget into the loss band, so they retrain per budget.

    Raises ValueError for unknown kinds so a new arm cannot be silently
    misclassified as budget-agnostic.
    """
    if arm_kind in _BUDGET_AGNOSTIC_KINDS:
        return True
    if arm_kind in _BUDGET_DEPENDENT_KINDS:
        return False
    raise ValueError(
        f"Unknown arm kind {arm_kind!r}: cannot classify budget dependence. "
        f"Add it to _BUDGET_AGNOSTIC_KINDS or _BUDGET_DEPENDENT_KINDS."
    )


def _data_cell_of(cell: dict, data_axes: list) -> dict:
    """Project a full cell onto its data-axis subset (the part that varies data)."""
    return {k: cell[k] for k in data_axes if k in cell}


def _data_cell_key(data_cell: dict) -> tuple:
    """Hashable, order-independent key for a data cell.

    Values that are lists (e.g. a band [alpha, beta]) are converted to tuples so
    the key is hashable.
    """
    def _h(v):
        return tuple(v) if isinstance(v, list) else v

    return tuple(sorted((k, _h(v)) for k, v in data_cell.items()))


def build_dataset_keys(
    experiment: str,
    exp_cfg: dict,
    seeds: int,
    arm_cfg_loader: "callable | None" = None,
) -> list[dict]:
    """Expand an experiment into dataset keys, each carrying its training combos.

    A dataset key = (experiment, data-cell, seed): the data is generated ONCE per
    key.  Every training combo attached to the key is trained on that shared
    in-memory dataset.

    A training combo is one (arm_name, concrete arm_spec, budgets) unit:
      * Budget-agnostic arms (ce / trivial / smoothap / ce_warmup_only) appear
        ONCE per distinct (arm, non-budget training-cell, sweep config) with a
        ``budgets`` list of EVERY budget that training-cell uses.  The arm trains
        once and coverage@budget is recorded for each budget.
      * Budget-dependent arms (pauc / pauc_cold) appear once per
        (arm, training-cell, sweep config, budget) with a single-element
        ``budgets`` list — the band is f(budget), so each budget is a separate
        training.

    The returned list is the foreach grain for ``train``.  Its length is the
    DATASET count (distinct data-cells x seeds), NOT the combo count.

    Each dataset-key dict has keys:
        experiment   str
        data_cell    dict   (data-axis subset of the cell)
        seed         int
        combos       list[dict]   training combos (see below)

    Each combo dict has keys:
        arm           str    arm config-group name
        arm_spec      dict   concrete arm_spec (pre training-cfg merge)
        budgets       list[float]   budgets to evaluate this model at
        budget_dep    bool   whether the arm retrains per budget
        train_cell    dict   the (non-data) training-cell this combo came from

    arm_cfg_loader : callable, optional
        ``arm_name -> arm_cfg dict``.  Defaults to ``_load_arm_cfg`` (reads the
        arm YAML).  Injecting a loader lets the contract tests exercise this
        function without pyyaml on the path.
    """
    if arm_cfg_loader is None:
        arm_cfg_loader = _load_arm_cfg

    axes = exp_cfg.get("axes", {})
    data_axes = list(exp_cfg.get("data_axes", []))
    arm_names = exp_cfg.get("arms", [])
    arm_overrides = exp_cfg.get("arm_overrides", {})

    cells = _cell_product(axes)

    # Group full cells by their data-cell projection.
    by_data: dict = {}
    for cell in cells:
        dkey = _data_cell_key(_data_cell_of(cell, data_axes))
        by_data.setdefault(dkey, []).append(cell)

    dataset_keys: list[dict] = []
    for dkey, group_cells in by_data.items():
        data_cell = _data_cell_of(group_cells[0], data_axes)
        for seed in range(seeds):
            combos = _build_combos_for_dataset(
                group_cells, arm_names, arm_overrides, exp_cfg, arm_cfg_loader
            )
            dataset_keys.append({
                "experiment": experiment,
                "data_cell": data_cell,
                "seed": seed,
                "combos": combos,
            })
    return dataset_keys


def _build_combos_for_dataset(
    group_cells: list,
    arm_names: list,
    arm_overrides: dict,
    exp_cfg: dict,
    arm_cfg_loader: "callable | None" = None,
) -> list[dict]:
    """Build the training-combo list for one dataset (all cells sharing data).

    Budget-agnostic arms are de-duplicated across budgets: a single combo per
    (arm, non-budget training-cell, sweep config) collects every budget into its
    ``budgets`` list.  Budget-dependent arms get one combo per budget.
    """
    # Accumulator for budget-agnostic combos: key -> combo dict (mutated to add
    # budgets).  Key excludes budget so the same model serves all budgets.
    if arm_cfg_loader is None:
        arm_cfg_loader = _load_arm_cfg

    agnostic: dict = {}
    dependent: list[dict] = []

    for arm_name in arm_names:
        arm_cfg_raw = arm_cfg_loader(arm_name)
        for cell in group_cells:
            budget = cell.get("budget", 0.005)
            concrete_specs = _expand_arm_configs(
                arm_name, arm_cfg_raw, cell, arm_overrides=arm_overrides
            )
            for spec in concrete_specs:
                spec = _apply_cell_to_arm(spec, cell, exp_cfg)
                kind = spec["kind"]
                agn = is_budget_agnostic_arm(kind)
                # Non-budget training-cell: cell minus budget (band stays; CE
                # ignores band but pauc-dependent handled separately).
                tc = {k: v for k, v in cell.items() if k != "budget"}
                if agn:
                    # Budget-agnostic arms (ce / trivial / smoothap) ignore both
                    # budget AND band, so strip both: the same model serves every
                    # (budget, band) combination — train once, eval at each budget.
                    agn_spec = {k: v for k, v in spec.items()
                                if k not in ("budget", "band")}
                    agn_tc = {k: v for k, v in tc.items() if k != "band"}
                    sig = _combo_signature(arm_name, agn_spec, exclude_budget=True)
                    if sig not in agnostic:
                        agnostic[sig] = {
                            "arm": arm_name,
                            "arm_spec": agn_spec,
                            "budgets": [],
                            "budget_dep": False,
                            "train_cell": agn_tc,
                        }
                    if budget not in agnostic[sig]["budgets"]:
                        agnostic[sig]["budgets"].append(budget)
                else:
                    dependent.append({
                        "arm": arm_name,
                        "arm_spec": spec,
                        "budgets": [budget],
                        "budget_dep": True,
                        "train_cell": tc,
                    })

    combos = list(agnostic.values()) + dependent
    for c in combos:
        c["budgets"] = sorted(set(c["budgets"]))
    return combos


def _combo_signature(arm_name: str, spec: dict, exclude_budget: bool) -> tuple:
    """Stable hashable signature for de-duplicating budget-agnostic combos.

    Two budget-agnostic specs that differ ONLY in budget collapse to the same
    signature, so the arm trains once and is evaluated at every budget.
    """
    def _h(v):
        if isinstance(v, list):
            return tuple(_h(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((k, _h(x)) for k, x in v.items()))
        return v

    items = sorted(
        (k, _h(v)) for k, v in spec.items()
        if not (exclude_budget and k == "budget")
    )
    return (arm_name,) + tuple(items)


# ===========================================================================
# Static-branch analysis helpers (pure; consumers of train records)
# ===========================================================================
#
# Each analysis step filters the aggregated records for ITS experiment and
# computes a result.  These filters are pure so they can be unit-tested without
# metaflow.  Mapping experiment-config name -> analysis:
#   cue_ablation        -> an_cue_specificity
#   mechanism_probe     -> an_mechanism
#   mechanism_transfer  -> an_transfer
#   band_vs_hnm         -> an_band_escape
#   alpha_widen         -> an_alpha_lever
#   confounder_sweep    -> an_surrogate
#   capacity_warmup     -> an_capacity
#   band_default_sweep  -> an_default_sweep

def filter_records(records: list, experiment: str) -> list:
    """Return the subset of records produced by the named experiment.

    Records without an 'experiment' tag are ignored.  An empty result means the
    experiment was not enabled — analysis steps pass through cleanly.
    """
    return [r for r in records if r.get("experiment") == experiment]


def _cell_of(rec: dict) -> dict:
    """Reconstruct a record's full cell (data_cell + train_cell + budget)."""
    cell = dict(rec.get("data_cell", {}))
    cell.update(rec.get("train_cell", {}))
    if "budget" in rec:
        cell["budget"] = rec["budget"]
    return cell


def _seed_paired_lift(records: list, arm_a: str, arm_b: str, cell_keys: list) -> dict:
    """Per-cell seed-paired coverage lift (arm_a - arm_b) with bootstrap CI.

    Pairs by seed (robust to foreach delivery order).  ``cell_keys`` selects the
    cell fields that define a cell (e.g. ['cue', 'budget']).  Returns a dict keyed
    by the frozen cell tuple -> {cell, lift_mean, lift_lo, lift_hi,
    ci_excludes_zero, n_seeds}.
    """
    by_key: dict = {}
    for r in records:
        cell = _cell_of(r)
        ck = tuple((k, cell.get(k)) for k in cell_keys)
        by_key.setdefault(ck, {}).setdefault(r["arm"], {})[r["seed"]] = (
            r["test"]["coverage"]
        )

    out: dict = {}
    for ck, arms in by_key.items():
        a = arms.get(arm_a, {})
        b = arms.get(arm_b, {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        lifts = np.array([a[s] - b[s] for s in shared])
        m, lo, hi = bootstrap_ci(lifts, n_resamples=1000, seed=0)
        out[ck] = {
            "cell": dict(ck),
            "lift_mean": m,
            "lift_lo": lo,
            "lift_hi": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_seeds": len(shared),
        }
    return out


def _mean_metric_by_cell(records: list, arm: str, cell_keys: list, metric: str) -> dict:
    """Mean of test[metric] for one arm, grouped by cell.  Returns ck -> mean."""
    acc: dict = {}
    for r in records:
        if r["arm"] != arm:
            continue
        cell = _cell_of(r)
        ck = tuple((k, cell.get(k)) for k in cell_keys)
        acc.setdefault(ck, []).append(r["test"][metric])
    return {ck: float(np.mean(v)) for ck, v in acc.items()}


def analyze_cue_specificity(records: list) -> list:
    """cue_ablation: per (cue, budget) coverage lift (PAUC - CE) vs AUROC lift.

    Reproduces 01's metric-specificity result: coverage lift is specific to
    nonlinear cues while AUROC lift is flat.
    """
    recs = filter_records(records, "cue_ablation")
    if not recs:
        return []
    keys = ["cue", "budget"]
    lifts = _seed_paired_lift(recs, "pauc_pairwise", "weighted_ce", keys)
    auroc_pauc = _mean_metric_by_cell(recs, "pauc_pairwise", keys, "auroc")
    auroc_ce = _mean_metric_by_cell(recs, "weighted_ce", keys, "auroc")

    rows = []
    for ck, lift in lifts.items():
        cell = lift["cell"]
        auroc_lift = None
        if ck in auroc_pauc and ck in auroc_ce:
            auroc_lift = auroc_pauc[ck] - auroc_ce[ck]
        rows.append({
            "cue": cell.get("cue"),
            "budget": cell.get("budget"),
            "coverage_lift_mean": lift["lift_mean"],
            "coverage_lift_lo": lift["lift_lo"],
            "coverage_lift_hi": lift["lift_hi"],
            "ci_excludes_zero": lift["ci_excludes_zero"],
            "auroc_lift": auroc_lift,
        })
    rows.sort(key=lambda r: (str(r["cue"]), r["budget"] if r["budget"] is not None else 0))
    return rows


def analyze_mechanism(records: list) -> dict:
    """mechanism_probe: gradient allocation, repr probe, gap-closure (reproduces 02).

    Computes (mean over seeds, on the nonlinear-product cell when present):
      grad_mass_ce / grad_mass_pauc   — neg-gradient mass on decoys (from records'
                                        grad_mass diagnostic)
      probe_ce / probe_pauc           — repr probe pos-vs-decoy AUC
      band_decoyfrac_pauc             — PAUC band decoy-fraction
      gap_vs_pauc                     — coverage gap (condition - PAUC) for the
                                        oracle / HNM concentrated-CE arms
    The grad_mass and repr_probe values come from each record's diagnostics block
    (recomputed in train from the stored model).  This function only AGGREGATES.
    """
    recs = filter_records(records, "mechanism_probe")
    if not recs:
        return {}

    def _diag_mean(arm: str, key: str):
        vals = [r["diagnostics"][key] for r in recs
                if r["arm"] == arm and r["diagnostics"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _cov_by_arm(arm: str):
        return [r["test"]["coverage"] for r in recs if r["arm"] == arm]

    # grad_mass is the same diagnostic dict on any arm's record; read from CE.
    grad_ce = _diag_mean("weighted_ce", "grad_mass_ce")
    grad_pauc = _diag_mean("weighted_ce", "grad_mass_pauc")
    probe_ce = _diag_mean("weighted_ce", "repr_probe_auc")
    probe_pauc = _diag_mean("pauc_pairwise", "repr_probe_auc")
    # Band precision: fraction of the PAUC band that IS decoys (denominator = band
    # members, not total decoys).  This reproduces exp 02's band_decoy_fraction and
    # the §3 claim "The PAUC band is 73% decoys against a 1.2% base rate."
    # Stored in diagnostics as 'bdg_decoyfrac' by _run_diagnostics.
    # Do NOT use 'bdg_in_band' here — that is decoy RECALL (used by an_band_escape /
    # exp 04) and is a different quantity on the same band.
    band_decoyfrac = _diag_mean("pauc_pairwise", "bdg_decoyfrac")

    pauc_cov = _cov_by_arm("pauc_pairwise")
    gap = {}
    if pauc_cov:
        pauc_mean = float(np.mean(pauc_cov))
        for arm in ("weighted_ce", "ce_oracle", "ce_hnm"):
            cov = _cov_by_arm(arm)
            if cov:
                gap[arm] = float(np.mean(cov)) - pauc_mean

    return {
        "grad_mass_ce": grad_ce,
        "grad_mass_pauc": grad_pauc,
        "probe_auc_ce": probe_ce,
        "probe_auc_pauc": probe_pauc,
        "band_decoyfrac_pauc": band_decoyfrac,
        "coverage": {
            arm: float(np.mean(_cov_by_arm(arm)))
            for arm in ("weighted_ce", "ce_oracle", "ce_hnm", "pauc_pairwise")
            if _cov_by_arm(arm)
        },
        "gap_vs_pauc": gap,
    }


def analyze_transfer(records: list) -> list:
    """mechanism_transfer: gap-closure across cue x budget (reproduces 03).

    Per (cue, budget) cell, reports the mean coverage of each condition and the
    HNM-vs-PAUC and oracle-vs-PAUC gaps.  best_vs_pauc is the best
    concentrated-CE arm minus PAUC (>0 => concentrated-CE beats PAUC).
    """
    recs = filter_records(records, "mechanism_transfer")
    if not recs:
        return []
    keys = ["cue", "budget"]
    conds = ["weighted_ce", "ce_oracle", "ce_hnm", "pauc_pairwise"]
    cov = {c: _mean_metric_by_cell(recs, c, keys, "coverage") for c in conds}

    all_cks = set()
    for c in conds:
        all_cks.update(cov[c])

    rows = []
    for ck in sorted(all_cks):
        cell = dict(ck)
        pauc = cov["pauc_pairwise"].get(ck)
        row = {"cue": cell.get("cue"), "budget": cell.get("budget")}
        for c in conds:
            row[c] = cov[c].get(ck)
        if pauc is not None:
            concentrated = {c: cov[c].get(ck) for c in ("ce_oracle", "ce_hnm")
                            if cov[c].get(ck) is not None}
            if concentrated:
                best = max(concentrated, key=concentrated.get)
                row["best_concentrated"] = best
                row["best_vs_pauc"] = concentrated[best] - pauc
                row["hnm_overshoots_pauc"] = bool(concentrated[best] - pauc > 0)
        rows.append(row)
    return rows


def analyze_band_escape(records: list) -> list:
    """band_vs_hnm: decoy geometry from stored CE scores (reproduces 04).

    Per (cue, budget), recomputes the in-band / in-top2 / above-band decoy
    fractions from the STORED CE test scores+grp, averaged over seeds with
    bootstrap CIs.  Uses band_decoy_geometry as the consumer.
    """
    recs = filter_records(records, "band_vs_hnm")
    if not recs:
        return []
    keys = ["cue", "budget"]
    by_cell: dict = {}
    for r in recs:
        if r["arm"] != "weighted_ce":
            continue
        if r.get("scores") is None or r.get("grp") is None:
            continue
        cell = _cell_of(r)
        ck = tuple((k, cell.get(k)) for k in keys)
        budget = cell.get("budget", 0.005)
        band = tuple(r["config"].get("band", [0.5, 1.5]))
        bg = band_decoy_geometry(np.asarray(r["scores"]), np.asarray(r["grp"]),
                                 budget, band)
        d = by_cell.setdefault(ck, {"in_band": [], "in_top2": [], "above_band": []})
        for k in ("in_band", "in_top2", "above_band"):
            d[k].append(bg[k])

    rows = []
    for ck, d in sorted(by_cell.items()):
        cell = dict(ck)
        row = {"cue": cell.get("cue"), "budget": cell.get("budget")}
        for k in ("in_band", "in_top2", "above_band"):
            m, lo, hi = bootstrap_ci(np.array(d[k]), n_resamples=1000, seed=0)
            row[k] = {"mean": m, "lo": lo, "hi": hi}
        rows.append(row)
    return rows


def analyze_alpha_lever(records: list) -> list:
    """alpha_widen: PAUC band-widening improvement per cell (reproduces 05).

    Per (cue, budget), compares pauc_wide (band [0.0, *]) to pauc_std (band
    [0.5, *]) and reports the seed-paired wide-minus-std coverage lift, plus each
    PAUC band's coverage vs CE-HNM.
    """
    recs = filter_records(records, "alpha_widen")
    if not recs:
        return []

    # Partition pauc records by band into std / wide.
    def _band_label(r):
        band = r["config"].get("band")
        if band is None:
            return None
        alpha_mult = band[0]
        return "wide" if alpha_mult == 0.0 else "std"

    keys = ["cue", "budget"]
    # Build per (cell, label, seed) coverage maps for pauc; and CE/HNM means.
    pauc_cov: dict = {}
    for r in recs:
        if r["arm"] != "pauc_pairwise":
            continue
        label = _band_label(r)
        if label is None:
            continue
        cell = _cell_of(r)
        ck = tuple((k, cell.get(k)) for k in keys)
        pauc_cov.setdefault(ck, {}).setdefault(label, {})[r["seed"]] = (
            r["test"]["coverage"]
        )

    hnm = _mean_metric_by_cell(recs, "ce_hnm", keys, "coverage")
    ce = _mean_metric_by_cell(recs, "weighted_ce", keys, "coverage")

    rows = []
    for ck, labels in sorted(pauc_cov.items()):
        cell = dict(ck)
        std = labels.get("std", {})
        wide = labels.get("wide", {})
        row = {
            "cue": cell.get("cue"),
            "budget": cell.get("budget"),
            "cov_pauc_std": float(np.mean(list(std.values()))) if std else None,
            "cov_pauc_wide": float(np.mean(list(wide.values()))) if wide else None,
            "cov_hnm": hnm.get(ck),
            "cov_ce": ce.get(ck),
        }
        shared = sorted(set(std) & set(wide))
        if shared:
            lifts = np.array([wide[s] - std[s] for s in shared])
            m, lo, hi = bootstrap_ci(lifts, n_resamples=1000, seed=0)
            row["wide_minus_std"] = {"mean": m, "lo": lo, "hi": hi,
                                     "ci_excludes_zero": bool(lo > 0 or hi < 0)}
        rows.append(row)
    return rows


def analyze_surrogate(records: list) -> list:
    """confounder_sweep: coverage by condition across the sweep (reproduces 08).

    Per (confounder_frac, pos_rate), reports mean coverage / auroc / aucpr for
    each condition (trivial / CE / trapezoid / smoothap / pairwise) and the
    seed-paired PAUC-minus-CE coverage lift.
    """
    recs = filter_records(records, "confounder_sweep")
    if not recs:
        return []
    keys = ["confounder_frac", "pos_rate"]
    conds = ["trivial", "weighted_ce", "pauc_pairwise", "pauc_trapezoid", "smoothap"]

    cov = {c: _mean_metric_by_cell(recs, c, keys, "coverage") for c in conds}
    auroc_m = {c: _mean_metric_by_cell(recs, c, keys, "auroc") for c in conds}
    aucpr_m = {c: _mean_metric_by_cell(recs, c, keys, "aucpr") for c in conds}
    lift = _seed_paired_lift(recs, "pauc_pairwise", "weighted_ce", keys)

    all_cks = set()
    for c in conds:
        all_cks.update(cov[c])

    rows = []
    for ck in sorted(all_cks):
        cell = dict(ck)
        row = {"confounder_frac": cell.get("confounder_frac"),
               "pos_rate": cell.get("pos_rate"), "conditions": {}}
        for c in conds:
            if ck in cov[c]:
                row["conditions"][c] = {
                    "coverage": cov[c][ck],
                    "auroc": auroc_m[c].get(ck),
                    "aucpr": aucpr_m[c].get(ck),
                }
        if ck in lift:
            row["pauc_minus_ce"] = {
                "coverage_lift": lift[ck]["lift_mean"],
                "lo": lift[ck]["lift_lo"],
                "hi": lift[ck]["lift_hi"],
                "ci_excludes_zero": lift[ck]["ci_excludes_zero"],
            }
        rows.append(row)
    return rows


def analyze_capacity(records: list) -> list:
    """capacity_warmup: warmup-vs-cold and capacity readings (reproduces 07).

    Per (pos_rate, decoy_frac, capacity), reports mean test coverage per
    condition and the seed-paired PAUC-minus-CE lift.
    """
    recs = filter_records(records, "capacity_warmup")
    if not recs:
        return []
    keys = ["pos_rate", "decoy_frac", "capacity"]
    conds = ["trivial", "weighted_ce", "pauc_pairwise"]
    cov = {c: _mean_metric_by_cell(recs, c, keys, "coverage") for c in conds}
    lift = _seed_paired_lift(recs, "pauc_pairwise", "weighted_ce", keys)

    all_cks = set()
    for c in conds:
        all_cks.update(cov[c])

    rows = []
    for ck in sorted(all_cks, key=lambda t: tuple(str(x) for x in t)):
        cell = dict(ck)
        row = {
            "pos_rate": cell.get("pos_rate"),
            "decoy_frac": cell.get("decoy_frac"),
            "capacity": cell.get("capacity"),
            "conditions": {c: cov[c].get(ck) for c in conds if ck in cov[c]},
        }
        if ck in lift:
            row["pauc_minus_ce"] = {
                "coverage_lift": lift[ck]["lift_mean"],
                "lo": lift[ck]["lift_lo"],
                "hi": lift[ck]["lift_hi"],
                "ci_excludes_zero": lift[ck]["ci_excludes_zero"],
            }
        rows.append(row)
    return rows


def analyze_default_sweep(records: list) -> dict:
    """band_default_sweep: best (alpha,beta) band per pos_rate (reproduces 06).

    For each pos_rate, builds the coverage grid over the swept bands and finds
    the best band; also finds the band most robust (highest mean coverage) across
    pos_rates.  Reference: CE-HNM coverage per pos_rate.
    """
    recs = filter_records(records, "band_default_sweep")
    if not recs:
        return {}

    # PAUC coverage by (pos_rate, band) averaged over seeds.
    by_pr_band: dict = {}
    for r in recs:
        if r["arm"] != "pauc_pairwise":
            continue
        cell = _cell_of(r)
        pr = cell.get("pos_rate")
        band = tuple(r["config"].get("band", [0.5, 1.5]))
        by_pr_band.setdefault(pr, {}).setdefault(band, []).append(
            r["test"]["coverage"]
        )

    hnm = _mean_metric_by_cell(recs, "ce_hnm", ["pos_rate"], "coverage")

    per_pos = []
    # band -> list of per-pos means (for robustness).
    band_means: dict = {}
    for pr in sorted(by_pr_band):
        grid = {band: float(np.mean(v)) for band, v in by_pr_band[pr].items()}
        best_band = max(grid, key=grid.get)
        per_pos.append({
            "pos_rate": pr,
            "cov_hnm": hnm.get((("pos_rate", pr),)),
            "grid": {f"a{b[0]}_b{b[1]}": v for b, v in grid.items()},
            "best": {"alpha_m": best_band[0], "beta_m": best_band[1],
                     "cov": grid[best_band]},
        })
        for band, v in grid.items():
            band_means.setdefault(band, []).append(v)

    robust = None
    if band_means:
        rb = max(band_means, key=lambda b: float(np.mean(band_means[b])))
        robust = {"alpha_m": rb[0], "beta_m": rb[1],
                  "mean_cov": float(np.mean(band_means[rb])),
                  "per_pos": [float(v) for v in band_means[rb]]}

    return {"per_pos": per_pos, "robust_band": robust}


# ===========================================================================
# Metaflow FlowSpec
# ===========================================================================
#
# Guard: only define the FlowSpec (and its @card-decorated steps) when metaflow
# is importable.  The contract tests import this module WITHOUT metaflow, so the
# whole class — including the @card decorator and metaflow.cards imports — must
# live inside this guard.  All orchestration imports (metaflow, hydra, matplotlib)
# stay lazy or guarded so ``import pauc_flow`` pulls only torch / numpy / sklearn.

try:
    from metaflow import FlowSpec, step, card, Config
    from metaflow.cards import Image, Markdown
    _METAFLOW_AVAILABLE = True
except ImportError:
    _METAFLOW_AVAILABLE = False


def _infer_experiment_name_fallback(exp_cfg: dict) -> str:
    """Best-effort experiment name from diagnostics/arms when 'name' is absent.

    This is the fallback used by start() when the experiment YAML does not
    carry an explicit 'name' key.  All standard experiment YAMLs now include
    'name', so this path is only exercised by ad-hoc / override-only runs.

    Hydra does not inject the config-group name into the loaded dict, so we
    derive a stable signature from the arm+axis contents.  In practice each
    run selects a single experiment group; the name tags records for the static
    analysis filters.
    """
    axes = set(exp_cfg.get("axes", {}).keys())
    arms = set(exp_cfg.get("arms", []))
    diags = set(exp_cfg.get("diagnostics", []))
    if "confounder_frac" in axes:
        return "confounder_sweep"
    if "capacity" in axes:
        return "capacity_warmup"
    if "pos_rate" in axes and "band" in axes:
        return "band_default_sweep"
    if "band" in axes:
        return "alpha_widen"
    if "grad_mass" in diags and "repr_probe" in diags:
        return "mechanism_probe"
    if "band_decoy_geometry" in diags:
        return "band_vs_hnm"
    if "ce_oracle" in arms or "ce_hnm" in arms:
        return "mechanism_transfer"
    return "cue_ablation"


# Experiment-config name -> (analysis function, analysis-step attribute, figure
# stem).  Used by the static analysis steps and by report().
_ANALYSIS_FIGURE_STEMS = {
    "cue_ablation": "fig_metric_specificity",
    "mechanism_probe": "fig_mechanism",
    "mechanism_transfer": "fig_mechanism_transfer",
    "band_vs_hnm": "fig_warmup_edge",
    "alpha_widen": "fig_alpha_widen",
    "confounder_sweep": "fig_surrogate",
    "capacity_warmup": "fig_capacity_lr",
    "band_default_sweep": "fig_default_sweep",
}

_FIGURES_DIR = _pathlib.Path(__file__).parent.parent / "figures"


def _render_to_png_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _write_figure(stem: str, fig) -> bytes:
    """Render a figure: write PNG into figures/ (derived output) and return bytes.

    The bytes are also attached to the step's @card by the caller.  No JSON or
    other loose result files are written — only figures/*.png and Metaflow
    artifacts/cards.
    """
    png = _render_to_png_bytes(fig)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (_FIGURES_DIR / f"{stem}.png").write_bytes(png)
    return png


if _METAFLOW_AVAILABLE:

    class PaucFlow(FlowSpec):
        """PAUC-vs-CE consolidated Metaflow pipeline (dataset-keyed).

        Pipeline shape::

            start -> train (foreach DATASET KEY) -> join -> select_by_val
                  -> aggregate
                  -> [ an_cue_specificity, an_mechanism, an_transfer,
                       an_band_escape, an_alpha_lever, an_surrogate,
                       an_capacity, an_default_sweep ]            (static branch)
                  -> join_analyses -> report -> end

        The foreach grain in ``train`` is the DATASET KEY (experiment x data-cell
        x seed), NOT the per-config combo.  Each dataset is generated once and all
        models that use it are trained on the shared in-memory tensors.
        Budget-agnostic arms (ce / trivial / smoothap) train once and are
        evaluated at every budget; budget-dependent arms (pauc) retrain per
        budget.

        No JSON files are written: results live in Metaflow artifacts and cards;
        figures are written to ``figures/*.png`` as derived report outputs.

        Smoke run (reduced):
            uv run lab/pauc_vs_ce_regimes/flow/pauc_flow.py \\
                --config-value cfg "hydra_overrides: [experiment=cue_ablation,seeds=1,geometry.n_train=5000,geometry.n_test=10000]" \\
                run --max-workers 4
        """

        cfg = Config(
            "cfg",
            default=str(_CONF_DIR / "config.yaml"),
            parser=_hydra_parser,
        )

        @step
        def start(self):
            """Compose config; expand enabled experiment into DATASET KEYS.

            Each dataset key is (experiment, data-cell, seed) and carries the list
            of training combos that use that dataset.  The foreach over dataset
            keys means the branch count equals the DATASET count, not the combo
            count.
            """
            cfg = self.cfg
            exp_cfg = cfg["experiment"]
            geo_cfg = cfg["geometry"]
            seeds = cfg["seeds"]
            experiment = exp_cfg.get("name") or _infer_experiment_name_fallback(exp_cfg)

            training_cfg = _resolve_training_cfg(cfg)
            self.bootstrap_cfg = cfg["bootstrap"]
            self.geometry_cfg = dict(geo_cfg)
            self.training_cfg = dict(training_cfg)
            self.experiment_cfg = dict(exp_cfg)
            self.experiment_name = experiment

            dataset_keys = build_dataset_keys(experiment, exp_cfg, seeds)
            n_combos = sum(len(dk["combos"]) for dk in dataset_keys)

            print(f"[start] experiment={experiment}", flush=True)
            print(
                f"[start] data_axes={exp_cfg.get('data_axes', [])} "
                f"axes={list(exp_cfg.get('axes', {}).keys())}",
                flush=True,
            )
            print(
                f"[start] DATASET KEYS (foreach grain)={len(dataset_keys)} "
                f"(seeds={seeds}); total training combos={n_combos}",
                flush=True,
            )
            print(f"[start] arms={exp_cfg.get('arms', [])}", flush=True)

            self.dataset_keys = dataset_keys
            self.next(self.train, foreach="dataset_keys")

        @staticmethod
        def _infer_experiment_name(exp_cfg: dict) -> str:
            """Best-effort experiment name fallback (delegates to module-level helper).

            Kept for backward compatibility.  start() now calls
            _infer_experiment_name_fallback directly.
            """
            return _infer_experiment_name_fallback(exp_cfg)

        @step
        def train(self):
            """Generate ONE dataset, train every combo on the shared tensors.

            Emits one record per (arm, config, budget).  Budget-agnostic arms are
            trained once and coverage@budget is recorded for each budget in the
            combo's ``budgets`` list; budget-dependent arms train per budget.
            Models are stored in the record only when the experiment requests a
            model-consuming diagnostic; scores+grp are stored only when requested.
            """
            import torch

            torch.set_num_threads(1)

            dk = self.input
            experiment = dk["experiment"]
            data_cell = dk["data_cell"]
            seed = dk["seed"]
            combos = dk["combos"]

            exp_cfg = self.experiment_cfg
            geo_cfg = self.geometry_cfg
            training_cfg = self.training_cfg
            diag_names = exp_cfg.get("diagnostics", [])
            requests_model = bool(exp_cfg.get("requests_model", False))
            requests_scores = bool(exp_cfg.get("requests_scores", False))

            key = _task_key(experiment, data_cell, seed)
            print(
                f"[train] {key} dataset combos={len(combos)} "
                f"(budget_dep={sum(c['budget_dep'] for c in combos)})",
                flush=True,
            )

            # Build the dataset ONCE (data-cell carries the data-axis overrides).
            data = _build_data_dict(geo_cfg, data_cell, seed)

            records = []
            for combo in combos:
                arm_name = combo["arm"]
                base_spec = combo["arm_spec"]
                budgets = combo["budgets"]
                train_cell = combo["train_cell"]

                if combo["budget_dep"]:
                    # One training per budget (band = f(budget)).
                    for budget in budgets:
                        spec = dict(base_spec)
                        spec["budget"] = budget
                        spec = _merge_training_into_arm(spec, training_cfg, train_cell)
                        model, info = train_arm(spec, data, seed)
                        records.append(self._make_record(
                            experiment, data_cell, train_cell, seed, arm_name,
                            spec, budget, model, info, data, diag_names,
                            requests_model, requests_scores,
                        ))
                else:
                    # Train ONCE; evaluate at every budget.
                    spec = _merge_training_into_arm(
                        dict(base_spec), training_cfg, train_cell
                    )
                    model, info = train_arm(spec, data, seed)
                    for budget in budgets:
                        records.append(self._make_record(
                            experiment, data_cell, train_cell, seed, arm_name,
                            spec, budget, model, info, data, diag_names,
                            requests_model, requests_scores,
                        ))
                    print(
                        f"[train] {key} arm={arm_name} trained ONCE, "
                        f"eval at budgets={budgets}",
                        flush=True,
                    )

            self.records = records
            self.next(self.join)

        @staticmethod
        def _make_record(
            experiment, data_cell, train_cell, seed, arm_name, spec, budget,
            model, info, data, diag_names, requests_model, requests_scores,
        ) -> dict:
            """Build one (arm, config, budget) record from a trained model.

            Metrics are evaluated AT the given budget (so budget-agnostic models
            yield per-budget coverage from the same scores).  Diagnostics are the
            per-budget consumers (band geometry depends on budget).
            """
            test_scores = scores_of(model, data["X_test"])
            y_test_np = data["_y_test_np"]
            test_cov = coverage(y_test_np, test_scores, budget)
            test_auroc = auroc(y_test_np, test_scores)
            test_aucpr = aucpr(y_test_np, test_scores)

            if data["X_val"] is None:
                val_cov = None
            elif info.get("best_val_cov") is not None:
                val_cov = float(info["best_val_cov"])
            else:
                val_scores = scores_of(model, data["X_val"])
                val_cov = coverage(data["_y_val_np"], val_scores, budget)

            # Diagnostics evaluated at this budget.
            spec_for_diag = dict(spec)
            spec_for_diag["budget"] = budget
            diag_results = _run_diagnostics(
                diag_names, model, data, spec_for_diag, train_cell
            )

            config_snapshot = {
                k: v for k, v in spec.items() if not k.startswith("_")
            }
            config_snapshot["budget"] = budget

            rec = {
                "experiment": experiment,
                "data_cell": data_cell,
                "train_cell": train_cell,
                "cell": {**data_cell, **train_cell, "budget": budget},
                "seed": seed,
                "arm": arm_name,
                "config": config_snapshot,
                "budget": budget,
                "val_coverage": val_cov,
                "test": {
                    "coverage": test_cov,
                    "auroc": test_auroc,
                    "aucpr": test_aucpr,
                },
                "diagnostics": diag_results,
            }
            if requests_model:
                rec["model"] = model
            if requests_scores:
                rec["scores"] = test_scores
                rec["grp"] = data.get("grp_test")
            return rec

        @step
        def join(self, inputs):
            """Flatten all dataset-branch records into a single list."""
            all_records = []
            for inp in inputs:
                all_records.extend(inp.records)
            self.all_records = all_records
            self.merge_artifacts(
                inputs,
                include=[
                    "bootstrap_cfg", "geometry_cfg", "training_cfg",
                    "experiment_cfg", "experiment_name",
                ],
            )
            print(f"[join] total records: {len(all_records)}", flush=True)
            self.next(self.select_by_val)

        @step
        def select_by_val(self):
            """Per (cell, arm, budget, seed): keep the config with best val_coverage.

            Records with ``val_coverage=None`` (sequential split, no val set) pass
            through.  Selection is keyed by the full cell AND budget so that
            budget-agnostic arms evaluated at multiple budgets are kept separate.
            """
            from collections import defaultdict

            grouped: dict = defaultdict(list)
            for rec in self.all_records:
                cell_key = tuple(sorted(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in rec["cell"].items()
                ))
                group_key = (cell_key, rec["arm"], rec["seed"])
                grouped[group_key].append(rec)

            selected = []
            for _, recs in grouped.items():
                selected.append(_select_by_val_pure(recs))

            self.selected_records = selected
            print(
                f"[select_by_val] {len(self.all_records)} -> {len(selected)} "
                f"after val-selection",
                flush=True,
            )
            self.next(self.aggregate)

        @step
        def aggregate(self):
            """Bootstrap CI aggregation over seeds per (cell, arm).

            Stores self.aggregate_results (per cell/arm coverage/auroc/aucpr CIs)
            and self.lift_results (pauc_pairwise - weighted_ce, seed-paired).
            """
            from collections import defaultdict

            n_resamples = self.bootstrap_cfg.get("n_resamples", 1000)
            boot_seed = self.bootstrap_cfg.get("seed", 0)

            def _cell_key(cell):
                return tuple(sorted(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in cell.items()
                ))

            grouped: dict = defaultdict(list)
            for rec in self.selected_records:
                grouped[(_cell_key(rec["cell"]), rec["arm"])].append(rec)

            agg_results = []
            for (cell_key, arm_name), recs in grouped.items():
                cov = np.array([r["test"]["coverage"] for r in recs])
                au = np.array([r["test"]["auroc"] for r in recs])
                ap = np.array([r["test"]["aucpr"] for r in recs])
                cm, clo, chi = bootstrap_ci(cov, n_resamples=n_resamples, seed=boot_seed)
                am, alo, ahi = bootstrap_ci(au, n_resamples=n_resamples, seed=boot_seed)
                pm, plo, phi = bootstrap_ci(ap, n_resamples=n_resamples, seed=boot_seed)
                agg_results.append({
                    "cell": dict(recs[0]["cell"]),
                    "arm": arm_name,
                    "n_seeds": len(recs),
                    "test_coverage": {"mean": cm, "lo": clo, "hi": chi},
                    "test_auroc": {"mean": am, "lo": alo, "hi": ahi},
                    "test_aucpr": {"mean": pm, "lo": plo, "hi": phi},
                })

            # Paired lift pauc_pairwise - weighted_ce, paired by seed per cell.
            cov_by_key: dict = {}
            for rec in self.selected_records:
                cov_by_key[(_cell_key(rec["cell"]), rec["arm"], rec["seed"])] = (
                    rec["test"]["coverage"]
                )
            all_cells = {_cell_key(rec["cell"]) for rec in self.selected_records}
            lift_results = []
            for ck in all_cells:
                pauc_seeds = {s for (c, a, s) in cov_by_key
                              if c == ck and a == "pauc_pairwise"}
                ce_seeds = {s for (c, a, s) in cov_by_key
                            if c == ck and a == "weighted_ce"}
                shared = sorted(pauc_seeds & ce_seeds)
                if not shared:
                    continue
                lifts = np.array([
                    cov_by_key[(ck, "pauc_pairwise", s)]
                    - cov_by_key[(ck, "weighted_ce", s)] for s in shared
                ])
                lm, llo, lhi = bootstrap_ci(lifts, n_resamples=n_resamples, seed=boot_seed)
                lift_results.append({
                    "cell": dict(ck),
                    "comparison": "pauc_pairwise - weighted_ce",
                    "metric": "test_coverage",
                    "lift_mean": lm, "lift_lo": llo, "lift_hi": lhi,
                    "ci_excludes_zero": bool(llo > 0 or lhi < 0),
                    "n_seeds": len(shared), "seeds": shared,
                })

            self.aggregate_results = agg_results
            self.lift_results = lift_results
            print(
                f"[aggregate] {len(agg_results)} (cell, arm) summaries, "
                f"{len(lift_results)} paired lifts",
                flush=True,
            )
            self.next(
                self.an_cue_specificity, self.an_mechanism, self.an_transfer,
                self.an_band_escape, self.an_alpha_lever, self.an_surrogate,
                self.an_capacity, self.an_default_sweep,
            )

        # ---- static analysis branch -------------------------------------------
        # Each step filters records for its experiment; passes through cleanly if
        # the experiment was not enabled (empty result, no card content, no PNG).

        def _attach_card_figure(self, title: str, stem: str, fig, summary_md: str):
            """Render figure -> figures/ PNG + attach to @card; store result."""
            from metaflow import current

            png = _write_figure(stem, fig)
            current.card.append(Markdown(f"## {title}"))
            if summary_md:
                current.card.append(Markdown(summary_md))
            current.card.append(Image(png, label=stem))

        @card
        @step
        def an_cue_specificity(self):
            """cue_ablation: coverage-lift vs AUROC-lift per cue (reproduces 01)."""
            from metaflow import current

            rows = analyze_cue_specificity(self.all_records)
            self.an_result = {"experiment": "cue_ablation", "rows": rows}
            if not rows:
                self.next(self.join_analyses)
                return
            fig = self._fig_cue_specificity(rows)
            md = "\n".join(
                f"- {r['cue']} @budget={r['budget']}: cov_lift="
                f"{r['coverage_lift_mean']:+.4f} "
                f"[{r['coverage_lift_lo']:+.4f},{r['coverage_lift_hi']:+.4f}]"
                f"{'*' if r['ci_excludes_zero'] else ''}, "
                f"auroc_lift={r['auroc_lift']:+.4f}" if r['auroc_lift'] is not None
                else f"- {r['cue']}: cov_lift {r['coverage_lift_mean']:+.4f}"
                for r in rows
            )
            self._attach_card_figure(
                "Cue specificity (coverage lift vs AUROC lift)",
                _ANALYSIS_FIGURE_STEMS["cue_ablation"], fig, md,
            )
            current.card.append(Markdown(f"`{len(rows)}` cells analyzed."))
            self.next(self.join_analyses)

        @card
        @step
        def an_mechanism(self):
            """mechanism_probe: grad mass, repr probe, gap-closure (reproduces 02)."""
            result = analyze_mechanism(self.all_records)
            self.an_result = {"experiment": "mechanism_probe", "result": result}
            if not result:
                self.next(self.join_analyses)
                return
            fig = self._fig_mechanism(result)
            md = (
                f"- band decoy-frac (PAUC band precision, fraction of band that is decoys): "
                f"{result['band_decoyfrac_pauc']}\n"
                f"- neg-grad mass on decoys CE: {result['grad_mass_ce']}\n"
                f"- neg-grad mass on decoys PAUC: {result['grad_mass_pauc']}\n"
                f"- probe AUC pos-vs-decoy CE: {result['probe_auc_ce']}\n"
                f"- probe AUC pos-vs-decoy PAUC: {result['probe_auc_pauc']}"
            )
            self._attach_card_figure(
                "Mechanism probe", _ANALYSIS_FIGURE_STEMS["mechanism_probe"], fig, md,
            )
            self.next(self.join_analyses)

        @card
        @step
        def an_transfer(self):
            """mechanism_transfer: gap-closure across cue x budget (reproduces 03)."""
            rows = analyze_transfer(self.all_records)
            self.an_result = {"experiment": "mechanism_transfer", "rows": rows}
            if not rows:
                self.next(self.join_analyses)
                return
            fig = self._fig_transfer(rows)
            md = "\n".join(
                f"- {r['cue']} @{r['budget']}: best={r.get('best_concentrated')} "
                f"best_vs_pauc={r.get('best_vs_pauc')}" for r in rows
            )
            self._attach_card_figure(
                "Mechanism transfer", _ANALYSIS_FIGURE_STEMS["mechanism_transfer"],
                fig, md,
            )
            self.next(self.join_analyses)

        @card
        @step
        def an_band_escape(self):
            """band_vs_hnm: decoy band geometry from CE scores (reproduces 04)."""
            rows = analyze_band_escape(self.all_records)
            self.an_result = {"experiment": "band_vs_hnm", "rows": rows}
            if not rows:
                self.next(self.join_analyses)
                return
            fig = self._fig_band_escape(rows)
            md = "\n".join(
                f"- {r['cue']} @{r['budget']}: in_band={r['in_band']['mean']:.2f} "
                f"in_top2={r['in_top2']['mean']:.2f} "
                f"above_band={r['above_band']['mean']:.2f}" for r in rows
            )
            self._attach_card_figure(
                "Band escape", _ANALYSIS_FIGURE_STEMS["band_vs_hnm"], fig, md,
            )
            self.next(self.join_analyses)

        @card
        @step
        def an_alpha_lever(self):
            """alpha_widen: band-widening (wide-minus-std) improvement (reproduces 05)."""
            rows = analyze_alpha_lever(self.all_records)
            self.an_result = {"experiment": "alpha_widen", "rows": rows}
            if not rows:
                self.next(self.join_analyses)
                return
            fig = self._fig_alpha_lever(rows)
            md = "\n".join(
                f"- {r['cue']} @{r['budget']}: std={r['cov_pauc_std']} "
                f"wide={r['cov_pauc_wide']} hnm={r['cov_hnm']}" for r in rows
            )
            self._attach_card_figure(
                "Alpha lever (band widening)", _ANALYSIS_FIGURE_STEMS["alpha_widen"],
                fig, md,
            )
            self.next(self.join_analyses)

        @card
        @step
        def an_surrogate(self):
            """confounder_sweep: coverage by condition across sweep (reproduces 08)."""
            rows = analyze_surrogate(self.all_records)
            self.an_result = {"experiment": "confounder_sweep", "rows": rows}
            if not rows:
                self.next(self.join_analyses)
                return
            fig = self._fig_surrogate(rows)
            md = "\n".join(
                f"- conf={r['confounder_frac']} pos={r['pos_rate']}: "
                + ", ".join(f"{c}={v['coverage']:.3f}"
                            for c, v in r["conditions"].items())
                for r in rows
            )
            self._attach_card_figure(
                "Surrogate (confounder sweep)",
                _ANALYSIS_FIGURE_STEMS["confounder_sweep"], fig, md,
            )
            self.next(self.join_analyses)

        @card
        @step
        def an_capacity(self):
            """capacity_warmup: capacity / warmup readings (reproduces 07)."""
            rows = analyze_capacity(self.all_records)
            self.an_result = {"experiment": "capacity_warmup", "rows": rows}
            if not rows:
                self.next(self.join_analyses)
                return
            fig = self._fig_capacity(rows)
            md = "\n".join(
                f"- pos={r['pos_rate']} decoy={r['decoy_frac']} cap={r['capacity']}: "
                + ", ".join(f"{c}={v:.3f}" for c, v in r["conditions"].items())
                for r in rows
            )
            self._attach_card_figure(
                "Capacity / warmup", _ANALYSIS_FIGURE_STEMS["capacity_warmup"], fig, md,
            )
            self.next(self.join_analyses)

        @card
        @step
        def an_default_sweep(self):
            """band_default_sweep: best band per pos_rate, robust band (reproduces 06)."""
            result = analyze_default_sweep(self.all_records)
            self.an_result = {"experiment": "band_default_sweep", "result": result}
            if not result:
                self.next(self.join_analyses)
                return
            fig = self._fig_default_sweep(result)
            md = "\n".join(
                f"- pos={c['pos_rate']}: best band a{c['best']['alpha_m']}"
                f"_b{c['best']['beta_m']} cov={c['best']['cov']:.3f}"
                for c in result["per_pos"]
            )
            if result.get("robust_band"):
                rb = result["robust_band"]
                md += (f"\n- robust band: a{rb['alpha_m']}_b{rb['beta_m']} "
                       f"mean_cov={rb['mean_cov']:.3f}")
            self._attach_card_figure(
                "Default band sweep", _ANALYSIS_FIGURE_STEMS["band_default_sweep"],
                fig, md,
            )
            self.next(self.join_analyses)

        # ---- figure builders (pure matplotlib; no I/O) ------------------------

        @staticmethod
        def _new_fig(w=7.0, h=4.0):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(w, h))
            return fig, ax

        def _fig_cue_specificity(self, rows):
            fig, ax = self._new_fig()
            labels = [f"{r['cue']}\n@{r['budget']}" for r in rows]
            x = np.arange(len(rows))
            cov = [r["coverage_lift_mean"] for r in rows]
            lo = [r["coverage_lift_mean"] - r["coverage_lift_lo"] for r in rows]
            hi = [r["coverage_lift_hi"] - r["coverage_lift_mean"] for r in rows]
            au = [r["auroc_lift"] if r["auroc_lift"] is not None else 0.0 for r in rows]
            ax.bar(x - 0.2, cov, width=0.4, yerr=[lo, hi], capsize=3,
                   label="coverage lift (PAUC-CE)")
            ax.bar(x + 0.2, au, width=0.4, label="AUROC lift (PAUC-CE)")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("lift"); ax.set_title("Cue specificity"); ax.legend()
            return fig

        def _fig_mechanism(self, result):
            fig, ax = self._new_fig()
            cov = result.get("coverage", {})
            arms = list(cov.keys())
            ax.bar(np.arange(len(arms)), [cov[a] for a in arms])
            ax.set_xticks(np.arange(len(arms)))
            ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("coverage@budget")
            ax.set_title(
                f"Mechanism: gap-closure | grad CE={result['grad_mass_ce']} "
                f"PAUC={result['grad_mass_pauc']}"
            )
            return fig

        def _fig_transfer(self, rows):
            fig, ax = self._new_fig()
            labels = [f"{r['cue']}\n@{r['budget']}" for r in rows]
            x = np.arange(len(rows))
            for arm, off in [("weighted_ce", -0.3), ("ce_hnm", -0.1),
                             ("ce_oracle", 0.1), ("pauc_pairwise", 0.3)]:
                vals = [r.get(arm) if r.get(arm) is not None else 0.0 for r in rows]
                ax.bar(x + off, vals, width=0.2, label=arm)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("coverage"); ax.set_title("Mechanism transfer"); ax.legend(fontsize=7)
            return fig

        def _fig_band_escape(self, rows):
            fig, ax = self._new_fig()
            labels = [f"{r['cue']}\n@{r['budget']}" for r in rows]
            x = np.arange(len(rows))
            for k, off in [("in_band", -0.25), ("in_top2", 0.0), ("above_band", 0.25)]:
                ax.bar(x + off, [r[k]["mean"] for r in rows], width=0.25, label=k)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("decoy fraction"); ax.set_title("Band escape"); ax.legend()
            return fig

        def _fig_alpha_lever(self, rows):
            fig, ax = self._new_fig()
            labels = [f"{r['cue']}\n@{r['budget']}" for r in rows]
            x = np.arange(len(rows))
            for k, off in [("cov_ce", -0.3), ("cov_hnm", -0.1),
                           ("cov_pauc_std", 0.1), ("cov_pauc_wide", 0.3)]:
                vals = [r.get(k) if r.get(k) is not None else 0.0 for r in rows]
                ax.bar(x + off, vals, width=0.2, label=k)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("coverage"); ax.set_title("Alpha widening"); ax.legend(fontsize=7)
            return fig

        def _fig_surrogate(self, rows):
            fig, ax = self._new_fig(8.0, 4.0)
            conds = ["trivial", "weighted_ce", "pauc_pairwise", "pauc_trapezoid", "smoothap"]
            labels = [f"c{r['confounder_frac']}\np{r['pos_rate']}" for r in rows]
            x = np.arange(len(rows))
            width = 0.15
            for i, c in enumerate(conds):
                vals = [r["conditions"].get(c, {}).get("coverage", 0.0) for r in rows]
                ax.bar(x + (i - 2) * width, vals, width=width, label=c)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("coverage@budget"); ax.set_title("Surrogate sweep")
            ax.legend(fontsize=7)
            return fig

        def _fig_capacity(self, rows):
            fig, ax = self._new_fig(8.0, 4.0)
            conds = ["trivial", "weighted_ce", "pauc_pairwise"]
            labels = [f"{r['capacity']}\np{r['pos_rate']}d{r['decoy_frac']}" for r in rows]
            x = np.arange(len(rows))
            for i, c in enumerate(conds):
                vals = [r["conditions"].get(c, 0.0) for r in rows]
                ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=c)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
            ax.set_ylabel("coverage"); ax.set_title("Capacity / warmup"); ax.legend()
            return fig

        def _fig_default_sweep(self, result):
            fig, ax = self._new_fig(8.0, 4.5)
            per_pos = result["per_pos"]
            band_keys = sorted({k for c in per_pos for k in c["grid"]})
            for c in per_pos:
                ys = [c["grid"].get(b, np.nan) for b in band_keys]
                ax.plot(range(len(band_keys)), ys, marker="o",
                        label=f"pos={c['pos_rate']}")
            ax.set_xticks(range(len(band_keys)))
            ax.set_xticklabels(band_keys, rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("coverage@budget"); ax.set_title("Default band sweep")
            ax.legend(fontsize=7)
            return fig

        # ---- join + report + end ----------------------------------------------

        @step
        def join_analyses(self, inputs):
            """Collect analysis artifacts from the static branch.

            The upstream artifacts (all_records etc.) are identical across the
            static branches by origin, but ``all_records`` can carry nn.Module
            models (requests_model experiments) which do not compare equal across
            pickled branch copies — so merge_artifacts would raise.  We therefore
            assign the propagated artifacts explicitly from the first input
            instead of merging them.

            Safety invariants:
              - Every static branch (an_*) is a pure reader: it only sets
                ``self.an_result`` and does not mutate shared upstream artifacts.
                This is why propagating artifacts from inputs[0] is safe —
                all branches received the same data and none modified it.
              - The propagated ``all_records`` must be present on every input
                (all branches fanned out from aggregate, which sets it).
            """
            # Assert the propagation precondition: every static branch must carry
            # all_records (set in aggregate before the fan-out).
            for i, inp in enumerate(inputs):
                assert hasattr(inp, "all_records"), (
                    f"join_analyses: inputs[{i}] is missing 'all_records'; "
                    "all static branches must originate from the aggregate step."
                )
            self.analyses = {
                inp.an_result["experiment"]: inp.an_result for inp in inputs
            }
            first = inputs[0]
            self.all_records = first.all_records
            self.selected_records = first.selected_records
            self.aggregate_results = first.aggregate_results
            self.lift_results = first.lift_results
            self.bootstrap_cfg = first.bootstrap_cfg
            self.geometry_cfg = first.geometry_cfg
            self.training_cfg = first.training_cfg
            self.experiment_cfg = first.experiment_cfg
            self.experiment_name = first.experiment_name
            n_nonempty = sum(
                1 for a in self.analyses.values()
                if a.get("rows") or a.get("result")
            )
            print(
                f"[join_analyses] {len(self.analyses)} analysis steps, "
                f"{n_nonempty} produced results",
                flush=True,
            )
            self.next(self.report)

        @step
        def report(self):
            """Print a concise summary of every analysis that produced results."""
            print("\n=== PaucFlow analyses ===", flush=True)
            for exp, a in self.analyses.items():
                payload = a.get("rows") or a.get("result")
                if not payload:
                    continue
                stem = _ANALYSIS_FIGURE_STEMS.get(exp, exp)
                print(f"\n--- {exp}  (figure: figures/{stem}.png) ---", flush=True)
                if exp == "cue_ablation":
                    for r in a["rows"]:
                        star = "*" if r["ci_excludes_zero"] else " "
                        au = (f"auroc_lift={r['auroc_lift']:+.4f}"
                              if r["auroc_lift"] is not None else "auroc_lift=n/a")
                        print(
                            f"  {str(r['cue']):>16} budget={r['budget']}  "
                            f"cov_lift={r['coverage_lift_mean']:+.4f} "
                            f"[{r['coverage_lift_lo']:+.4f},{r['coverage_lift_hi']:+.4f}]"
                            f"{star}  {au}",
                            flush=True,
                        )
                else:
                    print(f"  {payload}", flush=True)
            self.next(self.end)

        @step
        def end(self):
            """Final summary."""
            print("\n=== PaucFlow run complete ===", flush=True)
            print(f"experiment:            {self.experiment_name}", flush=True)
            print(f"total records:         {len(self.all_records)}", flush=True)
            print(f"val-selected records:  {len(self.selected_records)}", flush=True)
            print(
                f"aggregated summaries:  {len(self.aggregate_results)}",
                flush=True,
            )

    if __name__ == "__main__":
        PaucFlow()
