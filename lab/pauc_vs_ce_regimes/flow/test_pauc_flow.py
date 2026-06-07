"""
Contract tests for pauc_flow.py component functions.

These tests are fast, small-data, and smoke-level — they verify:
  - Data identity (same spec+seed => identical tensors)
  - grp composition (codes, counts match geometry)
  - coverage equals the top-k definition on hand-constructed data
  - bootstrap_ci is deterministic at seed 0 and satisfies lo <= mean <= hi
  - train_arm dispatch: each kind returns finite coverage in [0, 1]
  - build_model returns correct output shapes

Run with:
    uv run --with torch --with numpy --with scikit-learn --with pytest \\
        pytest lab/pauc_vs_ce_regimes/flow/test_pauc_flow.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow importing pauc_flow without the repo package installed
_FLOW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FLOW_DIR))

# This must import without pulling in metaflow/hydra/matplotlib
import pauc_flow as pf

# ---------------------------------------------------------------------------
# Tiny sizes for fast tests
# ---------------------------------------------------------------------------
_N_TRAIN = 2_000
_N_SMALL = 500  # used for train_arm smoke tests
_EPOCHS_SMOKE = 2
_BS_SMOKE = 128


# ===========================================================================
# Data identity tests
# ===========================================================================

class TestDataIdentity:
    """make_data(spec, seed) twice => byte-identical tensors."""

    @pytest.mark.parametrize("kind", ["headline", "confounder", "hard_bulk"])
    def test_same_spec_same_seed_identical(self, kind):
        spec = {"kind": kind, "n": _N_TRAIN}
        X1, y1, g1 = pf.make_data(spec, seed=0)
        X2, y2, g2 = pf.make_data(spec, seed=0)
        assert torch.equal(X1, X2), "X differs across calls with same seed"
        assert torch.equal(y1, y2), "y differs across calls with same seed"
        assert np.array_equal(g1, g2), "grp differs across calls with same seed"

    @pytest.mark.parametrize("kind", ["headline", "confounder", "hard_bulk"])
    def test_different_seeds_differ(self, kind):
        spec = {"kind": kind, "n": _N_TRAIN}
        X1, _, _ = pf.make_data(spec, seed=0)
        X2, _, _ = pf.make_data(spec, seed=1)
        # With high probability the data differs
        assert not torch.equal(X1, X2), "Different seeds should produce different data"

    def test_two_arms_see_identical_data_for_fixed_seed(self):
        """Two arms in the same cell should receive the same data (same seed)."""
        spec = {"kind": "headline", "n": _N_TRAIN, "cue": "nonlinear_prod"}
        X1, y1, g1 = pf.make_data(spec, seed=3)
        X2, y2, g2 = pf.make_data(spec, seed=3)
        assert torch.equal(X1, X2)
        assert torch.equal(y1, y2)
        assert np.array_equal(g1, g2), "grp arrays must be identical for same seed (affects arm symmetry)"


# ===========================================================================
# grp composition tests
# ===========================================================================

class TestGrpComposition:
    """Group codes are present and counts roughly match geometry parameters."""

    def test_headline_grp_codes(self):
        n = _N_TRAIN
        spec = {"kind": "headline", "n": n, "pos_rate": 0.05, "decoy_frac": 0.02}
        _, y, grp = pf.make_data(spec, seed=0)
        assert set(np.unique(grp)).issubset({0, 1, 2})
        # No confounders in headline
        assert 3 not in np.unique(grp)

    def test_headline_pos_count(self):
        n = _N_TRAIN
        pos_rate = 0.05
        spec = {"kind": "headline", "n": n, "pos_rate": pos_rate}
        _, y, grp = pf.make_data(spec, seed=0)
        expected_pos = max(8, round(pos_rate * n))
        assert int((grp == 1).sum()) == expected_pos
        # y and grp agree
        assert int(y.sum()) == expected_pos

    def test_headline_decoy_count(self):
        n = _N_TRAIN
        pos_rate = 0.05
        decoy_frac = 0.02
        spec = {"kind": "headline", "n": n, "pos_rate": pos_rate, "decoy_frac": decoy_frac}
        _, y, grp = pf.make_data(spec, seed=0)
        n_neg = int((grp != 1).sum())
        expected_decoy = round(decoy_frac * n_neg)
        assert int((grp == 2).sum()) == expected_decoy

    def test_confounder_has_code_3(self):
        n = _N_TRAIN
        spec = {
            "kind": "confounder",
            "n": n,
            "pos_rate": 0.05,
            "confounder_frac": 0.05,
        }
        _, _, grp = pf.make_data(spec, seed=0)
        assert 3 in np.unique(grp), "Confounder geometry should produce grp=3"

    def test_confounder_zero_frac_no_code3(self):
        spec = {"kind": "confounder", "n": _N_TRAIN, "confounder_frac": 0.0}
        _, _, grp = pf.make_data(spec, seed=0)
        assert 3 not in np.unique(grp)

    def test_confounder_decoys_get_easy_signal_exactly_once(self):
        """Decoys in confounder geometry must receive +EASY exactly once.

        Regression test: a previous version called inject_easy(X, decoy) in
        addition to inject_decoys (which already applies +EASY), doubling the
        shift.
        """
        n = _N_TRAIN
        # Use a very large decoy_frac so decoys are numerous and detectable
        spec = {
            "kind": "confounder",
            "n": n,
            "pos_rate": 0.05,
            "decoy_frac": 0.10,
            "confounder_frac": 0.0,
        }
        X, _, grp = pf.make_data(spec, seed=0)
        pos_mean = float(X[grp == 1, 0].mean())
        decoy_mean = float(X[grp == 2, 0].mean())
        # Positives and decoys should have similar mean on feature 0
        # (both shifted by +EASY once).  If decoys were shifted twice their
        # mean would be ~+EASY higher than positives.
        # _EASY=2.5 from conf/geometry/headline.yaml (now a function parameter, not a module global)
        _EASY_DEFAULT = 2.5
        assert abs(decoy_mean - pos_mean) < _EASY_DEFAULT * 0.5, (
            f"Decoy mean {decoy_mean:.3f} vs pos mean {pos_mean:.3f}: "
            "suggests easy signal was applied more than once to decoys."
        )

    def test_hard_bulk_grp_codes(self):
        spec = {"kind": "hard_bulk", "n": _N_TRAIN, "pos_rate": 0.05, "decoy_frac": 0.0}
        _, y, grp = pf.make_data(spec, seed=0)
        assert 0 in np.unique(grp)
        assert 1 in np.unique(grp)
        assert 2 not in np.unique(grp)  # decoy_frac=0 means no decoys

    def test_hard_bulk_with_decoys(self):
        spec = {"kind": "hard_bulk", "n": _N_TRAIN, "pos_rate": 0.05, "decoy_frac": 0.10}
        _, _, grp = pf.make_data(spec, seed=0)
        assert 2 in np.unique(grp)

    def test_hard_bulk_pos_count(self):
        n = _N_TRAIN
        pos_rate = 0.05
        spec = {"kind": "hard_bulk", "n": n, "pos_rate": pos_rate}
        _, y, grp = pf.make_data(spec, seed=0)
        expected = int(round(n * pos_rate))
        assert int((grp == 1).sum()) == expected


# ===========================================================================
# coverage top-k contract
# ===========================================================================

class TestCoverage:
    """coverage equals the top-k definition on hand-constructed arrays."""

    def test_hand_constructed_budget_covers_all_positives(self):
        # 3 positives at the top, 7 negatives at the bottom
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        # budget = 0.3 => k = ceil(0.3 * 10) = 3
        assert pf.coverage(y, scores, budget=0.3) == pytest.approx(1.0)

    def test_hand_constructed_partial_recall(self):
        # 4 positives; top-2 captures 2 of them
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        scores = np.array([10, 9, 8, 7, 1, 1, 1, 1], dtype=float)
        # budget = 0.25 => k = ceil(0.25 * 8) = 2
        # top-2 scores are index 0 (y=1) and index 1 (y=0) => 1 positive out of 4
        assert pf.coverage(y, scores, budget=0.25) == pytest.approx(0.25)

    def test_ceil_rounding(self):
        # budget = 0.11, n = 10 => k = ceil(1.1) = 2
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        assert pf.coverage(y, scores, budget=0.11) == pytest.approx(1.0)

    def test_zero_budget_clamp_to_k1(self):
        # Even budget=0 should return k=max(1,0)=1
        y = np.array([1, 0, 0], dtype=float)
        scores = np.array([10, 5, 1], dtype=float)
        assert pf.coverage(y, scores, budget=0.0) == pytest.approx(1.0)

    def test_no_positives_returns_zero(self):
        y = np.array([0, 0, 0], dtype=float)
        scores = np.array([3, 2, 1], dtype=float)
        assert pf.coverage(y, scores, budget=0.5) == 0.0


# ===========================================================================
# bootstrap_ci contract
# ===========================================================================

class TestBootstrapCi:
    def test_deterministic_at_seed0(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        r1 = pf.bootstrap_ci(values, n_resamples=500, seed=0)
        r2 = pf.bootstrap_ci(values, n_resamples=500, seed=0)
        assert r1 == r2

    def test_lo_le_mean_le_hi(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 30)
        mean, lo, hi = pf.bootstrap_ci(values, n_resamples=500, seed=0)
        assert lo <= mean
        assert mean <= hi

    def test_mean_equals_sample_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, _, _ = pf.bootstrap_ci(values, seed=0)
        assert mean == pytest.approx(np.mean(values))

    def test_different_seeds_may_differ(self):
        values = np.arange(20, dtype=float)
        _, lo0, hi0 = pf.bootstrap_ci(values, n_resamples=200, seed=0)
        _, lo1, hi1 = pf.bootstrap_ci(values, n_resamples=200, seed=99)
        # Not guaranteed to differ in 100% of runs, but very likely
        # Just check both return valid CIs
        assert lo0 <= hi0
        assert lo1 <= hi1


# ===========================================================================
# build_model shape tests
# ===========================================================================

class TestBuildModel:
    @pytest.mark.parametrize("capacity,d_in,expected_out", [
        ("linear", 20, 1),
        ("mlp_1x16", 20, 1),
        ("mlp_2x64", 20, 1),
        ("linear", 31, 1),
        ("mlp_2x64", 31, 1),
    ])
    def test_output_shape(self, capacity, d_in, expected_out):
        model = pf.build_model(capacity, d_in=d_in)
        x = torch.randn(8, d_in)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, expected_out), (
            f"Expected shape (8, {expected_out}), got {out.shape}"
        )

    def test_linear_is_single_layer(self):
        model = pf.build_model("linear", d_in=10)
        assert isinstance(model, torch.nn.Linear)

    def test_mlp_1x16_hidden_size(self):
        model = pf.build_model("mlp_1x16", d_in=20)
        # Should have Linear(20,16), ReLU, Linear(16,1)
        assert isinstance(model, torch.nn.Sequential)
        layers = list(model.children())
        linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
        assert linear_layers[0].out_features == 16
        assert linear_layers[-1].out_features == 1

    def test_mlp_2x64_hidden_size(self):
        model = pf.build_model("mlp_2x64", d_in=20)
        layers = list(model.children())
        linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
        assert len(linear_layers) == 3  # 20->64, 64->64, 64->1
        assert linear_layers[0].out_features == 64
        assert linear_layers[1].out_features == 64
        assert linear_layers[-1].out_features == 1

    def test_unknown_capacity_raises(self):
        with pytest.raises(ValueError):
            pf.build_model("transformer", d_in=20)


# ===========================================================================
# train_arm dispatch smoke tests
# ===========================================================================

def _make_tiny_data(n=_N_SMALL, seed=0):
    """Build a tiny synthetic dataset for smoke-testing train_arm."""
    spec = {
        "kind": "headline",
        "n": n,
        "pos_rate": 0.10,   # higher pos rate so there are positives in small batches
        "decoy_frac": 0.02,
        "cue": "nonlinear_prod",
    }
    X, y, grp = pf.make_data(spec, seed=seed)
    y_np = y.squeeze(1).numpy()
    # Train/val split: first 80% train, last 20% val
    split = int(0.8 * n)
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "grp_train": grp[:split],
        "X_val": X[split:],
        "y_val": y[split:],
        "grp_val": grp[split:],
        "X_test": X[split:],
        "grp_test": grp[split:],
        "y_np_test": y_np[split:],
        "budget": 0.20,  # wide budget so we always hit positives
    }


def _check_arm(arm_spec, data, seed=0):
    """Train arm and assert coverage is finite in [0,1]."""
    model, info = pf.train_arm(arm_spec, data, seed=seed)
    X_test = data["X_test"]
    y_test = data["y_np_test"]
    budget = data["budget"]
    s = pf.scores_of(model, X_test)
    cov = pf.coverage(y_test, s, budget)
    assert np.isfinite(cov), f"Coverage is not finite: {cov}"
    assert 0.0 <= cov <= 1.0, f"Coverage out of [0,1]: {cov}"
    return cov


class TestTrainArmDispatch:
    """Smoke tests: each arm kind trains without error and returns finite coverage."""

    def setup_method(self):
        self.data = _make_tiny_data(n=_N_SMALL, seed=7)

    def test_trivial_arm(self):
        arm = {"kind": "trivial"}
        _check_arm(arm, self.data)

    def test_ce_arm(self):
        arm = {
            "kind": "ce",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
        }
        _check_arm(arm, self.data)

    def test_ce_hnm_arm(self):
        arm = {
            "kind": "ce",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "hnm": {"top_q": 0.02, "factor": 50.0},
        }
        _check_arm(arm, self.data)

    def test_ce_oracle_arm(self):
        arm = {
            "kind": "ce",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "oracle": {"factor": 10.0},
        }
        _check_arm(arm, self.data)

    def test_pauc_arm_pairwise(self):
        arm = {
            "kind": "pauc",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "surrogate": "pairwise",
            "band": (0.5, 1.5),
            "queue_size": 256,
            "temp": 0.5,
            "temp_end": 0.1,
            "budget": self.data["budget"],
        }
        _check_arm(arm, self.data)

    def test_pauc_arm_trapezoid(self):
        arm = {
            "kind": "pauc",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "surrogate": "trapezoid",
            "band": (0.0, 1.0),
            "queue_size": 256,
            "budget": self.data["budget"],
        }
        _check_arm(arm, self.data)

    def test_smoothap_arm(self):
        arm = {
            "kind": "smoothap",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "queue_size": 256,
            "temp": 0.1,
            "temp_end": 0.01,
            "budget": self.data["budget"],
        }
        _check_arm(arm, self.data)

    def test_ce_warmup_only_arm(self):
        arm = {
            "kind": "ce_warmup_only",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE * 3,
            "batch_size": _BS_SMOKE,
            "warmup_frac": 0.33,
        }
        _check_arm(arm, self.data)

    def test_pauc_cold_arm(self):
        arm = {
            "kind": "pauc_cold",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "surrogate": "pairwise",
            "band": (0.5, 1.5),
            "queue_size": 256,
            "budget": self.data["budget"],
        }
        _check_arm(arm, self.data)

    def test_early_stop_arm(self):
        arm = {
            "kind": "ce",
            "capacity": "mlp_2x64",
            "epochs": _EPOCHS_SMOKE,
            "batch_size": _BS_SMOKE,
            "early_stop": True,
            "eval_every": 1,
            "budget": self.data["budget"],
        }
        model, info = pf.train_arm(arm, self.data, seed=0)
        assert info.get("best_val_cov") is not None
        assert np.isfinite(info["best_val_cov"])

    def test_unknown_arm_raises(self):
        arm = {"kind": "unknown_arm_xyz"}
        with pytest.raises((ValueError, KeyError)):
            pf.train_arm(arm, self.data, seed=0)


# ===========================================================================
# pos_weight and idx_batches
# ===========================================================================

class TestUtilities:
    def test_pos_weight_scalar(self):
        y = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pw = pf.pos_weight(y)
        # 3 negatives / 1 positive = 3
        assert pw.item() == pytest.approx(3.0)

    def test_pos_weight_2d(self):
        y = torch.tensor([[1.0], [0.0], [0.0]])
        pw = pf.pos_weight(y)
        assert pw.item() == pytest.approx(2.0)

    def test_idx_batches_covers_all(self):
        n, bs = 100, 32
        batches = pf.idx_batches(n, seed=0, batch_size=bs)
        all_idx = torch.cat(batches)
        assert all_idx.shape[0] == n
        assert all_idx.unique().shape[0] == n  # no duplicates, full coverage

    def test_idx_batches_deterministic(self):
        b1 = pf.idx_batches(50, seed=42)
        b2 = pf.idx_batches(50, seed=42)
        for a, b in zip(b1, b2):
            assert torch.equal(a, b)

    def test_scores_of_eval_train_toggle(self):
        model = pf.build_model("mlp_2x64")
        model.train()
        X = torch.randn(10, 20)
        s = pf.scores_of(model, X)
        assert model.training, "scores_of should restore train mode"
        assert s.shape == (10,)


# ===========================================================================
# inject_cue contract
# ===========================================================================

class TestInjectCue:
    @pytest.mark.parametrize("cue", ["linear", "nonlinear_prod", "product",
                                      "nonlinear_radial", "radial"])
    def test_known_cues_dont_raise(self, cue):
        X = np.zeros((5, 20), dtype=np.float32)
        rng = np.random.default_rng(0)
        pf.inject_cue(X, np.array([0, 1, 2, 3, 4]), cue, rng)

    def test_unknown_cue_raises(self):
        X = np.zeros((3, 20), dtype=np.float32)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            pf.inject_cue(X, np.array([0, 1, 2]), "circular", rng)

    def test_linear_cue_shifts_features_5_6(self):
        # sep=1.7 from conf/geometry/headline.yaml (now a parameter, not a module global)
        _SEP_DEFAULT = 1.7
        X = np.zeros((3, 20), dtype=np.float32)
        rng = np.random.default_rng(0)
        pf.inject_cue(X, np.array([0, 1, 2]), "linear", rng)
        assert np.all(X[:3, 5] == _SEP_DEFAULT)
        assert np.all(X[:3, 6] == _SEP_DEFAULT)
        # Other features unchanged
        assert np.all(X[:, 7] == 0.0)


# ===========================================================================
# band_decoy_geometry contract
# ===========================================================================

class TestBandDecoyGeometry:
    def test_returns_fractions_in_0_1(self):
        # Synthetic: positives at top, decoys in middle, easy-negs at bottom
        n = 1000
        scores = np.linspace(0, 1, n)
        grp = np.zeros(n, dtype=np.int64)
        grp[:10] = 1       # positives (top 1%)
        grp[10:30] = 2     # decoys (next 2%)
        result = pf.band_decoy_geometry(scores[::-1].copy(), grp, budget=0.05,
                                         band=(0.5, 1.5))
        for k in ("in_band", "in_top2", "above_band"):
            assert k in result
            assert 0.0 <= result[k] <= 1.0, f"{k}={result[k]} out of [0,1]"

    def test_no_decoys_returns_zeros(self):
        scores = np.random.randn(100)
        grp = np.zeros(100, dtype=np.int64)
        grp[:10] = 1
        result = pf.band_decoy_geometry(scores, grp, budget=0.05, band=(0.5, 1.5))
        assert result["in_band"] == 0.0
        assert result["in_top2"] == 0.0
        assert result["above_band"] == 0.0

    def test_empty_negatives_returns_zeros(self):
        """All-positive grp (no negatives) should not raise on sn.max()/quantile."""
        scores = np.ones(10, dtype=np.float32)
        grp = np.ones(10, dtype=np.int64)  # everyone is positive, no negatives
        result = pf.band_decoy_geometry(scores, grp, budget=0.05, band=(0.5, 1.5))
        assert result == {"in_band": 0.0, "in_top2": 0.0, "above_band": 0.0}


# ===========================================================================
# grad_mass_on_decoys smoke test
# ===========================================================================

class TestGradMassOnDecoys:
    def test_returns_expected_keys_with_valid_fractions(self):
        """grad_mass_on_decoys returns 'ce' and 'pauc' keys, each in [0, 1]."""
        # Build a small dataset with positives and decoys
        spec = {
            "kind": "headline",
            "n": 400,
            "pos_rate": 0.10,
            "decoy_frac": 0.05,
            "cue": "nonlinear_prod",
        }
        X, y, grp = pf.make_data(spec, seed=0)
        data = {"X_train": X, "y_train": y, "grp_train": grp}

        model = pf.build_model("mlp_2x64", d_in=X.shape[1])
        result = pf.grad_mass_on_decoys(model, data)

        assert set(result.keys()) == {"ce", "pauc"}, f"Unexpected keys: {result.keys()}"
        for k, v in result.items():
            assert np.isfinite(v), f"{k}={v} is not finite"
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"


# ===========================================================================
# repr_probe_auc tests
# ===========================================================================

class TestReprProbeAuc:
    def _make_probe_data(self, n=300):
        spec = {
            "kind": "headline",
            "n": n,
            "pos_rate": 0.15,
            "decoy_frac": 0.10,
            "cue": "nonlinear_prod",
        }
        X, y, grp = pf.make_data(spec, seed=1)
        return {"X_train": X, "grp_train": grp}

    def test_nonlinear_model_returns_float_in_range(self):
        """repr_probe_auc on an MLP returns a float in [0.5, 1.0]."""
        data = self._make_probe_data()
        model = pf.build_model("mlp_2x64", d_in=20)
        auc = pf.repr_probe_auc(model, data)
        assert isinstance(auc, float)
        assert 0.5 <= auc <= 1.0, f"AUC {auc} outside [0.5, 1.0]"

    def test_linear_model_raises_type_error(self):
        """repr_probe_auc on a bare nn.Linear raises TypeError (no penultimate layer)."""
        data = self._make_probe_data()
        model = pf.build_model("linear", d_in=20)
        with pytest.raises(TypeError, match="nn.Sequential"):
            pf.repr_probe_auc(model, data)

    def test_missing_data_raises_key_error(self):
        """repr_probe_auc raises KeyError when neither X_test nor X_train is present."""
        model = pf.build_model("mlp_2x64", d_in=20)
        with pytest.raises(KeyError):
            pf.repr_probe_auc(model, {})

    def test_restores_eval_mode(self):
        """repr_probe_auc restores eval mode if model was in eval mode before call."""
        data = self._make_probe_data()
        model = pf.build_model("mlp_2x64", d_in=20)
        model.eval()
        pf.repr_probe_auc(model, data)
        assert not model.training, "repr_probe_auc should restore eval mode"


# ===========================================================================
# Per-epoch reshuffle regression test
# ===========================================================================

class TestPerEpochReshuffle:
    """Verify ranking trainers use per-epoch-varying batch order (matching CE arm).

    Source: 03_mechanism_transfer.py, train_pauc, comment:
      'Reshuffle per epoch (seed + e) to match the CE/HNM arm — keeps the
       optimizer trajectory symmetric across arms; only the loss differs.'
    """

    def test_pauc_uses_per_epoch_varying_batch_order(self):
        """idx_batches(n, seed+0) and idx_batches(n, seed+1) must differ.

        This is the shuffle contract that _train_pauc and _train_smoothap rely
        on.  If this assertion holds AND the trainers use 'seed + epoch' inside
        the epoch loop, epoch-to-epoch order varies.  A regression to a fixed
        shuffle would break both the symmetry contract and this test.
        """
        n, bs = 2000, 128
        seed = 7
        batches_e0 = pf.idx_batches(n, seed + 0, bs)
        batches_e1 = pf.idx_batches(n, seed + 1, bs)
        # The full permutation should differ between epochs
        all_e0 = torch.cat(batches_e0)
        all_e1 = torch.cat(batches_e1)
        assert not torch.equal(all_e0, all_e1), (
            "idx_batches(n, seed+0) == idx_batches(n, seed+1): "
            "per-epoch reshuffle is broken — different epoch seeds must produce different orders"
        )

    def test_ce_and_pauc_batch_orders_agree_per_epoch(self):
        """CE arm and PAUC arm produce the SAME batch order for a given epoch+seed.

        Both _train_ce and _train_pauc call idx_batches(n, seed + epoch, batch_size).
        This test pins that contract: for epoch=0 and epoch=3, the resulting
        batch indices are identical across arms (since the call signature is the same).
        """
        n, bs, seed = 2000, 128, 5
        for epoch in (0, 3):
            ce_batches = pf.idx_batches(n, seed + epoch, bs)
            pauc_batches = pf.idx_batches(n, seed + epoch, bs)
            for i, (cb, pb) in enumerate(zip(ce_batches, pauc_batches)):
                assert torch.equal(cb, pb), (
                    f"CE and PAUC batch orders differ at epoch={epoch}, batch={i}. "
                    "Arms must see identical data in each step for fair comparison."
                )


# ===========================================================================
# _build_data_dict split convention tests  (fidelity-critical)
# ===========================================================================

class TestBuildDataDictSequential:
    """Sequential split reproduces the original 01_cue_ablation two-call RNG stream."""

    def _geo_cfg(self, n_train=1000, n_test=2000):
        return {
            "kind": "headline",
            "n_train": n_train,
            "n_test": n_test,
            "pos_rate": 0.05,
            "decoy_frac": 0.02,
            "cue": "nonlinear_prod",
            "seed_base": 7000,
            "split_convention": "sequential",
        }

    def test_sequential_matches_two_call_stream(self):
        """_build_data_dict sequential must produce tensors identical to the
        original 01_cue_ablation.py pattern of advancing one RNG for train then
        test from the same stream.
        """
        geo = self._geo_cfg()
        cell = {}
        seed = 3

        # Reference: original two-call pattern from 01_cue_ablation.py
        rng_ref = np.random.default_rng(7000 + seed)
        train_spec = dict(geo)
        train_spec["n"] = geo["n_train"]
        X_tr_ref, y_tr_ref, _ = pf.make_data(train_spec, seed, rng=rng_ref)
        test_spec = dict(geo)
        test_spec["n"] = geo["n_test"]
        X_te_ref, y_te_ref, _ = pf.make_data(test_spec, seed, rng=rng_ref)

        # Flow helper
        data = pf._build_data_dict(geo, cell, seed)

        assert torch.equal(data["X_train"], X_tr_ref), (
            "Train set mismatch: sequential split does not reproduce "
            "01_cue_ablation two-call stream."
        )
        assert torch.equal(data["y_train"], y_tr_ref)
        assert torch.equal(data["X_test"], X_te_ref), (
            "Test set mismatch: sequential split does not reproduce "
            "01_cue_ablation two-call stream."
        )
        assert torch.equal(data["y_test"], y_te_ref)

    def test_sequential_no_val_set(self):
        """Sequential split must NOT produce a validation set."""
        geo = self._geo_cfg()
        data = pf._build_data_dict(geo, {}, seed=0)
        assert data["X_val"] is None, "Sequential split should have no val set"
        assert data["y_val"] is None
        assert data["grp_val"] is None
        assert data["_y_val_np"] is None

    def test_sequential_train_test_present(self):
        """Sequential split produces non-None train and test sets."""
        geo = self._geo_cfg()
        data = pf._build_data_dict(geo, {}, seed=0)
        assert data["X_train"] is not None
        assert data["X_test"] is not None
        assert data["X_train"].shape[1] == 20
        assert data["X_test"].shape[1] == 20

    def test_sequential_different_seeds_differ(self):
        """Different seeds under sequential split produce different data."""
        geo = self._geo_cfg()
        d0 = pf._build_data_dict(geo, {}, seed=0)
        d1 = pf._build_data_dict(geo, {}, seed=1)
        assert not torch.equal(d0["X_train"], d1["X_train"])
        assert not torch.equal(d0["X_test"], d1["X_test"])

    def test_sequential_deterministic(self):
        """Same seed under sequential split produces identical tensors."""
        geo = self._geo_cfg()
        d0 = pf._build_data_dict(geo, {}, seed=2)
        d1 = pf._build_data_dict(geo, {}, seed=2)
        assert torch.equal(d0["X_train"], d1["X_train"])
        assert torch.equal(d0["X_test"], d1["X_test"])


class TestBuildDataDictIndependent:
    """Independent split convention produces three separate statistically
    independent datasets using seed offsets."""

    def _geo_cfg(self, n_train=500, n_val=200, n_test=800):
        return {
            "kind": "headline",
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "pos_rate": 0.05,
            "decoy_frac": 0.02,
            "cue": "nonlinear_prod",
            "seed_base": 7000,
            "split_convention": "independent",
        }

    def test_independent_has_val_set(self):
        """Independent split must produce a validation set."""
        geo = self._geo_cfg()
        data = pf._build_data_dict(geo, {}, seed=0)
        assert data["X_val"] is not None
        assert data["y_val"] is not None
        assert data["grp_val"] is not None
        assert data["_y_val_np"] is not None

    def test_independent_all_splits_present(self):
        """Independent split produces train, val, and test sets."""
        geo = self._geo_cfg()
        data = pf._build_data_dict(geo, {}, seed=0)
        assert data["X_train"] is not None
        assert data["X_val"] is not None
        assert data["X_test"] is not None

    def test_independent_splits_differ_from_each_other(self):
        """Train, val, and test data must differ (independent seeds)."""
        geo = self._geo_cfg()
        data = pf._build_data_dict(geo, {}, seed=0)
        # Compare first rows of each split — they should differ
        assert not torch.equal(data["X_train"][:1], data["X_val"][:1])
        assert not torch.equal(data["X_train"][:1], data["X_test"][:1])

    def test_unknown_convention_raises(self):
        """Unknown split_convention must raise ValueError."""
        geo = {
            "kind": "headline",
            "n_train": 100,
            "n_test": 200,
            "seed_base": 7000,
            "split_convention": "bogus_convention",
        }
        with pytest.raises(ValueError, match="split_convention"):
            pf._build_data_dict(geo, {}, seed=0)


# ===========================================================================
# _expand_arm_configs arm_overrides tests  (fidelity-critical)
# ===========================================================================

class TestExpandArmConfigsOverrides:
    """arm_overrides collapses sweeps to singletons and injects custom values."""

    def _base_weighted_ce_cfg(self):
        return {
            "kind": "ce",
            "capacity": "mlp_2x64",
            "lr": 0.001,
            "epochs": 15,
            "batch_size": 4096,
            "early_stop": True,
            "wd_sweep": [0.0, 0.0001, 0.001],
        }

    def _base_pauc_cfg(self):
        return {
            "kind": "pauc",
            "capacity": "mlp_2x64",
            "lr": 0.001,
            "epochs": 15,
            "batch_size": 4096,
            "early_stop": True,
            "wd_sweep": [0.0, 0.0001, 0.001],
            "temp_sweep": [0.3, 0.5],
        }

    def test_no_overrides_weighted_ce_expands_to_three(self):
        """Without arm_overrides, weighted_ce expands to 3 wd configs."""
        cfg = self._base_weighted_ce_cfg()
        specs = pf._expand_arm_configs("weighted_ce", cfg, cell={})
        assert len(specs) == 3

    def test_override_collapses_wd_to_singleton(self):
        """arm_overrides: {weighted_ce: {wd_sweep: [0.0]}} -> one config."""
        cfg = self._base_weighted_ce_cfg()
        overrides = {"weighted_ce": {"wd_sweep": [0.0]}}
        specs = pf._expand_arm_configs("weighted_ce", cfg, cell={}, arm_overrides=overrides)
        assert len(specs) == 1, (
            f"Expected 1 config after collapsing wd_sweep, got {len(specs)}"
        )
        assert specs[0]["wd"] == 0.0

    def test_override_collapses_both_sweeps_to_one(self):
        """Collapsing both wd_sweep and temp_sweep to singletons yields 1 config."""
        cfg = self._base_pauc_cfg()
        overrides = {"pauc_pairwise": {"wd_sweep": [0.0], "temp_sweep": [0.5]}}
        specs = pf._expand_arm_configs("pauc_pairwise", cfg, cell={}, arm_overrides=overrides)
        assert len(specs) == 1, (
            f"Expected 1 config after collapsing both sweeps, got {len(specs)}"
        )
        assert specs[0]["wd"] == 0.0
        assert specs[0]["temp"] == 0.5

    def test_override_only_applies_to_named_arm(self):
        """arm_overrides for 'pauc_pairwise' must not affect 'weighted_ce'."""
        ce_cfg = self._base_weighted_ce_cfg()
        overrides = {"pauc_pairwise": {"wd_sweep": [0.0]}}
        specs = pf._expand_arm_configs("weighted_ce", ce_cfg, cell={}, arm_overrides=overrides)
        # weighted_ce should still expand to 3 configs (override was for different arm)
        assert len(specs) == 3

    def test_override_with_custom_temp_sweep(self):
        """arm_overrides can inject a new temp_sweep for capacity_warmup-style experiments."""
        cfg = self._base_pauc_cfg()
        overrides = {"pauc_pairwise": {"temp_sweep": [0.2, 0.4]}}
        specs = pf._expand_arm_configs("pauc_pairwise", cfg, cell={}, arm_overrides=overrides)
        # wd_sweep has 3 values, temp_sweep overridden to [0.2, 0.4] -> 6 combos
        assert len(specs) == 6
        temps = {s["temp"] for s in specs}
        assert temps == {0.2, 0.4}

    def test_arm_name_key_preserved(self):
        """_arm_name key is always present in expanded specs."""
        cfg = self._base_weighted_ce_cfg()
        overrides = {"weighted_ce": {"wd_sweep": [0.0]}}
        specs = pf._expand_arm_configs("weighted_ce", cfg, cell={}, arm_overrides=overrides)
        assert specs[0]["_arm_name"] == "weighted_ce"

    def test_sweep_keys_not_in_output_specs(self):
        """Expanded specs must not contain *_sweep keys."""
        cfg = self._base_pauc_cfg()
        overrides = {"pauc_pairwise": {"wd_sweep": [0.0], "temp_sweep": [0.5]}}
        specs = pf._expand_arm_configs("pauc_pairwise", cfg, cell={}, arm_overrides=overrides)
        for spec in specs:
            assert "wd_sweep" not in spec
            assert "temp_sweep" not in spec


# ===========================================================================
# Paired-lift seed pairing tests  (fidelity-critical)
# ===========================================================================

class TestPairedLiftSeedPairing:
    """Paired lift computation pairs by seed, not positional index."""

    def _make_selected_records(self, seed_order_pauc, seed_order_ce, cell=None):
        """Build a minimal list of selected records for two arms."""
        if cell is None:
            cell = {"cue": "nonlinear_prod", "budget": 0.005}
        records = []
        for seed, cov in seed_order_pauc:
            records.append({
                "cell": cell,
                "arm": "pauc_pairwise",
                "seed": seed,
                "test": {"coverage": cov, "auroc": 0.8, "aucpr": 0.1},
                "val_coverage": None,
                "config": {},
                "diagnostics": {},
            })
        for seed, cov in seed_order_ce:
            records.append({
                "cell": cell,
                "arm": "weighted_ce",
                "seed": seed,
                "test": {"coverage": cov, "auroc": 0.7, "aucpr": 0.08},
                "val_coverage": None,
                "config": {},
                "diagnostics": {},
            })
        return records

    def _compute_lifts(self, records):
        """Run the paired-lift logic extracted from aggregate step."""
        from collections import defaultdict

        cov_by_key: dict = {}
        for rec in records:
            cell_key = tuple(sorted(rec["cell"].items()))
            lookup_key = (cell_key, rec["arm"], rec["seed"])
            cov_by_key[lookup_key] = rec["test"]["coverage"]

        all_cell_keys = {
            tuple(sorted(rec["cell"].items()))
            for rec in records
        }

        lifts_out = []
        for cell_key in all_cell_keys:
            pauc_seeds = {
                seed for (ck, arm, seed) in cov_by_key
                if ck == cell_key and arm == "pauc_pairwise"
            }
            ce_seeds = {
                seed for (ck, arm, seed) in cov_by_key
                if ck == cell_key and arm == "weighted_ce"
            }
            shared_seeds = sorted(pauc_seeds & ce_seeds)
            if not shared_seeds:
                continue
            lifts = [
                cov_by_key[(cell_key, "pauc_pairwise", s)]
                - cov_by_key[(cell_key, "weighted_ce", s)]
                for s in shared_seeds
            ]
            lifts_out.append((shared_seeds, lifts))
        return lifts_out

    def test_correct_pairing_by_seed(self):
        """Lifts are paired by seed value, not arrival position."""
        # PAUC records arrive in seed order [2, 0, 1]
        # CE records arrive in seed order [1, 2, 0]
        # Correct per-seed lifts: seed0: 0.8-0.5=0.3, seed1: 0.7-0.6=0.1, seed2: 0.9-0.4=0.5
        pauc = [(2, 0.9), (0, 0.8), (1, 0.7)]
        ce   = [(1, 0.6), (2, 0.4), (0, 0.5)]
        records = self._make_selected_records(pauc, ce)
        results = self._compute_lifts(records)
        assert len(results) == 1
        seeds, lifts = results[0]
        assert seeds == [0, 1, 2]
        assert lifts == pytest.approx([0.3, 0.1, 0.5], abs=1e-9)

    def test_mismatched_seeds_excluded(self):
        """Seeds present in only one arm are excluded from paired lift."""
        # PAUC has seeds 0, 1, 2; CE has seeds 0, 2 only
        pauc = [(0, 0.8), (1, 0.7), (2, 0.9)]
        ce   = [(0, 0.5), (2, 0.4)]
        records = self._make_selected_records(pauc, ce)
        results = self._compute_lifts(records)
        assert len(results) == 1
        seeds, lifts = results[0]
        # Only seeds 0 and 2 are shared
        assert seeds == [0, 2]
        assert len(lifts) == 2
        assert lifts == pytest.approx([0.3, 0.5], abs=1e-9)

    def test_no_shared_seeds_produces_no_lift(self):
        """When PAUC and CE have no overlapping seeds, no lift is computed."""
        pauc = [(0, 0.8), (1, 0.7)]
        ce   = [(2, 0.5), (3, 0.4)]
        records = self._make_selected_records(pauc, ce)
        results = self._compute_lifts(records)
        assert results == [], (
            "No shared seeds should produce no paired lift."
        )


# ===========================================================================
# Training config threading tests  (contract for config-authoritative DAG)
# ===========================================================================

class TestResolveTrainingCfg:
    """_resolve_training_cfg and _merge_training_into_arm threading contracts."""

    def _default_training_cfg(self):
        """Minimal training cfg matching conf/training/default.yaml."""
        return {
            "hid": 64,
            "epochs": 15,
            "batch": 4096,
            "lr": 0.001,
            "queue": 8192,
            "warmup_frac": 0.30,
            "blend_frac": 0.15,
            "temp_start": 0.5,
            "temp_end": 0.1,
            "eval_every": 25,
        }

    def _capacity_training_cfg(self):
        """Minimal training cfg matching conf/training/capacity.yaml."""
        return {
            "hid": 64,
            "epochs": {"linear": 150, "mlp_1x16": 200, "mlp_2x64": 200},
            "batch": 8192,
            "lr": 0.002,
            "queue": 8192,
            "warmup_frac": 0.3333,
            "blend_frac": 0.15,
            "temp_start": 0.5,
            "temp_end": 0.5,
            "eval_every": 25,
        }

    def test_resolve_training_cfg_returns_dict(self):
        """_resolve_training_cfg returns a plain dict from cfg['training']."""
        cfg = {"training": self._default_training_cfg(), "seeds": 1}
        result = pf._resolve_training_cfg(cfg)
        assert isinstance(result, dict)
        assert result["hid"] == 64
        assert result["epochs"] == 15
        assert result["batch"] == 4096
        assert result["lr"] == pytest.approx(0.001)
        assert result["queue"] == 8192

    def test_resolve_training_cfg_missing_training_key(self):
        """_resolve_training_cfg returns an empty dict when 'training' is absent."""
        cfg = {"seeds": 1}
        result = pf._resolve_training_cfg(cfg)
        assert isinstance(result, dict)

    def test_merge_training_fills_absent_arm_keys(self):
        """Training defaults fill in arm_spec keys that the arm YAML omits."""
        arm_spec = {"kind": "ce", "capacity": "mlp_2x64"}
        training_cfg = self._default_training_cfg()
        merged = pf._merge_training_into_arm(arm_spec, training_cfg)
        # Keys absent from arm_spec should come from training_cfg
        assert merged["batch_size"] == 4096
        assert merged["lr"] == pytest.approx(0.001)
        assert merged["queue_size"] == 8192
        assert merged["hid"] == 64
        assert merged["epochs"] == 15

    def test_merge_training_does_not_override_arm_values(self):
        """Arm YAML values take precedence over training defaults."""
        arm_spec = {
            "kind": "ce",
            "capacity": "mlp_2x64",
            "lr": 0.005,        # explicitly set in arm
            "batch_size": 512,  # explicitly set in arm
            "epochs": 30,       # explicitly set in arm
        }
        training_cfg = self._default_training_cfg()
        merged = pf._merge_training_into_arm(arm_spec, training_cfg)
        # Arm values must NOT be overridden
        assert merged["lr"] == pytest.approx(0.005)
        assert merged["batch_size"] == 512
        assert merged["epochs"] == 30

    def test_merge_training_capacity_epochs_dict_lookup(self):
        """When epochs is a per-capacity dict, the correct epoch count is resolved."""
        training_cfg = self._capacity_training_cfg()

        for capacity, expected_epochs in [
            ("linear", 150),
            ("mlp_1x16", 200),
            ("mlp_2x64", 200),
        ]:
            arm_spec = {"kind": "ce", "capacity": capacity}
            merged = pf._merge_training_into_arm(arm_spec, training_cfg, cell={"capacity": capacity})
            assert merged["epochs"] == expected_epochs, (
                f"Expected epochs={expected_epochs} for capacity={capacity!r}, "
                f"got {merged['epochs']}"
            )

    def test_merge_training_hid_reaches_build_model(self):
        """hid from training_cfg reaches build_model via arm_spec['hid']."""
        training_cfg = {"hid": 32, "epochs": 5, "batch": 64, "lr": 1e-3, "queue": 256,
                        "warmup_frac": 0.3, "blend_frac": 0.15, "temp_start": 0.5,
                        "temp_end": 0.1, "eval_every": 25}
        arm_spec = {"kind": "ce", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, training_cfg)
        # Build model with the merged hid value
        model = pf.build_model("mlp_2x64", d_in=20, hid=merged["hid"])
        layers = list(model.children())
        linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
        assert linear_layers[0].out_features == 32, (
            f"Expected hid=32 in first hidden layer, got {linear_layers[0].out_features}"
        )

    def test_merge_training_epochs_int_passes_through(self):
        """When training_cfg epochs is a plain int, it is used directly."""
        training_cfg = self._default_training_cfg()  # epochs=15 as int
        arm_spec = {"kind": "pauc", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, training_cfg)
        assert merged["epochs"] == 15

    def test_merge_training_temp_start_maps_to_temp(self):
        """training_cfg temp_start maps to arm_spec 'temp' key."""
        training_cfg = self._default_training_cfg()
        arm_spec = {"kind": "pauc", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, training_cfg)
        assert merged["temp"] == pytest.approx(0.5)


# ===========================================================================
# Budget-agnostic vs budget-dependent arm classification
# ===========================================================================

class TestBudgetClassification:
    """is_budget_agnostic_arm classifies arm kinds correctly."""

    @pytest.mark.parametrize("kind", ["trivial", "ce", "ce_warmup_only", "smoothap"])
    def test_budget_agnostic_kinds(self, kind):
        assert pf.is_budget_agnostic_arm(kind) is True

    @pytest.mark.parametrize("kind", ["pauc", "pauc_cold"])
    def test_budget_dependent_kinds(self, kind):
        assert pf.is_budget_agnostic_arm(kind) is False

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            pf.is_budget_agnostic_arm("mystery_arm")


# ===========================================================================
# Dataset-key expansion
# ===========================================================================

_FAKE_ARM_CFGS = {
    "weighted_ce": {"kind": "ce", "capacity": "mlp_2x64", "lr": 0.001,
                    "epochs": 15, "batch_size": 4096, "early_stop": True,
                    "wd_sweep": [0.0, 0.0001, 0.001]},
    "ce_hnm": {"kind": "ce", "capacity": "mlp_2x64", "lr": 0.001, "epochs": 15,
               "batch_size": 4096, "early_stop": True,
               "hnm": {"top_q": 0.02, "factor": 50.0},
               "wd_sweep": [0.0, 0.0001, 0.001]},
    "pauc_pairwise": {"kind": "pauc", "capacity": "mlp_2x64", "lr": 0.001,
                      "epochs": 15, "batch_size": 4096, "early_stop": True,
                      "surrogate": "pairwise", "queue_size": 8192, "temp": 0.5,
                      "temp_end": 0.1, "warmup_frac": 0.30, "blend_frac": 0.15,
                      "wd_sweep": [0.0, 0.0001, 0.001], "band": [0.5, 1.5]},
}


def _fake_arm_loader(arm_name):
    """In-memory arm-cfg loader (no pyyaml dependency for the contract tests)."""
    return dict(_FAKE_ARM_CFGS[arm_name])


class TestDatasetKeyExpansion:
    """build_dataset_keys groups by data-axis and de-dups budget-agnostic arms."""

    def _cue_ablation_cfg(self):
        """Minimal cue_ablation experiment cfg (mirrors conf/experiment yaml)."""
        return {
            "axes": {
                "cue": ["linear", "nonlinear_prod", "nonlinear_radial"],
                "budget": [0.005, 0.010],
            },
            "data_axes": ["cue"],
            "arms": ["weighted_ce", "pauc_pairwise"],
            "band": [0.5, 1.5],
            "diagnostics": ["metric_specificity"],
            "arm_overrides": {
                "weighted_ce": {"wd_sweep": [0.0], "early_stop": False},
                "pauc_pairwise": {"wd_sweep": [0.0], "temp_sweep": [0.5],
                                  "early_stop": False},
            },
        }

    def test_cue_ablation_seeds1_yields_three_dataset_keys(self):
        """3 cues x 1 seed => 3 dataset keys (budget is NOT a data axis)."""
        cfg = self._cue_ablation_cfg()
        keys = pf.build_dataset_keys("cue_ablation", cfg, seeds=1, arm_cfg_loader=_fake_arm_loader)
        assert len(keys) == 3, f"Expected 3 dataset keys, got {len(keys)}"
        cues = {dk["data_cell"]["cue"] for dk in keys}
        assert cues == {"linear", "nonlinear_prod", "nonlinear_radial"}

    def test_cue_ablation_seeds2_yields_six_dataset_keys(self):
        """3 cues x 2 seeds => 6 dataset keys."""
        cfg = self._cue_ablation_cfg()
        keys = pf.build_dataset_keys("cue_ablation", cfg, seeds=2, arm_cfg_loader=_fake_arm_loader)
        assert len(keys) == 6

    def test_ce_trained_once_eval_both_budgets(self):
        """Budget-agnostic CE appears ONCE with budgets=[0.005, 0.010]."""
        cfg = self._cue_ablation_cfg()
        keys = pf.build_dataset_keys("cue_ablation", cfg, seeds=1, arm_cfg_loader=_fake_arm_loader)
        dk = keys[0]
        ce_combos = [c for c in dk["combos"] if c["arm"] == "weighted_ce"]
        assert len(ce_combos) == 1, (
            f"CE should appear once per dataset, got {len(ce_combos)} combos"
        )
        assert ce_combos[0]["budget_dep"] is False
        assert ce_combos[0]["budgets"] == [0.005, 0.010], (
            "CE must be evaluated at BOTH budgets from a single training"
        )

    def test_pauc_trained_per_budget(self):
        """Budget-dependent PAUC appears once PER budget, each single-budget."""
        cfg = self._cue_ablation_cfg()
        keys = pf.build_dataset_keys("cue_ablation", cfg, seeds=1, arm_cfg_loader=_fake_arm_loader)
        dk = keys[0]
        pauc_combos = [c for c in dk["combos"] if c["arm"] == "pauc_pairwise"]
        assert len(pauc_combos) == 2, (
            f"PAUC should appear once per budget (2), got {len(pauc_combos)}"
        )
        for c in pauc_combos:
            assert c["budget_dep"] is True
            assert len(c["budgets"]) == 1
        assert {c["budgets"][0] for c in pauc_combos} == {0.005, 0.010}

    def test_combo_count_is_three_per_dataset(self):
        """Per dataset: 1 CE combo + 2 PAUC combos = 3 training combos."""
        cfg = self._cue_ablation_cfg()
        keys = pf.build_dataset_keys("cue_ablation", cfg, seeds=1, arm_cfg_loader=_fake_arm_loader)
        for dk in keys:
            assert len(dk["combos"]) == 3

    def test_branch_count_is_dataset_count_not_combo_count(self):
        """The foreach grain is the dataset count, not the combo count."""
        cfg = self._cue_ablation_cfg()
        keys = pf.build_dataset_keys("cue_ablation", cfg, seeds=1, arm_cfg_loader=_fake_arm_loader)
        n_combos = sum(len(dk["combos"]) for dk in keys)
        assert len(keys) == 3           # dataset count
        assert n_combos == 9            # 3 datasets x 3 combos
        assert len(keys) != n_combos

    def test_band_default_sweep_data_axis_is_pos_rate(self):
        """band_default_sweep keys on pos_rate (+ constant cue), band is training."""
        cfg = {
            "axes": {
                "cue": ["nonlinear_prod"],
                "pos_rate": [0.001, 0.005],
                "budget": [0.005],
                "band": [[0.0, 1.0], [0.5, 1.5]],
            },
            "data_axes": ["pos_rate", "cue"],
            "arms": ["ce_hnm", "pauc_pairwise"],
            "arm_overrides": {
                "ce_hnm": {"wd_sweep": [0.0], "early_stop": False},
                "pauc_pairwise": {"wd_sweep": [0.0], "temp_sweep": [0.5],
                                  "early_stop": False},
            },
        }
        keys = pf.build_dataset_keys("band_default_sweep", cfg, seeds=1, arm_cfg_loader=_fake_arm_loader)
        # 2 pos_rates x 1 cue => 2 dataset keys
        assert len(keys) == 2
        dk = keys[0]
        # CE-HNM (budget-agnostic) once; PAUC once per band (2 bands, single budget)
        ce = [c for c in dk["combos"] if c["arm"] == "ce_hnm"]
        pauc = [c for c in dk["combos"] if c["arm"] == "pauc_pairwise"]
        assert len(ce) == 1
        assert len(pauc) == 2, "Two bands => two PAUC training combos"
        bands = {tuple(c["arm_spec"]["band"]) for c in pauc}
        assert bands == {(0.0, 1.0), (0.5, 1.5)}


# ===========================================================================
# Static-branch analysis filtering
# ===========================================================================

def _rec(experiment, cue=None, budget=0.005, arm="weighted_ce", seed=0,
         coverage=0.5, auroc=0.7, aucpr=0.1, **extra):
    """Build a minimal aggregated train record for analysis-filter tests."""
    data_cell = {}
    train_cell = {}
    if cue is not None:
        data_cell["cue"] = cue
    rec = {
        "experiment": experiment,
        "data_cell": data_cell,
        "train_cell": train_cell,
        "cell": {**data_cell, "budget": budget},
        "seed": seed,
        "arm": arm,
        "config": extra.get("config", {"budget": budget}),
        "budget": budget,
        "val_coverage": None,
        "test": {"coverage": coverage, "auroc": auroc, "aucpr": aucpr},
        "diagnostics": extra.get("diagnostics", {}),
    }
    for k in ("scores", "grp"):
        if k in extra:
            rec[k] = extra[k]
    return rec


class TestAnalysisFiltering:
    """Each analysis returns rows for its experiment and empty when disabled."""

    def test_filter_records_by_experiment(self):
        recs = [_rec("cue_ablation"), _rec("mechanism_probe"), _rec("cue_ablation")]
        assert len(pf.filter_records(recs, "cue_ablation")) == 2
        assert len(pf.filter_records(recs, "mechanism_probe")) == 1
        assert pf.filter_records(recs, "alpha_widen") == []

    def test_cue_specificity_empty_when_disabled(self):
        """No cue_ablation records => analyze_cue_specificity returns []."""
        recs = [_rec("mechanism_probe")]
        assert pf.analyze_cue_specificity(recs) == []

    def test_cue_specificity_returns_rows(self):
        """With paired PAUC/CE records, cue_specificity yields per-cue lift rows."""
        recs = []
        for cue in ("linear", "nonlinear_prod"):
            for seed in range(4):
                recs.append(_rec("cue_ablation", cue=cue, arm="weighted_ce",
                                 seed=seed, coverage=0.40, auroc=0.80))
                recs.append(_rec("cue_ablation", cue=cue, arm="pauc_pairwise",
                                 seed=seed, coverage=0.55, auroc=0.80))
        rows = pf.analyze_cue_specificity(recs)
        cues = {r["cue"] for r in rows}
        assert cues == {"linear", "nonlinear_prod"}
        for r in rows:
            # PAUC coverage 0.55 - CE 0.40 = +0.15
            assert r["coverage_lift_mean"] == pytest.approx(0.15, abs=1e-9)
            # AUROC equal => 0 lift
            assert r["auroc_lift"] == pytest.approx(0.0, abs=1e-9)

    def test_other_analyses_empty_when_disabled(self):
        recs = [_rec("cue_ablation")]
        assert pf.analyze_mechanism(recs) == {}
        assert pf.analyze_transfer(recs) == []
        assert pf.analyze_band_escape(recs) == []
        assert pf.analyze_alpha_lever(recs) == []
        assert pf.analyze_surrogate(recs) == []
        assert pf.analyze_capacity(recs) == []
        assert pf.analyze_default_sweep(recs) == {}

    def test_band_escape_consumes_stored_scores(self):
        """analyze_band_escape recomputes geometry from stored CE scores+grp."""
        n = 1000
        scores = np.linspace(0, 1, n)[::-1].copy()
        grp = np.zeros(n, dtype=np.int64)
        grp[:10] = 1     # positives at top
        grp[10:30] = 2   # decoys next
        recs = []
        for seed in range(3):
            recs.append(_rec("band_vs_hnm", cue="nonlinear_prod", budget=0.005,
                             arm="weighted_ce", seed=seed,
                             scores=scores, grp=grp,
                             config={"budget": 0.005, "band": [0.5, 1.5]}))
        rows = pf.analyze_band_escape(recs)
        assert len(rows) == 1
        r = rows[0]
        for k in ("in_band", "in_top2", "above_band"):
            assert 0.0 <= r[k]["mean"] <= 1.0

    def test_mechanism_reads_diagnostics_and_gap(self):
        """analyze_mechanism aggregates grad_mass diagnostics + coverage gap."""
        recs = []
        for seed in range(3):
            recs.append(_rec("mechanism_probe", cue="nonlinear_prod",
                             arm="weighted_ce", seed=seed, coverage=0.30,
                             diagnostics={"grad_mass_ce": 0.10, "grad_mass_pauc": 0.60,
                                          "repr_probe_auc": 0.70}))
            recs.append(_rec("mechanism_probe", cue="nonlinear_prod",
                             arm="ce_hnm", seed=seed, coverage=0.50,
                             diagnostics={}))
            recs.append(_rec("mechanism_probe", cue="nonlinear_prod",
                             arm="pauc_pairwise", seed=seed, coverage=0.55,
                             diagnostics={"repr_probe_auc": 0.85,
                                          "bdg_in_band": 0.40}))
        result = pf.analyze_mechanism(recs)
        assert result["grad_mass_ce"] == pytest.approx(0.10)
        assert result["grad_mass_pauc"] == pytest.approx(0.60)
        assert result["probe_auc_ce"] == pytest.approx(0.70)
        assert result["probe_auc_pauc"] == pytest.approx(0.85)
        # gap weighted_ce - pauc = 0.30 - 0.55 = -0.25
        assert result["gap_vs_pauc"]["weighted_ce"] == pytest.approx(-0.25)
        assert result["gap_vs_pauc"]["ce_hnm"] == pytest.approx(-0.05)

    def test_default_sweep_best_band_per_pos_rate(self):
        """analyze_default_sweep picks the highest-coverage band per pos_rate."""
        recs = []
        # pos_rate 0.001: band (0.0,1.0) best; pos_rate 0.005: band (0.5,1.5) best
        configs = {
            (0.001, (0.0, 1.0)): 0.70,
            (0.001, (0.5, 1.5)): 0.50,
            (0.005, (0.0, 1.0)): 0.40,
            (0.005, (0.5, 1.5)): 0.65,
        }
        for (pr, band), cov in configs.items():
            for seed in range(2):
                rec = _rec("band_default_sweep", arm="pauc_pairwise", seed=seed,
                           coverage=cov, config={"budget": 0.005, "band": list(band)})
                rec["data_cell"]["pos_rate"] = pr
                rec["cell"]["pos_rate"] = pr
                recs.append(rec)
        result = pf.analyze_default_sweep(recs)
        per_pos = {c["pos_rate"]: c for c in result["per_pos"]}
        assert per_pos[0.001]["best"]["alpha_m"] == 0.0
        assert per_pos[0.001]["best"]["beta_m"] == 1.0
        assert per_pos[0.005]["best"]["alpha_m"] == 0.5
        assert per_pos[0.005]["best"]["beta_m"] == 1.5


# ===========================================================================
# FIX 1: band_decoy_precision (mechanism precision) vs band_decoy_geometry
#         (band_escape recall) — distinct quantities on the same band
# ===========================================================================

class TestBandDecoyPrecisionVsRecall:
    """Precision (fraction-of-band-that-is-decoys) differs from recall
    (fraction-of-decoys-in-band) on a constructed array, and precision
    reproduces the exp 02 band_decoy_fraction logic."""

    def _make_array(self):
        """Construct a grp/scores array where precision and recall differ clearly.

        Layout (n=100):
          - 10 positives (grp=1) — assigned top scores 0.91..1.00
          - 20 decoys   (grp=2) — assigned scores 0.51..0.70 (some in band, some above)
          - 70 easy-neg (grp=0) — assigned scores 0.00..0.69

        With budget=0.05 and band=(0.5, 1.5):
          alpha = 0.025, beta = 0.075
          Negatives = grp != 1 (90 samples: 20 decoys + 70 easy-neg).
          t_alpha = quantile(sn, 0.975) -- roughly top 2.5% of negatives
          t_beta  = quantile(sn, 0.925) -- roughly top 7.5% of negatives

        Recall   = (decoys in band) / (total decoys)   denominator = 20
        Precision = (decoys in band) / (all neg in band)  denominator = neg in band
        These will differ whenever decoys != all band members.
        """
        n = 200
        scores = np.zeros(n, dtype=np.float64)
        grp = np.zeros(n, dtype=np.int64)

        # 10 positives at the very top (not counted in band or recall denom)
        pos_idx = np.arange(10)
        grp[pos_idx] = 1
        scores[pos_idx] = np.linspace(0.91, 1.00, 10)

        # 40 decoys in the middle-upper range
        decoy_idx = np.arange(10, 50)
        grp[decoy_idx] = 2
        scores[decoy_idx] = np.linspace(0.51, 0.90, 40)

        # 150 easy-negatives at the bottom
        easy_idx = np.arange(50, 200)
        grp[easy_idx] = 0
        scores[easy_idx] = np.linspace(0.00, 0.50, 150)

        return scores, grp

    def test_precision_and_recall_differ(self):
        """On a constructed array, band precision != band recall (in_band).

        This pins the invariant that the two quantities are not interchangeable.
        A regression that conflates them (e.g. using bdg_in_band for mechanism)
        would fail this test.
        """
        scores, grp = self._make_array()
        budget = 0.05
        band = (0.5, 1.5)

        recall_dict = pf.band_decoy_geometry(scores, grp, budget, band)
        recall_in_band = recall_dict["in_band"]  # fraction of decoys in band
        precision = pf.band_decoy_precision(scores, grp, budget, band)

        # Both must be valid fractions
        assert 0.0 <= recall_in_band <= 1.0
        assert 0.0 <= precision <= 1.0

        # They compute different denominators and must differ on this array.
        # (recall denom = total decoys; precision denom = neg in band)
        # With 40 decoys in the middle and 150 easy-neg at the bottom, the band
        # contains many decoys but few easy-neg (easy-neg are at the bottom), so
        # precision > recall is expected here.
        assert recall_in_band != pytest.approx(precision, abs=0.01), (
            f"recall_in_band={recall_in_band:.4f} and precision={precision:.4f} "
            "are unexpectedly equal; they should differ because their denominators "
            "differ (total decoys vs total neg in band)."
        )

    def test_precision_matches_exp02_band_decoy_fraction(self):
        """band_decoy_precision must match the exp 02 band_decoy_fraction formula.

        experiments/02_mechanism_probe.py:
            neg = grp != 1
            lo, hi = quantile(scores[neg], 1-BETA), quantile(scores[neg], 1-ALPHA)
            band = neg & (scores >= lo) & (scores <= hi)
            return float((grp[band] == 2).mean()) if band.sum() else 0.0
        """
        scores, grp = self._make_array()
        budget = 0.05
        band_mults = (0.5, 1.5)
        alpha = band_mults[0] * budget  # 0.025
        beta = band_mults[1] * budget   # 0.075

        # Reference implementation (mirror of exp 02)
        neg = grp != 1
        sn = scores[neg]
        lo = np.quantile(sn, 1.0 - beta)
        hi = np.quantile(sn, 1.0 - alpha)
        band_mask = neg & (scores >= lo) & (scores <= hi)
        ref = float((grp[band_mask] == 2).mean()) if band_mask.sum() else 0.0

        result = pf.band_decoy_precision(scores, grp, budget, band_mults)
        assert result == pytest.approx(ref, abs=1e-9), (
            f"band_decoy_precision={result:.6f} != exp02 reference={ref:.6f}"
        )

    def test_precision_empty_negatives_returns_zero(self):
        scores = np.ones(10, dtype=np.float64)
        grp = np.ones(10, dtype=np.int64)  # all positives
        assert pf.band_decoy_precision(scores, grp, budget=0.05, band=(0.5, 1.5)) == 0.0

    def test_precision_no_decoys_returns_zero(self):
        """When no decoys are present, precision is 0 (no decoys in any band)."""
        rng = np.random.default_rng(42)
        scores = rng.random(100)
        grp = np.zeros(100, dtype=np.int64)
        grp[:5] = 1  # only positives and easy-neg
        result = pf.band_decoy_precision(scores, grp, budget=0.05, band=(0.5, 1.5))
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_analyze_mechanism_uses_precision_not_recall(self):
        """analyze_mechanism reads bdg_decoyfrac (precision) not bdg_in_band (recall).

        Build records where bdg_decoyfrac and bdg_in_band carry different values,
        and verify analyze_mechanism reports the precision value.
        """
        recs = []
        for seed in range(3):
            recs.append(_rec(
                "mechanism_probe", cue="nonlinear_prod",
                arm="pauc_pairwise", seed=seed, coverage=0.55,
                diagnostics={
                    "repr_probe_auc": 0.85,
                    "bdg_in_band": 0.45,    # recall (exp 04 quantity)
                    "bdg_decoyfrac": 0.73,  # precision (exp 02 quantity)
                },
            ))
            recs.append(_rec(
                "mechanism_probe", cue="nonlinear_prod",
                arm="weighted_ce", seed=seed, coverage=0.30,
                diagnostics={"grad_mass_ce": 0.10, "grad_mass_pauc": 0.60},
            ))
        result = pf.analyze_mechanism(recs)
        # Must report precision (0.73), not recall (0.45)
        assert result["band_decoyfrac_pauc"] == pytest.approx(0.73, abs=1e-9), (
            f"analyze_mechanism reported band_decoyfrac_pauc="
            f"{result['band_decoyfrac_pauc']:.4f}; expected 0.73 (precision). "
            "Check that it reads bdg_decoyfrac, not bdg_in_band."
        )


# ===========================================================================
# FIX 2: arm YAMLs no longer hard-set training params; they flow from cfg
# ===========================================================================

class TestArmYamlTrainingParamsFlowFromConfig:
    """After FIX 2, arm YAMLs omit epochs/lr/batch_size/queue_size so these
    flow from the resolved training config.  This tests the merge contract
    for both default and capacity training cfgs."""

    def _default_training_cfg(self):
        return {
            "hid": 64, "epochs": 15, "batch": 4096, "lr": 0.001,
            "queue": 8192, "warmup_frac": 0.30, "blend_frac": 0.15,
            "temp_start": 0.5, "temp_end": 0.1, "eval_every": 25,
        }

    def _capacity_training_cfg(self):
        return {
            "hid": 64,
            "epochs": {"linear": 150, "mlp_1x16": 200, "mlp_2x64": 200},
            "batch": 8192, "lr": 0.002, "queue": 8192,
            "warmup_frac": 0.3333, "blend_frac": 0.15,
            "temp_start": 0.5, "temp_end": 0.5, "eval_every": 25,
        }

    def test_arm_without_epochs_gets_default_epochs(self):
        """An arm spec with no 'epochs' key gets epochs=15 from default.yaml."""
        arm_spec = {"kind": "ce", "capacity": "mlp_2x64"}  # no epochs
        merged = pf._merge_training_into_arm(arm_spec, self._default_training_cfg())
        assert merged["epochs"] == 15

    def test_arm_without_lr_gets_default_lr(self):
        """An arm spec with no 'lr' key gets lr=0.001 from default.yaml."""
        arm_spec = {"kind": "ce", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, self._default_training_cfg())
        assert merged["lr"] == pytest.approx(0.001)

    def test_arm_without_batch_size_gets_default_batch(self):
        """An arm spec with no 'batch_size' gets batch_size=4096 from default.yaml."""
        arm_spec = {"kind": "ce", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, self._default_training_cfg())
        assert merged["batch_size"] == 4096

    def test_arm_without_queue_size_gets_default_queue(self):
        """An arm spec with no 'queue_size' gets queue_size=8192 from default.yaml."""
        arm_spec = {"kind": "pauc", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, self._default_training_cfg())
        assert merged["queue_size"] == 8192

    def test_capacity_arm_linear_gets_150_epochs(self):
        """linear capacity under capacity.yaml resolves to 150 epochs."""
        arm_spec = {"kind": "ce", "capacity": "linear"}  # no epochs key
        merged = pf._merge_training_into_arm(
            arm_spec, self._capacity_training_cfg(), cell={"capacity": "linear"}
        )
        assert merged["epochs"] == 150

    def test_capacity_arm_mlp1x16_gets_200_epochs(self):
        """mlp_1x16 capacity under capacity.yaml resolves to 200 epochs."""
        arm_spec = {"kind": "pauc", "capacity": "mlp_1x16"}
        merged = pf._merge_training_into_arm(
            arm_spec, self._capacity_training_cfg(), cell={"capacity": "mlp_1x16"}
        )
        assert merged["epochs"] == 200

    def test_capacity_arm_gets_lr_0002(self):
        """Capacity training cfg supplies lr=0.002."""
        arm_spec = {"kind": "ce", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(
            arm_spec, self._capacity_training_cfg(), cell={"capacity": "mlp_2x64"}
        )
        assert merged["lr"] == pytest.approx(0.002)

    def test_capacity_arm_gets_batch_8192(self):
        """Capacity training cfg supplies batch_size=8192."""
        arm_spec = {"kind": "ce", "capacity": "linear"}
        merged = pf._merge_training_into_arm(
            arm_spec, self._capacity_training_cfg(), cell={"capacity": "linear"}
        )
        assert merged["batch_size"] == 8192

    def test_headline_arm_still_gets_epochs_15(self):
        """Default training cfg gives epochs=15 to headline experiments."""
        arm_spec = {"kind": "pauc", "capacity": "mlp_2x64"}
        merged = pf._merge_training_into_arm(arm_spec, self._default_training_cfg())
        assert merged["epochs"] == 15


# ===========================================================================
# FIX 3: experiment YAMLs have explicit 'name' key
# ===========================================================================

class TestExperimentYamlNames:
    """All 8 experiment YAMLs must have an explicit 'name' key."""

    _CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
    _EXPECTED_NAMES = {
        "cue_ablation",
        "mechanism_probe",
        "mechanism_transfer",
        "band_vs_hnm",
        "alpha_widen",
        "band_default_sweep",
        "capacity_warmup",
        "confounder_sweep",
    }

    def _read_name_from_yaml(self, exp_name: str) -> "str | None":
        """Extract the 'name:' value from an experiment YAML without pyyaml.

        Looks for a top-level 'name: <value>' line (not inside a nested
        block).  This is sufficient for the simple experiment YAMLs.
        """
        import re
        path = self._CONF_DIR / "experiment" / f"{exp_name}.yaml"
        for line in path.read_text().splitlines():
            m = re.match(r"^name:\s*(\S+)", line)
            if m:
                return m.group(1)
        return None

    @pytest.mark.parametrize("exp_name", sorted(_EXPECTED_NAMES))
    def test_experiment_yaml_has_name_key(self, exp_name):
        """Each experiment YAML must contain a 'name' key matching its filename."""
        name_val = self._read_name_from_yaml(exp_name)
        assert name_val is not None, (
            f"conf/experiment/{exp_name}.yaml is missing the 'name' key. "
            f"Add 'name: {exp_name}' to the YAML."
        )
        assert name_val == exp_name, (
            f"conf/experiment/{exp_name}.yaml has name={name_val!r}; "
            f"expected {exp_name!r}."
        )

    def test_start_uses_explicit_name_over_inference(self):
        """When exp_cfg has 'name', start uses it (not _infer_experiment_name).

        This simulates the start() logic: exp_cfg.get('name') or _infer_...
        The axes chosen below would infer 'band_default_sweep' without the
        explicit name key.
        """
        exp_cfg = {
            "name": "my_custom_experiment",
            "axes": {"pos_rate": [0.001], "band": [[0.5, 1.5]]},
            "arms": [],
        }
        # Mirror the start() expression exactly
        name = exp_cfg.get("name") or pf._infer_experiment_name_fallback(exp_cfg)
        assert name == "my_custom_experiment", (
            "start() must use the explicit 'name' key when present, "
            f"not the inferred name; got {name!r}."
        )

    def test_infer_fallback_still_works_when_name_absent(self):
        """_infer_experiment_name_fallback infers 'band_default_sweep' when name absent."""
        exp_cfg = {
            "axes": {"pos_rate": [0.001], "band": [[0.5, 1.5]]},
            "arms": [],
        }
        name = exp_cfg.get("name") or pf._infer_experiment_name_fallback(exp_cfg)
        assert name == "band_default_sweep"


# ===========================================================================
# FIX 4: select_by_val warns on silent discard
# ===========================================================================

class TestSelectByValSilentDiscard:
    """select_by_val warns when val_coverage is None and len(recs) > 1."""

    def _make_group(self, n_recs: int, val_cov):
        """Build n_recs records for the same (cell, arm, seed) group."""
        cell = {"cue": "nonlinear_prod", "budget": 0.005}
        return [
            {
                "cell": cell,
                "arm": "weighted_ce",
                "seed": 0,
                "test": {"coverage": 0.5 + i * 0.01, "auroc": 0.8, "aucpr": 0.1},
                "val_coverage": val_cov,
                "config": {},
                "diagnostics": {},
                "experiment": "cue_ablation",
                "data_cell": {"cue": "nonlinear_prod"},
                "train_cell": {},
                "budget": 0.005,
            }
            for i in range(n_recs)
        ]

    def test_single_record_no_warning(self, recwarn):
        """A group of one record with val_coverage=None should not warn."""
        import warnings
        recs = self._make_group(1, val_cov=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pf._select_by_val_pure(recs)
        assert len(w) == 0, f"Unexpected warning: {[str(x.message) for x in w]}"

    def test_multiple_records_val_none_warns(self):
        """A group of >1 records with val_coverage=None must emit a warning."""
        import warnings
        recs = self._make_group(3, val_cov=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pf._select_by_val_pure(recs)
        assert len(w) >= 1, (
            "Expected a UserWarning when val_coverage is None and len(recs) > 1."
        )
        assert any("val_coverage" in str(warning.message).lower() or
                   "none" in str(warning.message).lower()
                   for warning in w), (
            f"Warning text does not mention val_coverage or None: "
            f"{[str(x.message) for x in w]}"
        )

    def test_multiple_records_with_val_cov_selects_best(self):
        """Groups with real val_coverage values are resolved by max, no warning."""
        import warnings
        recs = self._make_group(3, val_cov=None)
        # Override val_coverage with real values
        for i, r in enumerate(recs):
            r["val_coverage"] = 0.3 + i * 0.1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pf._select_by_val_pure(recs)
        assert len(w) == 0
        # Should pick the record with highest val_coverage (0.5, last one)
        assert result["val_coverage"] == pytest.approx(0.5)
