"""Unit tests for imbalanced_losses._quantile._quantile."""

from __future__ import annotations

import math

import pytest
import torch

from imbalanced_losses import _quantile as quantile_module

DTYPES = pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)


# ---------------------------------------------------------------------------
# _quantile: torch.quantile, with an exact sort-based path above its size limit
# ---------------------------------------------------------------------------

QUANTILE_MODES = ("linear", "lower", "higher", "nearest", "midpoint")
QUANTILE_MODE_PARAM = pytest.mark.parametrize("mode", QUANTILE_MODES)
QUANTILE_SIZES = pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 40, 1001])
# Dyadic levels: q * (n - 1) is exact in float32, so torch.quantile's own
# float32 rank arithmetic and the fallback's float64 rank arithmetic agree.
DYADIC_LEVELS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
# Levels whose float32 product with (n - 1) is rounded by torch.quantile.
INEXACT_LEVELS = [0.05, 0.2, 0.9, 0.995]


def _assert_same_values(actual, expected):
    """Same dtype, shape and device; equal values with NaNs at the same places."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.device == expected.device
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan], expected[~nan])


def _tied_sample(n, dtype, seed=0):
    """``n`` scores rounded to one decimal, so ties are common."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * 3).round(decimals=1).to(dtype)


@pytest.fixture
def always_fallback(monkeypatch):
    """Route every call, whatever its size, through the sort-based path."""
    monkeypatch.setattr(quantile_module, "_QUANTILE_MAX_NUMEL", 0)


class TestQuantileHelper:
    def test_limit_is_the_largest_input_torch_quantile_accepts(self):
        assert quantile_module._QUANTILE_MAX_NUMEL == 2**24

    @DTYPES
    @QUANTILE_MODE_PARAM
    @QUANTILE_SIZES
    def test_at_or_below_the_limit_is_torch_quantile(self, dtype, mode, n):
        x = _tied_sample(n, dtype)
        for q in DYADIC_LEVELS + INEXACT_LEVELS:
            _assert_same_values(
                quantile_module._quantile(x, q, interpolation=mode),
                torch.quantile(x, q, interpolation=mode),
            )

    @DTYPES
    @QUANTILE_MODE_PARAM
    @QUANTILE_SIZES
    def test_fallback_equals_torch_quantile_scalar_q(
        self, always_fallback, dtype, mode, n
    ):
        x = _tied_sample(n, dtype)
        for q in DYADIC_LEVELS:
            expected = torch.quantile(x, q, interpolation=mode)
            actual = quantile_module._quantile(x, q, interpolation=mode)
            assert actual.ndim == 0
            _assert_same_values(actual, expected)

    @DTYPES
    @QUANTILE_MODE_PARAM
    @QUANTILE_SIZES
    def test_fallback_equals_torch_quantile_vector_q(
        self, always_fallback, dtype, mode, n
    ):
        x = _tied_sample(n, dtype)
        q = torch.tensor(DYADIC_LEVELS, dtype=dtype)
        actual = quantile_module._quantile(x, q, interpolation=mode)
        assert actual.shape == q.shape
        _assert_same_values(actual, torch.quantile(x, q, interpolation=mode))

    @QUANTILE_MODE_PARAM
    @QUANTILE_SIZES
    def test_fallback_equals_torch_quantile_float64_any_level(
        self, always_fallback, mode, n
    ):
        # In float64 torch.quantile's rank arithmetic is the fallback's, so
        # every level agrees bit for bit.
        x = _tied_sample(n, torch.float64)
        q = torch.tensor(DYADIC_LEVELS + INEXACT_LEVELS, dtype=torch.float64)
        _assert_same_values(
            quantile_module._quantile(x, q, interpolation=mode),
            torch.quantile(x, q, interpolation=mode),
        )
        for level in q.tolist():
            _assert_same_values(
                quantile_module._quantile(x, level, interpolation=mode),
                torch.quantile(x, level, interpolation=mode),
            )

    @pytest.mark.parametrize("mode", ["lower", "higher", "nearest"])
    @QUANTILE_SIZES
    def test_fallback_float32_any_level_picks_the_float64_rank(
        self, always_fallback, mode, n
    ):
        # torch.quantile rounds the float32 product q * (n - 1) to float32
        # (0.995f * 1000 -> 995.0 where the exact product is 995.0000047),
        # which can pick the neighbouring order statistic.  The fallback keeps
        # the rank in float64, so it equals torch.quantile on the float64
        # input with the same float32-valued level.
        x = _tied_sample(n, torch.float32)
        q = torch.tensor(DYADIC_LEVELS + INEXACT_LEVELS, dtype=torch.float32)
        expected = torch.quantile(x.double(), q.double(), interpolation=mode).float()
        _assert_same_values(quantile_module._quantile(x, q, interpolation=mode), expected)

    @pytest.mark.parametrize("mode", ["linear", "midpoint"])
    @QUANTILE_SIZES
    def test_fallback_float32_any_level_interpolates_the_float64_rank(
        self, always_fallback, mode, n
    ):
        # Interpolation cannot be bitwise equal to torch.quantile in float32
        # for these levels.  "linear": torch's weight is the float32-rounded
        # rank minus its floor, the fallback's is the float64 one; measured
        # over this grid they differ by up to 3.8e-6 in value.  "midpoint":
        # when the rounded rank crosses an integer, torch averages a
        # different pair of order statistics (measured difference up to 0.45
        # here, half the gap between two of them).  Against the float64
        # computation with the same float32-valued level, the fallback is
        # within two float32 ulps (measured 4.8e-7 at magnitude 12).
        x = _tied_sample(n, torch.float32)
        q = torch.tensor(DYADIC_LEVELS + INEXACT_LEVELS, dtype=torch.float32)
        actual = quantile_module._quantile(x, q, interpolation=mode)
        assert actual.dtype == torch.float32
        reference = torch.quantile(x.double(), q.double(), interpolation=mode).float()
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)
        if mode == "linear":
            torch.testing.assert_close(
                actual, torch.quantile(x, q, interpolation=mode), rtol=0, atol=1e-5
            )

    @DTYPES
    @QUANTILE_MODE_PARAM
    @pytest.mark.parametrize("n", [1, 2, 5, 9])
    @pytest.mark.parametrize("nan_at", [0, -1])
    def test_a_nan_makes_every_level_nan(self, always_fallback, dtype, mode, n, nan_at):
        x = _tied_sample(n, dtype)
        x[nan_at] = float("nan")
        q = torch.tensor(DYADIC_LEVELS, dtype=dtype)
        for level in (0.0, 0.5, 1.0):
            actual = quantile_module._quantile(x, level, interpolation=mode)
            assert torch.isnan(actual)
            _assert_same_values(actual, torch.quantile(x, level, interpolation=mode))
        actual = quantile_module._quantile(x, q, interpolation=mode)
        assert torch.isnan(actual).all()
        _assert_same_values(actual, torch.quantile(x, q, interpolation=mode))

    @DTYPES
    @QUANTILE_MODE_PARAM
    def test_infinite_scores_match_torch_quantile(self, always_fallback, dtype, mode):
        x = torch.tensor([float("-inf"), -1.0, 0.0, 2.0, float("inf")], dtype=dtype)
        q = torch.tensor(DYADIC_LEVELS, dtype=dtype)
        _assert_same_values(
            quantile_module._quantile(x, q, interpolation=mode),
            torch.quantile(x, q, interpolation=mode),
        )

    def test_nearest_rounds_half_ranks_to_even_like_torch(self, always_fallback):
        # Ranks 0.5, 1.5, 2.5 and 3.5 of n = 8 (q = k / 14 is exact in float64).
        x = torch.arange(8, dtype=torch.float64)
        q = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64) / 7
        result = quantile_module._quantile(x, q, interpolation="nearest")
        assert result.tolist() == [0.0, 2.0, 2.0, 4.0]
        _assert_same_values(result, torch.quantile(x, q, interpolation="nearest"))

    @DTYPES
    def test_unsorted_input_is_not_mutated_and_nd_input_is_flattened(
        self, always_fallback, dtype
    ):
        x = _tied_sample(12, dtype).reshape(3, 4)
        before = x.clone()
        _assert_same_values(
            quantile_module._quantile(x, 0.5, interpolation="higher"),
            torch.quantile(x, 0.5, interpolation="higher"),
        )
        assert torch.equal(x, before)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_level_outside_unit_interval_raises_like_torch(self, always_fallback, bad):
        x = _tied_sample(5, torch.float32)
        with pytest.raises(RuntimeError):
            torch.quantile(x, bad)
        with pytest.raises(RuntimeError):
            quantile_module._quantile(x, bad)
        with pytest.raises(RuntimeError):
            quantile_module._quantile(x, torch.tensor([0.5, bad]))

    def test_real_size_input_above_the_torch_limit(self):
        # n - 1 = 2**24 + 3 is not a float32 integer: a float32 rank for
        # q = 1.0 rounds to 2**24 + 4, one past the last index, so this size
        # fails unless the rank is formed in float64.
        m = 2**22 + 1
        n = 4 * m
        # Each of 0 .. 2**22 four times, descending (all exact in float32):
        # the order statistic of rank r is r // 4 - 2**22.
        x = torch.arange(m, dtype=torch.float32).repeat_interleave(4).neg_()
        assert x.numel() == n
        with pytest.raises(RuntimeError, match="too large"):
            torch.quantile(x, 0.5)

        def statistic(rank):
            return float(rank // 4 - 2**22)

        levels = [0.0, 0.25, 0.5, 0.995, 1.0]
        # The rank is the float32-valued level times n - 1 in float64.
        ranks = [torch.tensor(q, dtype=torch.float32).item() * (n - 1) for q in levels]
        q = torch.tensor(levels, dtype=torch.float32)
        for mode, pick in (
            ("lower", math.floor), ("higher", math.ceil), ("nearest", round),
        ):
            result = quantile_module._quantile(x, q, interpolation=mode)
            assert result.dtype == torch.float32
            assert result.tolist() == [statistic(pick(r)) for r in ranks], mode
        assert result.tolist()[-1] == 0.0  # q = 1.0 is the last order statistic
        # The fractional rank of 0.995 lies between two integers, so the
        # interpolated value lies between the two picks.
        linear = quantile_module._quantile(x, 0.995, interpolation="linear").item()
        assert statistic(math.floor(ranks[3])) <= linear <= statistic(math.ceil(ranks[3]))
        # Ranks 2**23 + 1 and 2**23 + 2 both hold 2**21 - 2**22.
        assert quantile_module._quantile(x, 0.5, interpolation="midpoint").item() == (
            2**21 - 2**22
        )

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    @QUANTILE_MODE_PARAM
    def test_fallback_on_mps_equals_the_cpu_computation(self, always_fallback, mode):
        # MPS has no float64, so the ranks must be formed on the CPU.
        x = _tied_sample(1001, torch.float32)
        q = torch.tensor(DYADIC_LEVELS + INEXACT_LEVELS, dtype=torch.float32)
        actual = quantile_module._quantile(x.to("mps"), q.to("mps"), interpolation=mode)
        assert actual.device.type == "mps"
        expected = quantile_module._quantile(x, q, interpolation=mode)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        scalar = quantile_module._quantile(x.to("mps"), 0.995, interpolation=mode)
        assert scalar.ndim == 0
        torch.testing.assert_close(
            scalar.cpu(), quantile_module._quantile(x, 0.995, interpolation=mode),
            rtol=0, atol=0,
        )

    def test_always_fallback_routes_around_torch_quantile(
        self, always_fallback, monkeypatch
    ):
        def _forbidden(*args, **kwargs):
            raise AssertionError("torch.quantile was called on the sort route")

        monkeypatch.setattr(torch, "quantile", _forbidden)
        result = quantile_module._quantile(_tied_sample(9, torch.float64), 0.5)
        assert result.ndim == 0
