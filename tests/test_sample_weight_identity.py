"""
Locks the current (pre-``sample_weight``) numerical behavior of every loss.

``TestUnweightedBitwise`` replays the exact grid ``tests/_golden_sample_weight.py``
used to build ``tests/fixtures/sample_weight_golden.pt`` and asserts bitwise
(``torch.equal``) identity against it. This is the guarantee behind spec S7
"Bitwise unweighted identity"
(``.claude/output/specs/2026-09-22-sample-weight-design.md``): every task in
the ``sample_weight`` plan must keep this test green suite-wide, because none
of them may change what runs when ``sample_weight`` is never supplied on the
call and the queue has never held a weighted row.

The grid is built with the library's CURRENT forward signatures -- no
``sample_weight`` keyword exists yet at capture time -- and this test calls
those same signatures, so it stays valid unchanged both before and after the
``sample_weight`` feature lands.
"""

from __future__ import annotations

import pytest
import torch

from _golden_sample_weight import (
    DTYPES,
    FIXTURE_PATH,
    SEED,
    build_grid,
    run_entry,
)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).rsplit(".", maxsplit=1)[-1]


@pytest.fixture(scope="module")
def golden() -> dict[str, object]:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"{FIXTURE_PATH} is missing. Regenerate on the UNMODIFIED tree with "
            f"'uv run python tests/_golden_sample_weight.py'."
        )
    return torch.load(FIXTURE_PATH, weights_only=True)


class TestUnweightedBitwise:
    """Every (config, dtype, step, field) replays to a bit-identical tensor."""

    def test_fixture_metadata(self, golden):
        assert golden["_meta_seed"] == SEED
        assert isinstance(golden["_meta_torch_version"], str)
        assert golden["_meta_num_configs"] == len(build_grid())
        assert golden["_meta_num_tensor_entries"] > 0

    @pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_str)
    def test_replay_matches_fixture(self, golden, dtype):
        configs = build_grid()
        dtype_str = _dtype_str(dtype)

        expected_keys_for_dtype = {
            k for k in golden if not k.startswith("_meta_") and f"|dtype={dtype_str}|" in k
        }
        checked_keys: set[str] = set()

        for config_index, config in enumerate(configs):
            step_results = run_entry(config, config_index, dtype)
            for step_key, step_result in step_results.items():
                prefix = f"{config.name}|dtype={dtype_str}|step={step_key}"
                for field_name, actual in step_result.items():
                    key = f"{prefix}|{field_name}"
                    assert key in golden, f"fixture is missing key {key!r}"
                    expected = golden[key]
                    assert actual.shape == expected.shape, (
                        f"{key}: shape drifted from the golden fixture "
                        f"({tuple(actual.shape)} vs {tuple(expected.shape)})."
                    )
                    assert actual.dtype == expected.dtype, (
                        f"{key}: dtype drifted from the golden fixture "
                        f"({actual.dtype} vs {expected.dtype})."
                    )
                    assert torch.equal(actual, expected), (
                        f"{key}: replay diverged from the golden fixture -- "
                        f"the unweighted code path changed at the bit level."
                    )
                    checked_keys.add(key)

        # Every fixture entry for this dtype was exercised, and nothing more.
        assert checked_keys == expected_keys_for_dtype
