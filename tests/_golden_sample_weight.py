"""
Golden-fixture builder for the sample_weight bitwise-identity guarantee.

Captures loss values, ``return_per_class`` vectors (where the loss exposes
one), and ``logits.grad`` for a fixed grid of configurations run on the
UNMODIFIED library, before any ``sample_weight`` source change lands.
``tests/test_sample_weight_identity.py::TestUnweightedBitwise`` replays the
exact same grid against the changed tree and asserts bitwise (``torch.equal``)
identity -- this is the guarantee behind spec S7 "Bitwise unweighted
identity" (see ``.claude/output/specs/2026-09-22-sample-weight-design.md``).

The grid covers, per the task-0 brief:

- All 5 public losses (SigmoidFocalLoss, SoftmaxFocalLoss, SmoothAPLoss,
  RecallAtQuantileLoss, PAUCAtBudgetLoss), CPU, float32 and float64.
- SoftmaxFocalLoss: reduction in {mean, mean_positive, sum, none}, with
  ignore_index rows present. SigmoidFocalLoss: reduction in {mean, sum,
  none} (mean_positive does not apply to sigmoid).
- SmoothAPLoss / RecallAtQuantileLoss: queue_size in {0, 64} over 3
  consecutive steps (so queue merge/enqueue paths are exercised), crossed
  with iid_mask present/absent; ignore_index rows present every step.
- PAUCAtBudgetLoss: surrogate in {trapezoid, pairwise}, pos_numerator in
  {pool, live}, budget_basis in {fpr, population}, queue_size in {0, 64},
  iid_mask present/absent -- every value of every axis appears in at least
  one of 8 series (not a full 32-way cross product, to keep the fixture
  small and the capture fast).

Every config is built so every class stays valid (at least one positive,
one negative, and -- for PAUC -- iid negatives in every class) at every
step; ``run_entry`` raises if that guarantee is ever violated, because an
invalid class would inject ``nan`` into a stored tensor and ``torch.equal``
treats ``nan != nan``.

Usage
-----
Regenerate the fixture (only ever run against the unmodified tree, i.e.
before the ``sample_weight`` feature touches any loss under
``src/imbalanced_losses/``)::

    uv run python tests/_golden_sample_weight.py

``tests/test_sample_weight_identity.py`` imports ``build_grid`` and
``run_entry`` from this module so the replay uses the exact same inputs the
capture used -- the fixture stores only outputs, not inputs, to stay small.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import torch

from imbalanced_losses import (
    PAUCAtBudgetLoss,
    RecallAtQuantileLoss,
    SigmoidFocalLoss,
    SmoothAPLoss,
    SoftmaxFocalLoss,
)

SEED = 42
DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float64)
FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_weight_golden.pt"

NUM_CLASSES = 4
N_ROWS = 64
N_IGNORED_ROWS = 4
IGNORE_INDEX = -100
QUEUE_SIZE_SMALL = 64
QUEUE_STEPS = 3

GRID_DESCRIPTION = (
    "5 losses (SigmoidFocalLoss, SoftmaxFocalLoss, SmoothAPLoss, "
    "RecallAtQuantileLoss, PAUCAtBudgetLoss) x CPU x float32/float64. "
    "SigmoidFocalLoss: reduction in {mean,sum,none}. SoftmaxFocalLoss: "
    "reduction in {mean,mean_positive,sum,none}, ignore_index rows present. "
    "SmoothAPLoss / RecallAtQuantileLoss: queue_size in {0,64} over 3 "
    "consecutive steps, crossed with iid_mask present/absent, ignore_index "
    "rows present every step. PAUCAtBudgetLoss: 8 series spanning "
    "surrogate in {trapezoid,pairwise}, pos_numerator in {pool,live}, "
    "budget_basis in {fpr,population}, queue_size in {0,64}, iid_mask "
    "present/absent -- every axis value covered at least once, not a full "
    "cross product."
)


@dataclasses.dataclass
class StepCall:
    """Inputs for one forward call, dtype-independent (logits added at replay)."""

    targets: torch.Tensor
    iid_mask: torch.Tensor | None


@dataclasses.dataclass
class GoldenConfig:
    """One loss instance (constructor kwargs) + its multi-step call sequence."""

    name: str
    loss_cls: type
    loss_kwargs: dict[str, Any]
    logits_shape: tuple[int, int]
    steps: list[StepCall]
    return_per_class: bool
    float_targets: bool = False  # True only for SigmoidFocalLoss (0/1 labels)


def _balanced_targets(gen: torch.Generator, n: int, num_classes: int) -> torch.Tensor:
    """Round-robin class assignment, shuffled -- guarantees exact per-class balance."""
    base = torch.arange(n) % num_classes
    perm = torch.randperm(n, generator=gen)
    return base[perm]


def _with_ignored(targets: torch.Tensor, gen: torch.Generator, n_ignored: int) -> torch.Tensor:
    out = targets.clone()
    if n_ignored:
        idx = torch.randperm(out.size(0), generator=gen)[:n_ignored]
        out[idx] = IGNORE_INDEX
    return out


def _iid_mask(gen: torch.Generator, n: int) -> torch.Tensor:
    return torch.rand(n, generator=gen) > 0.3


def _ranking_steps(gen: torch.Generator, n_steps: int, with_iid_mask: bool) -> list[StepCall]:
    steps = []
    for _ in range(n_steps):
        targets = _with_ignored(
            _balanced_targets(gen, N_ROWS, NUM_CLASSES), gen, N_IGNORED_ROWS
        )
        mask = _iid_mask(gen, N_ROWS) if with_iid_mask else None
        steps.append(StepCall(targets=targets, iid_mask=mask))
    return steps


def build_grid() -> list[GoldenConfig]:
    """Build the full configuration grid. Deterministic given SEED."""
    gen = torch.Generator().manual_seed(SEED)
    configs: list[GoldenConfig] = []

    # ---- SigmoidFocalLoss: multi-label 0/1 targets, same shape as logits --
    for reduction in ("mean", "sum", "none"):
        targets = (torch.rand(N_ROWS, NUM_CLASSES, generator=gen) < 0.3).float()
        configs.append(
            GoldenConfig(
                name=f"SigmoidFocalLoss[reduction={reduction}]",
                loss_cls=SigmoidFocalLoss,
                loss_kwargs={"reduction": reduction},
                logits_shape=(N_ROWS, NUM_CLASSES),
                steps=[StepCall(targets=targets, iid_mask=None)],
                return_per_class=False,
                float_targets=True,
            )
        )

    # ---- SoftmaxFocalLoss: integer targets, ignore_index rows present -----
    for reduction in ("mean", "mean_positive", "sum", "none"):
        targets = _with_ignored(
            _balanced_targets(gen, N_ROWS, NUM_CLASSES), gen, N_IGNORED_ROWS
        )
        configs.append(
            GoldenConfig(
                name=f"SoftmaxFocalLoss[reduction={reduction}]",
                loss_cls=SoftmaxFocalLoss,
                loss_kwargs={"reduction": reduction},
                logits_shape=(N_ROWS, NUM_CLASSES),
                steps=[StepCall(targets=targets, iid_mask=None)],
                return_per_class=False,
            )
        )

    # ---- SmoothAPLoss / RecallAtQuantileLoss: queue_size x iid_mask -------
    for loss_cls, extra_kwargs in (
        (SmoothAPLoss, {}),
        (RecallAtQuantileLoss, {"quantile": 0.3}),
    ):
        for queue_size in (0, QUEUE_SIZE_SMALL):
            for with_iid_mask in (False, True):
                configs.append(
                    GoldenConfig(
                        name=(
                            f"{loss_cls.__name__}[queue_size={queue_size},"
                            f"iid_mask={with_iid_mask}]"
                        ),
                        loss_cls=loss_cls,
                        loss_kwargs={
                            "num_classes": NUM_CLASSES,
                            "queue_size": queue_size,
                            **extra_kwargs,
                        },
                        logits_shape=(N_ROWS, NUM_CLASSES),
                        steps=_ranking_steps(gen, QUEUE_STEPS, with_iid_mask),
                        return_per_class=True,
                    )
                )

    # ---- PAUCAtBudgetLoss: 8 series, every axis value covered at least once
    # (surrogate, pos_numerator, budget_basis, queue_size, iid_mask_present)
    pauc_series = [
        ("trapezoid", "pool", "fpr", 0, False),
        ("trapezoid", "pool", "fpr", QUEUE_SIZE_SMALL, False),
        ("trapezoid", "pool", "fpr", QUEUE_SIZE_SMALL, True),
        ("pairwise", "pool", "fpr", QUEUE_SIZE_SMALL, False),
        ("trapezoid", "live", "fpr", QUEUE_SIZE_SMALL, False),
        ("trapezoid", "pool", "population", QUEUE_SIZE_SMALL, False),
        ("pairwise", "live", "population", QUEUE_SIZE_SMALL, True),
        ("trapezoid", "pool", "fpr", 0, True),
    ]
    for surrogate, pos_numerator, budget_basis, queue_size, with_iid_mask in pauc_series:
        configs.append(
            GoldenConfig(
                name=(
                    f"PAUCAtBudgetLoss[surrogate={surrogate},"
                    f"pos_numerator={pos_numerator},budget_basis={budget_basis},"
                    f"queue_size={queue_size},iid_mask={with_iid_mask}]"
                ),
                loss_cls=PAUCAtBudgetLoss,
                loss_kwargs={
                    "num_classes": NUM_CLASSES,
                    # Wide band (matches existing test convention) so the
                    # small N=64 pool never trips the degenerate-scale or
                    # empty-band guards.
                    "alpha": 0.1,
                    "beta": 0.5,
                    "surrogate": surrogate,
                    "pos_numerator": pos_numerator,
                    "budget_basis": budget_basis,
                    "queue_size": queue_size,
                },
                logits_shape=(N_ROWS, NUM_CLASSES),
                steps=_ranking_steps(gen, QUEUE_STEPS, with_iid_mask),
                return_per_class=True,
            )
        )

    return configs


def _make_logits(
    config_index: int, step_index: int, dtype: torch.dtype, shape: tuple[int, int]
) -> torch.Tensor:
    """Deterministic logits for one (config, step, dtype) triple.

    ``config_index`` is the position of the config in ``build_grid()``'s
    return list; callers on both the capture and replay side must pass the
    same index for the same config so the inputs line up.
    """
    seed = SEED * 1_000_003 + config_index * 97 + step_index
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=dtype)


def run_entry(
    config: GoldenConfig, config_index: int, dtype: torch.dtype
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Instantiate ``config.loss_cls`` once and replay every step in sequence.

    The loss instance (and its memory queue, for queued losses) carries
    state across steps within this call, matching how the queue evolves
    across consecutive training steps.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        Keyed by step index (as ``str``); each value has ``"loss"``,
        ``"grad"``, and -- when ``config.return_per_class``-- ``"per_class"``.
    """
    loss_fn = config.loss_cls(**config.loss_kwargs).to(dtype)
    loss_fn.train()

    results: dict[str, dict[str, torch.Tensor]] = {}
    for step_index, step in enumerate(config.steps):
        logits = _make_logits(config_index, step_index, dtype, config.logits_shape)
        logits.requires_grad_(True)

        targets = step.targets.to(dtype) if config.float_targets else step.targets

        forward_kwargs: dict[str, Any] = {}
        if step.iid_mask is not None:
            forward_kwargs["iid_mask"] = step.iid_mask
        if config.return_per_class:
            forward_kwargs["return_per_class"] = True

        out = loss_fn(logits, targets, **forward_kwargs)

        if isinstance(out, tuple):
            loss_val, per_class, valid_vec = out[0], out[1], out[2]
            if not bool(valid_vec.all()):
                raise RuntimeError(
                    f"{config.name} step {step_index} (dtype={dtype}): a class "
                    f"went invalid/degenerate. The golden grid must keep every "
                    f"class valid at every step -- an invalid class injects nan "
                    f"into a stored tensor, and torch.equal treats nan != nan. "
                    f"Adjust the data generation (balance, band width, or "
                    f"ignore_index count) so this class stays valid."
                )
        else:
            loss_val, per_class = out, None

        backward_target = loss_val if loss_val.ndim == 0 else loss_val.sum()
        backward_target.backward()

        entry: dict[str, torch.Tensor] = {
            "loss": loss_val.detach().clone(),
            "grad": logits.grad.detach().clone(),
        }
        if per_class is not None:
            entry["per_class"] = per_class.detach().clone()
        results[str(step_index)] = entry

    return results


def capture_fixture() -> dict[str, Any]:
    """Run the full grid and flatten every (config, dtype, step, field) into one dict."""
    configs = build_grid()
    fixture: dict[str, Any] = {}

    for dtype in DTYPES:
        dtype_str = str(dtype).rsplit(".", maxsplit=1)[-1]
        for config_index, config in enumerate(configs):
            step_results = run_entry(config, config_index, dtype)
            for step_key, step_result in step_results.items():
                prefix = f"{config.name}|dtype={dtype_str}|step={step_key}"
                for field_name, tensor in step_result.items():
                    fixture[f"{prefix}|{field_name}"] = tensor

    # str(...): torch.__version__ is a TorchVersion (str subclass) whose type
    # is not on the weights_only=True safe-globals allowlist -- coerce to a
    # plain str so torch.load(weights_only=True) can read the fixture back.
    fixture["_meta_torch_version"] = str(torch.__version__)
    fixture["_meta_seed"] = SEED
    fixture["_meta_grid_description"] = GRID_DESCRIPTION
    fixture["_meta_num_configs"] = len(configs)
    fixture["_meta_num_tensor_entries"] = sum(
        1 for k in fixture if not k.startswith("_meta_")
    )
    return fixture


def main() -> None:
    fixture = capture_fixture()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, FIXTURE_PATH)
    size_kb = FIXTURE_PATH.stat().st_size / 1024
    print(
        f"Wrote {FIXTURE_PATH} ({size_kb:.1f} KiB, "
        f"{fixture['_meta_num_tensor_entries']} tensor entries, "
        f"{fixture['_meta_num_configs']} configs, torch {torch.__version__})"
    )


if __name__ == "__main__":
    main()
