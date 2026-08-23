"""Restart-safety tests for long G2.3B execution.

These cover execution durability only. They assert nothing about the scientific
protocol, which is frozen and tested in ``test_g2_3b_protocol.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from defectgen.training.engine import capture_random_states, restore_random_states
from defectgen.training.final_baseline import EarlyStopping
from defectgen.training.g2_3b_protocol import (
    ARM_GAN_1500,
    ARM_PREVALENCE_MATCHED_REAL,
    ARM_STANDARD_REAL,
    EVALUATION_SPLIT,
    G2_3B_VERSION,
    BudgetPlan,
    assert_completed_arm_compatible,
    assert_durable_counters,
    assert_resume_compatible,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    expected_updates_after_epoch,
    identity_mismatches,
    resume_start_epoch,
    run_identity,
)
from defectgen.training.numerics import NumericalStepController
from scripts.train_g2_3b_utility import _load_config, train_arm


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/g2_3b_utility_confirmation.json"
PLAN = BudgetPlan(optimizer_updates_per_epoch=496, batch_size=4, maximum_epochs=12)


@pytest.fixture(scope="module")
def config() -> dict:
    return _load_config(CONFIG_PATH)


def _identity(**overrides) -> dict:
    identity = run_identity(
        arm=ARM_GAN_1500,
        seed=45,
        schedule_sha256="a" * 64,
        initialization_sha256="b" * 64,
        config_sha256="c" * 64,
        plan=PLAN,
    )
    identity.update(overrides)
    return identity


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #


def test_atomic_torch_save_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "arm_last.pt"
    atomic_torch_save(path, {"epoch": 4, "tensor": torch.arange(6)})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 4
    assert torch.equal(payload["tensor"], torch.arange(6))
    assert not list(tmp_path.glob(".*tmp"))


def test_atomic_torch_save_overwrites_in_place(tmp_path: Path) -> None:
    path = tmp_path / "arm_last.pt"
    atomic_torch_save(path, {"epoch": 1})
    atomic_torch_save(path, {"epoch": 2})
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_a_failed_atomic_save_leaves_the_previous_checkpoint_intact(tmp_path: Path) -> None:
    path = tmp_path / "arm_last.pt"
    atomic_torch_save(path, {"epoch": 7, "good": True})

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("serialization failed midway")

    with pytest.raises(RuntimeError):
        atomic_torch_save(path, {"epoch": 8, "bad": Unpicklable()})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 7
    assert payload["good"] is True
    # No temporary residue is left behind to be mistaken for durable state.
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_json_round_trips_and_leaves_no_residue(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    atomic_write_json(path, {"arm": ARM_GAN_1500, "optimizer_updates": 5952})
    assert json.loads(path.read_text(encoding="utf-8"))["optimizer_updates"] == 5952
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_json_failure_preserves_the_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    atomic_write_json(path, {"epoch": 3})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"epoch": object()})
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 3
    assert list(tmp_path.iterdir()) == [path]


# --------------------------------------------------------------------------- #
# Run identity and resume compatibility
# --------------------------------------------------------------------------- #


def test_run_identity_pins_everything_a_restart_must_preserve() -> None:
    identity = _identity()
    assert identity["experiment_version"] == G2_3B_VERSION
    assert identity["arm"] == ARM_GAN_1500
    assert identity["seed"] == 45
    assert identity["schedule_sha256"] == "a" * 64
    assert identity["initialization_sha256"] == "b" * 64
    assert identity["config_sha256"] == "c" * 64
    assert identity["total_optimizer_updates"] == 5952
    assert identity["optimizer_updates_per_epoch"] == 496
    assert identity["maximum_epochs"] == 12
    assert identity["batch_size"] == 4


def test_run_identity_refuses_an_unknown_arm() -> None:
    with pytest.raises(ValueError):
        run_identity(
            arm="gan_2000", seed=45, schedule_sha256="a", initialization_sha256="b",
            config_sha256="c", plan=PLAN,
        )


def test_identical_durable_state_resumes() -> None:
    assert_resume_compatible(_identity(), _identity())
    assert identity_mismatches(_identity(), _identity()) == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("schedule_sha256", "d" * 64),
        ("initialization_sha256", "e" * 64),
        ("config_sha256", "f" * 64),
        ("seed", 46),
        ("arm", ARM_STANDARD_REAL),
        ("experiment_version", "some_other_version"),
        ("total_optimizer_updates", 2000),
        ("optimizer_updates_per_epoch", 500),
        ("maximum_epochs", 11),
        ("batch_size", 8),
    ],
)
def test_incompatible_durable_state_refuses_to_resume(field: str, value) -> None:
    with pytest.raises(RuntimeError) as error:
        assert_resume_compatible(_identity(**{field: value}), _identity())
    assert field in str(error.value)


def test_durable_state_missing_an_identity_field_refuses_to_resume() -> None:
    observed = _identity()
    observed.pop("schedule_sha256")
    with pytest.raises(RuntimeError):
        assert_resume_compatible(observed, _identity())
    # Durable state written before identities existed has no identity at all.
    with pytest.raises(RuntimeError):
        assert_resume_compatible({}, _identity())


# --------------------------------------------------------------------------- #
# Completed epochs and completed arms are never rerun or overwritten
# --------------------------------------------------------------------------- #


def test_resume_never_reruns_a_completed_epoch() -> None:
    assert resume_start_epoch(0, PLAN) == 1
    assert resume_start_epoch(7, PLAN) == 8
    # A fully completed arm resumes past the loop entirely, into evaluation only.
    assert resume_start_epoch(PLAN.maximum_epochs, PLAN) == PLAN.maximum_epochs + 1


def test_resume_refuses_an_out_of_range_durable_epoch() -> None:
    with pytest.raises(ValueError):
        resume_start_epoch(-1, PLAN)
    with pytest.raises(ValueError):
        resume_start_epoch(PLAN.maximum_epochs + 1, PLAN)


def test_durable_counters_must_agree_with_the_completed_epochs() -> None:
    assert expected_updates_after_epoch(0, PLAN) == 0
    assert expected_updates_after_epoch(7, PLAN) == 3472
    assert expected_updates_after_epoch(12, PLAN) == 5952
    assert_durable_counters(
        {"optimizer_step_executed": 3472, "optimizer_step_skipped": 0},
        last_completed_epoch=7,
        plan=PLAN,
    )


def test_truncated_or_skipped_durable_counters_are_fatal() -> None:
    with pytest.raises(RuntimeError):
        assert_durable_counters(
            {"optimizer_step_executed": 3400, "optimizer_step_skipped": 0},
            last_completed_epoch=7,
            plan=PLAN,
        )
    with pytest.raises(RuntimeError):
        assert_durable_counters(
            {"optimizer_step_executed": 3472, "optimizer_step_skipped": 1},
            last_completed_epoch=7,
            plan=PLAN,
        )


def _completed_report(**overrides) -> dict:
    report = {
        **_identity(),
        "evaluation_split": EVALUATION_SPLIT,
        "optimizer_updates": 5952,
        "skipped_updates": 0,
        "official_test_samples_loaded": 0,
    }
    report.update(overrides)
    return report


def test_a_matching_completed_arm_may_be_reused() -> None:
    assert_completed_arm_compatible(_completed_report(), _identity())


@pytest.mark.parametrize(
    "overrides",
    [
        {"optimizer_updates": 5951},
        {"optimizer_updates": 2000},
        {"skipped_updates": 1},
        {"official_test_samples_loaded": 3},
        {"evaluation_split": "train"},
        {"schedule_sha256": "z" * 64},
        {"seed": 47},
        {"arm": ARM_PREVALENCE_MATCHED_REAL},
        {"config_sha256": "z" * 64},
    ],
)
def test_an_incompatible_completed_arm_is_never_reused(overrides: dict) -> None:
    with pytest.raises(RuntimeError):
        assert_completed_arm_compatible(_completed_report(**overrides), _identity())


def test_train_arm_refuses_to_overwrite_a_completed_arm(tmp_path: Path, config) -> None:
    report_dir = tmp_path / "seed45"
    report_dir.mkdir(parents=True)
    (report_dir / f"{ARM_GAN_1500}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError) as error:
        train_arm(
            config,
            arm=ARM_GAN_1500,
            seed=45,
            training=None,
            validation=None,
            synthetic=None,
            schedule=[],
            strata={},
            report_dir=report_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            resume=False,
        )
    assert "Refusing to overwrite" in str(error.value)


def test_resume_is_automatic_and_not_opt_in() -> None:
    source = (REPO_ROOT / "scripts/train_g2_3b_utility.py").read_text(encoding="utf-8")
    # The durable-state branch must not be gated on the --resume flag, otherwise a
    # restart without it would silently discard completed epochs.
    assert "if last_path.is_file():" in source
    assert "if resume and last_path.is_file():" not in source


# --------------------------------------------------------------------------- #
# Resumed state preserves model, optimizer, scheduler, epoch, best, and RNG
# --------------------------------------------------------------------------- #


def _tiny_training_state(seed: int = 45):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.GroupNorm(2, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    controller = NumericalStepController(
        optimizer, precision_mode="bf16", gradient_clip_max_norm=1.0, automatic_fp32_retry=False
    )
    stopping = EarlyStopping(patience=4, minimum_delta=0.0)
    return model, optimizer, scheduler, controller, stopping


def test_durable_state_round_trips_every_resumable_component(tmp_path: Path) -> None:
    model, optimizer, scheduler, controller, stopping = _tiny_training_state()
    # Advance every component so restored values are distinguishable from fresh.
    loss = model(torch.randn(2, 3, 8, 8)).square().mean()
    loss.backward()
    optimizer.step()
    for value in (0.9, 0.8, 0.85, 0.86, 0.87):
        scheduler.step(value)
        stopping.step(value)
    controller.counters.optimizer_step_executed = 3472
    controller.counters.attempted_batches = 3472
    identity = _identity()
    state = {
        "epoch": 7,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "numerical_state": controller.state_dict(),
        "early_stopping_state": stopping.state_dict(),
        "random_states": capture_random_states(),
        "epoch_records": [{"epoch": index} for index in range(1, 8)],
        "best_validation": {"epoch": 5, "validation_total_loss": 0.8},
        "run_identity": identity,
    }
    path = tmp_path / "gan_1500_last.pt"
    atomic_torch_save(path, state)
    controller.close()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_resume_compatible(payload["run_identity"], identity)
    assert_durable_counters(
        payload["numerical_state"]["counters"], last_completed_epoch=7, plan=PLAN
    )

    restored_model, restored_optimizer, restored_scheduler, restored_controller, restored_stop = (
        _tiny_training_state(seed=999)
    )
    assert not torch.equal(
        restored_model[0].weight, model[0].weight
    ), "the fresh comparison model must start different"
    restored_model.load_state_dict(payload["model_state"])
    restored_optimizer.load_state_dict(payload["optimizer_state"])
    restored_scheduler.load_state_dict(payload["scheduler_state"])
    restored_controller.load_state_dict(payload["numerical_state"])
    restored_stop.load_state_dict(payload["early_stopping_state"])

    assert torch.equal(restored_model[0].weight, model[0].weight)
    assert restored_optimizer.state_dict()["param_groups"][0]["lr"] == (
        optimizer.state_dict()["param_groups"][0]["lr"]
    )
    for group_original, group_restored in zip(
        optimizer.state_dict()["state"].values(),
        restored_optimizer.state_dict()["state"].values(),
    ):
        assert group_original["step"] == group_restored["step"]
        assert torch.equal(group_original["exp_avg"], group_restored["exp_avg"])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_controller.state_dict()["counters"] == controller.state_dict()["counters"]
    assert restored_stop.state_dict() == stopping.state_dict()
    assert resume_start_epoch(int(payload["epoch"]), PLAN) == 8
    assert payload["best_validation"] == {"epoch": 5, "validation_total_loss": 0.8}
    assert len(payload["epoch_records"]) == 7
    restored_controller.close()


def test_random_state_capture_and_restore_is_exact() -> None:
    torch.manual_seed(1234)
    states = capture_random_states()
    first = torch.randn(5)
    drifted = torch.randn(5)
    assert not torch.equal(first, drifted)
    restore_random_states(states)
    assert torch.equal(torch.randn(5), first)


def test_learning_rate_schedule_survives_a_restart() -> None:
    """The reduced LR must survive a restart via the optimizer, not the scheduler.

    ``ReduceLROnPlateau.load_state_dict`` restores only the plateau bookkeeping;
    it never writes a rate back into ``optimizer.param_groups``. Saving scheduler
    state without optimizer state would silently reset every resumed arm to the
    initial 3e-4, so the resume path must restore both, and this test pins that.
    """
    _, optimizer, scheduler, controller, _ = _tiny_training_state()
    for value in (1.0, 0.9, 0.9, 0.9, 0.9):
        scheduler.step(value)
    reduced = optimizer.param_groups[0]["lr"]
    assert reduced < 3e-4, "the plateau scheduler must have reduced the rate"
    saved_scheduler = scheduler.state_dict()
    saved_optimizer = optimizer.state_dict()
    assert saved_optimizer["param_groups"][0]["lr"] == pytest.approx(reduced)
    controller.close()

    # Scheduler state alone is NOT sufficient.
    _, scheduler_only, lonely_scheduler, lonely_controller, _ = _tiny_training_state(seed=7)
    lonely_scheduler.load_state_dict(saved_scheduler)
    assert scheduler_only.param_groups[0]["lr"] == pytest.approx(3e-4)
    lonely_controller.close()

    # Optimizer plus scheduler, which is what the resume path restores, is exact.
    _, fresh_optimizer, fresh_scheduler, fresh_controller, _ = _tiny_training_state(seed=7)
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    fresh_optimizer.load_state_dict(saved_optimizer)
    fresh_scheduler.load_state_dict(saved_scheduler)
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(reduced)
    assert fresh_scheduler.state_dict() == saved_scheduler
    fresh_controller.close()


def test_resume_path_restores_optimizer_state_before_scheduler_state() -> None:
    source = (REPO_ROOT / "scripts/train_g2_3b_utility.py").read_text(encoding="utf-8")
    optimizer_line = source.index('optimizer.load_state_dict(payload["optimizer_state"])')
    scheduler_line = source.index('scheduler.load_state_dict(payload["scheduler_state"])')
    assert optimizer_line < scheduler_line
    assert 'restore_random_states(payload["random_states"])' in source


def test_resume_verifies_the_restored_learning_rate_against_the_epoch_record() -> None:
    source = (REPO_ROOT / "scripts/train_g2_3b_utility.py").read_text(encoding="utf-8")
    assert "assert_restored_learning_rate" in source


# --------------------------------------------------------------------------- #
# Durability changes did not alter the precommitted plan
# --------------------------------------------------------------------------- #


def test_the_precommitted_plan_is_unchanged_by_the_durability_work(config) -> None:
    committed = json.loads(
        (REPO_ROOT / "reports/g2_3b/plan/precommitted_plan.json").read_text(encoding="utf-8")
    )
    assert committed["confirmation_gate"] == config["confirmation_gate"]
    assert committed["budget"]["total_optimizer_updates"] == 5952
    assert committed["budget"]["optimizer_updates_per_epoch"] == 496
    assert committed["budget"]["maximum_epochs"] == 12
    assert committed["detector_optimizer_updates_executed"] == 0
    for seed in (45, 46, 47):
        assert committed["seeds"][str(seed)]["effective_defective_fraction_by_arm"] == {
            ARM_STANDARD_REAL: 0.5,
            ARM_PREVALENCE_MATCHED_REAL: 0.625,
            ARM_GAN_1500: 0.625,
        }


def test_schedule_identity_is_stable_across_processes(config) -> None:
    """The frozen plan's schedule hashes must still be reproducible verbatim."""
    from scripts.train_g2_3b_utility import _class_pools, _dataset
    from defectgen.training.g2_3b_protocol import build_arm_schedule, budget_plan

    committed = json.loads(
        (REPO_ROOT / "reports/g2_3b/plan/precommitted_plan.json").read_text(encoding="utf-8")
    )
    plan = budget_plan(config["training"])
    training = _dataset(config, "train")
    normal, defective = _class_pools(training)
    for seed in (45, 46, 47):
        for arm in (ARM_STANDARD_REAL, ARM_PREVALENCE_MATCHED_REAL, ARM_GAN_1500):
            entries = build_arm_schedule(
                arm, seed=seed, plan=plan, normal_pool=normal, defective_pool=defective,
                synthetic_pool_size=512,
            )
            recomputed = canonical_sha256([entry.__dict__ for entry in entries])
            assert recomputed == committed["seeds"][str(seed)]["arms"][arm]["schedule_sha256"]


def test_config_is_byte_identical_to_the_committed_protocol(config) -> None:
    on_disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert canonical_sha256(on_disk) == canonical_sha256(config)
    frozen = copy.deepcopy(config)
    frozen["training"]["optimizer_updates_per_epoch"] = 400
    assert canonical_sha256(frozen) != canonical_sha256(config)
