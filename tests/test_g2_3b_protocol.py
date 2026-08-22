"""Focused tests for the precommitted G2.3B utility protocol and scaffolding."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from defectgen.training.failure_diagnostics import model_state_sha256
from defectgen.training.g2_3_diagnostic import OfficialTestAccessError
from defectgen.training.g2_3b_protocol import (
    ARMS,
    ARM_GAN_1500,
    ARM_PREVALENCE_MATCHED_REAL,
    ARM_STANDARD_REAL,
    BATCH_PATTERNS,
    EVALUATION_SPLIT,
    FORBIDDEN_GAN_VARIANTS,
    G2_2_TERMINAL_DECISION,
    G2_3B_SEEDS,
    G2_3B_VERSION,
    PRIMARY_CANDIDATE,
    PRIMARY_CONTROL,
    SECONDARY_CANDIDATE,
    SECONDARY_CONTROL,
    SOURCE_CATEGORIES,
    SOURCE_DEFECTIVE_REAL,
    SOURCE_NORMAL_REAL,
    SOURCE_SYNTHETIC,
    TARGET_FRACTIONS,
    TRAINING_SPLIT,
    BudgetPlan,
    arm_comparison,
    arm_slot_counts,
    assert_allowed_gan_variant,
    assert_equal_budgets,
    assert_evaluation_split,
    assert_permitted_split,
    assert_train_only_provenance,
    budget_plan,
    build_arm_schedule,
    canonical_sha256,
    confirmation_decision,
    deterministic_class_stream,
    effective_class_balance,
    per_epoch_composition,
    schedule_composition,
    schedule_payload,
    select_operating_threshold,
    shared_class_stream_prefixes,
    stream_lengths,
    threshold_grid,
    validate_batch_patterns,
    verify_frozen_synthetic_identity,
)
from scripts.train_g2_3b_utility import _build_model, _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/g2_3b_utility_confirmation.json"


@pytest.fixture(scope="module")
def config() -> dict:
    return _load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def plan(config) -> BudgetPlan:
    return budget_plan(config["training"])


def _small_pools() -> tuple[list[int], list[int]]:
    """Ten normal and four defective real identities, disjoint dataset indices."""
    return list(range(0, 10)), list(range(10, 14))


def _tiny_plan() -> BudgetPlan:
    return BudgetPlan(optimizer_updates_per_epoch=8, batch_size=4, maximum_epochs=3)


# --------------------------------------------------------------------------- #
# Exact class/source composition
# --------------------------------------------------------------------------- #


def test_precommitted_arm_fractions_are_the_requested_design() -> None:
    assert TARGET_FRACTIONS[ARM_STANDARD_REAL] == {
        SOURCE_NORMAL_REAL: 0.5,
        SOURCE_DEFECTIVE_REAL: 0.5,
        SOURCE_SYNTHETIC: 0.0,
    }
    assert TARGET_FRACTIONS[ARM_PREVALENCE_MATCHED_REAL] == {
        SOURCE_NORMAL_REAL: 0.375,
        SOURCE_DEFECTIVE_REAL: 0.625,
        SOURCE_SYNTHETIC: 0.0,
    }
    assert TARGET_FRACTIONS[ARM_GAN_1500] == {
        SOURCE_NORMAL_REAL: 0.375,
        SOURCE_DEFECTIVE_REAL: 0.375,
        SOURCE_SYNTHETIC: 0.25,
    }


def test_batch_patterns_realize_their_fractions_exactly(plan) -> None:
    validate_batch_patterns(plan)
    for arm in ARMS:
        unit = [token for batch in BATCH_PATTERNS[arm] for token in batch]
        counts = Counter(unit)
        for category in SOURCE_CATEGORIES:
            assert counts.get(category, 0) / len(unit) == TARGET_FRACTIONS[arm][category]


def test_gan_arm_carries_exactly_one_synthetic_sample_per_batch() -> None:
    for batch in BATCH_PATTERNS[ARM_GAN_1500]:
        assert Counter(batch)[SOURCE_SYNTHETIC] == 1
    for arm in (ARM_STANDARD_REAL, ARM_PREVALENCE_MATCHED_REAL):
        for batch in BATCH_PATTERNS[arm]:
            assert SOURCE_SYNTHETIC not in batch


def test_per_epoch_slot_counts_are_exact_integers(plan) -> None:
    assert arm_slot_counts(ARM_STANDARD_REAL, plan) == {
        SOURCE_NORMAL_REAL: 992,
        SOURCE_DEFECTIVE_REAL: 992,
        SOURCE_SYNTHETIC: 0,
    }
    assert arm_slot_counts(ARM_PREVALENCE_MATCHED_REAL, plan) == {
        SOURCE_NORMAL_REAL: 744,
        SOURCE_DEFECTIVE_REAL: 1240,
        SOURCE_SYNTHETIC: 0,
    }
    assert arm_slot_counts(ARM_GAN_1500, plan) == {
        SOURCE_NORMAL_REAL: 744,
        SOURCE_DEFECTIVE_REAL: 744,
        SOURCE_SYNTHETIC: 496,
    }


def test_prevalence_matched_and_gan_arms_share_effective_defect_prevalence(plan) -> None:
    control = effective_class_balance(arm_slot_counts(ARM_PREVALENCE_MATCHED_REAL, plan))
    candidate = effective_class_balance(arm_slot_counts(ARM_GAN_1500, plan))
    standard = effective_class_balance(arm_slot_counts(ARM_STANDARD_REAL, plan))
    assert control["effective_defective_fraction"] == pytest.approx(0.625)
    assert candidate["effective_defective_fraction"] == pytest.approx(0.625)
    assert standard["effective_defective_fraction"] == pytest.approx(0.5)
    # This is exactly the G2.2 confound the primary comparison removes.
    assert candidate["effective_defective_fraction"] == control["effective_defective_fraction"]
    assert candidate["synthetic_samples"] != control["synthetic_samples"]


def test_prevalence_matched_arm_uses_real_defective_samples_only(plan) -> None:
    counts = arm_slot_counts(ARM_PREVALENCE_MATCHED_REAL, plan)
    assert counts[SOURCE_SYNTHETIC] == 0
    extra = counts[SOURCE_DEFECTIVE_REAL] - arm_slot_counts(ARM_GAN_1500, plan)[
        SOURCE_DEFECTIVE_REAL
    ]
    assert extra == arm_slot_counts(ARM_GAN_1500, plan)[SOURCE_SYNTHETIC]


def test_schedule_composition_recount_matches_the_precommitted_design() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    labels = [False] * 10 + [True] * 4
    for arm in ARMS:
        entries = schedule_payload(
            build_arm_schedule(
                arm, seed=45, plan=tiny, normal_pool=normal, defective_pool=defective,
                synthetic_pool_size=6,
            )
        )
        observed = schedule_composition(entries, labels)
        expected = arm_slot_counts(arm, tiny)
        assert observed["normal_real_samples"] == expected[SOURCE_NORMAL_REAL] * tiny.maximum_epochs
        assert (
            observed["defective_real_samples"]
            == expected[SOURCE_DEFECTIVE_REAL] * tiny.maximum_epochs
        )
        assert observed["synthetic_samples"] == expected[SOURCE_SYNTHETIC] * tiny.maximum_epochs
        assert observed["normal_real_fraction"] == pytest.approx(
            TARGET_FRACTIONS[arm][SOURCE_NORMAL_REAL]
        )
        assert observed["defective_real_fraction"] == pytest.approx(
            TARGET_FRACTIONS[arm][SOURCE_DEFECTIVE_REAL]
        )
        assert observed["synthetic_fraction"] == pytest.approx(
            TARGET_FRACTIONS[arm][SOURCE_SYNTHETIC]
        )


def test_schedule_composition_rejects_a_mislabelled_class_slot() -> None:
    entries = [{"source": SOURCE_DEFECTIVE_REAL, "source_index": 0, "epoch": 1}]
    with pytest.raises(RuntimeError):
        schedule_composition(entries, [False, True])


def test_schedule_composition_rejects_an_unknown_source() -> None:
    entries = [{"source": "mystery", "source_index": 0, "epoch": 1}]
    with pytest.raises(ValueError):
        schedule_composition(entries, [False, True])


def test_per_epoch_composition_is_identical_in_every_epoch() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    entries = schedule_payload(
        build_arm_schedule(
            ARM_GAN_1500, seed=46, plan=tiny, normal_pool=normal, defective_pool=defective,
            synthetic_pool_size=6,
        )
    )
    rows = per_epoch_composition(entries, tiny)
    assert len(rows) == tiny.maximum_epochs
    expected = arm_slot_counts(ARM_GAN_1500, tiny)
    for row in rows:
        assert row["optimizer_updates"] == tiny.optimizer_updates_per_epoch
        for category in SOURCE_CATEGORIES:
            assert row[category] == expected[category]


def test_committed_plan_records_the_exact_precommitted_composition() -> None:
    path = REPO_ROOT / "reports/g2_3b/plan/precommitted_plan.json"
    if not path.is_file():
        pytest.skip("The G2.3B plan has not been produced in this worktree")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["detector_optimizer_updates_executed"] == 0
    assert report["budget"]["matches_historical_mature_budget"] is True
    for seed in G2_3B_SEEDS:
        seed_report = report["seeds"][str(seed)]
        assert seed_report["effective_defective_fraction_by_arm"] == {
            ARM_STANDARD_REAL: 0.5,
            ARM_PREVALENCE_MATCHED_REAL: 0.625,
            ARM_GAN_1500: 0.625,
        }
        for arm in ARMS:
            entry = seed_report["arms"][arm]
            assert entry["observed_matches_precommitted_composition"] is True
            assert entry["slot_count"] == 23808
            assert entry["optimizer_updates"] == 5952
        assert all(seed_report["class_stream_prefixes_shared_across_arms"].values())


# --------------------------------------------------------------------------- #
# Equal update budgets and mature-budget arithmetic
# --------------------------------------------------------------------------- #


def test_budget_reproduces_the_historical_mature_total(config, plan) -> None:
    assert plan.optimizer_updates_per_epoch == 496
    assert plan.maximum_epochs == 12
    assert plan.total_optimizer_updates == 5952
    assert plan.total_optimizer_updates == int(
        config["immutable_inputs"]["historical_baseline_total_updates"]
    )
    assert plan.slots_per_epoch == 1984
    assert plan.total_slots == 23808
    assert config["training"]["skipped_updates_allowed"] == 0


def test_budget_plan_rejects_inconsistent_totals(config) -> None:
    broken = copy.deepcopy(config["training"])
    broken["total_optimizer_updates"] = 6000
    with pytest.raises(ValueError):
        budget_plan(broken)
    broken = copy.deepcopy(config["training"])
    broken["sample_slots_per_epoch"] = 2000
    with pytest.raises(ValueError):
        budget_plan(broken)


def test_every_arm_receives_an_identical_budget() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    schedules = {
        arm: build_arm_schedule(
            arm, seed=47, plan=tiny, normal_pool=normal, defective_pool=defective,
            synthetic_pool_size=6,
        )
        for arm in ARMS
    }
    assert_equal_budgets(schedules, tiny)
    assert len({len(entries) for entries in schedules.values()}) == 1
    for entries in schedules.values():
        assert len({entry.optimizer_step for entry in entries}) == tiny.total_optimizer_updates
        assert len(entries) == tiny.total_slots


def test_unequal_budgets_are_rejected() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    schedules = {
        arm: build_arm_schedule(
            arm, seed=47, plan=tiny, normal_pool=normal, defective_pool=defective,
            synthetic_pool_size=6,
        )
        for arm in ARMS
    }
    schedules[ARM_GAN_1500] = schedules[ARM_GAN_1500][:-4]
    with pytest.raises(RuntimeError):
        assert_equal_budgets(schedules, tiny)


def test_batch_pattern_must_tile_the_epoch() -> None:
    with pytest.raises(ValueError):
        validate_batch_patterns(
            BudgetPlan(optimizer_updates_per_epoch=7, batch_size=4, maximum_epochs=1)
        )
    with pytest.raises(ValueError):
        validate_batch_patterns(
            BudgetPlan(optimizer_updates_per_epoch=8, batch_size=8, maximum_epochs=1)
        )


# --------------------------------------------------------------------------- #
# Initialization equality
# --------------------------------------------------------------------------- #


def _small_model_config(config: dict) -> dict:
    reduced = copy.deepcopy(config)
    reduced["training"]["model"]["base_channels"] = 4
    return reduced


def test_arms_of_one_seed_share_an_identical_initialization(config) -> None:
    reduced = _small_model_config(config)
    device = torch.device("cpu")
    hashes = {arm: _build_model(reduced, 45, device)[1] for arm in ARMS}
    assert len(set(hashes.values())) == 1


def test_different_seeds_produce_different_initializations(config) -> None:
    reduced = _small_model_config(config)
    device = torch.device("cpu")
    hashes = {seed: _build_model(reduced, seed, device)[1] for seed in G2_3B_SEEDS}
    assert len(set(hashes.values())) == len(G2_3B_SEEDS)


def test_initialization_hash_is_a_function_of_the_weights(config) -> None:
    reduced = _small_model_config(config)
    model, initialization = _build_model(reduced, 46, torch.device("cpu"))
    assert model_state_sha256(model) == initialization
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    assert model_state_sha256(model) != initialization


# --------------------------------------------------------------------------- #
# Deterministic schedules
# --------------------------------------------------------------------------- #


def test_class_streams_are_deterministic_and_arm_independent() -> None:
    first = deterministic_class_stream(209, 40, seed=45, epoch=3, stream=SOURCE_DEFECTIVE_REAL)
    second = deterministic_class_stream(209, 40, seed=45, epoch=3, stream=SOURCE_DEFECTIVE_REAL)
    assert first == second
    assert all(0 <= value < 209 for value in first)
    # A longer request extends the same stream only if the key is the same; the
    # key contains no arm, so arms cannot perturb each other's draws.
    assert deterministic_class_stream(
        209, 40, seed=45, epoch=4, stream=SOURCE_DEFECTIVE_REAL
    ) != first
    assert deterministic_class_stream(
        209, 40, seed=46, epoch=3, stream=SOURCE_DEFECTIVE_REAL
    ) != first
    assert deterministic_class_stream(
        209, 40, seed=45, epoch=3, stream=SOURCE_NORMAL_REAL
    ) != first


def test_class_stream_draws_with_replacement() -> None:
    stream = deterministic_class_stream(4, 200, seed=45, epoch=1, stream=SOURCE_DEFECTIVE_REAL)
    assert len(stream) == 200
    assert max(Counter(stream).values()) > 1


def test_schedule_is_bitwise_reproducible() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()

    def _build():
        return schedule_payload(
            build_arm_schedule(
                ARM_GAN_1500, seed=45, plan=tiny, normal_pool=normal, defective_pool=defective,
                synthetic_pool_size=6,
            )
        )

    assert canonical_sha256(_build()) == canonical_sha256(_build())


def test_arms_consume_prefixes_of_one_shared_class_stream() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    schedules = {
        arm: schedule_payload(
            build_arm_schedule(
                arm, seed=45, plan=tiny, normal_pool=normal, defective_pool=defective,
                synthetic_pool_size=6,
            )
        )
        for arm in ARMS
    }
    for epoch in range(1, tiny.maximum_epochs + 1):
        for category in (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL):
            assert shared_class_stream_prefixes(schedules, category=category, epoch=epoch)
    lengths = stream_lengths(tiny)
    assert lengths[SOURCE_NORMAL_REAL] == arm_slot_counts(ARM_STANDARD_REAL, tiny)[
        SOURCE_NORMAL_REAL
    ]
    assert lengths[SOURCE_DEFECTIVE_REAL] == arm_slot_counts(ARM_PREVALENCE_MATCHED_REAL, tiny)[
        SOURCE_DEFECTIVE_REAL
    ]


def test_shared_prefix_detector_rejects_divergent_streams() -> None:
    schedules = {
        "a": [{"source": SOURCE_NORMAL_REAL, "epoch": 1, "pool_position": 1}],
        "b": [{"source": SOURCE_NORMAL_REAL, "epoch": 1, "pool_position": 2}],
    }
    assert not shared_class_stream_prefixes(schedules, category=SOURCE_NORMAL_REAL, epoch=1)


def test_schedule_entries_carry_a_consistent_epoch_and_step_index() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    entries = build_arm_schedule(
        ARM_PREVALENCE_MATCHED_REAL, seed=45, plan=tiny, normal_pool=normal,
        defective_pool=defective, synthetic_pool_size=6,
    )
    for entry in entries:
        assert entry.augmentation_epoch == entry.epoch
        expected_epoch = (entry.optimizer_step - 1) // tiny.optimizer_updates_per_epoch + 1
        assert entry.epoch == expected_epoch
        assert 0 <= entry.batch_position < tiny.batch_size
    assert [entry.position for entry in entries] == list(range(len(entries)))


def test_gan_arm_requires_a_synthetic_pool() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    with pytest.raises(ValueError):
        build_arm_schedule(
            ARM_GAN_1500, seed=45, plan=tiny, normal_pool=normal, defective_pool=defective,
            synthetic_pool_size=0,
        )


def test_unknown_arm_is_refused() -> None:
    tiny = _tiny_plan()
    normal, defective = _small_pools()
    with pytest.raises(ValueError):
        build_arm_schedule(
            "gan_2000", seed=45, plan=tiny, normal_pool=normal, defective_pool=defective,
            synthetic_pool_size=6,
        )


# --------------------------------------------------------------------------- #
# Synthetic checkpoint / manifest identity
# --------------------------------------------------------------------------- #


def _synthetic_fixture(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "joint_1500.pt"
    checkpoint.write_bytes(b"frozen-gan-checkpoint")
    from defectgen.training.g2_3b_protocol import file_sha256

    checkpoint_hash = file_sha256(checkpoint)
    image = np.full((8, 6, 3), 33, dtype=np.uint8)
    mask = np.zeros((8, 6), dtype=np.uint8)
    mask[3:5, 2:4] = 255
    valid = np.zeros((8, 6), dtype=np.uint8)
    valid[1:7, :] = 255
    paths = {}
    for name, array in (("image.png", image), ("mask.png", mask), ("valid.png", valid)):
        Image.fromarray(array).save(tmp_path / name)
        paths[name] = file_sha256(tmp_path / name)
    row = {
        "sample_id": "g2-2-synthetic-000000",
        "official_split": "train",
        "development_split": "train",
        "checkpoint_step": 1500,
        "checkpoint_sha256": checkpoint_hash,
        "image_path": "image.png",
        "image_sha256": paths["image.png"],
        "mask_path": "mask.png",
        "mask_sha256": paths["mask.png"],
        "valid_region_path": "valid.png",
        "valid_region_sha256": paths["valid.png"],
        "source_provenance": {
            "template": {
                "sample_id": "train-t",
                "official_split": "train",
                "development_split": "train",
            },
            "background": {
                "sample_id": "train-b",
                "official_split": "train",
                "development_split": "train",
            },
        },
    }
    manifest = {
        "variant": "checkpoint_1500",
        "row_count": 1,
        "official_test_source_count": 0,
        "detector_validation_source_count": 0,
        "rows": [row],
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pairing = {"checkpoint_hashes_after": {"checkpoint_1500": checkpoint_hash}}
    pairing["content_sha256"] = canonical_sha256(pairing)
    (tmp_path / "pairing.json").write_text(json.dumps(pairing), encoding="utf-8")
    settings = {
        "variant": "checkpoint_1500",
        "manifest_path": "manifest.json",
        "pairing_report_path": "pairing.json",
        "expected_sample_count": 1,
        "regenerate": False,
        "frozen_gan_checkpoint_path": "joint_1500.pt",
        "frozen_gan_checkpoint_sha256": checkpoint_hash,
        "frozen_gan_checkpoint_step": 1500,
        "frozen_manifest_content_sha256": manifest["content_sha256"],
        "frozen_pairing_report_content_sha256": pairing["content_sha256"],
        "verify_row_file_hashes": True,
        "require_every_sample_defective": True,
    }
    return settings


def test_frozen_synthetic_identity_verifies(tmp_path: Path) -> None:
    settings = _synthetic_fixture(tmp_path)
    report = verify_frozen_synthetic_identity(tmp_path, settings)
    assert report["variant"] == "checkpoint_1500"
    assert report["row_count"] == 1
    assert report["row_files_verified"] == 3
    assert report["rows_verified_defective"] == 1
    assert report["regenerated"] is False


def test_changed_gan_checkpoint_is_rejected(tmp_path: Path) -> None:
    settings = _synthetic_fixture(tmp_path)
    (tmp_path / "joint_1500.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        verify_frozen_synthetic_identity(tmp_path, settings)


def test_changed_synthetic_image_is_rejected(tmp_path: Path) -> None:
    settings = _synthetic_fixture(tmp_path)
    Image.fromarray(np.full((8, 6, 3), 99, dtype=np.uint8)).save(tmp_path / "image.png")
    with pytest.raises(RuntimeError):
        verify_frozen_synthetic_identity(tmp_path, settings)


def test_empty_synthetic_mask_is_rejected(tmp_path: Path) -> None:
    settings = _synthetic_fixture(tmp_path)
    blank = np.zeros((8, 6), dtype=np.uint8)
    Image.fromarray(blank).save(tmp_path / "mask.png")
    from defectgen.training.g2_3b_protocol import file_sha256

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("content_sha256")
    manifest["rows"][0]["mask_sha256"] = file_sha256(tmp_path / "mask.png")
    manifest["content_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    settings["frozen_manifest_content_sha256"] = manifest["content_sha256"]
    with pytest.raises(RuntimeError):
        verify_frozen_synthetic_identity(tmp_path, settings)


def test_regeneration_request_is_refused(tmp_path: Path) -> None:
    settings = _synthetic_fixture(tmp_path)
    settings["regenerate"] = True
    with pytest.raises(RuntimeError):
        verify_frozen_synthetic_identity(tmp_path, settings)


@pytest.mark.parametrize("variant", FORBIDDEN_GAN_VARIANTS)
def test_forbidden_gan_checkpoints_are_refused(variant: str) -> None:
    with pytest.raises(RuntimeError):
        assert_allowed_gan_variant(variant)


def test_only_checkpoint_1500_is_an_allowed_gan_variant() -> None:
    assert assert_allowed_gan_variant("checkpoint_1500") == "checkpoint_1500"
    assert FORBIDDEN_GAN_VARIANTS == ("checkpoint_1000", "checkpoint_2000")
    with pytest.raises(ValueError):
        assert_allowed_gan_variant("checkpoint_1750")


def test_non_training_synthetic_provenance_is_refused() -> None:
    row = {
        "official_split": "train",
        "development_split": EVALUATION_SPLIT,
        "source_provenance": {
            "template": {"official_split": "train", "development_split": "train"},
            "background": {"official_split": "train", "development_split": "train"},
        },
    }
    with pytest.raises(RuntimeError):
        assert_train_only_provenance([row])


def test_config_pins_the_frozen_g2_2_synthetic_identity(config) -> None:
    settings = config["synthetic"]
    assert settings["variant"] == "checkpoint_1500"
    assert settings["regenerate"] is False
    assert settings["expected_sample_count"] == 512
    assert (
        settings["frozen_gan_checkpoint_sha256"]
        == "5af1c6aafabcc0444117aa43209dcab168e57f4489259728e8f9066a4fdf1c81"
    )
    assert (
        settings["frozen_manifest_content_sha256"]
        == "9eba21b4347dcdafafd9d0f90dd06b297cb58b2f7ee58f1887fed7a4cd62ca91"
    )
    assert (
        settings["frozen_pairing_report_content_sha256"]
        == "540a4637936c25ae9fd3678732bbc9d81e75f066e584e6d3ee078768f491ed33"
    )


# --------------------------------------------------------------------------- #
# Validation-only threshold selection
# --------------------------------------------------------------------------- #


def test_threshold_grid_is_deterministic_and_inside_the_open_unit_interval(config) -> None:
    grid = threshold_grid(config["threshold_selection"])
    assert len(grid) == 99
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.99)
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)
    assert 0.5 in grid
    assert threshold_grid(config["threshold_selection"]) == grid


def test_threshold_grid_covers_the_historical_baseline_sweep(config) -> None:
    grid = threshold_grid(config["threshold_selection"])
    historical = [value / 100 for value in range(5, 100, 5)]
    assert all(any(abs(value - point) < 1e-12 for point in grid) for value in historical)


def test_threshold_selection_is_validation_only(config) -> None:
    assert config["threshold_selection"]["data_source"] == EVALUATION_SPLIT
    assert assert_evaluation_split(EVALUATION_SPLIT) == EVALUATION_SPLIT
    with pytest.raises(ValueError):
        assert_evaluation_split(TRAINING_SPLIT)


def _sweep_row(threshold: float, dice: float, defective: float = 0.0, precision: float = 0.0):
    return {
        "threshold": threshold,
        "global_dice": dice,
        "mean_defective_image_dice": defective,
        "pixel_precision": precision,
    }


def test_threshold_selection_maximizes_global_dice() -> None:
    grid = [0.1, 0.2, 0.3]
    rows = [_sweep_row(0.1, 0.4), _sweep_row(0.2, 0.7), _sweep_row(0.3, 0.6)]
    selection = select_operating_threshold(rows, grid)
    assert selection["selected_threshold"] == pytest.approx(0.2)
    assert selection["data_source"] == EVALUATION_SPLIT


def test_threshold_selection_tie_breaking_is_fully_deterministic() -> None:
    grid = [0.1, 0.2, 0.3, 0.4]
    rows = [
        _sweep_row(0.1, 0.7, defective=0.1, precision=0.9),
        _sweep_row(0.2, 0.7, defective=0.5, precision=0.1),
        _sweep_row(0.3, 0.7, defective=0.5, precision=0.4),
        _sweep_row(0.4, 0.7, defective=0.5, precision=0.4),
    ]
    # Equal Dice -> higher defective-image Dice; then higher precision; then the
    # smallest threshold among the survivors.
    assert select_operating_threshold(rows, grid)["selected_threshold"] == pytest.approx(0.3)


def test_threshold_selection_rejects_a_sweep_off_the_precommitted_grid() -> None:
    grid = [0.1, 0.2, 0.3]
    rows = [_sweep_row(0.1, 0.4), _sweep_row(0.25, 0.7), _sweep_row(0.3, 0.6)]
    with pytest.raises(RuntimeError):
        select_operating_threshold(rows, grid)


def test_threshold_selection_rejects_an_empty_sweep() -> None:
    with pytest.raises(ValueError):
        select_operating_threshold([], [0.1])


def test_the_same_rule_applies_to_every_arm_and_seed(config) -> None:
    settings = config["threshold_selection"]
    assert settings["policy"] == "precommitted_identical_rule_for_every_arm_and_seed"
    assert settings["objective"] == "maximum_validation_global_dice"
    assert settings["secondary_fixed_threshold"] == 0.5
    assert (
        settings["secondary_fixed_threshold_role"]
        == "continuity_evidence_only_never_a_gate_input"
    )
    assert settings["threshold_independent_metric"] == "pixel_pr_auc"


# --------------------------------------------------------------------------- #
# Official-test refusal
# --------------------------------------------------------------------------- #


def test_permitted_splits_are_development_train_and_validation() -> None:
    assert assert_permitted_split(TRAINING_SPLIT) == TRAINING_SPLIT
    assert assert_permitted_split(EVALUATION_SPLIT) == EVALUATION_SPLIT


def test_official_held_out_split_is_refused() -> None:
    from defectgen.training.g2_3b_protocol import FORBIDDEN_SPLIT

    with pytest.raises(OfficialTestAccessError):
        assert_permitted_split(FORBIDDEN_SPLIT)
    with pytest.raises(OfficialTestAccessError):
        assert_evaluation_split(FORBIDDEN_SPLIT)


def test_dataset_builder_refuses_the_official_held_out_split(config) -> None:
    from defectgen.training.g2_3b_protocol import FORBIDDEN_SPLIT
    from scripts.train_g2_3b_utility import _dataset

    with pytest.raises(OfficialTestAccessError):
        _dataset(config, FORBIDDEN_SPLIT)


def test_unknown_split_is_a_plain_value_error() -> None:
    with pytest.raises(ValueError) as error:
        assert_permitted_split("holdout")
    assert not isinstance(error.value, OfficialTestAccessError)


def test_script_exposes_no_official_evaluation_mode() -> None:
    source = (REPO_ROOT / "scripts/train_g2_3b_utility.py").read_text(encoding="utf-8")
    assert '"test"' not in source
    assert "'test'" not in source
    assert "official-test" not in source
    assert '("plan", "train", "confirm")' in source


def test_config_forbids_official_access_even_after_a_pass(config) -> None:
    policy = config["access_policy"]
    assert policy["official_test_allowed"] is False
    assert policy["official_test_allowed_after_confirmation"] is False
    assert policy["gan_updates_allowed"] is False
    assert policy["regenerate_synthetic_allowed"] is False
    assert policy["modifies_g2_2_artifacts"] is False
    assert policy["modifies_g2_3a_artifacts"] is False
    assert policy["evaluation_splits_allowed"] == [EVALUATION_SPLIT]
    assert policy["training_splits_allowed"] == [TRAINING_SPLIT]
    assert config["immutable_inputs"]["g2_2_terminal_decision"] == G2_2_TERMINAL_DECISION


def test_config_loader_rejects_a_relaxed_access_policy(config) -> None:
    import tempfile

    broken = copy.deepcopy(config)
    broken["access_policy"]["official_test_allowed"] = True
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "broken.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(RuntimeError):
            _load_config(path)


def test_config_loader_rejects_changed_seeds_or_arms(config) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        for mutate in (
            lambda payload: payload.__setitem__("seeds", [42, 43, 44]),
            lambda payload: payload["confirmation_gate"].__setitem__(
                "primary_control", ARM_STANDARD_REAL
            ),
            lambda payload: payload["confirmation_gate"].__setitem__(
                "frozen_before_training", False
            ),
        ):
            broken = copy.deepcopy(config)
            mutate(broken)
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with pytest.raises((ValueError, RuntimeError)):
                _load_config(path)


# --------------------------------------------------------------------------- #
# Confirmation-gate arithmetic
# --------------------------------------------------------------------------- #


def _metrics(dice, iou, precision, recall, fpr, pr_auc_value):
    return {
        "global_dice": dice,
        "global_iou": iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "normal_image_false_positive_rate": fpr,
        "pixel_pr_auc": pr_auc_value,
    }


def test_arm_comparison_computes_signed_deltas() -> None:
    candidate = _metrics(0.80, 0.68, 0.85, 0.75, 0.04, 0.82)
    control = _metrics(0.78, 0.66, 0.84, 0.76, 0.05, 0.80)
    comparison = arm_comparison(candidate, control)
    assert comparison["global_dice_gain"] == pytest.approx(0.02)
    assert comparison["global_iou_gain"] == pytest.approx(0.02)
    assert comparison["pixel_precision_delta"] == pytest.approx(0.01)
    assert comparison["pixel_recall_delta"] == pytest.approx(-0.01)
    assert comparison["normal_fpr_delta"] == pytest.approx(-0.01)
    assert comparison["pixel_pr_auc_gain"] == pytest.approx(0.02)


def test_arm_comparison_requires_every_gate_metric() -> None:
    candidate = _metrics(0.8, 0.68, 0.85, 0.75, 0.04, 0.82)
    control = dict(candidate)
    control.pop("pixel_pr_auc")
    with pytest.raises(KeyError):
        arm_comparison(candidate, control)


def _comparison(dice, iou=0.01, precision=0.0, recall=0.0, fpr=0.0, pr_auc_value=0.02):
    return {
        "global_dice_gain": dice,
        "global_iou_gain": iou,
        "pixel_precision_delta": precision,
        "pixel_recall_delta": recall,
        "normal_fpr_delta": fpr,
        "pixel_pr_auc_gain": pr_auc_value,
    }


def test_gate_passes_a_clean_three_seed_win(config) -> None:
    rules = config["confirmation_gate"]
    confirmed, aggregate = confirmation_decision(
        [_comparison(0.03), _comparison(0.02), _comparison(0.025)], rules=rules
    )
    assert confirmed is True
    assert aggregate["positive_dice_seeds"] == 3
    assert aggregate["positive_pr_auc_seeds"] == 3
    assert aggregate["failed_criteria"] == []
    assert aggregate["decision"] == "confirmed_gan_1500_utility_beyond_prevalence"
    assert aggregate["official_test_authorized_by_this_decision"] is False


def test_gate_fails_a_material_mean_recall_regression(config) -> None:
    rules = config["confirmation_gate"]
    confirmed, aggregate = confirmation_decision(
        [
            _comparison(0.03, recall=-0.05),
            _comparison(0.02, recall=-0.02),
            _comparison(0.03, recall=0.01),
        ],
        rules=rules,
    )
    assert confirmed is False
    assert "mean_recall_regression" in aggregate["failed_criteria"]
    assert aggregate["decision"] == "stop_not_confirmed_g2_3b"


def test_gate_fails_when_only_one_seed_gains_dice(config) -> None:
    rules = config["confirmation_gate"]
    confirmed, aggregate = confirmation_decision(
        [_comparison(0.09), _comparison(-0.01), _comparison(-0.02)], rules=rules
    )
    assert confirmed is False
    assert aggregate["positive_dice_seeds"] == 1
    assert "positive_dice_seeds" in aggregate["failed_criteria"]


def test_gate_fails_a_flat_pr_auc_even_when_dice_gains(config) -> None:
    rules = config["confirmation_gate"]
    confirmed, aggregate = confirmation_decision(
        [
            _comparison(0.03, pr_auc_value=0.001),
            _comparison(0.03, pr_auc_value=0.002),
            _comparison(0.03, pr_auc_value=0.000),
        ],
        rules=rules,
    )
    assert confirmed is False
    assert "mean_pixel_pr_auc_gain" in aggregate["failed_criteria"]


def test_gate_fails_a_precision_or_fpr_regression(config) -> None:
    rules = config["confirmation_gate"]
    _, precision_failure = confirmation_decision(
        [_comparison(0.03, precision=-0.05)] * 3, rules=rules
    )
    assert "mean_precision_regression" in precision_failure["failed_criteria"]
    _, fpr_failure = confirmation_decision([_comparison(0.03, fpr=0.10)] * 3, rules=rules)
    assert "mean_normal_fpr_regression" in fpr_failure["failed_criteria"]


def test_gate_requires_exactly_three_seeds(config) -> None:
    rules = config["confirmation_gate"]
    with pytest.raises(ValueError):
        confirmation_decision([_comparison(0.03), _comparison(0.03)], rules=rules)
    with pytest.raises(ValueError):
        confirmation_decision([_comparison(0.03)] * 4, rules=rules)


def test_gate_thresholds_are_not_weaker_than_the_frozen_g2_2_rules(config) -> None:
    g2_2 = json.loads(
        (REPO_ROOT / "configs/g2_2_detector_utility.json").read_text(encoding="utf-8")
    )["selection"]
    gate = config["confirmation_gate"]
    assert gate["minimum_mean_global_dice_gain"] >= g2_2["minimum_global_dice_gain"]
    assert gate["minimum_mean_global_iou_gain"] >= g2_2["minimum_global_iou_gain"]
    assert gate["maximum_mean_normal_fpr_regression"] <= g2_2["maximum_normal_fpr_regression"]
    assert gate["maximum_mean_precision_regression"] <= g2_2["maximum_precision_regression"]
    assert gate["maximum_mean_recall_regression"] <= g2_2["maximum_recall_regression"]
    assert gate["minimum_positive_dice_seeds"] >= 2
    # PR-AUC is an added criterion, never a substitution.
    assert gate["minimum_mean_pixel_pr_auc_gain"] > 0
    assert gate["minimum_positive_pr_auc_seeds"] >= 2
    assert gate["justification"]["no_value_was_chosen_from_any_g2_3b_result"] is True


def test_primary_comparison_is_gan_against_the_prevalence_matched_control(config) -> None:
    assert PRIMARY_CANDIDATE == ARM_GAN_1500
    assert PRIMARY_CONTROL == ARM_PREVALENCE_MATCHED_REAL
    assert SECONDARY_CANDIDATE == ARM_STANDARD_REAL
    assert SECONDARY_CONTROL == ARM_PREVALENCE_MATCHED_REAL
    assert config["comparisons"]["primary"]["gated"] is True
    assert config["comparisons"]["secondary"]["gated"] is False
    assert config["confirmation_gate"]["seeds"] == [45, 46, 47]
    assert G2_3B_SEEDS == (45, 46, 47)
    assert G2_3B_VERSION == config["experiment_version"]
