"""Focused tests for the G2.3A post-hoc, validation-only diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from defectgen.training.g2_3_diagnostic import (
    ALLOWED_DIAGNOSTIC_SPLITS,
    DIAGNOSTIC_CHECKPOINTS,
    FORBIDDEN_DIAGNOSTIC_SPLITS,
    G2_2_TERMINAL_DECISION,
    G2_3_VERSION,
    OfficialTestAccessError,
    PixelHistogram,
    assert_no_forbidden_provenance,
    assert_validation_only_split,
    baseline_update_position,
    build_logit_grid,
    build_probability_grid,
    canonical_sha256,
    checkpoint_identity,
    class_prevalence_confound,
    curve_point,
    dispersion,
    expected_schedule_composition,
    match_threshold_index,
    pr_auc,
    schedule_composition,
    summarize_mask_records,
    synthetic_mask_record,
    threshold_curve,
    write_curve_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Validation-only dataset access and refusal of official-test construction
# --------------------------------------------------------------------------- #


def test_validation_split_is_the_only_permitted_diagnostic_split() -> None:
    assert ALLOWED_DIAGNOSTIC_SPLITS == frozenset({"validation"})
    assert assert_validation_only_split("validation") == "validation"


@pytest.mark.parametrize("split", sorted(FORBIDDEN_DIAGNOSTIC_SPLITS))
def test_official_test_construction_is_refused(split: str) -> None:
    with pytest.raises(OfficialTestAccessError):
        assert_validation_only_split(split)


def test_training_split_is_not_a_diagnostic_evaluation_split() -> None:
    with pytest.raises(ValueError):
        assert_validation_only_split("train")
    # The refusal for "train" must not be an OfficialTestAccessError, which is
    # reserved for the untouched official test split.
    with pytest.raises(ValueError) as error:
        assert_validation_only_split("train")
    assert not isinstance(error.value, OfficialTestAccessError)


def test_diagnostic_config_forbids_every_training_and_test_route() -> None:
    config = json.loads(
        (REPO_ROOT / "configs/g2_3_diagnostic.json").read_text(encoding="utf-8")
    )
    assert config["experiment_version"] == G2_3_VERSION
    assert config["classification"] == "POST_HOC_DIAGNOSTIC"
    assert config["immutable_inputs"]["g2_2_terminal_decision"] == G2_2_TERMINAL_DECISION
    policy = config["access_policy"]
    assert policy["evaluation_splits_allowed"] == ["validation"]
    for name in (
        "training_allowed",
        "gan_updates_allowed",
        "detector_updates_allowed",
        "regenerate_synthetic_allowed",
        "official_test_allowed",
        "modifies_g2_2_artifacts",
        "checkpoint_2000_evaluated",
        "checkpoint_1000_selected",
    ):
        assert policy[name] is False


def test_diagnostic_script_never_names_an_official_test_split() -> None:
    source = (REPO_ROOT / "scripts/run_g2_3_diagnostic.py").read_text(encoding="utf-8")
    # The diagnostic never constructs development_split="test" in any form.
    assert '"test"' not in source
    assert "'test'" not in source
    assert "official-test" not in source


def test_provenance_guard_proves_train_only_membership_without_reading_test() -> None:
    report = assert_no_forbidden_provenance(
        ["train-1", "train-2", "train-1"],
        training_ids=frozenset({"train-1", "train-2"}),
        validation_ids=frozenset({"validation-9"}),
    )
    assert report["unique_source_identities"] == 2
    assert report["outside_development_train"] == 0
    assert report["detector_validation_overlap"] == 0
    assert report["official_test_rows_read"] == 0


def test_provenance_guard_rejects_a_validation_source() -> None:
    with pytest.raises(RuntimeError):
        assert_no_forbidden_provenance(
            ["validation-9"],
            training_ids=frozenset({"train-1"}),
            validation_ids=frozenset({"validation-9"}),
        )


# --------------------------------------------------------------------------- #
# Checkpoint identity
# --------------------------------------------------------------------------- #


def test_checkpoint_identity_records_hash_size_and_payload_fields(tmp_path: Path) -> None:
    path = tmp_path / "arm.pt"
    torch.save({"variant": "real_only", "seed": 43, "optimizer_updates": 2000}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = checkpoint_identity(path, payload)
    assert identity["size_bytes"] == path.stat().st_size
    assert len(identity["sha256"]) == 64
    assert identity["variant"] == "real_only"
    assert identity["seed"] == 43
    assert identity["optimizer_updates"] == 2000
    assert identity["sha256"] == checkpoint_identity(path, payload)["sha256"]


def test_checkpoint_identity_detects_a_different_file(tmp_path: Path) -> None:
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    assert checkpoint_identity(first)["sha256"] != checkpoint_identity(second)["sha256"]


def test_missing_checkpoint_is_a_hard_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint_identity(tmp_path / "absent.pt")


def test_diagnostic_checkpoint_set_is_exactly_the_authorized_six() -> None:
    assert DIAGNOSTIC_CHECKPOINTS == (
        (42, "real_only"),
        (42, "checkpoint_1500"),
        (43, "real_only"),
        (43, "checkpoint_1500"),
        (44, "real_only"),
        (44, "checkpoint_1500"),
    )
    # Checkpoint 1,000 and checkpoint 2,000 are never diagnostic inputs.
    assert all("checkpoint_1000" != variant for _, variant in DIAGNOSTIC_CHECKPOINTS)
    assert all("checkpoint_2000" != variant for _, variant in DIAGNOSTIC_CHECKPOINTS)


# --------------------------------------------------------------------------- #
# Deterministic threshold metrics
# --------------------------------------------------------------------------- #


def test_probability_grid_is_strictly_increasing_and_contains_0_5() -> None:
    grid = build_probability_grid()
    assert grid.ndim == 1
    assert np.all(np.diff(grid) > 0)
    assert np.count_nonzero(grid == 0.5) == 1
    assert 0.0 < grid[0] and grid[-1] < 1.0


def test_histogram_survivor_counts_are_exact() -> None:
    grid = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    histogram = PixelHistogram(grid)
    histogram.update(np.array([0.1, 0.2, 0.5, 0.7, 0.9]))
    # values >= 0.2 -> 4, >= 0.5 -> 3, >= 0.8 -> 1
    assert histogram.survivors().tolist() == [4, 3, 1]
    assert histogram.total == 5
    assert histogram.summary()["minimum"] == pytest.approx(0.1)
    assert histogram.summary()["maximum"] == pytest.approx(0.9)
    assert histogram.summary()["mean"] == pytest.approx(0.48)


def test_histogram_is_order_independent_and_additive() -> None:
    grid = build_probability_grid(probability_step=0.05, logit_limit=4.0, logit_step=1.0)
    values = np.linspace(0.01, 0.99, 97)
    first = PixelHistogram(grid)
    first.update(values)
    second = PixelHistogram(grid)
    for chunk in np.array_split(values[::-1], 7):
        second.update(chunk)
    assert first.survivors().tolist() == second.survivors().tolist()
    assert first.total == second.total


def test_histogram_rejects_an_unsorted_grid() -> None:
    with pytest.raises(ValueError):
        PixelHistogram(np.array([0.5, 0.2, 0.8]))


def test_threshold_curve_matches_a_hand_computed_confusion_matrix() -> None:
    grid = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    positive = PixelHistogram(grid)
    negative = PixelHistogram(grid)
    positive.update(np.array([0.9, 0.6, 0.4, 0.1]))
    negative.update(np.array([0.8, 0.3, 0.2, 0.05, 0.05, 0.05]))
    curve = threshold_curve(
        positive, negative, normal_image_maxima=[0.8, 0.3, 0.05], defective_image_maxima=[0.9, 0.4]
    )
    index = 1  # threshold 0.5
    point = curve_point(curve, index)
    assert point["true_positive_pixels"] == 2
    assert point["false_positive_pixels"] == 1
    assert point["false_negative_pixels"] == 2
    assert point["recall"] == pytest.approx(0.5)
    assert point["precision"] == pytest.approx(2 / 3)
    assert point["dice"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 2))
    assert point["iou"] == pytest.approx(2 / (2 + 1 + 2))
    assert point["normal_image_false_positive_rate"] == pytest.approx(1 / 3)
    assert point["defective_images_zero_detected_pixels"] == 1


def test_threshold_curve_is_monotone_in_the_expected_directions() -> None:
    grid = build_probability_grid(probability_step=0.01, logit_limit=6.0, logit_step=0.5)
    rng = np.random.default_rng(20260822)
    positive = PixelHistogram(grid)
    negative = PixelHistogram(grid)
    positive.update(rng.beta(5.0, 2.0, size=4000))
    negative.update(rng.beta(2.0, 5.0, size=9000))
    curve = threshold_curve(positive, negative)
    assert np.all(np.diff(curve["recall"]) <= 1e-12)
    assert np.all(np.diff(curve["true_positive_pixels"]) <= 0)
    assert np.all(np.diff(curve["false_positive_pixels"]) <= 0)
    assert 0.0 <= pr_auc(curve["recall"], curve["precision"]) <= 1.0


def test_threshold_curve_is_bitwise_reproducible() -> None:
    grid = build_probability_grid(probability_step=0.01, logit_limit=6.0, logit_step=0.5)

    def _build() -> dict[str, np.ndarray]:
        positive = PixelHistogram(grid)
        negative = PixelHistogram(grid)
        positive.update(np.linspace(0.05, 0.95, 501))
        negative.update(np.linspace(0.01, 0.6, 801))
        return threshold_curve(positive, negative, normal_image_maxima=[0.2, 0.4, 0.9])

    first, second = _build(), _build()
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_curve_csv_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    grid = build_probability_grid(probability_step=0.05, logit_limit=4.0, logit_step=1.0)
    positive, negative = PixelHistogram(grid), PixelHistogram(grid)
    positive.update(np.linspace(0.1, 0.9, 51))
    negative.update(np.linspace(0.0001, 0.5, 151))
    curve = threshold_curve(positive, negative, normal_image_maxima=[0.1, 0.6])
    first = write_curve_csv(tmp_path / "curve.csv", curve)
    second = write_curve_csv(tmp_path / "curve_again.csv", curve)
    assert first == second
    curve["dice"] = curve["dice"] + 0.5
    assert write_curve_csv(tmp_path / "changed.csv", curve) != first


def test_match_threshold_index_finds_the_closest_operating_point() -> None:
    values = np.array([0.9, 0.7, 0.55, 0.3, 0.1])
    assert match_threshold_index(values, 0.56) == 2
    assert match_threshold_index(values, 0.9) == 0
    assert match_threshold_index(values, 0.0) == 4


def test_match_threshold_index_breaks_ties_toward_the_larger_threshold() -> None:
    # Curves are indexed by ascending threshold, so the last tied index is the
    # largest threshold that achieves the target.
    values = np.array([0.8, 0.5, 0.5, 0.5, 0.2])
    assert match_threshold_index(values, 0.5) == 3


def test_pr_auc_of_a_perfect_separation_is_one() -> None:
    grid = build_probability_grid(probability_step=0.01, logit_limit=6.0, logit_step=0.5)
    positive, negative = PixelHistogram(grid), PixelHistogram(grid)
    positive.update(np.full(500, 0.99))
    negative.update(np.full(500, 0.001))
    curve = threshold_curve(positive, negative)
    assert pr_auc(curve["recall"], curve["precision"]) == pytest.approx(1.0, abs=1e-6)


def test_pr_auc_uses_the_step_rule_not_trapezoidal_interpolation() -> None:
    # Ascending threshold, so recall descends. Step rule: 1.0*(1.0-0.5) + 0.8*(0.5-0.0).
    recall = np.array([1.0, 0.5, 0.0])
    precision = np.array([1.0, 0.8, 0.0])
    assert pr_auc(recall, precision) == pytest.approx(0.9)


def test_pr_auc_rejects_a_non_monotone_recall_curve() -> None:
    with pytest.raises(ValueError):
        pr_auc(np.array([0.2, 0.9, 0.1]), np.array([0.5, 0.5, 0.5]))


def test_logit_grid_is_symmetric_and_contains_zero() -> None:
    grid = build_logit_grid(limit=2.0, step=0.5)
    assert grid.tolist() == [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


def test_dispersion_reports_sample_standard_deviation() -> None:
    result = dispersion([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert result["standard_deviation"] == pytest.approx(1.0)
    assert result["range"] == pytest.approx(2.0)


def test_baseline_update_position_brackets_the_g2_2_budget() -> None:
    rows = [
        {
            "epoch": index,
            "learning_rate": 0.0003,
            "train_total_loss": 1.0 / index,
            "validation_total_loss": 1.0 / index,
            "validation_global_dice_at_0_5": 0.1 * index,
            "validation_global_iou_at_0_5": 0.05 * index,
            "validation_pixel_precision_at_0_5": 0.2,
            "validation_pixel_recall_at_0_5": 0.3,
        }
        for index in range(1, 13)
    ]
    position = baseline_update_position(rows, updates=2000, updates_per_epoch=496)
    assert position["equivalent_baseline_epochs"] == pytest.approx(2000 / 496)
    assert position["bracketing_epoch_below"]["epoch"] == 4
    assert position["bracketing_epoch_above"]["epoch"] == 5
    assert position["baseline_total_updates"] == 5952
    assert position["fraction_of_baseline_budget"] == pytest.approx(2000 / 5952)


# --------------------------------------------------------------------------- #
# Schedule composition calculations
# --------------------------------------------------------------------------- #


def _entries(sources: list[tuple[str, int]]) -> list[dict[str, object]]:
    return [
        {"position": index, "source": source, "source_index": value}
        for index, (source, value) in enumerate(sources)
    ]


def test_schedule_composition_counts_each_class_exactly() -> None:
    labels = [False, True, False, True]
    entries = _entries(
        [("real", 0), ("real", 1), ("synthetic", 7), ("real", 3), ("synthetic", 8)]
    )
    result = schedule_composition(entries, labels)
    assert result["total_sample_slots"] == 5
    assert result["normal_real_samples"] == 1
    assert result["defective_real_samples"] == 2
    assert result["synthetic_samples"] == 2
    assert result["total_effective_defective_samples"] == 4
    assert result["total_effective_normal_samples"] == 1
    assert result["effective_defective_fraction"] == pytest.approx(0.8)
    assert result["distinct_real_identities_used"] == 3


def test_schedule_composition_honours_a_non_defective_synthetic_assumption() -> None:
    labels = [False, True]
    entries = _entries([("real", 0), ("real", 1), ("synthetic", 0), ("synthetic", 1)])
    result = schedule_composition(entries, labels, synthetic_defective=False)
    assert result["total_effective_defective_samples"] == 1
    assert result["total_effective_normal_samples"] == 3


def test_schedule_composition_rejects_an_out_of_range_real_index() -> None:
    with pytest.raises(IndexError):
        schedule_composition(_entries([("real", 5)]), [False, True])


def test_schedule_composition_rejects_an_unknown_source() -> None:
    with pytest.raises(ValueError):
        schedule_composition(_entries([("mystery", 0)]), [False, True])


def test_expected_composition_reproduces_the_frozen_g2_2_design() -> None:
    control = expected_schedule_composition(
        variant="real_only", optimizer_updates=2000, batch_size=4, synthetic_fraction=0.25
    )
    arm = expected_schedule_composition(
        variant="checkpoint_1500", optimizer_updates=2000, batch_size=4, synthetic_fraction=0.25
    )
    assert control == {
        "total_sample_slots": 8000,
        "normal_real_samples": 4000,
        "defective_real_samples": 4000,
        "synthetic_samples": 0,
        "total_effective_defective_samples": 4000,
        "total_effective_normal_samples": 4000,
        "effective_defective_fraction": 0.5,
    }
    assert arm == {
        "total_sample_slots": 8000,
        "normal_real_samples": 3000,
        "defective_real_samples": 3000,
        "synthetic_samples": 2000,
        "total_effective_defective_samples": 5000,
        "total_effective_normal_samples": 3000,
        "effective_defective_fraction": 0.625,
    }
    assert arm["normal_real_samples"] / arm["total_sample_slots"] == pytest.approx(0.375)
    assert arm["defective_real_samples"] / arm["total_sample_slots"] == pytest.approx(0.375)
    assert arm["synthetic_samples"] / arm["total_sample_slots"] == pytest.approx(0.25)


def test_class_prevalence_confound_is_detected_and_quantified() -> None:
    control = expected_schedule_composition(
        variant="real_only", optimizer_updates=2000, batch_size=4, synthetic_fraction=0.25
    )
    arm = expected_schedule_composition(
        variant="checkpoint_1500", optimizer_updates=2000, batch_size=4, synthetic_fraction=0.25
    )
    result = class_prevalence_confound(control, arm)
    assert result["is_class_prevalence_confound"] is True
    assert result["effective_defective_fraction_delta"] == pytest.approx(0.125)


def test_identical_prevalence_is_not_flagged_as_a_confound() -> None:
    control = {"effective_defective_fraction": 0.5}
    assert class_prevalence_confound(control, control)["is_class_prevalence_confound"] is False


def test_recorded_g2_2_schedules_have_the_audited_composition() -> None:
    """The committed audit must agree with the live schedule/label recomputation."""
    audit_path = REPO_ROOT / "reports/g2_3/diagnostic/schedule_composition_audit.json"
    if not audit_path.is_file():
        pytest.skip("G2.3A schedule audit has not been produced in this worktree")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["synthetic_samples_are_all_defective"] is True
    for name, arm in audit["arms"].items():
        assert arm["content_hash_matches"] is True, name
        assert arm["observed_matches_expected"] is True, name
        observed = arm["observed"]
        if arm["variant"] == "real_only":
            assert observed["effective_defective_fraction"] == pytest.approx(0.5)
        else:
            assert observed["normal_real_samples"] == 3000
            assert observed["defective_real_samples"] == 3000
            assert observed["synthetic_samples"] == 2000
            assert observed["effective_defective_fraction"] == pytest.approx(0.625)
    assert audit["classification_result"]["is_class_prevalence_confound"] is True


# --------------------------------------------------------------------------- #
# Synthetic-mask validity checks
# --------------------------------------------------------------------------- #


def _mask_and_valid(positive: tuple[slice, slice] | None, outside: bool = False):
    mask = np.zeros((8, 6), dtype=bool)
    valid = np.zeros((8, 6), dtype=bool)
    valid[1:7, :] = True
    if positive is not None:
        mask[positive] = True
    if outside:
        mask[0, 0] = True
    return mask, valid


def test_valid_synthetic_mask_passes_every_hard_check() -> None:
    mask, valid = _mask_and_valid((slice(2, 4), slice(1, 3)))
    record = synthetic_mask_record(
        sample_id="s0", mask=mask, valid=valid, image_shape=(8, 6)
    )
    assert record["has_positive_valid_defect_pixel"] is True
    assert record["support_inside_valid_region"] is True
    assert record["positive_valid_defect_pixels"] == 4
    assert record["support_outside_valid_region"] == 0


def test_empty_synthetic_mask_is_reported_as_non_defective() -> None:
    mask, valid = _mask_and_valid(None)
    record = synthetic_mask_record(sample_id="s1", mask=mask, valid=valid, image_shape=(8, 6))
    assert record["has_positive_valid_defect_pixel"] is False
    summary = summarize_mask_records([record])
    assert summary["every_synthetic_sample_is_defective"] is False
    assert summary["empty_mask_sample_ids"] == ["s1"]


def test_support_outside_the_valid_region_is_reported() -> None:
    mask, valid = _mask_and_valid((slice(2, 4), slice(1, 3)), outside=True)
    record = synthetic_mask_record(sample_id="s2", mask=mask, valid=valid, image_shape=(8, 6))
    assert record["support_inside_valid_region"] is False
    assert record["support_outside_valid_region"] == 1
    summary = summarize_mask_records([record])
    assert summary["all_support_inside_valid_region"] is False
    assert summary["support_outside_valid_sample_ids"] == ["s2"]


def test_misaligned_mask_and_valid_region_is_a_hard_failure() -> None:
    mask = np.zeros((8, 6), dtype=bool)
    valid = np.ones((8, 5), dtype=bool)
    with pytest.raises(RuntimeError):
        synthetic_mask_record(sample_id="s3", mask=mask, valid=valid, image_shape=(8, 6))


def test_misaligned_image_and_mask_is_a_hard_failure() -> None:
    mask, valid = _mask_and_valid((slice(2, 4), slice(1, 3)))
    with pytest.raises(RuntimeError):
        synthetic_mask_record(sample_id="s4", mask=mask, valid=valid, image_shape=(9, 6))


def test_mask_summary_reports_robust_positive_pixel_statistics(tmp_path: Path) -> None:
    records = []
    for index, size in enumerate((1, 4, 9)):
        mask, valid = _mask_and_valid((slice(2, 2 + size // 3 + 1), slice(1, 1 + size // 3 + 1)))
        Image.fromarray((mask * 255).astype(np.uint8)).save(tmp_path / f"m{index}.png")
        records.append(
            synthetic_mask_record(
                sample_id=f"s{index}", mask=mask, valid=valid, image_shape=(8, 6)
            )
        )
    summary = summarize_mask_records(records)
    assert summary["masks_checked"] == 3
    assert summary["all_have_positive_valid_defect_pixel"] is True
    assert summary["minimum_positive_valid_defect_pixels"] >= 1


def test_materialized_synthetic_masks_are_all_defective_and_inside_validity() -> None:
    """Re-read the committed integrity audit for all 512 local synthetic masks."""
    audit_path = REPO_ROOT / "reports/g2_3/diagnostic/synthetic_mask_integrity.json"
    if not audit_path.is_file():
        pytest.skip("G2.3A synthetic-mask audit has not been produced in this worktree")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["masks_read_from_disk"] == 512
    assert audit["hard_check"]["all_have_positive_valid_defect_pixel"] is True
    assert audit["hard_check"]["all_support_inside_valid_region"] is True
    assert audit["hard_check"]["minimum_positive_valid_defect_pixels"] >= 1
    assert audit["provenance"]["detector_validation_overlap"] == 0
    assert audit["provenance"]["official_test_rows_read"] == 0
    assert audit["rows_declaring_train_only_split"] == 512
    records_path = REPO_ROOT / audit["per_sample_records_path"]
    if not records_path.is_file():
        pytest.skip("Bulk per-mask records are a local artifact and are absent here")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert len(records["records"]) == 512
    assert records["content_sha256"] == canonical_sha256(records["records"])
    assert audit["per_sample_records_sha256"] == records["content_sha256"]
    assert all(row["has_positive_valid_defect_pixel"] for row in records["records"])
    assert all(row["support_outside_valid_region"] == 0 for row in records["records"])


# --------------------------------------------------------------------------- #
# The diagnostic may never restate the G2.2 outcome
# --------------------------------------------------------------------------- #


def test_diagnostic_summary_leaves_the_g2_2_decision_untouched() -> None:
    summary_path = REPO_ROOT / "reports/g2_3/diagnostic/diagnostic_summary.json"
    if not summary_path.is_file():
        pytest.skip("G2.3A summary has not been produced in this worktree")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["g2_2_decision"] == G2_2_TERMINAL_DECISION
    assert summary["g2_2_decision_changed_by_this_phase"] is False
    assert summary["official_test_access_count"] == 0
    assert summary["gan_optimizer_updates"] == 0
    assert summary["detector_optimizer_updates"] == 0
    assert summary["synthetic_samples_regenerated"] == 0
    assert all(summary["constraints_honoured"].values())


def test_diagnostic_threshold_metrics_reproduce_the_recorded_g2_2_values() -> None:
    path = REPO_ROOT / "reports/g2_3/diagnostic/threshold_calibration.json"
    if not path.is_file():
        pytest.skip("G2.3A threshold diagnostics have not been produced in this worktree")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["official_test_samples_loaded"] == 0
    assert report["evaluation_split"] == "validation"
    assert report["validation_sample_count"] == 350
    for key, row in report["checkpoints"].items():
        assert row["reproduction_max_absolute_delta"] == pytest.approx(0.0, abs=1e-9), key
        assert row["checkpoint"]["deterministic_repeat_logits_bitwise_equal"] is True, key
        assert row["checkpoint"]["optimizer_updates"] == 2000, key
        assert row["metrics_at_0_5_from_curve"]["threshold"] == 0.5, key
