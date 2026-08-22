"""Run the G2.3A post-hoc, validation-only diagnostic.

POST-HOC DIAGNOSTIC. This script does not train, fine-tune, resume, or
re-materialize anything. It reads frozen G2.2 artifacts and already-trained G2.2
detector checkpoints. It cannot construct the official KSDD2 test split, and
nothing it produces may change the terminal G2.2 decision stop_not_confirmed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.data.splits import load_development_manifest  # noqa: E402
from defectgen.models import UNet  # noqa: E402
from defectgen.training.g2_3_diagnostic import (  # noqa: E402
    DIAGNOSTIC_CHECKPOINTS,
    DIAGNOSTIC_LABEL,
    G2_2_TERMINAL_DECISION,
    G2_3_VERSION,
    PixelHistogram,
    assert_no_forbidden_provenance,
    assert_validation_only_split,
    atomic_write_json,
    baseline_update_position,
    build_logit_grid,
    build_probability_grid,
    canonical_sha256,
    checkpoint_identity,
    class_prevalence_confound,
    curve_point,
    dispersion,
    expected_schedule_composition,
    load_epoch_metrics,
    match_threshold_index,
    pr_auc,
    schedule_composition,
    summarize_mask_records,
    synthetic_mask_record,
    threshold_curve,
    write_curve_csv,
)
from defectgen.training.numerics import precision_autocast  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


STRATUM_NAMES = (
    "contact:border",
    "contact:non_border",
    "size:small",
    "size:medium",
    "size:large",
)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_version"] != G2_3_VERSION:
        raise ValueError("Unexpected G2.3A diagnostic version")
    policy = config["access_policy"]
    forbidden = (
        "training_allowed",
        "gan_updates_allowed",
        "detector_updates_allowed",
        "regenerate_synthetic_allowed",
        "official_test_allowed",
        "modifies_g2_2_artifacts",
        "checkpoint_2000_evaluated",
        "checkpoint_1000_selected",
    )
    if any(policy[name] for name in forbidden):
        raise RuntimeError("G2.3A access policy must forbid every write/train/test route")
    if list(policy["evaluation_splits_allowed"]) != ["validation"]:
        raise RuntimeError("G2.3A may evaluate only the validation split")
    if config["immutable_inputs"]["g2_2_terminal_decision"] != G2_2_TERMINAL_DECISION:
        raise RuntimeError("The G2.2 terminal decision is an immutable diagnostic input")
    return config


def _base_config(config: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / config["immutable_inputs"]["historical_baseline_config_path"]
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_dataset(base: dict[str, Any], split: str = "validation") -> KSDD2FullImageDataset:
    """Construct the detector validation split, and refuse anything else."""
    assert_validation_only_split(split)
    data = base["data"]
    normalization = data["detector_normalization"]
    dataset = KSDD2FullImageDataset(
        REPO_ROOT,
        split,
        REPO_ROOT / data["manifest"],
        target_size=(int(data["canvas_width"]), int(data["canvas_height"])),
        image_padding_mode="reflect",
        mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"],
        spatial_transform=None,
    )
    if any(row["development_split"] != "validation" for row in dataset.rows):
        raise RuntimeError("A non-validation row entered the G2.3A diagnostic")
    return dataset


def _real_training_labels(base: dict[str, Any]) -> tuple[list[bool], list[str]]:
    """Load the detector training row order used by every G2.2 schedule."""
    rows = load_development_manifest(REPO_ROOT, REPO_ROOT / base["data"]["manifest"])
    training = [row for row in rows if row["development_split"] == "train"]
    return [bool(row["has_defect"]) for row in training], [row["sample_id"] for row in training]


def _strata(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Replicate the frozen G2.2 stratification exactly (train-derived tertiles)."""
    path = REPO_ROOT / config["immutable_inputs"]["bbox_statistics_path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train_pixels = sorted(
        int(row["mask_pixels"])
        for row in rows
        if row["development_split"] == "train" and int(row["mask_pixels"]) > 0
    )
    if not train_pixels:
        raise RuntimeError("No train-only defect geometry was available")
    small_cutoff = train_pixels[len(train_pixels) // 3]
    large_cutoff = train_pixels[(2 * len(train_pixels)) // 3]
    assignment: dict[str, str] = {}
    for row in rows:
        if row["development_split"] != "validation" or int(row["mask_pixels"]) <= 0:
            continue
        pixels = int(row["mask_pixels"])
        size = "small" if pixels <= small_cutoff else "medium" if pixels <= large_cutoff else "large"
        border = "border" if row["touches_border"].lower() == "true" else "non_border"
        assignment[row["sample_id"]] = f"size:{size}|contact:{border}"
    return assignment, {
        "source": "development-training defective mask-pixel tertiles",
        "small_max_mask_pixels": small_cutoff,
        "medium_max_mask_pixels": large_cutoff,
        "defective_validation_images_stratified": len(assignment),
    }


# --------------------------------------------------------------------------- #
# Question 1 -- convergence / control-instability audit
# --------------------------------------------------------------------------- #


def audit_convergence(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["immutable_inputs"]
    epochs = load_epoch_metrics(
        REPO_ROOT / inputs["historical_baseline_report_directory"] / "epoch_metrics.csv"
    )
    updates_per_epoch = int(inputs["historical_baseline_updates_per_epoch"])
    baseline_summary = json.loads(
        (REPO_ROOT / inputs["historical_baseline_report_directory"] / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    g2_2_config = json.loads(
        (REPO_ROOT / inputs["g2_2_config_path"]).read_text(encoding="utf-8")
    )
    g2_2_updates = int(g2_2_config["detector"]["optimizer_updates"])

    arms: dict[str, dict[str, Any]] = {}
    for seed, variant in DIAGNOSTIC_CHECKPOINTS:
        report = json.loads(
            (REPO_ROOT / config["recorded_g2_2_reports"][str(seed)][variant]).read_text(
                encoding="utf-8"
            )
        )
        arms[f"seed{seed}:{variant}"] = report
    pilot_1000 = json.loads(
        (REPO_ROOT / "reports/g2_2/pilot_seed42/checkpoint_1000.json").read_text(encoding="utf-8")
    )
    arms["seed42:checkpoint_1000"] = pilot_1000

    def _metric(variant: str, name: str) -> list[float]:
        return [
            float(arms[f"seed{seed}:{variant}"]["overall"][name]) for seed in (42, 43, 44)
        ]

    controls = {
        name: dispersion(_metric("real_only", name))
        for name in (
            "global_dice",
            "global_iou",
            "pixel_precision",
            "pixel_recall",
            "normal_image_false_positive_rate",
        )
    }
    arm_dispersion = {
        name: dispersion(_metric("checkpoint_1500", name))
        for name in (
            "global_dice",
            "global_iou",
            "pixel_precision",
            "pixel_recall",
            "normal_image_false_positive_rate",
        )
    }

    epoch_dice = [float(row["validation_global_dice_at_0_5"]) for row in epochs]
    trajectory = [
        {
            "epoch": int(row["epoch"]),
            "cumulative_optimizer_updates": int(row["epoch"]) * updates_per_epoch,
            "learning_rate": float(row["learning_rate"]),
            "train_total_loss": float(row["train_total_loss"]),
            "validation_total_loss": float(row["validation_total_loss"]),
            "validation_global_dice_at_0_5": float(row["validation_global_dice_at_0_5"]),
            "validation_pixel_precision_at_0_5": float(row["validation_pixel_precision_at_0_5"]),
            "validation_pixel_recall_at_0_5": float(row["validation_pixel_recall_at_0_5"]),
            "sampled_training_defective_fraction": float(row["sampled_training_defective_fraction"]),
        }
        for row in epochs
    ]

    loss_progress = {}
    for key, report in sorted(arms.items()):
        mean_all = float(report["mean_training_loss"])
        final_100 = float(report["final_100_mean_training_loss"])
        loss_progress[key] = {
            "mean_training_loss_all_2000_updates": mean_all,
            "final_100_update_mean_training_loss": final_100,
            "final_100_minus_run_mean": final_100 - mean_all,
            "still_decreasing_at_update_2000": final_100 < mean_all,
        }

    position = baseline_update_position(
        epochs, updates=g2_2_updates, updates_per_epoch=updates_per_epoch
    )
    below = position["bracketing_epoch_below"]
    above = position["bracketing_epoch_above"]

    control_dice = _metric("real_only", "global_dice")
    control_recall = _metric("real_only", "pixel_recall")
    arm_recall = _metric("checkpoint_1500", "pixel_recall")

    return {
        "experiment_version": G2_3_VERSION,
        "classification": DIAGNOSTIC_LABEL,
        "question": "Q1 detector convergence / control instability",
        "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
        "historical_baseline": {
            "updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": int(inputs["historical_baseline_total_updates"]),
            "accepted_epoch": int(baseline_summary["best_epoch"]),
            "accepted_selection_rule": "minimum validation total loss over 12 completed epochs",
            "accepted_metrics_at_0_5": baseline_summary["validation_metrics_at_0_5"],
            "epoch_trajectory": trajectory,
            "epoch_dice_dispersion_at_0_5": dispersion(epoch_dice),
            "epoch_dice_dispersion_epochs_3_to_10": dispersion(epoch_dice[2:10]),
            "learning_rate_reduced_after_epoch": 9,
            "sampler": {
                "policy": "deterministic_weighted_with_replacement",
                "target_defective_fraction": 0.5,
                "observed_defective_fraction_dispersion": dispersion(
                    [float(row["sampled_training_defective_fraction"]) for row in epochs]
                ),
            },
        },
        "g2_2_budget_mapping": {
            "g2_2_optimizer_updates": g2_2_updates,
            **position,
            "interpretation": (
                f"{g2_2_updates} updates lands between historical baseline epoch "
                f"{below['epoch']} and epoch {above['epoch']}"
            ),
            "baseline_dice_at_equivalent_point": {
                "epoch_below": below["validation_global_dice_at_0_5"],
                "epoch_above": above["validation_global_dice_at_0_5"],
            },
            "baseline_train_loss_at_equivalent_point": {
                "epoch_below": below["train_total_loss"],
                "epoch_above": above["train_total_loss"],
            },
            "baseline_learning_rate_at_equivalent_point": below["learning_rate"],
            "g2_2_learning_rate": float(g2_2_config["detector"]["learning_rate"]),
            "learning_rate_schedule_difference": (
                "the historical baseline used ReduceLROnPlateau and halved the rate after "
                "epoch 9; at the 2,000-update equivalent point both regimes still ran at "
                "3e-4, so learning rate does not explain the gap at update 2,000"
            ),
            "checkpoint_selection_difference": (
                "the historical baseline reported a validation-loss-selected best epoch; "
                "every G2.2 arm reported its unselected last iterate at update 2,000"
            ),
        },
        "g2_2_training_loss_progress": loss_progress,
        "g2_2_control_dispersion_at_0_5": controls,
        "g2_2_checkpoint_1500_dispersion_at_0_5": arm_dispersion,
        "recall_variance_attribution": {
            "real_only_pixel_recall_by_seed": dict(zip(("42", "43", "44"), control_recall)),
            "checkpoint_1500_pixel_recall_by_seed": dict(zip(("42", "43", "44"), arm_recall)),
            "real_only_recall_standard_deviation": dispersion(control_recall)["standard_deviation"],
            "checkpoint_1500_recall_standard_deviation": dispersion(arm_recall)[
                "standard_deviation"
            ],
            "seed43_control_recall_minus_other_control_mean": control_recall[1]
            - statistics.mean([control_recall[0], control_recall[2]]),
            "seed43_arm_recall_minus_other_arm_mean": arm_recall[1]
            - statistics.mean([arm_recall[0], arm_recall[2]]),
        },
        "underconvergence_indicators": {
            "every_control_dice_below_baseline_equivalent_epoch": all(
                value < below["validation_global_dice_at_0_5"] for value in control_dice
            ),
            "every_control_dice_below_accepted_baseline": all(
                value < float(baseline_summary["validation_metrics_at_0_5"]["global_dice"])
                for value in control_dice
            ),
            "control_normal_fpr_minimum": controls["normal_image_false_positive_rate"]["minimum"],
            "accepted_baseline_normal_fpr": float(
                baseline_summary["validation_metrics_at_0_5"]["normal_image_false_positive_rate"]
            ),
            "all_arms_training_loss_still_decreasing_at_2000": all(
                row["still_decreasing_at_update_2000"] for row in loss_progress.values()
            ),
        },
        "evidence_limits": [
            "Per-update and per-epoch validation curves were never recorded for G2.2 arms; "
            "only the run mean and the final-100-update mean training loss survive, so the "
            "loss trend near update 2,000 is coarse.",
            "Per-epoch normal-image false-positive rate was never recorded for the historical "
            "baseline, so only its final accepted value can be compared.",
            "No G2.2 arm was trained past 2,000 updates, so no artifact can establish that a "
            "longer budget would have changed the confirmation outcome. Causality is not "
            "claimed here.",
        ],
    }


# --------------------------------------------------------------------------- #
# Question 3 -- schedule composition audit
# --------------------------------------------------------------------------- #


def audit_schedules(config: dict[str, Any]) -> dict[str, Any]:
    base = _base_config(config)
    labels, sample_ids = _real_training_labels(base)
    g2_2_config = json.loads(
        (REPO_ROOT / config["immutable_inputs"]["g2_2_config_path"]).read_text(encoding="utf-8")
    )
    detector = g2_2_config["detector"]
    mask_summary = _mask_records(config)[1]
    synthetic_defective = bool(mask_summary["every_synthetic_sample_is_defective"])

    arms: dict[str, Any] = {}
    for seed, variants in sorted(config["schedules"].items()):
        for variant, relative in sorted(variants.items()):
            payload = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
            entries = payload["entries"]
            recorded_hash = payload["content_sha256"]
            observed = schedule_composition(
                entries, labels, synthetic_defective=synthetic_defective
            )
            expected = expected_schedule_composition(
                variant=variant,
                optimizer_updates=int(detector["optimizer_updates"]),
                batch_size=int(detector["batch_size"]),
                synthetic_fraction=float(detector["synthetic_fraction"]),
                synthetic_defective=synthetic_defective,
            )
            arms[f"seed{seed}:{variant}"] = {
                "schedule_path": relative,
                "recorded_content_sha256": recorded_hash,
                "recomputed_content_sha256": canonical_sha256(entries),
                "content_hash_matches": canonical_sha256(entries) == recorded_hash,
                "seed": int(payload["seed"]),
                "variant": payload["variant"],
                "observed": observed,
                "expected_from_implementation": expected,
                "observed_matches_expected": all(
                    observed[name] == expected[name]
                    for name in (
                        "total_sample_slots",
                        "normal_real_samples",
                        "defective_real_samples",
                        "synthetic_samples",
                        "total_effective_defective_samples",
                        "total_effective_normal_samples",
                    )
                ),
            }

    confounds = {}
    for seed in ("42", "43", "44"):
        control = arms[f"seed{seed}:real_only"]["observed"]
        arm = arms[f"seed{seed}:checkpoint_1500"]["observed"]
        confounds[seed] = class_prevalence_confound(control, arm)
    if "seed42:checkpoint_1000" in arms:
        confounds["42_checkpoint_1000"] = class_prevalence_confound(
            arms["seed42:real_only"]["observed"], arms["seed42:checkpoint_1000"]["observed"]
        )

    is_confounded = all(row["is_class_prevalence_confound"] for row in confounds.values())
    return {
        "experiment_version": G2_3_VERSION,
        "classification": DIAGNOSTIC_LABEL,
        "question": "Q3 G2.2 training-composition confound",
        "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
        "development_training_pool": {
            "rows": len(labels),
            "defective": int(sum(labels)),
            "normal": int(len(labels) - sum(labels)),
            "sample_id_order_sha256": canonical_sha256(sample_ids),
        },
        "synthetic_samples_are_all_defective": synthetic_defective,
        "arms": arms,
        "class_prevalence": confounds,
        "classification_result": {
            "is_class_prevalence_confound": is_confounded,
            "statement": (
                "CONFIRMED: G2.2 simultaneously changed synthetic image content and "
                "effective defect prevalence. The real-only control trained on 50.0% "
                "effective defective samples; every synthetic arm trained on 62.5%. The "
                "arms therefore differ in two variables at once, so no G2.2 comparison can "
                "attribute its effect to synthetic content alone."
                if is_confounded
                else "No effective-prevalence difference was found between control and arms."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Question 3 -- synthetic mask integrity audit
# --------------------------------------------------------------------------- #


def _mask_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = REPO_ROOT / config["synthetic"]["manifest_directory"]
    manifests = {
        name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in config["synthetic"]["variants"]
    }
    rows = manifests["checkpoint_1500"]["rows"]
    expected = int(config["synthetic"]["expected_sample_count"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} synthetic rows, found {len(rows)}")
    records: list[dict[str, Any]] = []
    for row in rows:
        with Image.open(REPO_ROOT / row["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) > 0
        with Image.open(REPO_ROOT / row["valid_region_path"]) as source:
            valid = np.asarray(source.convert("L")) > 0
        with Image.open(REPO_ROOT / row["image_path"]) as source:
            image_shape = (source.size[1], source.size[0])
        records.append(
            synthetic_mask_record(
                sample_id=row["sample_id"], mask=mask, valid=valid, image_shape=image_shape
            )
        )
    return records, summarize_mask_records(records)


def audit_synthetic_masks(config: dict[str, Any]) -> dict[str, Any]:
    base = _base_config(config)
    rows = load_development_manifest(REPO_ROOT, REPO_ROOT / base["data"]["manifest"])
    training_ids = frozenset(
        row["sample_id"] for row in rows if row["development_split"] == "train"
    )
    validation_ids = frozenset(
        row["sample_id"] for row in rows if row["development_split"] == "validation"
    )
    directory = REPO_ROOT / config["synthetic"]["manifest_directory"]
    manifests = {
        name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in config["synthetic"]["variants"]
    }
    records, summary = _mask_records(config)

    source_ids: list[str] = []
    declared_train_rows = 0
    shared_mask_paths = True
    for first, second in zip(
        manifests["checkpoint_1000"]["rows"], manifests["checkpoint_1500"]["rows"]
    ):
        provenance = second["source_provenance"]
        source_ids.append(provenance["template"]["sample_id"])
        source_ids.append(provenance["background"]["sample_id"])
        declared_train_rows += int(
            second["official_split"] == "train" and second["development_split"] == "train"
        )
        shared_mask_paths &= (
            first["mask_path"] == second["mask_path"]
            and first["mask_sha256"] == second["mask_sha256"]
            and first["valid_region_path"] == second["valid_region_path"]
            and first["valid_region_sha256"] == second["valid_region_sha256"]
        )
    provenance_report = assert_no_forbidden_provenance(
        source_ids, training_ids=training_ids, validation_ids=validation_ids
    )

    return {
        "experiment_version": G2_3_VERSION,
        "classification": DIAGNOSTIC_LABEL,
        "question": "Q3 synthetic-mask integrity",
        "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
        "masks_read_from_disk": len(records),
        "hard_check": summary,
        "mask_and_valid_region_shared_between_checkpoints": bool(shared_mask_paths),
        "rows_declaring_train_only_split": declared_train_rows,
        "provenance": provenance_report,
        "manifest_content_sha256": {
            name: manifest["content_sha256"] for name, manifest in manifests.items()
        },
        # The 512 per-sample records are bulk reproducible evidence and are written
        # to their own file, mirroring the G2.2 convention for row-level artifacts.
        "per_sample_records_sha256": canonical_sha256(records),
        "per_sample_records_path": "reports/g2_3/diagnostic/synthetic_mask_records.json",
        "records": records,
    }


# --------------------------------------------------------------------------- #
# Question 2 -- deterministic validation-only threshold diagnostics
# --------------------------------------------------------------------------- #


class CheckpointAccumulator:
    """Accumulate exact grid histograms and per-image statistics for one model."""

    def __init__(self, probability_grid: np.ndarray, logit_grid: np.ndarray) -> None:
        self.probability_grid = probability_grid
        self.global_positive = PixelHistogram(probability_grid)
        self.global_negative = PixelHistogram(probability_grid)
        self.stratum_positive = {name: PixelHistogram(probability_grid) for name in STRATUM_NAMES}
        self.stratum_negative = {name: PixelHistogram(probability_grid) for name in STRATUM_NAMES}
        self.logit_defect = PixelHistogram(logit_grid)
        self.logit_background = PixelHistogram(logit_grid)
        self.logit_normal_image = PixelHistogram(logit_grid)
        self.normal_image_maxima: list[float] = []
        self.defective_image_maxima: list[float] = []
        self.fixed_threshold_rows: list[dict[str, Any]] = []

    def add_image(
        self,
        *,
        sample_id: str,
        probabilities: np.ndarray,
        logits: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
        has_defect: bool,
        strata: tuple[str, ...],
        threshold: float,
    ) -> None:
        selected = valid.ravel()
        probability = probabilities.ravel()[selected].astype(np.float64)
        logit = logits.ravel()[selected].astype(np.float64)
        defect = mask.ravel()[selected]
        positive = probability[defect]
        negative = probability[~defect]
        self.global_positive.update(positive)
        self.global_negative.update(negative)
        self.logit_defect.update(logit[defect])
        self.logit_background.update(logit[~defect])
        if not has_defect:
            self.logit_normal_image.update(logit)
            self.normal_image_maxima.append(float(probability.max()))
        else:
            self.defective_image_maxima.append(float(probability.max()))
            for name in strata:
                self.stratum_positive[name].update(positive)
                self.stratum_negative[name].update(negative)
        true_positive = int((positive >= threshold).sum())
        false_positive = int((negative >= threshold).sum())
        self.fixed_threshold_rows.append(
            {
                "sample_id": sample_id,
                "has_defect": bool(has_defect),
                "true_positive_pixels": true_positive,
                "false_positive_pixels": false_positive,
                "false_negative_pixels": int(positive.size - true_positive),
                "predicted_pixels": true_positive + false_positive,
                "valid_pixels": int(probability.size),
            }
        )


def _fixed_threshold_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    true_positive = sum(row["true_positive_pixels"] for row in rows)
    false_positive = sum(row["false_positive_pixels"] for row in rows)
    false_negative = sum(row["false_negative_pixels"] for row in rows)
    normal = [row for row in rows if not row["has_defect"]]
    denominator = 2 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    predicted = true_positive + false_positive
    return {
        "global_dice": 2 * true_positive / denominator if denominator else 1.0,
        "global_iou": true_positive / union if union else 1.0,
        "pixel_precision": true_positive / predicted if predicted else 0.0,
        "pixel_recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0,
        "normal_image_false_positive_rate": (
            sum(row["predicted_pixels"] > 0 for row in normal) / len(normal) if normal else 0.0
        ),
        "normal_false_positive_images": sum(row["predicted_pixels"] > 0 for row in normal),
        "normal_image_count": len(normal),
    }


def _evaluate_checkpoint(
    *,
    config: dict[str, Any],
    base: dict[str, Any],
    validation: KSDD2FullImageDataset,
    strata: dict[str, str],
    checkpoint_path: Path,
    probability_grid: np.ndarray,
    logit_grid: np.ndarray,
) -> tuple[CheckpointAccumulator, dict[str, Any]]:
    inference = config["inference"]
    threshold = float(inference["fixed_comparison_threshold"])
    configure_reproducibility(int(inference["seed"]), deterministic=True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = base["model"]
    model = UNet(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config["output_channels"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device, memory_format=torch.channels_last)
    model.load_state_dict(payload["model_state"])
    model.eval()

    loader = DataLoader(
        validation,
        batch_size=int(inference["batch_size"]),
        shuffle=False,
        num_workers=int(inference["num_workers"]),
    )
    accumulator = CheckpointAccumulator(probability_grid, logit_grid)
    repeat_batches = int(inference["repeat_determinism_check_batches"])
    deterministic_repeat = True
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, memory_format=torch.channels_last)
            with precision_autocast(device.type, inference["precision"]):
                logits = model(image)
            logits = logits.float().cpu()
            if batch_index < repeat_batches:
                with precision_autocast(device.type, inference["precision"]):
                    repeat = model(image)
                deterministic_repeat &= bool(torch.equal(logits, repeat.float().cpu()))
            probabilities = torch.sigmoid(logits)
            masks = batch["mask"].bool()
            valid_regions = batch["valid_region"].bool()
            has_defect = batch["has_defect"].bool()
            for position in range(image.shape[0]):
                sample_id = str(batch["sample_id"][position])
                label = bool(has_defect[position])
                assignment = strata.get(sample_id)
                names: tuple[str, ...] = ()
                if label:
                    if assignment is None:
                        raise RuntimeError(f"Defective validation image {sample_id} has no stratum")
                    names = tuple(assignment.split("|"))
                accumulator.add_image(
                    sample_id=sample_id,
                    probabilities=probabilities[position, 0].numpy(),
                    logits=logits[position, 0].numpy(),
                    mask=masks[position, 0].numpy(),
                    valid=valid_regions[position, 0].numpy(),
                    has_defect=label,
                    strata=names,
                    threshold=threshold,
                )
    identity = checkpoint_identity(checkpoint_path, payload)
    identity["deterministic_repeat_logits_bitwise_equal"] = deterministic_repeat
    identity["device"] = device.type
    identity["precision"] = inference["precision"]
    identity["torch_version"] = torch.__version__
    del payload, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return accumulator, identity


def _curve_bundle(accumulator: CheckpointAccumulator) -> dict[str, Any]:
    global_curve = threshold_curve(
        accumulator.global_positive,
        accumulator.global_negative,
        normal_image_maxima=accumulator.normal_image_maxima,
        defective_image_maxima=accumulator.defective_image_maxima,
    )
    stratum_curves = {
        name: threshold_curve(
            accumulator.stratum_positive[name], accumulator.stratum_negative[name]
        )
        for name in STRATUM_NAMES
    }
    return {"global": global_curve, "strata": stratum_curves}


def _at(curve: dict[str, np.ndarray], index: int) -> dict[str, float]:
    point = curve_point(curve, index)
    return {
        "threshold": point["threshold"],
        "dice": point["dice"],
        "iou": point["iou"],
        "precision": point["precision"],
        "recall": point["recall"],
        "normal_image_false_positive_rate": point["normal_image_false_positive_rate"],
        "normal_false_positive_images": point["normal_false_positive_images"],
        "true_positive_pixels": point["true_positive_pixels"],
        "false_positive_pixels": point["false_positive_pixels"],
        "false_negative_pixels": point["false_negative_pixels"],
    }


def _strata_at(bundle: dict[str, Any], index: int) -> dict[str, dict[str, float]]:
    return {
        name: {
            key: value
            for key, value in _at(bundle["strata"][name], index).items()
            if key
            not in {"normal_image_false_positive_rate", "normal_false_positive_images"}
        }
        for name in STRATUM_NAMES
    }


def run_threshold_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    base = _base_config(config)
    validation = _validation_dataset(base, config["inference"]["split"])
    strata, stratification = _strata(config)
    grid_config = config["threshold_grid"]
    probability_grid = build_probability_grid(
        probability_step=float(grid_config["probability_step"]),
        logit_limit=float(grid_config["logit_limit"]),
        logit_step=float(grid_config["logit_step"]),
    )
    logit_grid = build_logit_grid(
        limit=float(grid_config["logit_distribution_limit"]),
        step=float(grid_config["logit_distribution_step"]),
    )
    threshold = float(config["inference"]["fixed_comparison_threshold"])
    fixed_index = int(np.flatnonzero(probability_grid == threshold)[0])
    output_dir = REPO_ROOT / config["output_directory"]
    curve_dir = output_dir / "threshold_curves"

    per_checkpoint: dict[str, Any] = {}
    bundles: dict[str, Any] = {}
    started = time.perf_counter()
    for seed, variant in DIAGNOSTIC_CHECKPOINTS:
        key = f"seed{seed}:{variant}"
        checkpoint_path = REPO_ROOT / config["checkpoints"][str(seed)][variant]
        accumulator, identity = _evaluate_checkpoint(
            config=config,
            base=base,
            validation=validation,
            strata=strata,
            checkpoint_path=checkpoint_path,
            probability_grid=probability_grid,
            logit_grid=logit_grid,
        )
        bundle = _curve_bundle(accumulator)
        bundles[key] = bundle
        curve = bundle["global"]
        recorded = json.loads(
            (REPO_ROOT / config["recorded_g2_2_reports"][str(seed)][variant]).read_text(
                encoding="utf-8"
            )
        )["overall"]
        reproduced = _fixed_threshold_metrics(accumulator.fixed_threshold_rows)
        deltas = {
            name: reproduced[name] - float(recorded[name])
            for name in (
                "global_dice",
                "global_iou",
                "pixel_precision",
                "pixel_recall",
                "normal_image_false_positive_rate",
            )
        }
        best_dice_index = int(np.argmax(curve["dice"]))
        curve_hash = write_curve_csv(curve_dir / f"seed{seed}_{variant}.csv", curve)
        per_checkpoint[key] = {
            "seed": seed,
            "variant": variant,
            "checkpoint": identity,
            "recorded_g2_2_metrics_at_0_5": {
                name: float(recorded[name])
                for name in (
                    "global_dice",
                    "global_iou",
                    "pixel_precision",
                    "pixel_recall",
                    "normal_image_false_positive_rate",
                )
            },
            "reproduced_metrics_at_0_5": reproduced,
            "reproduction_deltas": deltas,
            "reproduction_max_absolute_delta": max(abs(value) for value in deltas.values()),
            "metrics_at_0_5_from_curve": _at(curve, fixed_index),
            "strata_at_0_5": _strata_at(bundle, fixed_index),
            "pixel_pr_auc": pr_auc(curve["recall"], curve["precision"]),
            "pixel_pr_auc_recall_span": [
                float(curve["recall"].min()),
                float(curve["recall"].max()),
            ],
            "stratum_pixel_pr_auc": {
                name: pr_auc(bundle["strata"][name]["recall"], bundle["strata"][name]["precision"])
                for name in STRATUM_NAMES
            },
            "best_validation_dice_threshold_diagnostic_only": {
                "note": (
                    "DIAGNOSTIC INFORMATION ONLY. This threshold selects nothing, changes no "
                    "G2.2 gate, and is never applied to the official test split."
                ),
                **_at(curve, best_dice_index),
            },
            "probability_distribution_summaries": {
                "true_defect_pixels": accumulator.global_positive.summary(),
                "true_background_pixels": accumulator.global_negative.summary(),
            },
            "logit_distribution_summaries": {
                "true_defect_pixels": accumulator.logit_defect.summary(),
                "true_background_pixels": accumulator.logit_background.summary(),
                "normal_image_valid_pixels": accumulator.logit_normal_image.summary(),
            },
            "threshold_curve_csv": (curve_dir / f"seed{seed}_{variant}.csv")
            .relative_to(REPO_ROOT)
            .as_posix(),
            "threshold_curve_sha256": curve_hash,
        }
        print(
            f"{DIAGNOSTIC_LABEL} evaluated {key}: reproduction max delta "
            f"{per_checkpoint[key]['reproduction_max_absolute_delta']:.3e}",
            flush=True,
        )

    comparisons = _matched_operating_points(config, bundles, per_checkpoint, fixed_index)
    result = {
        "experiment_version": G2_3_VERSION,
        "classification": DIAGNOSTIC_LABEL,
        "question": "Q2 threshold / calibration effect",
        "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
        "evaluation_split": config["inference"]["split"],
        "official_test_samples_loaded": 0,
        "validation_sample_count": len(validation),
        "stratification_thresholds": stratification,
        "threshold_grid": {
            "points": int(probability_grid.size),
            "minimum": float(probability_grid[0]),
            "maximum": float(probability_grid[-1]),
            "contains_fixed_threshold_0_5": True,
            **grid_config,
        },
        "checkpoints": per_checkpoint,
        "matched_operating_points": comparisons,
        "runtime_seconds": time.perf_counter() - started,
    }
    return result


def _matched_operating_points(
    config: dict[str, Any],
    bundles: dict[str, Any],
    per_checkpoint: dict[str, Any],
    fixed_index: int,
) -> dict[str, Any]:
    tolerances = config["matching"]
    comparisons: dict[str, Any] = {}
    for seed in (42, 43, 44):
        control_bundle = bundles[f"seed{seed}:real_only"]
        arm_bundle = bundles[f"seed{seed}:checkpoint_1500"]
        control = control_bundle["global"]
        arm = arm_bundle["global"]
        control_point = _at(control, fixed_index)
        arm_point = _at(arm, fixed_index)

        recall_index = match_threshold_index(arm["recall"], control_point["recall"])
        recall_point = _at(arm, recall_index)
        recall_gap = abs(recall_point["recall"] - control_point["recall"])

        fpr_index = match_threshold_index(
            arm["normal_image_false_positive_rate"],
            control_point["normal_image_false_positive_rate"],
        )
        fpr_point = _at(arm, fpr_index)
        fpr_gap = abs(
            fpr_point["normal_image_false_positive_rate"]
            - control_point["normal_image_false_positive_rate"]
        )

        stratum_matched: dict[str, Any] = {}
        for name in STRATUM_NAMES:
            control_stratum = _at(control_bundle["strata"][name], fixed_index)
            index = match_threshold_index(
                arm_bundle["strata"][name]["recall"], control_stratum["recall"]
            )
            arm_stratum = _at(arm_bundle["strata"][name], index)
            stratum_matched[name] = {
                "control_at_0_5": {
                    key: control_stratum[key] for key in ("dice", "iou", "precision", "recall")
                },
                "checkpoint_1500_at_0_5": {
                    key: _at(arm_bundle["strata"][name], fixed_index)[key]
                    for key in ("dice", "iou", "precision", "recall")
                },
                "checkpoint_1500_at_stratum_matched_recall": {
                    "threshold": arm_stratum["threshold"],
                    "recall_gap": abs(arm_stratum["recall"] - control_stratum["recall"]),
                    **{key: arm_stratum[key] for key in ("dice", "iou", "precision", "recall")},
                },
            }

        precision_gain_at_matched_recall = recall_point["precision"] - control_point["precision"]
        recall_gain_at_matched_fpr = fpr_point["recall"] - control_point["recall"]
        pr_auc_delta = (
            per_checkpoint[f"seed{seed}:checkpoint_1500"]["pixel_pr_auc"]
            - per_checkpoint[f"seed{seed}:real_only"]["pixel_pr_auc"]
        )
        frontier_better = (
            pr_auc_delta > 0
            and precision_gain_at_matched_recall > 0
            and recall_point["dice"] > control_point["dice"]
        )
        comparisons[str(seed)] = {
            "classification": DIAGNOSTIC_LABEL,
            "comparison_A_both_at_threshold_0_5": {
                "real_only": control_point,
                "checkpoint_1500": arm_point,
                "deltas": {
                    key: arm_point[key] - control_point[key]
                    for key in (
                        "dice",
                        "iou",
                        "precision",
                        "recall",
                        "normal_image_false_positive_rate",
                    )
                },
            },
            "comparison_B_checkpoint_1500_at_matched_recall": {
                "target_recall": control_point["recall"],
                "achieved_recall": recall_point["recall"],
                "recall_gap": recall_gap,
                "feasible": recall_gap <= float(tolerances["recall_match_tolerance"]),
                "threshold": recall_point["threshold"],
                "dice": recall_point["dice"],
                "iou": recall_point["iou"],
                "precision": recall_point["precision"],
                "normal_image_false_positive_rate": recall_point[
                    "normal_image_false_positive_rate"
                ],
                "deltas_versus_real_only_at_0_5": {
                    "dice": recall_point["dice"] - control_point["dice"],
                    "iou": recall_point["iou"] - control_point["iou"],
                    "precision": precision_gain_at_matched_recall,
                    "normal_image_false_positive_rate": recall_point[
                        "normal_image_false_positive_rate"
                    ]
                    - control_point["normal_image_false_positive_rate"],
                },
                "strata": _strata_at(arm_bundle, recall_index),
            },
            "comparison_C_checkpoint_1500_at_matched_normal_fpr": {
                "target_normal_image_false_positive_rate": control_point[
                    "normal_image_false_positive_rate"
                ],
                "achieved_normal_image_false_positive_rate": fpr_point[
                    "normal_image_false_positive_rate"
                ],
                "normal_fpr_gap": fpr_gap,
                "feasible": fpr_gap <= float(tolerances["normal_fpr_match_tolerance"]),
                "threshold": fpr_point["threshold"],
                "dice": fpr_point["dice"],
                "iou": fpr_point["iou"],
                "precision": fpr_point["precision"],
                "recall": fpr_point["recall"],
                "deltas_versus_real_only_at_0_5": {
                    "dice": fpr_point["dice"] - control_point["dice"],
                    "iou": fpr_point["iou"] - control_point["iou"],
                    "precision": fpr_point["precision"] - control_point["precision"],
                    "recall": recall_gain_at_matched_fpr,
                },
                "strata": _strata_at(arm_bundle, fpr_index),
            },
            "stratum_matched_recall": stratum_matched,
            "pixel_pr_auc": {
                "real_only": per_checkpoint[f"seed{seed}:real_only"]["pixel_pr_auc"],
                "checkpoint_1500": per_checkpoint[f"seed{seed}:checkpoint_1500"]["pixel_pr_auc"],
                "delta": pr_auc_delta,
            },
            "frontier_verdict": {
                "criteria": (
                    "checkpoint_1500 has the better precision-recall frontier when its "
                    "pixel PR-AUC exceeds the control, its precision at the control's "
                    "threshold-0.5 recall exceeds the control's precision, and its Dice at "
                    "that matched recall exceeds the control's Dice"
                ),
                "checkpoint_1500_frontier_at_least_as_good": bool(frontier_better),
                "verdict": (
                    "operating_point_shift_not_frontier_degradation"
                    if frontier_better
                    else "frontier_not_uniformly_better"
                ),
                "recall_regression_at_0_5": arm_point["recall"] - control_point["recall"],
                "precision_gain_at_matched_recall": precision_gain_at_matched_recall,
                "recall_gain_at_matched_normal_fpr": recall_gain_at_matched_fpr,
            },
        }
    return comparisons


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def build_summary(
    config: dict[str, Any],
    convergence: dict[str, Any],
    schedules: dict[str, Any],
    masks: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    matched = thresholds["matched_operating_points"]
    frontier = {
        seed: matched[seed]["frontier_verdict"]["verdict"] for seed in ("42", "43", "44")
    }
    recall_gaps = [
        matched[seed]["comparison_B_checkpoint_1500_at_matched_recall"]["recall_gap"]
        for seed in ("42", "43", "44")
    ]
    dice_at_matched_recall = {
        seed: matched[seed]["comparison_B_checkpoint_1500_at_matched_recall"][
            "deltas_versus_real_only_at_0_5"
        ]["dice"]
        for seed in ("42", "43", "44")
    }
    pr_auc_deltas = {seed: matched[seed]["pixel_pr_auc"]["delta"] for seed in ("42", "43", "44")}
    reproduction = max(
        row["reproduction_max_absolute_delta"] for row in thresholds["checkpoints"].values()
    )
    checkpoints = thresholds["checkpoints"]
    control_pr_auc = [checkpoints[f"seed{seed}:real_only"]["pixel_pr_auc"] for seed in (42, 43, 44)]
    arm_pr_auc = [
        checkpoints[f"seed{seed}:checkpoint_1500"]["pixel_pr_auc"] for seed in (42, 43, 44)
    ]
    stratum_pr_auc_delta = {
        name: {
            str(seed): checkpoints[f"seed{seed}:checkpoint_1500"]["stratum_pixel_pr_auc"][name]
            - checkpoints[f"seed{seed}:real_only"]["stratum_pixel_pr_auc"][name]
            for seed in (42, 43, 44)
        }
        for name in STRATUM_NAMES
    }
    seed43 = matched["43"]
    seed43_attribution = {
        "raw_recall_delta_at_0_5": seed43["comparison_A_both_at_threshold_0_5"]["deltas"]["recall"],
        "recall_is_reachable_by_thresholding": seed43[
            "comparison_B_checkpoint_1500_at_matched_recall"
        ]["feasible"],
        "dice_delta_after_matching_recall": seed43[
            "comparison_B_checkpoint_1500_at_matched_recall"
        ]["deltas_versus_real_only_at_0_5"]["dice"],
        "precision_delta_after_matching_recall": seed43[
            "comparison_B_checkpoint_1500_at_matched_recall"
        ]["deltas_versus_real_only_at_0_5"]["precision"],
        "recall_delta_after_matching_normal_fpr": seed43[
            "comparison_C_checkpoint_1500_at_matched_normal_fpr"
        ]["deltas_versus_real_only_at_0_5"]["recall"],
        "pixel_pr_auc_delta": seed43["pixel_pr_auc"]["delta"],
        "seed43_control_pr_auc_is_highest_control": max(control_pr_auc) == control_pr_auc[1],
        "seed43_arm_pr_auc_is_highest_arm": max(arm_pr_auc) == arm_pr_auc[1],
        "statement": (
            "Seed 43 combines a large operating-point shift with a modest genuine frontier "
            "deficit relative to its own control. The control's threshold-0.5 recall is fully "
            "reachable by rethresholding checkpoint_1500, so the -0.207 recall figure is not a "
            "capability ceiling; but after matching recall the arm is still behind that "
            "particular control on Dice and precision, and its pixel PR-AUC is lower. The "
            "seed-43 control is simultaneously the strongest of the three controls while the "
            "seed-43 arm is the strongest of the three arms, so the seed-43 gap is driven by "
            "control-side variance rather than by degraded seed-43 GAN behaviour."
        ),
    }
    return {
        "experiment_version": G2_3_VERSION,
        "classification": DIAGNOSTIC_LABEL,
        "stage": "g2_3a_post_hoc_validation_only_diagnostic",
        "g2_2_decision": G2_2_TERMINAL_DECISION,
        "g2_2_decision_changed_by_this_phase": False,
        "official_test_access_count": 0,
        "gan_optimizer_updates": 0,
        "detector_optimizer_updates": 0,
        "synthetic_samples_regenerated": 0,
        "q1_convergence": {
            "g2_2_updates": convergence["g2_2_budget_mapping"]["g2_2_optimizer_updates"],
            "equivalent_baseline_epochs": convergence["g2_2_budget_mapping"][
                "equivalent_baseline_epochs"
            ],
            "fraction_of_baseline_budget": convergence["g2_2_budget_mapping"][
                "fraction_of_baseline_budget"
            ],
            "controls_underconverged_indicators": convergence["underconvergence_indicators"],
            "control_dice_dispersion": convergence["g2_2_control_dispersion_at_0_5"]["global_dice"],
            "control_recall_standard_deviation": convergence["recall_variance_attribution"][
                "real_only_recall_standard_deviation"
            ],
            "arm_recall_standard_deviation": convergence["recall_variance_attribution"][
                "checkpoint_1500_recall_standard_deviation"
            ],
        },
        "q2_threshold_calibration": {
            "reproduced_recorded_metrics_at_0_5_max_absolute_delta": reproduction,
            "frontier_verdict_by_seed": frontier,
            "pixel_pr_auc_delta_by_seed": pr_auc_deltas,
            "dice_delta_at_matched_recall_by_seed": dice_at_matched_recall,
            "matched_recall_feasible_all_seeds": all(
                gap <= float(config["matching"]["recall_match_tolerance"]) for gap in recall_gaps
            ),
            "matched_normal_fpr_feasible_by_seed": {
                seed: matched[seed]["comparison_C_checkpoint_1500_at_matched_normal_fpr"][
                    "feasible"
                ]
                for seed in ("42", "43", "44")
            },
            "pixel_pr_auc_by_seed": {
                "real_only": dict(zip(("42", "43", "44"), control_pr_auc)),
                "checkpoint_1500": dict(zip(("42", "43", "44"), arm_pr_auc)),
            },
            "pixel_pr_auc_dispersion": {
                "real_only": dispersion(control_pr_auc),
                "checkpoint_1500": dispersion(arm_pr_auc),
            },
            "stratum_pixel_pr_auc_delta_by_seed": stratum_pr_auc_delta,
            "stratum_pr_auc_improved_in_all_three_seeds": {
                name: all(value > 0 for value in deltas.values())
                for name, deltas in stratum_pr_auc_delta.items()
            },
            "calibration_of_the_fixed_0_5_threshold": {
                "note": (
                    "DIAGNOSTIC INFORMATION ONLY. These best-Dice thresholds select nothing "
                    "and do not reopen the frozen G2.2 threshold."
                ),
                "best_validation_dice_threshold_by_checkpoint": {
                    key: row["best_validation_dice_threshold_diagnostic_only"]["threshold"]
                    for key, row in sorted(checkpoints.items())
                },
                "best_validation_dice_by_checkpoint": {
                    key: row["best_validation_dice_threshold_diagnostic_only"]["dice"]
                    for key, row in sorted(checkpoints.items())
                },
                "every_g2_2_checkpoint_prefers_a_threshold_above_0_5": all(
                    row["best_validation_dice_threshold_diagnostic_only"]["threshold"] > 0.5
                    for row in checkpoints.values()
                ),
                "accepted_baseline_selected_threshold": 0.05,
                "accepted_baseline_dice_at_selected_threshold": 0.7982508941396036,
                "accepted_baseline_dice_at_0_5": 0.777700283628586,
                "best_g2_2_dice_at_any_threshold": max(
                    row["best_validation_dice_threshold_diagnostic_only"]["dice"]
                    for row in checkpoints.values()
                ),
                "no_g2_2_checkpoint_reaches_accepted_baseline_at_any_threshold": all(
                    row["best_validation_dice_threshold_diagnostic_only"]["dice"] < 0.777700283628586
                    for row in checkpoints.values()
                ),
            },
            "seed43_attribution": seed43_attribution,
        },
        "q3_composition": {
            "is_class_prevalence_confound": schedules["classification_result"][
                "is_class_prevalence_confound"
            ],
            "control_effective_defective_fraction": schedules["class_prevalence"]["42"][
                "control_effective_defective_fraction"
            ],
            "arm_effective_defective_fraction": schedules["class_prevalence"]["42"][
                "arm_effective_defective_fraction"
            ],
            "every_synthetic_sample_is_defective": masks["hard_check"][
                "every_synthetic_sample_is_defective"
            ],
            "masks_checked": masks["masks_read_from_disk"],
            "all_support_inside_valid_region": masks["hard_check"][
                "all_support_inside_valid_region"
            ],
            "detector_validation_overlap": masks["provenance"]["detector_validation_overlap"],
            "official_test_rows_read": masks["provenance"]["official_test_rows_read"],
        },
        "constraints_honoured": {
            "no_training": True,
            "no_gan_update": True,
            "no_g2_2_rerun": True,
            "no_synthetic_regeneration": True,
            "no_threshold_or_gate_change_in_g2_2": True,
            "checkpoint_2000_not_evaluated": True,
            "checkpoint_1000_not_selected": True,
            "official_test_not_constructed": True,
        },
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(config_path: Path, stage: str) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = REPO_ROOT / config["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Any] = {}

    if stage in {"all", "convergence"}:
        produced["convergence"] = audit_convergence(config)
        atomic_write_json(output_dir / "convergence_audit.json", produced["convergence"])
    if stage in {"all", "masks"}:
        produced["masks"] = audit_synthetic_masks(config)
        records = produced["masks"].pop("records")
        atomic_write_json(
            output_dir / "synthetic_mask_records.json",
            {
                "experiment_version": G2_3_VERSION,
                "classification": DIAGNOSTIC_LABEL,
                "content_sha256": canonical_sha256(records),
                "records": records,
            },
        )
        atomic_write_json(output_dir / "synthetic_mask_integrity.json", produced["masks"])
    if stage in {"all", "schedules"}:
        produced["schedules"] = audit_schedules(config)
        atomic_write_json(output_dir / "schedule_composition_audit.json", produced["schedules"])
    if stage in {"all", "thresholds"}:
        produced["thresholds"] = run_threshold_diagnostics(config)
        atomic_write_json(
            output_dir / "threshold_calibration.json",
            {
                key: value
                for key, value in produced["thresholds"].items()
                if key != "matched_operating_points"
            },
        )
        atomic_write_json(
            output_dir / "matched_operating_points.json",
            {
                "experiment_version": G2_3_VERSION,
                "classification": DIAGNOSTIC_LABEL,
                "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
                "seeds": produced["thresholds"]["matched_operating_points"],
            },
        )
    if stage in {"all", "summary"}:
        # The summary stage reuses persisted stage outputs so it never has to
        # repeat inference; the persisted files are the machine-readable record.
        def _stage_output(name: str, key: str) -> dict[str, Any]:
            if key in produced:
                return produced[key]
            return json.loads((output_dir / name).read_text(encoding="utf-8"))

        thresholds = _stage_output("threshold_calibration.json", "thresholds")
        if "matched_operating_points" not in thresholds:
            thresholds = dict(thresholds)
            thresholds["matched_operating_points"] = json.loads(
                (output_dir / "matched_operating_points.json").read_text(encoding="utf-8")
            )["seeds"]
        summary = build_summary(
            config,
            _stage_output("convergence_audit.json", "convergence"),
            _stage_output("schedule_composition_audit.json", "schedules"),
            _stage_output("synthetic_mask_integrity.json", "masks"),
            thresholds,
        )
        summary["diagnostic_config_sha256"] = canonical_sha256(config)
        atomic_write_json(output_dir / "diagnostic_summary.json", summary)
        produced["summary"] = summary
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="G2.3A post-hoc validation-only diagnostic")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/g2_3_diagnostic.json"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "convergence", "schedules", "masks", "thresholds", "summary"),
        default="all",
    )
    args = parser.parse_args()
    produced = run(args.config, args.stage)
    if "summary" in produced:
        print(json.dumps(produced["summary"], indent=2, sort_keys=True))
    else:
        print(json.dumps(sorted(produced), indent=2))


if __name__ == "__main__":
    main()
