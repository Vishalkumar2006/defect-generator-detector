"""Run the precommitted G2.3B mature-budget, prevalence-controlled utility experiment.

Modes:

    plan    build and hash every arm schedule, re-verify the frozen synthetic
            identity, audit composition, and freeze the confirmation gate. Trains
            nothing and runs on CPU.
    train   train one seed's three arms to the mature budget.
    confirm aggregate seeds 45/46/47 and apply the frozen gate.

There is no mode that reaches the official held-out KSDD2 split. G2.3B has no
code path that can construct, count, inspect, or evaluate it; a future action
there requires separate authorization after a precommitted G2.3B confirmation
PASS. Every split name this script can pass to a loader is checked by
``assert_permitted_split``, which admits development train and validation only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.augmentation import SynchronizedRandomFlips  # noqa: E402
from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.models import UNet, count_parameters  # noqa: E402
from defectgen.training.failure_diagnostics import model_state_sha256  # noqa: E402
from defectgen.training.engine import capture_random_states, restore_random_states  # noqa: E402
from defectgen.training.final_baseline import EarlyStopping  # noqa: E402
from defectgen.training.g2_2_utility import stratified_validation_metrics  # noqa: E402
from defectgen.training.g2_3_diagnostic import (  # noqa: E402
    PixelHistogram,
    build_probability_grid,
    pr_auc,
    threshold_curve,
)
from defectgen.training.g2_3b_protocol import (  # noqa: E402
    ARMS,
    EVALUATION_SPLIT,
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
    TRAINING_SPLIT,
    FrozenSyntheticDataset,
    ScheduledCompositionDataset,
    arm_comparison,
    arm_slot_counts,
    assert_completed_arm_compatible,
    assert_durable_counters,
    assert_equal_budgets,
    assert_resume_compatible,
    assert_restored_learning_rate,
    assert_evaluation_split,
    assert_permitted_split,
    batch_pattern,
    atomic_torch_save,
    atomic_write_json,
    budget_plan,
    canonical_sha256,
    confirmation_decision,
    effective_class_balance,
    per_epoch_composition,
    resume_start_epoch,
    run_identity,
    schedule_composition,
    schedule_payload,
    select_operating_threshold,
    shared_class_stream_prefixes,
    stream_lengths,
    threshold_grid,
    validate_batch_patterns,
    verify_frozen_synthetic_identity,
    build_arm_schedule,
)
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.metrics import detailed_validation_metrics  # noqa: E402
from defectgen.training.numerics import NumericalStepController, precision_autocast  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_version"] != G2_3B_VERSION:
        raise ValueError("Unexpected G2.3B experiment version")
    policy = config["access_policy"]
    for name in (
        "official_test_allowed",
        "official_test_allowed_after_confirmation",
        "gan_updates_allowed",
        "regenerate_synthetic_allowed",
        "modifies_g2_2_artifacts",
        "modifies_g2_3a_artifacts",
    ):
        if policy[name]:
            raise RuntimeError(f"G2.3B access policy must keep {name} false")
    if list(policy["evaluation_splits_allowed"]) != [EVALUATION_SPLIT]:
        raise RuntimeError("G2.3B may evaluate the validation split only")
    if list(policy["training_splits_allowed"]) != [TRAINING_SPLIT]:
        raise RuntimeError("G2.3B may train on the development training split only")
    if config["immutable_inputs"]["g2_2_terminal_decision"] != G2_2_TERMINAL_DECISION:
        raise RuntimeError("The G2.2 terminal decision is an immutable G2.3B input")
    if tuple(int(value) for value in config["seeds"]) != G2_3B_SEEDS:
        raise ValueError(f"G2.3B seeds are frozen at {G2_3B_SEEDS}")
    if tuple(sorted(config["arms"])) != tuple(sorted(ARMS)):
        raise ValueError("G2.3B arms are frozen")
    if not config["confirmation_gate"]["frozen_before_training"]:
        raise RuntimeError("The G2.3B gate must be frozen before any training")
    if config["confirmation_gate"]["primary_candidate"] != PRIMARY_CANDIDATE:
        raise ValueError("The primary candidate arm is frozen")
    if config["confirmation_gate"]["primary_control"] != PRIMARY_CONTROL:
        raise ValueError("The primary control arm is frozen")
    plan = budget_plan(config["training"])
    validate_batch_patterns(plan)
    for arm, settings in config["arms"].items():
        counts = arm_slot_counts(arm, plan)
        balance = effective_class_balance(counts)
        for field, category in (
            ("normal_real_fraction", SOURCE_NORMAL_REAL),
            ("defective_real_fraction", SOURCE_DEFECTIVE_REAL),
            ("synthetic_fraction", SOURCE_SYNTHETIC),
        ):
            if abs(float(settings[field]) - balance[field]) > 1e-12:
                raise ValueError(f"{arm}: configured {field} disagrees with its batch pattern")
        if [list(batch) for batch in settings["batch_pattern"]] != [
            list(batch) for batch in batch_pattern(arm)
        ]:
            raise ValueError(f"{arm}: configured batch pattern disagrees with the frozen pattern")
    return config


def _dataset(config: dict[str, Any], split: str) -> KSDD2FullImageDataset:
    assert_permitted_split(split)
    data = config["data"]
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
    if any(row["development_split"] != split for row in dataset.rows):
        raise RuntimeError(f"A foreign row entered the G2.3B {split} split")
    return dataset


def _class_pools(dataset: KSDD2FullImageDataset) -> tuple[list[int], list[int]]:
    labels = dataset.labels
    normal = [index for index, label in enumerate(labels) if not label]
    defective = [index for index, label in enumerate(labels) if label]
    if not normal or not defective:
        raise RuntimeError("Both class pools must be non-empty")
    return normal, defective


def _synthetic_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (REPO_ROOT / config["synthetic"]["manifest_path"]).read_text(encoding="utf-8")
    )


def _strata(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Replicate the frozen G2.2 stratification (development-training tertiles)."""
    path = REPO_ROOT / config["immutable_inputs"]["bbox_statistics_path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train_pixels = sorted(
        int(row["mask_pixels"])
        for row in rows
        if row["development_split"] == TRAINING_SPLIT and int(row["mask_pixels"]) > 0
    )
    if not train_pixels:
        raise RuntimeError("No train-only defect geometry was available")
    small_cutoff = train_pixels[len(train_pixels) // 3]
    large_cutoff = train_pixels[(2 * len(train_pixels)) // 3]
    mapping: dict[str, str] = {}
    for row in rows:
        if row["development_split"] != EVALUATION_SPLIT or int(row["mask_pixels"]) <= 0:
            continue
        pixels = int(row["mask_pixels"])
        size = "small" if pixels <= small_cutoff else "medium" if pixels <= large_cutoff else "large"
        border = "border" if row["touches_border"].lower() == "true" else "non_border"
        mapping[row["sample_id"] + ":size"] = f"size:{size}"
        mapping[row["sample_id"] + ":border"] = f"contact:{border}"
    return mapping, {
        "source": "development-training defective mask-pixel tertiles",
        "small_max_mask_pixels": small_cutoff,
        "medium_max_mask_pixels": large_cutoff,
    }


def _expanded_stratified(per_image, sample_ids, mapping) -> dict[str, Any]:
    rows, ids, selected = [], [], {}
    for row, sample_id in zip(per_image, sample_ids):
        if not bool(row["has_defect"]):
            continue
        for kind in ("size", "border"):
            key = f"{sample_id}:{kind}"
            if key in mapping:
                rows.append(row)
                ids.append(key)
                selected[key] = mapping[key]
    return stratified_validation_metrics(rows, ids, selected)


# --------------------------------------------------------------------------- #
# Plan mode -- freeze the protocol before any training
# --------------------------------------------------------------------------- #


def build_plan(config: dict[str, Any], *, write_expanded: bool = True) -> dict[str, Any]:
    plan = budget_plan(config["training"])
    training = _dataset(config, TRAINING_SPLIT)
    validation = _dataset(config, EVALUATION_SPLIT)
    assert_evaluation_split(EVALUATION_SPLIT)
    normal_pool, defective_pool = _class_pools(training)
    synthetic_identity = verify_frozen_synthetic_identity(REPO_ROOT, config["synthetic"])
    synthetic_count = int(synthetic_identity["row_count"])
    labels = training.labels
    plan_dir = REPO_ROOT / config["paths"]["plan_directory"]

    seeds: dict[str, Any] = {}
    for seed in G2_3B_SEEDS:
        schedules = {
            arm: build_arm_schedule(
                arm,
                seed=seed,
                plan=plan,
                normal_pool=normal_pool,
                defective_pool=defective_pool,
                synthetic_pool_size=synthetic_count,
            )
            for arm in ARMS
        }
        assert_equal_budgets(schedules, plan)
        payloads = {arm: schedule_payload(entries) for arm, entries in schedules.items()}
        arms: dict[str, Any] = {}
        for arm in ARMS:
            payload = payloads[arm]
            observed = schedule_composition(payload, labels)
            expected = effective_class_balance(arm_slot_counts(arm, plan))
            expected_total = {
                key: value * plan.maximum_epochs
                for key, value in arm_slot_counts(arm, plan).items()
            }
            matches = all(
                observed[field] == expected_total[category]
                for field, category in (
                    ("normal_real_samples", SOURCE_NORMAL_REAL),
                    ("defective_real_samples", SOURCE_DEFECTIVE_REAL),
                    ("synthetic_samples", SOURCE_SYNTHETIC),
                )
            )
            schedule_hash = canonical_sha256(payload)
            if write_expanded:
                atomic_write_json(
                    plan_dir / f"seed{seed}" / f"{arm}_schedule.json",
                    {
                        "experiment_version": G2_3B_VERSION,
                        "seed": seed,
                        "arm": arm,
                        "entries": payload,
                        "content_sha256": schedule_hash,
                    },
                )
            arms[arm] = {
                "schedule_sha256": schedule_hash,
                "slot_count": len(payload),
                "optimizer_updates": plan.total_optimizer_updates,
                "per_epoch_slot_counts": arm_slot_counts(arm, plan),
                "per_epoch_target_fractions": {
                    "normal_real_fraction": expected["normal_real_fraction"],
                    "defective_real_fraction": expected["defective_real_fraction"],
                    "synthetic_fraction": expected["synthetic_fraction"],
                },
                "observed_composition": observed,
                "observed_matches_precommitted_composition": matches,
                "per_epoch_composition": per_epoch_composition(payload, plan),
            }
        seeds[str(seed)] = {
            "arms": arms,
            "class_stream_prefixes_shared_across_arms": {
                category: all(
                    shared_class_stream_prefixes(payloads, category=category, epoch=epoch)
                    for epoch in range(1, plan.maximum_epochs + 1)
                )
                for category in SOURCE_CATEGORIES
            },
            "effective_defective_fraction_by_arm": {
                arm: arms[arm]["observed_composition"]["effective_defective_fraction"]
                for arm in ARMS
            },
        }

    report = {
        "experiment_version": G2_3B_VERSION,
        "stage": "precommitted_plan_no_training_executed",
        "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
        "detector_optimizer_updates_executed": 0,
        "gan_optimizer_updates": 0,
        "synthetic_samples_regenerated": 0,
        "official_test_samples_loaded": 0,
        "budget": {
            "optimizer_updates_per_epoch": plan.optimizer_updates_per_epoch,
            "batch_size": plan.batch_size,
            "sample_slots_per_epoch": plan.slots_per_epoch,
            "maximum_epochs": plan.maximum_epochs,
            "total_optimizer_updates": plan.total_optimizer_updates,
            "total_sample_slots": plan.total_slots,
            "historical_baseline_total_updates": int(
                config["immutable_inputs"]["historical_baseline_total_updates"]
            ),
            "matches_historical_mature_budget": plan.total_optimizer_updates
            == int(config["immutable_inputs"]["historical_baseline_total_updates"]),
        },
        "pools": {
            "development_training_rows": len(training),
            "development_training_normal": len(normal_pool),
            "development_training_defective": len(defective_pool),
            "development_validation_rows": len(validation),
            "synthetic_rows": synthetic_count,
            "class_stream_lengths_per_epoch": stream_lengths(plan),
        },
        "frozen_synthetic_identity": synthetic_identity,
        "threshold_selection": {
            **{key: value for key, value in config["threshold_selection"].items()},
            "grid_values_sha256": canonical_sha256(threshold_grid(config["threshold_selection"])),
            "grid_point_count": len(threshold_grid(config["threshold_selection"])),
            "contains_secondary_fixed_threshold": 0.5
            in threshold_grid(config["threshold_selection"]),
        },
        "confirmation_gate": config["confirmation_gate"],
        "comparisons": config["comparisons"],
        "seeds": seeds,
        "expanded_schedules_written": bool(write_expanded),
    }
    report["plan_content_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "plan_content_sha256"}
    )
    atomic_write_json(plan_dir / "precommitted_plan.json", report)
    return report


# --------------------------------------------------------------------------- #
# Evaluation -- validation only, one precommitted threshold rule
# --------------------------------------------------------------------------- #


def _collect_validation_outputs(model, loader, criterion, device, precision):
    model.eval()
    probabilities, targets, valid_regions, labels, sample_ids = [], [], [], [], []
    loss_sum, samples = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, memory_format=torch.channels_last)
            mask = batch["mask"].to(device)
            valid = batch["valid_region"].to(device)
            with precision_autocast(device.type, precision):
                logits = model(image)
            components = criterion.components(logits.float(), mask.float(), valid.float())
            if not torch.isfinite(components["total"]):
                raise RuntimeError("Non-finite validation loss; automatic fp32 retry is disabled")
            loss_sum += float(components["total"].item()) * image.shape[0]
            samples += image.shape[0]
            probabilities.append(torch.sigmoid(logits.float()).cpu())
            targets.append(mask.bool().cpu())
            valid_regions.append(valid.bool().cpu())
            labels.append(batch["has_defect"].bool().cpu())
            sample_ids.extend(str(value) for value in batch["sample_id"])
    return (
        loss_sum / max(samples, 1),
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(valid_regions),
        torch.cat(labels),
        sample_ids,
    )


def _pixel_pr_auc(probabilities, targets, valid_regions, grid_settings) -> dict[str, Any]:
    grid = build_probability_grid(
        probability_step=float(grid_settings["probability_step"]),
        logit_limit=float(grid_settings["logit_limit"]),
        logit_step=float(grid_settings["logit_step"]),
    )
    positive, negative = PixelHistogram(grid), PixelHistogram(grid)
    for index in range(probabilities.shape[0]):
        selected = valid_regions[index, 0].numpy().ravel()
        probability = probabilities[index, 0].numpy().ravel()[selected].astype(np.float64)
        defect = targets[index, 0].numpy().ravel()[selected]
        positive.update(probability[defect])
        negative.update(probability[~defect])
    curve = threshold_curve(positive, negative)
    return {
        "pixel_pr_auc": pr_auc(curve["recall"], curve["precision"]),
        "estimator": "step_rule_average_precision",
        "grid_points": int(grid.size),
        "recall_span": [float(curve["recall"].min()), float(curve["recall"].max())],
    }


def _reportable(metrics: dict[str, Any], stratified: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": float(metrics["threshold"]),
        "global_dice": float(metrics["global_dice"]),
        "global_iou": float(metrics["global_iou"]),
        "pixel_precision": float(metrics["pixel_precision"]),
        "pixel_recall": float(metrics["pixel_recall"]),
        "normal_image_false_positive_rate": float(metrics["normal_image_false_positive_rate"]),
        "image_recall": float(metrics["image_recall"]),
        "image_precision": float(metrics["image_precision"]),
        "image_f1": float(metrics["image_f1"]),
        "mean_defective_image_dice": float(metrics["mean_defective_image_dice"]),
        "defective_images_zero_detected_pixels": int(
            metrics["defective_images_zero_detected_pixels"]
        ),
        "stratified_dice_and_recall": {
            group: {"dice": float(row["dice"]), "recall": float(row["recall"]), "images": int(row["images"])}
            for group, row in sorted(stratified.items())
        },
    }


def evaluate_arm(
    model, validation, criterion, *, config: dict[str, Any], device, strata
) -> dict[str, Any]:
    settings = config["threshold_selection"]
    assert_evaluation_split(str(settings["data_source"]))
    grid = threshold_grid(settings)
    loader = DataLoader(
        validation,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    loss, probabilities, targets, valid_regions, labels, sample_ids = _collect_validation_outputs(
        model, loader, criterion, device, config["training"]["precision"]["mode"]
    )
    sweep_rows = []
    per_image_by_threshold: dict[float, Any] = {}
    for threshold in grid:
        metrics, per_image = detailed_validation_metrics(
            probabilities, targets, valid_regions, labels, threshold
        )
        sweep_rows.append(metrics)
        per_image_by_threshold[float(threshold)] = per_image
    selection = select_operating_threshold(sweep_rows, grid)
    selected = float(selection["selected_threshold"])
    selected_metrics = dict(selection["selected_row"])
    fixed = float(settings["secondary_fixed_threshold"])
    if fixed not in per_image_by_threshold:
        raise RuntimeError("The secondary fixed threshold must lie on the precommitted grid")
    fixed_metrics = next(row for row in sweep_rows if float(row["threshold"]) == fixed)
    result = {
        "validation_loss": loss,
        "validation_sample_count": len(sample_ids),
        "validation_sample_ids_sha256": canonical_sha256(sample_ids),
        "official_test_samples_loaded": 0,
        "threshold_selection": {
            key: value for key, value in selection.items() if key != "selected_row"
        },
        "at_selected_threshold": _reportable(
            selected_metrics,
            _expanded_stratified(per_image_by_threshold[selected], sample_ids, strata),
        ),
        "at_fixed_threshold_0_5_secondary_continuity_only": _reportable(
            fixed_metrics,
            _expanded_stratified(per_image_by_threshold[fixed], sample_ids, strata),
        ),
        "threshold_sweep": sweep_rows,
        **_pixel_pr_auc(probabilities, targets, valid_regions, settings["pr_auc_grid"]),
    }
    del probabilities, targets, valid_regions, labels, per_image_by_threshold
    return result


# --------------------------------------------------------------------------- #
# Training -- mature stabilized-BF16 semantics, exact equal budgets
# --------------------------------------------------------------------------- #


def _build_model(config: dict[str, Any], seed: int, device) -> tuple[torch.nn.Module, str]:
    configure_reproducibility(seed, deterministic=True, warn_only=True)
    model_config = config["training"]["model"]
    model = UNet(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config["output_channels"]),
        base_channels=int(model_config["base_channels"]),
    )
    initialization = model_state_sha256(model)
    return model.to(device, memory_format=torch.channels_last), initialization


def train_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    training,
    validation,
    synthetic,
    schedule,
    strata,
    report_dir: Path,
    checkpoint_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    settings = config["training"]
    plan = budget_plan(settings)
    # A finished arm is immutable evidence. Reuse goes through run_training, so
    # reaching train_arm with a completed report means something is about to
    # overwrite it. Checked first, before any device or model work.
    completed_path = report_dir / f"{arm}.json"
    if completed_path.is_file():
        raise RuntimeError(
            f"Refusing to overwrite the completed immutable arm report {completed_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; G2.3B refuses a CPU fallback for mature training")
    if settings["precision"]["mode"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16; refusing to begin G2.3B")
    if settings["precision"]["grad_scaler"]:
        raise ValueError("The BF16 protocol must not enable GradScaler")
    device = torch.device("cuda:0")
    model, initialization = _build_model(config, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["optimizer"]["learning_rate"]),
        weight_decay=float(settings["optimizer"]["weight_decay"]),
    )
    scheduler_settings = settings["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_settings["mode"],
        factor=float(scheduler_settings["factor"]),
        patience=int(scheduler_settings["patience"]),
        min_lr=float(scheduler_settings["minimum_learning_rate"]),
    )
    criterion = CombinedBCEDiceLoss(
        bce_weight=float(settings["loss"]["bce_weight"]),
        dice_weight=float(settings["loss"]["dice_weight"]),
        pos_weight=float(settings["loss"]["pos_weight"]),
    )
    controller = NumericalStepController(
        optimizer,
        precision_mode=settings["precision"]["mode"],
        gradient_clip_max_norm=float(settings["precision"]["gradient_clip_max_norm"]),
        automatic_fp32_retry=False,
    )
    stopping_settings = settings["early_stopping"]
    if stopping_settings["enforced"]:
        raise RuntimeError("G2.3B early stopping must remain monitor-only to keep budgets equal")
    stopping = EarlyStopping(
        patience=int(stopping_settings["patience"]),
        minimum_delta=float(stopping_settings["minimum_delta"]),
    )
    augmentation = settings["augmentation"]
    transform = SynchronizedRandomFlips(
        horizontal_probability=float(augmentation["horizontal_flip_probability"]),
        vertical_probability=float(augmentation["vertical_flip_probability"]),
        seed=seed,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{arm}_best.pt"
    last_path = checkpoint_dir / f"{arm}_last.pt"
    schedule_hash = canonical_sha256(schedule_payload(schedule))
    identity = run_identity(
        arm=arm,
        seed=seed,
        schedule_sha256=schedule_hash,
        initialization_sha256=initialization,
        config_sha256=canonical_sha256(config),
        plan=plan,
    )

    start_epoch = 1
    records: list[dict[str, Any]] = []
    best = {"epoch": 0, "validation_total_loss": math.inf}
    # Resume is automatic, never opt-in: if durable state exists it is either
    # compatible and resumed from, or incompatible and fatal. There is no path
    # that silently discards completed epochs and starts over.
    if last_path.is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        assert_resume_compatible(payload.get("run_identity", {}), identity)
        assert_durable_counters(
            payload["numerical_state"]["counters"],
            last_completed_epoch=int(payload["epoch"]),
            plan=plan,
        )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        controller.load_state_dict(payload["numerical_state"])
        stopping.load_state_dict(payload["early_stopping_state"])
        restore_random_states(payload["random_states"])
        records = list(payload["epoch_records"])
        best = dict(payload["best_validation"])
        assert_restored_learning_rate(optimizer.param_groups[0]["lr"], records)
        start_epoch = resume_start_epoch(int(payload["epoch"]), plan)
        print(
            f"resuming {arm} seed={seed} from durable epoch {payload['epoch']}; "
            f"next epoch {start_epoch}/{plan.maximum_epochs}",
            flush=True,
        )
    elif resume:
        print(f"no durable state for {arm} seed={seed}; starting from epoch 1", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, plan.maximum_epochs + 1):
        epoch_start = time.perf_counter()
        slice_start = (epoch - 1) * plan.slots_per_epoch
        epoch_schedule = schedule[slice_start : slice_start + plan.slots_per_epoch]
        loader = DataLoader(
            ScheduledCompositionDataset(training, synthetic, epoch_schedule, transform),
            batch_size=plan.batch_size,
            shuffle=False,
            num_workers=int(settings["num_workers"]),
            pin_memory=bool(settings["pin_memory"] and torch.cuda.is_available()),
        )
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        losses: list[float] = []
        source_counts = {category: 0 for category in SOURCE_CATEGORIES}
        defective_slots = 0
        for batch_index, batch in enumerate(loader, start=1):
            telemetry = controller.run_batch(
                model,
                batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last),
                batch["mask"].to(device, non_blocking=True),
                batch["valid_region"].to(device, non_blocking=True),
                criterion,
            )
            if not telemetry.optimizer_step_executed or telemetry.optimizer_updates_this_batch != 1:
                raise RuntimeError(
                    f"{arm} seed={seed} epoch={epoch}: a skipped or duplicated update is fatal "
                    f"at batch {batch_index}"
                )
            losses.append(float(telemetry.total_loss))
            for source in batch["schedule_source"]:
                source_counts[str(source)] += 1
            defective_slots += int(batch["has_defect"].sum().item())
        if len(losses) != plan.optimizer_updates_per_epoch:
            raise RuntimeError(f"{arm}: epoch {epoch} did not deliver the exact update budget")
        validation_loss, probabilities, targets, valid_regions, labels, _ = (
            _collect_validation_outputs(
                model,
                DataLoader(validation, batch_size=plan.batch_size, shuffle=False, num_workers=0),
                criterion,
                device,
                settings["precision"]["mode"],
            )
        )
        metrics, _ = detailed_validation_metrics(
            probabilities, targets, valid_regions, labels, 0.5
        )
        del probabilities, targets, valid_regions, labels
        scheduler.step(validation_loss)
        # Monitor-only: the historical baseline configured early stopping but never
        # triggered it. Acting on it here would break the equal-budget invariant.
        would_stop = stopping.step(validation_loss)
        improved = validation_loss < float(best["validation_total_loss"])
        if improved:
            best = {"epoch": epoch, "validation_total_loss": validation_loss}
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_total_loss": statistics.mean(losses),
            "final_100_train_loss": statistics.mean(losses[-100:]),
            "validation_total_loss": validation_loss,
            "validation_global_dice_at_0_5": float(metrics["global_dice"]),
            "validation_pixel_precision_at_0_5": float(metrics["pixel_precision"]),
            "validation_pixel_recall_at_0_5": float(metrics["pixel_recall"]),
            "optimizer_updates": len(losses),
            "scheduled_defective_slot_fraction": defective_slots / plan.slots_per_epoch,
            "source_counts": dict(source_counts),
            "early_stopping_would_have_triggered": bool(would_stop),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        records.append(record)
        state = {
            "experiment_version": G2_3B_VERSION,
            "arm": arm,
            "seed": seed,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "numerical_state": controller.state_dict(),
            "early_stopping_state": stopping.state_dict(),
            "random_states": capture_random_states(),
            "epoch_records": records,
            "best_validation": best,
            "initialization_sha256": initialization,
            "schedule_sha256": schedule_hash,
            "run_identity": identity,
        }
        # best first, then last: if the process dies between the two writes the
        # durable epoch only ever moves backwards, never past a missing best.
        if improved:
            atomic_torch_save(best_path, state)
        atomic_torch_save(last_path, state)
        atomic_write_json(report_dir / f"{arm}_epochs.json", records)
        print(
            f"{arm} seed={seed} epoch={epoch}/{plan.maximum_epochs} "
            f"train={record['train_total_loss']:.6f} validation={validation_loss:.6f} "
            f"dice@0.5={metrics['global_dice']:.6f} lr={learning_rate:.6g}",
            flush=True,
        )

    counters = controller.state_dict()["counters"]
    if counters["optimizer_step_executed"] != plan.total_optimizer_updates:
        raise RuntimeError("The arm did not receive the exact precommitted optimizer budget")
    if counters["optimizer_step_skipped"]:
        raise RuntimeError("A skipped optimizer update is fatal in G2.3B")
    controller.close()

    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    evaluation = evaluate_arm(
        model, validation, criterion, config=config, device=device, strata=strata
    )
    result = {
        **identity,
        "evaluation_split": EVALUATION_SPLIT,
        "final_model_sha256": model_state_sha256(model),
        "selected_epoch": int(payload["epoch"]),
        "checkpoint_selection": settings["checkpoint_selection"],
        "optimizer_updates": int(counters["optimizer_step_executed"]),
        "attempted_batches": int(counters["attempted_batches"]),
        "skipped_updates": int(counters["optimizer_step_skipped"]),
        "model_parameter_count": count_parameters(model),
        "epoch_records": records,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_best": best_path.relative_to(REPO_ROOT).as_posix(),
        **evaluation,
    }
    atomic_write_json(report_dir / f"{arm}.json", result)
    return result


def run_training(config: dict[str, Any], seed: int, *, resume: bool = False) -> dict[str, Any]:
    if seed not in G2_3B_SEEDS:
        raise ValueError(f"G2.3B seeds are frozen at {G2_3B_SEEDS}")
    plan = budget_plan(config["training"])
    training = _dataset(config, TRAINING_SPLIT)
    validation = _dataset(config, EVALUATION_SPLIT)
    strata, _ = _strata(config)
    normal_pool, defective_pool = _class_pools(training)
    identity = verify_frozen_synthetic_identity(REPO_ROOT, config["synthetic"])
    manifest = _synthetic_manifest(config)
    normalization = config["data"]["detector_normalization"]
    synthetic = FrozenSyntheticDataset(
        REPO_ROOT,
        manifest,
        mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"],
    )
    schedules = {
        arm: build_arm_schedule(
            arm,
            seed=seed,
            plan=plan,
            normal_pool=normal_pool,
            defective_pool=defective_pool,
            synthetic_pool_size=len(synthetic),
        )
        for arm in ARMS
    }
    assert_equal_budgets(schedules, plan)
    report_dir = REPO_ROOT / config["paths"]["report_directory"] / f"seed{seed}"
    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_directory"] / f"seed{seed}"
    results = {}
    config_hash = canonical_sha256(config)
    for arm in ARMS:
        completed = report_dir / f"{arm}.json"
        if completed.is_file():
            report = json.loads(completed.read_text(encoding="utf-8"))
            # A completed arm is immutable evidence, but only if it belongs to
            # this exact configuration, schedule, seed, and budget.
            assert_completed_arm_compatible(
                report,
                run_identity(
                    arm=arm,
                    seed=seed,
                    schedule_sha256=canonical_sha256(schedule_payload(schedules[arm])),
                    initialization_sha256=report.get("initialization_sha256", ""),
                    config_sha256=config_hash,
                    plan=plan,
                ),
            )
            print(f"reusing verified completed immutable arm: seed{seed}:{arm}", flush=True)
            results[arm] = report
            continue
        results[arm] = train_arm(
            config,
            arm=arm,
            seed=seed,
            training=training,
            validation=validation,
            synthetic=synthetic if arm == PRIMARY_CANDIDATE else None,
            schedule=schedules[arm],
            strata=strata,
            report_dir=report_dir,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )
    initializations = {result["initialization_sha256"] for result in results.values()}
    if len(initializations) != 1:
        raise RuntimeError(f"Seed {seed} arms did not share one initialization")
    budgets = {result["optimizer_updates"] for result in results.values()}
    if budgets != {plan.total_optimizer_updates}:
        raise RuntimeError(f"Seed {seed} arms did not share one optimizer budget: {budgets}")
    summary = {
        "experiment_version": G2_3B_VERSION,
        "stage": "single_seed_three_arm_training",
        "seed": seed,
        "shared_initialization_sha256": initializations.pop(),
        "optimizer_updates_per_arm": plan.total_optimizer_updates,
        "frozen_synthetic_identity": identity,
        "arms": results,
        "official_test_access_count": 0,
        "gan_optimizer_updates": 0,
    }
    atomic_write_json(report_dir / "seed_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Confirmation
# --------------------------------------------------------------------------- #


def run_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    report_root = REPO_ROOT / config["paths"]["report_directory"]
    rules = config["confirmation_gate"]
    seeds: dict[str, Any] = {}
    primary: list[dict[str, float]] = []
    secondary: list[dict[str, float]] = []
    for seed in G2_3B_SEEDS:
        path = report_root / f"seed{seed}" / "seed_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Seed {seed} has not completed: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        arms = summary["arms"]
        metrics = {
            arm: {
                **arms[arm]["at_selected_threshold"],
                "pixel_pr_auc": arms[arm]["pixel_pr_auc"],
            }
            for arm in ARMS
        }
        primary_comparison = arm_comparison(
            metrics[PRIMARY_CANDIDATE], metrics[PRIMARY_CONTROL]
        )
        secondary_comparison = arm_comparison(
            metrics[SECONDARY_CANDIDATE], metrics[SECONDARY_CONTROL]
        )
        primary.append(primary_comparison)
        secondary.append(secondary_comparison)
        seeds[str(seed)] = {
            "selected_thresholds": {
                arm: arms[arm]["threshold_selection"]["selected_threshold"] for arm in ARMS
            },
            "metrics_at_selected_threshold": metrics,
            "metrics_at_fixed_threshold_0_5_secondary_continuity_only": {
                arm: arms[arm]["at_fixed_threshold_0_5_secondary_continuity_only"] for arm in ARMS
            },
            "primary_comparison_gan_1500_minus_prevalence_matched_real": primary_comparison,
            "secondary_comparison_standard_real_minus_prevalence_matched_real": secondary_comparison,
        }
    confirmed, aggregate = confirmation_decision(primary, rules=rules)
    secondary_means = {
        name: float(np.mean([row[name] for row in secondary])) for name in secondary[0]
    }
    result = {
        "experiment_version": G2_3B_VERSION,
        "stage": "three_seed_primary_confirmation",
        "seeds": list(G2_3B_SEEDS),
        "primary_comparison": {
            "candidate": PRIMARY_CANDIDATE,
            "control": PRIMARY_CONTROL,
            "gated": True,
        },
        "secondary_comparison": {
            "candidate": SECONDARY_CANDIDATE,
            "control": SECONDARY_CONTROL,
            "gated": False,
            "mean_deltas": secondary_means,
        },
        "frozen_gate": rules,
        "seed_results": seeds,
        "aggregate": aggregate,
        "confirmed": confirmed,
        "decision": aggregate["decision"],
        "official_test_access_count": 0,
        "official_test_authorized_by_this_decision": False,
        "gan_optimizer_updates": 0,
        "g2_2_terminal_decision_unchanged": G2_2_TERMINAL_DECISION,
    }
    atomic_write_json(report_root / "confirmation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="G2.3B precommitted utility experiment")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/g2_3b_utility_confirmation.json"
    )
    parser.add_argument("--mode", choices=("plan", "train", "confirm"), default="plan")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.mode == "plan":
        payload = build_plan(config)
    elif args.mode == "train":
        if args.seed is None:
            raise SystemExit("--seed is required for --mode train")
        payload = run_training(config, int(args.seed), resume=args.resume)
    else:
        payload = run_confirmation(config)
    print(json.dumps({key: value for key, value in payload.items() if key != "seeds"}, indent=2)[:4000])


if __name__ == "__main__":
    raise SystemExit(main())
