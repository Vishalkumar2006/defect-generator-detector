"""Run the gated G2.2 equal-budget detector utility experiment."""

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
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.augmentation import SynchronizedRandomFlips  # noqa: E402
from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.models import UNet  # noqa: E402
from defectgen.training.failure_diagnostics import model_state_sha256  # noqa: E402
from defectgen.training.g2_2_utility import (  # noqa: E402
    G2_2_VERSION,
    ScheduledMixtureDataset,
    SyntheticDetectorDataset,
    assert_paired_manifests,
    atomic_write_json,
    build_equal_budget_schedule,
    canonical_sha256,
    confirmation_decision,
    meaningful_winner,
    stratified_validation_metrics,
)
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.metrics import detailed_validation_metrics  # noqa: E402
from defectgen.training.numerics import NumericalStepController, precision_autocast  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_version"] != G2_2_VERSION:
        raise ValueError("Unexpected G2.2 experiment version")
    detector = config["detector"]
    if float(detector["synthetic_fraction"]) != 0.25 or int(detector["batch_size"]) != 4:
        raise ValueError("G2.2 is frozen to one synthetic sample in every four-sample batch")
    if not detector["constant_learning_rate"]:
        raise ValueError("G2.2 requires the same constant learning-rate schedule in every arm")
    if config["official_test"]["enabled_during_pilot"] or config["official_test"]["enabled_during_confirmation"]:
        raise ValueError("Official test access must be disabled during pilot and confirmation")
    return config


def _base_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads((REPO_ROOT / config["detector"]["base_config_path"]).read_text(encoding="utf-8"))


def _real_dataset(base: dict[str, Any], split: str) -> KSDD2FullImageDataset:
    if split not in {"train", "validation"}:
        raise RuntimeError("Pilot/confirmation may construct only detector train or validation")
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
    if any(row["development_split"] not in {"train", "validation"} for row in dataset.rows):
        raise RuntimeError("Official-test row entered G2.2 pilot/confirmation")
    return dataset


def _load_synthetic_manifests(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    directory = REPO_ROOT / config["gan"]["manifest_directory"]
    manifests = {
        name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("checkpoint_1000", "checkpoint_1500")
    }
    for name, manifest in manifests.items():
        content_hash = manifest.pop("content_sha256")
        if canonical_sha256(manifest) != content_hash:
            raise RuntimeError(f"Synthetic manifest content hash failed for {name}")
        manifest["content_sha256"] = content_hash
        if manifest["official_test_source_count"] or manifest["detector_validation_source_count"]:
            raise RuntimeError("Forbidden source rows are declared in a synthetic manifest")
    assert_paired_manifests(manifests["checkpoint_1000"]["rows"], manifests["checkpoint_1500"]["rows"])
    return manifests


def _strata() -> tuple[dict[str, str], dict[str, Any]]:
    path = REPO_ROOT / "reports/preprocessing/bbox_statistics.csv"
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
    result: dict[str, str] = {}
    for row in rows:
        if row["development_split"] != "validation" or int(row["mask_pixels"]) <= 0:
            continue
        pixels = int(row["mask_pixels"])
        size = "small" if pixels <= small_cutoff else "medium" if pixels <= large_cutoff else "large"
        border = "border" if row["touches_border"].lower() == "true" else "non_border"
        # Each validation defect participates in both requested stratifications.
        result[row["sample_id"] + ":size"] = f"size:{size}"
        result[row["sample_id"] + ":border"] = f"contact:{border}"
    return result, {
        "source": "development-training defective mask-pixel tertiles",
        "small_max_mask_pixels": small_cutoff,
        "medium_max_mask_pixels": large_cutoff,
    }


def _expanded_stratified(per_image, sample_ids, mapping):
    expanded_rows, expanded_ids, expanded_mapping = [], [], {}
    for row, sample_id in zip(per_image, sample_ids):
        if not bool(row["has_defect"]):
            continue
        for kind in ("size", "border"):
            key = f"{sample_id}:{kind}"
            if key in mapping:
                expanded_rows.append(row)
                expanded_ids.append(key)
                expanded_mapping[key] = mapping[key]
    return stratified_validation_metrics(expanded_rows, expanded_ids, expanded_mapping)


def _evaluate(model, validation, *, device, precision, threshold, batch_size, official_test=False):
    loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    probabilities, targets, valid_regions, labels, sample_ids = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, memory_format=torch.channels_last)
            with precision_autocast(device.type, precision):
                logits = model(image)
            probabilities.append(torch.sigmoid(logits.float()).cpu())
            targets.append(batch["mask"].bool())
            valid_regions.append(batch["valid_region"].bool())
            labels.append(batch["has_defect"].bool())
            sample_ids.extend(str(value) for value in batch["sample_id"])
    overall, per_image = detailed_validation_metrics(
        torch.cat(probabilities), torch.cat(targets), torch.cat(valid_regions), torch.cat(labels), threshold
    )
    result = {
        "overall": overall,
        "validation_sample_count": len(sample_ids),
        "validation_sample_ids_sha256": canonical_sha256(sample_ids),
        "official_test_samples_loaded": len(sample_ids) if official_test else 0,
    }
    if official_test:
        result["validation_sample_count"] = 0
        result["official_test_sample_count"] = len(sample_ids)
        result["official_test_sample_ids_sha256"] = result.pop("validation_sample_ids_sha256")
    else:
        strata, thresholds = _strata()
        result["stratified"] = _expanded_stratified(per_image, sample_ids, strata)
        result["stratification_thresholds"] = thresholds
    return result


def _train_variant(
    config: dict[str, Any],
    base: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    *,
    variant: str,
    seed: int,
    report_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    detector = config["detector"]
    configure_reproducibility(seed, deterministic=True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = _real_dataset(base, "train")
    validation = _real_dataset(base, "validation")
    normalization = base["data"]["detector_normalization"]
    synthetic = None
    synthetic_count = int(config["gan"]["synthetic_sample_count"])
    if variant != "real_only":
        synthetic = SyntheticDetectorDataset(
            REPO_ROOT,
            manifests[variant],
            mean=normalization["mean"],
            standard_deviation=normalization["standard_deviation"],
        )
    schedule = build_equal_budget_schedule(
        real.labels,
        optimizer_updates=int(detector["optimizer_updates"]),
        batch_size=int(detector["batch_size"]),
        seed=seed,
        synthetic_fraction=float(detector["synthetic_fraction"]),
        synthetic_count=synthetic_count,
        variant=variant,
    )
    schedule_payload = [entry.__dict__ for entry in schedule]
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / f"{variant}_schedule.json", {
        "experiment_version": G2_2_VERSION,
        "variant": variant,
        "seed": seed,
        "entries": schedule_payload,
        "content_sha256": canonical_sha256(schedule_payload),
    })
    transform = SynchronizedRandomFlips(
        horizontal_probability=float(base["augmentation"]["horizontal_flip_probability"]),
        vertical_probability=float(base["augmentation"]["vertical_flip_probability"]),
        seed=seed,
    )
    training = ScheduledMixtureDataset(real, synthetic, schedule, transform)
    loader = DataLoader(
        training,
        batch_size=int(detector["batch_size"]),
        shuffle=False,
        num_workers=int(detector["num_workers"]),
        pin_memory=bool(detector["pin_memory"] and torch.cuda.is_available()),
    )
    model_config = base["model"]
    model = UNet(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config["output_channels"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device, memory_format=torch.channels_last)
    initialization_sha256 = model_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(detector["learning_rate"]),
        weight_decay=float(detector["weight_decay"]),
    )
    loss_config = base["loss"]
    criterion = CombinedBCEDiceLoss(
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
        pos_weight=float(loss_config["pos_weight"]),
    )
    controller = NumericalStepController(
        optimizer,
        precision_mode=detector["precision"],
        gradient_clip_max_norm=float(detector["gradient_clip_max_norm"]),
        automatic_fp32_retry=False,
    )
    start = time.perf_counter()
    losses: list[float] = []
    source_counts = {"real": 0, "synthetic": 0}
    try:
        for batch_index, batch in enumerate(loader, start=1):
            telemetry = controller.run_batch(
                model,
                batch["image"].to(
                    device, non_blocking=True, memory_format=torch.channels_last
                ),
                batch["mask"].to(device, non_blocking=True),
                batch["valid_region"].to(device, non_blocking=True),
                criterion,
            )
            if not telemetry.optimizer_step_executed or telemetry.optimizer_updates_this_batch != 1:
                raise RuntimeError(f"Equal-budget optimizer update failed at batch {batch_index}")
            losses.append(float(telemetry.total_loss))
            for source in batch["schedule_source"]:
                source_counts[str(source)] += 1
            if batch_index % 100 == 0:
                atomic_write_json(
                    report_dir / f"{variant}_progress.json",
                    {
                        "variant": variant,
                        "seed": seed,
                        "completed_optimizer_updates": batch_index,
                        "target_optimizer_updates": int(detector["optimizer_updates"]),
                        "latest_100_mean_loss": statistics.mean(losses[-100:]),
                    },
                )
                print(f"{variant} seed={seed} update={batch_index}/{detector['optimizer_updates']} loss={statistics.mean(losses[-100:]):.6f}", flush=True)
    finally:
        controller.close()
    counters = controller.state_dict()["counters"]
    if counters["optimizer_step_executed"] != int(detector["optimizer_updates"]):
        raise RuntimeError("Arm did not receive the exact precommitted optimizer budget")
    metrics = _evaluate(
        model,
        validation,
        device=device,
        precision=detector["precision"],
        threshold=float(detector["threshold"]),
        batch_size=int(detector["batch_size"]),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}.pt"
    torch.save({
        "experiment_version": G2_2_VERSION,
        "variant": variant,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_updates": int(detector["optimizer_updates"]),
        "initialization_sha256": initialization_sha256,
        "schedule_sha256": canonical_sha256(schedule_payload),
        "metrics": metrics,
    }, checkpoint_path)
    result = {
        "variant": variant,
        "seed": seed,
        "initialization_sha256": initialization_sha256,
        "final_model_sha256": model_state_sha256(model),
        "optimizer_updates": int(counters["optimizer_step_executed"]),
        "attempted_batches": int(counters["attempted_batches"]),
        "skipped_updates": int(counters["optimizer_step_skipped"]),
        "constant_learning_rate": float(detector["learning_rate"]),
        "source_counts": source_counts,
        "observed_synthetic_fraction": source_counts["synthetic"] / sum(source_counts.values()),
        "mean_training_loss": statistics.mean(losses),
        "final_100_mean_training_loss": statistics.mean(losses[-100:]),
        "runtime_seconds": time.perf_counter() - start,
        **metrics,
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
    }
    atomic_write_json(report_dir / f"{variant}.json", result)
    return result


def run_pilot(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    base = _base_config(config)
    manifests = _load_synthetic_manifests(config)
    detector = config["detector"]
    seed = int(detector["pilot_seed"])
    report_dir = REPO_ROOT / detector["pilot_report_directory"]
    checkpoint_dir = REPO_ROOT / detector["pilot_checkpoint_directory"]
    results = {}
    for variant in detector["variants"]:
        completed_report = report_dir / f"{variant}.json"
        completed_checkpoint = checkpoint_dir / f"{variant}.pt"
        if completed_report.is_file() and completed_checkpoint.is_file():
            result = json.loads(completed_report.read_text(encoding="utf-8"))
            if (
                result.get("seed") != seed
                or result.get("variant") != variant
                or result.get("optimizer_updates") != int(detector["optimizer_updates"])
                or result.get("skipped_updates") != 0
                or result.get("official_test_samples_loaded") != 0
            ):
                raise RuntimeError(f"Existing completed result is incompatible: {completed_report}")
            print(f"reusing completed immutable arm: {variant}", flush=True)
            results[variant] = result
        else:
            if completed_report.exists() != completed_checkpoint.exists():
                raise RuntimeError(f"Incomplete result/checkpoint pair for {variant}")
            results[variant] = _train_variant(
                config,
                base,
                manifests,
                variant=variant,
                seed=seed,
                report_dir=report_dir,
                checkpoint_dir=checkpoint_dir,
            )
    initialization_hashes = {result["initialization_sha256"] for result in results.values()}
    if len(initialization_hashes) != 1:
        raise RuntimeError("Detector variants did not start from identical initialization")
    control = results["real_only"]
    comparisons = {}
    winners = []
    for variant in ("checkpoint_1000", "checkpoint_1500"):
        passes, comparison = meaningful_winner(results[variant], control, rules=config["selection"])
        comparisons[variant] = comparison
        if passes:
            winners.append(variant)
    selected = max(winners, key=lambda name: results[name]["overall"]["global_dice"], default=None)
    summary = {
        "experiment_version": G2_2_VERSION,
        "stage": "bounded_one_seed_pilot",
        "seed": seed,
        "equal_optimizer_updates_per_arm": int(detector["optimizer_updates"]),
        "fixed_threshold": float(detector["threshold"]),
        "results": results,
        "comparisons_to_real_only": comparisons,
        "meaningful_winners": winners,
        "selected_for_confirmation": selected,
        "decision": "confirm_over_three_seeds" if selected else "stop_no_meaningful_synthetic_gain",
        "official_test_access_count": 0,
        "gan_optimizer_updates": 0,
    }
    atomic_write_json(report_dir / "pilot_summary.json", summary)
    return summary


def run_confirmation(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    base = _base_config(config)
    manifests = _load_synthetic_manifests(config)
    detector = config["detector"]
    pilot_path = REPO_ROOT / detector["pilot_report_directory"] / "pilot_summary.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    winner = pilot.get("selected_for_confirmation")
    if winner not in {"checkpoint_1000", "checkpoint_1500"}:
        raise RuntimeError("Pilot did not authorize a synthetic confirmation arm")
    seeds = [int(value) for value in detector["confirmation_seeds"]]
    if seeds != [42, 43, 44]:
        raise ValueError("G2.2 confirmation seeds must remain [42,43,44]")
    seed_results: dict[str, Any] = {}
    for seed in seeds:
        if seed == int(detector["pilot_seed"]):
            arms = {name: pilot["results"][name] for name in ("real_only", winner)}
        else:
            report_dir = REPO_ROOT / detector["confirmation_report_directory"] / f"seed{seed}"
            checkpoint_dir = REPO_ROOT / detector["confirmation_checkpoint_directory"] / f"seed{seed}"
            arms = {
                variant: _train_variant(
                    config, base, manifests, variant=variant, seed=seed,
                    report_dir=report_dir, checkpoint_dir=checkpoint_dir,
                )
                for variant in ("real_only", winner)
            }
        if len({arms[name]["initialization_sha256"] for name in arms}) != 1:
            raise RuntimeError(f"Confirmation seed {seed} did not use identical initialization")
        _, comparison = meaningful_winner(arms[winner], arms["real_only"], rules=config["selection"])
        seed_results[str(seed)] = {"arms": arms, "comparison": comparison}
    comparisons = [seed_results[str(seed)]["comparison"] for seed in seeds]
    confirmed, aggregate = confirmation_decision(comparisons, rules=config["selection"])
    summary = {
        "experiment_version": G2_2_VERSION,
        "stage": "three_seed_confirmation",
        "winner_configuration": winner,
        "seeds": seeds,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "confirmed": confirmed,
        "decision": "authorize_single_official_test" if confirmed else "stop_not_confirmed",
        "official_test_access_count": 0,
        "gan_optimizer_updates": 0,
    }
    output = REPO_ROOT / detector["confirmation_report_directory"] / "confirmation_summary.json"
    atomic_write_json(output, summary)
    return summary


def run_official_test(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    detector = config["detector"]
    confirmation_path = REPO_ROOT / detector["confirmation_report_directory"] / "confirmation_summary.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if not confirmation.get("confirmed") or confirmation.get("decision") != "authorize_single_official_test":
        raise RuntimeError("Three-seed confirmation did not authorize official-test access")
    output_dir = REPO_ROOT / "reports/g2_2/official_test"
    output_path = output_dir / "evaluation.json"
    if output_path.exists():
        raise FileExistsError("The one permitted official-test evaluation already exists")
    seed = int(config["official_test"]["precommitted_evaluation_seed"])
    if seed != int(detector["pilot_seed"]):
        raise ValueError("Official-test checkpoint seed differs from the precommitted pilot seed")
    winner = confirmation["winner_configuration"]
    checkpoint_path = REPO_ROOT / detector["pilot_checkpoint_directory"] / f"{winner}.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["variant"] != winner or int(payload["seed"]) != seed:
        raise RuntimeError("Official-test detector checkpoint identity mismatch")
    base = _base_config(config)
    data = base["data"]
    normalization = data["detector_normalization"]
    # This is the only function in G2.2 allowed to construct development_split=test,
    # which is the untouched official KSDD2 test split.
    official = KSDD2FullImageDataset(
        REPO_ROOT, "test", REPO_ROOT / data["manifest"],
        target_size=(int(data["canvas_width"]), int(data["canvas_height"])),
        image_padding_mode="reflect", mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"], spatial_transform=None,
    )
    if any(row["official_split"] != "test" or row["development_split"] != "test" for row in official.rows):
        raise RuntimeError("Official-test dataset contains a non-test row")
    configure_reproducibility(seed, deterministic=True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = base["model"]
    model = UNet(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config["output_channels"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device, memory_format=torch.channels_last)
    model.load_state_dict(payload["model_state"])
    result = {
        "experiment_version": G2_2_VERSION,
        "stage": "single_final_official_test",
        "winner_configuration": winner,
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_seed": seed,
        "threshold": float(detector["threshold"]),
        "single_evaluation": True,
        **_evaluate(
            model, official, device=device, precision=detector["precision"],
            threshold=float(detector["threshold"]), batch_size=int(detector["batch_size"]),
            official_test=True,
        ),
    }
    atomic_write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/g2_2_detector_utility.json")
    parser.add_argument("--mode", choices=("pilot", "confirmation", "official-test"), default="pilot")
    args = parser.parse_args()
    action = {
        "pilot": run_pilot,
        "confirmation": run_confirmation,
        "official-test": run_official_test,
    }[args.mode]
    print(json.dumps(action(args.config), indent=2))


if __name__ == "__main__":
    main()
