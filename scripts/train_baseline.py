"""Train one validation-only real-image baseline candidate with resumable state."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.data.sampling import deterministic_weighted_sampler, seed_worker  # noqa: E402
from defectgen.models import UNet, count_parameters  # noqa: E402
from defectgen.training.engine import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
    write_metric_logs,
)
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.metrics import (  # noqa: E402
    detailed_validation_metrics,
    validation_threshold_sweep,
)
from defectgen.training.numerics import NumericalStepController, precision_autocast  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


EPOCH_FIELDS = [
    "epoch",
    "train_bce",
    "train_dice",
    "train_total_loss",
    "validation_bce",
    "validation_dice",
    "validation_total_loss",
    "global_dice",
    "global_iou",
    "pixel_precision",
    "pixel_recall",
    "mean_defective_image_dice",
    "median_defective_image_dice",
    "defective_images_zero_detected_pixels",
    "mean_predicted_defect_fraction_normal_images",
    "normal_image_false_positive_rate",
    "image_precision",
    "image_recall",
    "image_f1",
    "sampled_training_defective_fraction",
    "nonfinite_forward_loss",
    "nonfinite_gradient",
    "amp_overflow_scale_drop",
    "optimizer_step_executed",
    "optimizer_step_skipped",
    "fp32_retry_attempted",
    "fp32_retry_executed",
    "epoch_seconds",
    "peak_allocated_vram_bytes",
    "peak_reserved_vram_bytes",
]


def candidate_configuration(base: dict[str, Any], pilot: dict[str, Any], pos_weight: float) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["training"]["epochs"] = int(pilot["maximum_epochs"])
    config["training"]["scheduler"] = pilot["scheduler"]
    config["loss"]["pos_weight"] = float(pos_weight)
    config["pilot"] = {
        "phase": pilot["phase"],
        "checkpoint_selection": pilot["checkpoint_selection"],
        "metric_threshold_during_training": pilot["metric_threshold_during_training"],
        "threshold_sweep": pilot["threshold_sweep"],
        "official_test_evaluation": False,
    }
    return config


def candidate_directories(pilot: dict[str, Any], pos_weight: float) -> tuple[Path, Path]:
    """Return isolated report/checkpoint roots for one loss candidate."""
    label = f"pw_{int(pos_weight)}"
    return REPO_ROOT / pilot["report_root"] / label, REPO_ROOT / pilot["checkpoint_root"] / label


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_datasets(config: dict[str, Any]):
    normalization = config["detector_normalization"]
    common = {
        "repo_root": REPO_ROOT,
        "manifest_path": REPO_ROOT / config["paths"]["manifest"],
        "target_size": (config["model"]["input_width"], config["model"]["input_height"]),
        "image_padding_mode": config["model"]["image_padding_mode"],
        "mean": normalization["mean"],
        "standard_deviation": normalization["standard_deviation"],
        "augmentation": None,
    }
    training = KSDD2FullImageDataset(development_split="train", **common)
    validation = KSDD2FullImageDataset(development_split="validation", **common)
    if len(training) != 1981 or len(validation) != 350:
        raise ValueError(f"Unexpected development sizes: training={len(training)}, validation={len(validation)}")
    if any(row["development_split"] == "test" for row in training.rows + validation.rows):
        raise RuntimeError("Official test row entered a pilot dataset")
    return training, validation


def _build_loaders(config, training, validation):
    seed = int(config["seed"])
    sampler = deterministic_weighted_sampler(training.labels, seed=seed, num_samples=len(training))
    loader_generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "pin_memory": bool(config["training"]["pin_memory"] and torch.cuda.is_available()),
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
    }
    train_loader = DataLoader(training, sampler=sampler, **common)
    validation_loader = DataLoader(validation, shuffle=False, **common)
    return train_loader, validation_loader, sampler, loader_generator


def _train_epoch(model, loader, criterion, controller, device):
    model.train()
    sums = {"bce": 0.0, "dice": 0.0, "total": 0.0}
    samples = defective = successful_samples = 0
    counter_before = controller.state_dict()["counters"]
    anomalies = []
    for batch_index, batch in enumerate(loader, start=1):
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid = batch["valid_region"].to(device, non_blocking=True)
        telemetry = controller.run_batch(model, inputs, targets, valid, criterion)
        batch_size = len(inputs)
        if telemetry.optimizer_step_executed:
            sums["bce"] += telemetry.bce_loss * batch_size
            sums["dice"] += telemetry.dice_loss * batch_size
            sums["total"] += telemetry.total_loss * batch_size
            successful_samples += batch_size
        if telemetry.is_anomaly:
            anomalies.append({"batch_index": batch_index, **telemetry.to_dict()})
        samples += batch_size
        defective += int(batch["has_defect"].sum().item())
    if successful_samples == 0:
        raise RuntimeError("Every optimizer attempt in the epoch was skipped")
    counter_after = controller.state_dict()["counters"]
    events = {key: int(counter_after[key]) - int(counter_before[key]) for key in counter_after}
    return (
        {key: value / successful_samples for key, value in sums.items()},
        defective / samples,
        events,
        anomalies,
    )


def _validate(model, loader, criterion, device, precision_mode, keep_outputs: bool = True):
    model.eval()
    sums = {"bce": 0.0, "dice": 0.0, "total": 0.0}
    samples = 0
    probabilities = []
    targets = []
    valid_regions = []
    labels = []
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            batch_targets = batch["mask"].to(device, non_blocking=True)
            batch_valid = batch["valid_region"].to(device, non_blocking=True)
            with precision_autocast(device.type, precision_mode):
                logits = model(inputs)
            components = criterion.components(logits.float(), batch_targets.float(), batch_valid.float())
            if not all(torch.isfinite(value) for value in components.values()):
                raise RuntimeError("Non-finite validation loss; automatic fp32 retry is disabled")
            batch_size = len(inputs)
            for key in sums:
                sums[key] += float(components[key].item()) * batch_size
            samples += batch_size
            if keep_outputs:
                probabilities.append(torch.sigmoid(logits.float()).cpu())
                targets.append(batch_targets.bool().cpu())
                valid_regions.append(batch_valid.bool().cpu())
                labels.append(batch["has_defect"].bool().cpu())
                sample_ids.extend(batch["sample_id"])
    losses = {key: value / samples for key, value in sums.items()}
    if not keep_outputs:
        return losses, None
    output = {
        "probabilities": torch.cat(probabilities),
        "targets": torch.cat(targets),
        "valid_regions": torch.cat(valid_regions),
        "labels": torch.cat(labels),
        "sample_ids": sample_ids,
    }
    return losses, output


def _save_epoch_plots(records: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [record["epoch"] for record in records]
    axes[0].plot(epochs, [record["train_total_loss"] for record in records], marker="o", label="train")
    axes[0].plot(epochs, [record["validation_total_loss"] for record in records], marker="o", label="validation")
    axes[0].set_title("Combined BCE + Dice loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [record["global_dice"] for record in records], marker="o", label="global Dice")
    axes[1].plot(
        epochs,
        [record["mean_defective_image_dice"] for record in records],
        marker="o",
        label="mean defective-image Dice",
    )
    axes[1].set_title("Validation metrics at threshold 0.5")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def _write_threshold_outputs(rows, best_global, best_defective, report_dir: Path) -> None:
    fields = list(rows[0])
    with (report_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "threshold_sweep.json").write_text(
        json.dumps(
            {
                "data_source": "validation only",
                "rows": rows,
                "best_global_dice": best_global,
                "best_mean_defective_image_dice": best_defective,
                "objectives_select_different_thresholds": best_global["threshold"] != best_defective["threshold"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_candidate(
    config: dict[str, Any],
    *,
    report_dir: Path,
    checkpoint_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; D1 pilot refuses CPU fallback")
    if config["training"]["augmentation_enabled"]:
        raise ValueError("Augmentation must remain disabled for the loss pilot")
    if config["training"]["batch_size"] != 4:
        raise ValueError("The verified physical batch size must remain 4")
    configure_reproducibility(config["seed"], deterministic=True, warn_only=True)
    training, validation = _build_datasets(config)
    train_loader, validation_loader, sampler, loader_generator = _build_loaders(config, training, validation)
    model = UNet(base_channels=config["model"]["base_channels"])
    initialization_hash = model_state_sha256(model)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = None
    if config["training"]["scheduler"] != "none":
        raise ValueError("D1 frozen scheduler must be 'none'")
    criterion = CombinedBCEDiceLoss(
        config["loss"]["bce_weight"], config["loss"]["dice_weight"], config["loss"]["pos_weight"]
    )
    mixed_precision = bool(config["training"]["mixed_precision"])
    precision_mode = config["training"].get("precision_mode", "fp16" if mixed_precision else "fp32")
    if precision_mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16")
    scaler = torch.amp.GradScaler("cuda") if precision_mode == "fp16" else None
    controller = NumericalStepController(
        optimizer,
        precision_mode=precision_mode,
        scaler=scaler,
        gradient_clip_max_norm=config["training"].get("gradient_clip_max_norm"),
        automatic_fp32_retry=False,
    )
    start_epoch = 1
    records: list[dict[str, Any]] = []
    best = {"epoch": 0, "validation_total_loss": math.inf}
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint is absent: {last_checkpoint}")
        payload = load_training_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_configuration=config,
            sampler_generator=sampler.generator,
            loader_generator=loader_generator,
            map_location=device,
            numerical_controller=controller,
        )
        start_epoch = int(payload["epoch"]) + 1
        records = payload["metric_records"]
        for index, record in enumerate(records):
            legacy_gradient = int(record.get("nonfinite_gradient_steps", 0))
            legacy_retry = int(record.get("amp_forward_fallback_batches", 0))
            record.setdefault("nonfinite_forward_loss", legacy_retry)
            record.setdefault("nonfinite_gradient", legacy_gradient)
            record.setdefault("amp_overflow_scale_drop", 0)  # D1 did not record scale history.
            record.setdefault("optimizer_step_executed", math.ceil(1981 / 4) - legacy_gradient)
            record.setdefault("optimizer_step_skipped", legacy_gradient)
            record.setdefault("fp32_retry_attempted", legacy_retry)
            record.setdefault("fp32_retry_executed", legacy_retry)
            records[index] = {field: record[field] for field in EPOCH_FIELDS}
        best = payload["best_validation"]

    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    overall_peak_allocated = 0
    overall_peak_reserved = 0
    anomaly_path = report_dir / "numerical_anomalies.json"
    anomalies = json.loads(anomaly_path.read_text(encoding="utf-8")) if resume and anomaly_path.is_file() else []
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        epoch_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        train_losses, sampled_fraction, numerical_events, epoch_anomalies = _train_epoch(
            model, train_loader, criterion, controller, device
        )
        anomalies.extend({"epoch": epoch, **anomaly} for anomaly in epoch_anomalies)
        validation_losses, outputs = _validate(
            model, validation_loader, criterion, device, precision_mode, keep_outputs=True
        )
        metrics, _ = detailed_validation_metrics(
            outputs["probabilities"],
            outputs["targets"],
            outputs["valid_regions"],
            outputs["labels"],
            threshold=0.5,
        )
        elapsed = time.perf_counter() - epoch_start
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        overall_peak_allocated = max(overall_peak_allocated, peak_allocated)
        overall_peak_reserved = max(overall_peak_reserved, peak_reserved)
        record = {
            "epoch": epoch,
            "train_bce": train_losses["bce"],
            "train_dice": train_losses["dice"],
            "train_total_loss": train_losses["total"],
            "validation_bce": validation_losses["bce"],
            "validation_dice": validation_losses["dice"],
            "validation_total_loss": validation_losses["total"],
            **{key: metrics[key] for key in EPOCH_FIELDS if key in metrics},
            "sampled_training_defective_fraction": sampled_fraction,
            **{key: numerical_events[key] for key in EPOCH_FIELDS if key in numerical_events},
            "epoch_seconds": elapsed,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
        }
        record = {field: record[field] for field in EPOCH_FIELDS}
        records.append(record)
        if validation_losses["total"] < best["validation_total_loss"]:
            best = {"epoch": epoch, "validation_total_loss": validation_losses["total"]}
            save_training_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                configuration=config,
                best_validation=best,
                sampler_generator=sampler.generator,
                loader_generator=loader_generator,
                metric_records=records,
                numerical_controller=controller,
            )
        save_training_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            configuration=config,
            best_validation=best,
            sampler_generator=sampler.generator,
            loader_generator=loader_generator,
            metric_records=records,
            numerical_controller=controller,
        )
        write_metric_logs(records, report_dir / "epoch_metrics.csv", report_dir / "epoch_metrics.json")
        anomaly_path.write_text(json.dumps(anomalies, indent=2) + "\n", encoding="utf-8")
        _save_epoch_plots(records, report_dir / "loss_curves.png")
        print(
            f'epoch {epoch}/{config["training"]["epochs"]}: train={train_losses["total"]:.6f}, '
            f'validation={validation_losses["total"]:.6f}, global_dice={metrics["global_dice"]:.6f}, '
            f'defective_dice={metrics["mean_defective_image_dice"]:.6f}, seconds={elapsed:.1f}',
            flush=True,
        )
        del outputs

    # Evaluate the checkpoint selected solely by validation combined loss.
    payload = load_training_checkpoint(
        best_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_configuration=config,
        sampler_generator=sampler.generator,
        loader_generator=loader_generator,
        map_location=device,
        numerical_controller=controller,
    )
    best_losses, outputs = _validate(
        model, validation_loader, criterion, device, precision_mode, keep_outputs=True
    )
    sweep, best_global, best_defective = validation_threshold_sweep(
        outputs["probabilities"], outputs["targets"], outputs["valid_regions"], outputs["labels"]
    )
    _write_threshold_outputs(sweep, best_global, best_defective, report_dir)
    metrics_at_global, per_image = detailed_validation_metrics(
        outputs["probabilities"],
        outputs["targets"],
        outputs["valid_regions"],
        outputs["labels"],
        threshold=float(best_global["threshold"]),
    )
    for row, sample_id in zip(per_image, outputs["sample_ids"]):
        row["sample_id"] = sample_id
    per_image_fields = ["sample_id", *[field for field in per_image[0] if field != "sample_id"]]
    with (report_dir / "per_image_metrics_best_global_threshold.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(per_image)
    post_training_seconds = time.perf_counter() - total_start
    # Epoch durations in restored records make this meaningful after a resume.
    total_seconds = sum(float(record["epoch_seconds"]) for record in records) + max(
        0.0, post_training_seconds - sum(float(record["epoch_seconds"]) for record in records[start_epoch - 1 :])
    )
    summary = {
        "status": "PASS - VALIDATION-ONLY SCREENING CANDIDATE",
        "pos_weight": config["loss"]["pos_weight"],
        "seed": config["seed"],
        "initialization_sha256": initialization_hash,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "official_test_samples_loaded": 0,
        "augmentation_enabled": False,
        "epochs_completed": len(records),
        "best_epoch": payload["epoch"],
        "best_validation_loss": best_losses["total"],
        "best_validation_loss_components": best_losses,
        "threshold_0_5_metrics_at_best_epoch": detailed_validation_metrics(
            outputs["probabilities"], outputs["targets"], outputs["valid_regions"], outputs["labels"], 0.5
        )[0],
        "best_global_dice_threshold": best_global,
        "best_mean_defective_dice_threshold": best_defective,
        "threshold_objectives_differ": best_global["threshold"] != best_defective["threshold"],
        "per_image_metrics_threshold": metrics_at_global["threshold"],
        "runtime_seconds": total_seconds,
        "peak_allocated_vram_bytes": max(
            overall_peak_allocated, max(record["peak_allocated_vram_bytes"] for record in records)
        ),
        "peak_reserved_vram_bytes": max(
            overall_peak_reserved, max(record["peak_reserved_vram_bytes"] for record in records)
        ),
        "numerical_counters": {
            key: sum(int(record[key]) for record in records)
            for key in (
                "nonfinite_forward_loss",
                "nonfinite_gradient",
                "amp_overflow_scale_drop",
                "optimizer_step_executed",
                "optimizer_step_skipped",
                "fp32_retry_attempted",
                "fp32_retry_executed",
            )
        },
        # Compatibility aliases for report readers; automatic retry is now disabled.
        "nonfinite_gradient_steps": sum(int(record["nonfinite_gradient"]) for record in records),
        "amp_forward_fallback_batches": 0,
        "amp_validation_fallback_batches": 0,
        "model_parameter_count": count_parameters(model),
        "checkpoint_last": last_checkpoint.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_best": best_checkpoint.relative_to(REPO_ROOT).as_posix(),
        "resume_supported": True,
        "configuration": config,
        "notes": [
            "Validation only; official test was not constructed or evaluated.",
            "This is an imbalance-loss screening candidate, not a final baseline result.",
        ],
    }
    (report_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    controller.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos-weight", type=float, required=True, choices=(1.0, 5.0, 10.0, 20.0))
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs" / "baseline.json")
    parser.add_argument("--pilot-config", type=Path, default=REPO_ROOT / "configs" / "loss_pilot.json")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base = json.loads(args.base_config.read_text(encoding="utf-8"))
        pilot = json.loads(args.pilot_config.read_text(encoding="utf-8"))
        config = candidate_configuration(base, pilot, args.pos_weight)
        report_dir, checkpoint_dir = candidate_directories(pilot, args.pos_weight)
        if report_dir.is_dir() and not args.resume and any(report_dir.iterdir()):
            raise FileExistsError(f"Experiment report directory is not empty: {report_dir}; use --resume")
        summary = run_candidate(
            config, report_dir=report_dir, checkpoint_dir=checkpoint_dir, resume=args.resume
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f'candidate pw={summary["pos_weight"]} PASS: best_epoch={summary["best_epoch"]}, '
        f'best_validation_loss={summary["best_validation_loss"]:.6f}, runtime={summary["runtime_seconds"]:.1f}s',
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
