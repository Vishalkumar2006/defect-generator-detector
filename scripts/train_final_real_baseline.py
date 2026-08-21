"""Train and finalize the frozen E1 real-only segmentation baseline."""

from __future__ import annotations

import argparse
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
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.augmentation import SynchronizedRandomFlips  # noqa: E402
from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.data.sampling import deterministic_weighted_sampler, seed_worker  # noqa: E402
from defectgen.models import UNet, count_parameters  # noqa: E402
from defectgen.training.engine import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
    update_checkpoint_metadata,
    write_metric_logs,
)
from defectgen.training.final_baseline import (  # noqa: E402
    EarlyStopping,
    HISTORICAL_FP16_IDENTITY,
    PostTrainingValidationGate,
    STABILIZED_BF16_IDENTITY,
    build_plateau_scheduler,
    configuration_fingerprint,
    configuration_identity,
    load_final_baseline_configuration,
    threshold_candidates,
    validate_final_baseline_configuration,
)
from defectgen.training.failure_diagnostics import (  # noqa: E402
    atomic_write_json,
    git_worktree_state,
    model_state_sha256,
    nonfinite_components,
    write_numerical_failure_report,
)
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.metrics import detailed_validation_metrics, validation_threshold_sweep  # noqa: E402
from defectgen.training.numerics import NumericalStepController, precision_autocast  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


EPOCH_FIELDS = [
    "epoch",
    "precision_mode",
    "train_bce",
    "train_dice",
    "train_total_loss",
    "validation_bce",
    "validation_dice",
    "validation_total_loss",
    "validation_global_dice_at_0_5",
    "validation_global_iou_at_0_5",
    "validation_pixel_precision_at_0_5",
    "validation_pixel_recall_at_0_5",
    "learning_rate",
    "next_learning_rate",
    "sampled_training_defective_fraction",
    "attempted_batches",
    "optimizer_step_executed",
    "optimizer_step_skipped",
    "nonfinite_forward_loss",
    "nonfinite_gradient",
    "amp_overflow_scale_drop",
    "fp32_retry_attempted",
    "fp32_retry_executed",
    "grad_scaler_applicable",
    "grad_scaler_scale_initial",
    "grad_scaler_scale",
    "gradient_clip_max_norm",
    "maximum_training_gradient_norm",
    "maximum_pre_clipping_gradient_norm",
    "maximum_post_clipping_gradient_norm",
    "maximum_training_absolute_logit",
    "maximum_validation_absolute_logit",
    "epoch_seconds",
    "peak_allocated_vram_bytes",
    "peak_reserved_vram_bytes",
]


def _build_datasets(configuration: dict[str, Any]):
    """Construct only development train and validation; never construct official test."""
    data = configuration["data"]
    normalization = data["detector_normalization"]
    common = {
        "repo_root": REPO_ROOT,
        "manifest_path": REPO_ROOT / data["manifest"],
        "target_size": (int(data["canvas_width"]), int(data["canvas_height"])),
        "image_padding_mode": "reflect",
        "mean": normalization["mean"],
        "standard_deviation": normalization["standard_deviation"],
    }
    augmentation = configuration["augmentation"]
    training = KSDD2FullImageDataset(
        development_split=data["training_split"],
        spatial_transform=SynchronizedRandomFlips(
            horizontal_probability=float(augmentation["horizontal_flip_probability"]),
            vertical_probability=float(augmentation["vertical_flip_probability"]),
            seed=int(configuration["seed"]),
        ),
        **common,
    )
    validation = KSDD2FullImageDataset(
        development_split=data["validation_split"], spatial_transform=None, **common
    )
    if any(row["development_split"] == "test" for row in training.rows + validation.rows):
        raise RuntimeError("Official test row entered the E1 experiment")
    return training, validation


def _build_loaders(configuration, training, validation):
    seed = int(configuration["seed"])
    sampler = deterministic_weighted_sampler(training.labels, seed=seed, num_samples=len(training))
    loader_generator = torch.Generator().manual_seed(seed)
    settings = configuration["training"]
    common = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": int(settings["num_workers"]),
        "pin_memory": bool(settings["pin_memory"] and torch.cuda.is_available()),
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
    }
    return (
        DataLoader(training, sampler=sampler, **common),
        DataLoader(validation, shuffle=False, **common),
        sampler,
        loader_generator,
    )


def _dataset_reproducibility_metadata(configuration, training, validation) -> dict[str, Any]:
    manifest_path = REPO_ROOT / configuration["data"]["manifest"]
    split_payload = {
        "training": [row["sample_id"] for row in training.rows],
        "validation": [row["sample_id"] for row in validation.rows],
    }
    split_bytes = json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "manifest_relative_path": configuration["data"]["manifest"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_splits_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "training_split": configuration["data"]["training_split"],
        "validation_split": configuration["data"]["validation_split"],
        "training_samples": len(training),
        "validation_samples": len(validation),
        "official_test_samples_loaded": 0,
    }


def _validate_stabilized_output_identity(
    configuration: dict[str, Any], report_dir: Path, checkpoint_dir: Path, *, resume: bool
) -> None:
    identity = configuration_identity(configuration)
    if identity == HISTORICAL_FP16_IDENTITY:
        raise ValueError("The historical failed FP16 experiment is preserved and cannot be run or resumed")
    if identity != STABILIZED_BF16_IDENTITY:
        raise ValueError(f"Unsupported final baseline identity: {identity}")
    expected_report = (REPO_ROOT / configuration["paths"]["report_directory"]).resolve()
    expected_checkpoint = (REPO_ROOT / configuration["paths"]["checkpoint_directory"]).resolve()
    if report_dir.resolve() != expected_report or checkpoint_dir.resolve() != expected_checkpoint:
        raise ValueError("Run directories do not match the frozen stabilized experiment identity")
    historical = {
        (REPO_ROOT / "reports" / "final_real_baseline").resolve(),
        (REPO_ROOT / "checkpoints" / "final_real_baseline").resolve(),
    }
    if report_dir.resolve() in historical or checkpoint_dir.resolve() in historical:
        raise ValueError("Stabilized BF16 outputs may not use the preserved FP16 directories")
    if not resume:
        for path in (report_dir, checkpoint_dir):
            if path.is_dir() and any(path.iterdir()):
                raise FileExistsError(f"Fresh-run directory is not empty: {path}")
    elif not (checkpoint_dir / "last.pt").is_file():
        raise FileNotFoundError(f"Cannot resume; checkpoint is absent: {checkpoint_dir / 'last.pt'}")


def _train_epoch(model, loader, criterion, controller, device, *, failure_context=None):
    model.train()
    sums = {"bce": 0.0, "dice": 0.0, "total": 0.0}
    samples = defective = successful_samples = 0
    counter_before = controller.state_dict()["counters"]
    maximum_pre_clipping_gradient_norm = 0.0
    maximum_post_clipping_gradient_norm = 0.0
    most_recent_gradient_norm: float | None = None
    maximum_absolute_logit = 0.0
    anomalies: list[dict[str, Any]] = []
    last_batch_index = 0
    last_sample_ids: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        last_batch_index = batch_index
        last_sample_ids = [str(value) for value in batch.get("sample_id", [])]
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid = batch["valid_region"].to(device, non_blocking=True)
        try:
            telemetry = controller.run_batch(model, inputs, targets, valid, criterion)
        except Exception as error:
            if failure_context is not None:
                write_numerical_failure_report(
                    **failure_context,
                    phase="training",
                    batch_index=batch_index,
                    sample_ids=last_sample_ids,
                    precision_mode=controller.precision_mode,
                    logits=None,
                    loss_components=None,
                    scaler_scale=(float(controller.scaler.get_scale()) if controller.scaler else None),
                    most_recent_gradient_norm=most_recent_gradient_norm,
                    error=str(error),
                    explicit_nonfinite_component="training_step_exception",
                )
            raise
        if telemetry.optimizer_updates_this_batch > 1:
            if failure_context is not None:
                write_numerical_failure_report(
                    **failure_context,
                    phase="training",
                    batch_index=batch_index,
                    sample_ids=last_sample_ids,
                    precision_mode=controller.precision_mode,
                    logits=None,
                    loss_components={
                        "bce": telemetry.bce_loss,
                        "dice": telemetry.dice_loss,
                        "total": telemetry.total_loss,
                    },
                    scaler_scale=telemetry.scale_after,
                    most_recent_gradient_norm=telemetry.unscaled_gradient_norm,
                    error="A training attempt executed more than one optimizer update",
                    explicit_nonfinite_component="optimizer_update_invariant",
                )
            raise RuntimeError("A training attempt executed more than one optimizer update")
        batch_size = len(inputs)
        if telemetry.optimizer_step_executed:
            for key in sums:
                sums[key] += getattr(telemetry, f"{key}_loss") * batch_size
            successful_samples += batch_size
        if telemetry.unscaled_gradient_norm is not None:
            most_recent_gradient_norm = telemetry.pre_clipping_gradient_norm
            maximum_pre_clipping_gradient_norm = max(
                maximum_pre_clipping_gradient_norm, telemetry.pre_clipping_gradient_norm
            )
            maximum_post_clipping_gradient_norm = max(
                maximum_post_clipping_gradient_norm, telemetry.post_clipping_gradient_norm
            )
        maximum_absolute_logit = max(maximum_absolute_logit, telemetry.maximum_absolute_logit)
        if telemetry.is_anomaly:
            anomalies.append({"batch_index": batch_index, **telemetry.to_dict()})
        samples += batch_size
        defective += int(batch["has_defect"].sum().item())
    if successful_samples == 0:
        if failure_context is not None:
            write_numerical_failure_report(
                **failure_context,
                phase="training",
                batch_index=last_batch_index,
                sample_ids=last_sample_ids,
                precision_mode=controller.precision_mode,
                logits=None,
                loss_components=None,
                scaler_scale=(float(controller.scaler.get_scale()) if controller.scaler else None),
                most_recent_gradient_norm=most_recent_gradient_norm,
                error="Every optimizer attempt in the epoch was skipped",
                explicit_nonfinite_component="all_optimizer_attempts_skipped",
            )
        raise RuntimeError("Every optimizer attempt in the epoch was skipped")
    after = controller.state_dict()["counters"]
    events = {key: int(after[key]) - int(counter_before[key]) for key in after}
    return (
        {key: value / successful_samples for key, value in sums.items()},
        defective / samples,
        events,
        anomalies,
        maximum_pre_clipping_gradient_norm,
        maximum_post_clipping_gradient_norm,
        most_recent_gradient_norm,
        maximum_absolute_logit,
    )


def _validate(
    model,
    loader,
    criterion,
    device,
    precision_mode,
    *,
    keep_outputs: bool,
    failure_context=None,
    scaler_scale: float | None = None,
    most_recent_gradient_norm: float | None = None,
):
    model.eval()
    sums = {"bce": 0.0, "dice": 0.0, "total": 0.0}
    samples = 0
    maximum_absolute_logit = 0.0
    collected: dict[str, list[Any]] = {
        "probabilities": [],
        "targets": [],
        "valid_regions": [],
        "labels": [],
        "sample_ids": [],
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_region"].to(device, non_blocking=True)
            with precision_autocast(device.type, precision_mode):
                logits = model(inputs)
            components = criterion.components(logits.float(), targets.float(), valid.float())
            failures = nonfinite_components(logits, components)
            if failures:
                message = (
                    f"Non-finite validation component(s): {', '.join(failures)}; "
                    "automatic fp32 retry is disabled"
                )
                if failure_context is not None:
                    write_numerical_failure_report(
                        **failure_context,
                        phase="validation",
                        batch_index=batch_index,
                        sample_ids=[str(value) for value in batch.get("sample_id", [])],
                        precision_mode=precision_mode,
                        logits=logits,
                        loss_components=components,
                        scaler_scale=scaler_scale,
                        most_recent_gradient_norm=most_recent_gradient_norm,
                        error=message,
                    )
                raise RuntimeError(message)
            batch_size = len(inputs)
            for key in sums:
                sums[key] += float(components[key].item()) * batch_size
            samples += batch_size
            maximum_absolute_logit = max(
                maximum_absolute_logit, float(logits.detach().float().abs().max().item())
            )
            if keep_outputs:
                collected["probabilities"].append(torch.sigmoid(logits.float()).cpu())
                collected["targets"].append(targets.bool().cpu())
                collected["valid_regions"].append(valid.bool().cpu())
                collected["labels"].append(batch["has_defect"].bool().cpu())
                collected["sample_ids"].extend(batch["sample_id"])
    losses = {key: value / samples for key, value in sums.items()}
    if not keep_outputs:
        return losses, None, maximum_absolute_logit
    outputs = {
        key: (values if key == "sample_ids" else torch.cat(values))
        for key, values in collected.items()
    }
    return losses, outputs, maximum_absolute_logit


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_fixed_overlays(
    validation,
    outputs: dict[str, Any],
    fixed_ids_path: Path,
    threshold: float,
    output_path: Path,
) -> None:
    fixed = json.loads(fixed_ids_path.read_text(encoding="utf-8"))
    output_indexes = {sample_id: index for index, sample_id in enumerate(outputs["sample_ids"])}
    dataset_indexes = {row["sample_id"]: index for index, row in enumerate(validation.rows)}
    rows = [(label, sample_id) for label, sample_id in fixed.items()]
    if any(sample_id not in output_indexes or sample_id not in dataset_indexes for _, sample_id in rows):
        raise ValueError("A fixed diagnostic ID is absent from E1 validation outputs")
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), constrained_layout=True)
    mean = validation.mean
    deviation = validation.standard_deviation
    for row_index, (label, sample_id) in enumerate(rows):
        sample = validation[dataset_indexes[sample_id]]
        image = sample["image"]
        if mean is not None and deviation is not None:
            image = image * deviation + mean
        probability = outputs["probabilities"][output_indexes[sample_id], 0]
        truth = outputs["targets"][output_indexes[sample_id], 0]
        prediction = probability >= threshold
        panels = (image.permute(1, 2, 0).clamp(0, 1), truth, probability, prediction)
        titles = (f"{label}\n{sample_id}", "ground truth", "probability", f"prediction @{threshold:.2f}")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axes[row_index, column].imshow(panel, cmap=None if column == 0 else "magma")
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
    figure.savefig(output_path, dpi=140, facecolor="white")
    plt.close(figure)


def _save_epoch_plot(records: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in records]
    axes[0].plot(epochs, [row["train_total_loss"] for row in records], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_total_loss"] for row in records], marker="o", label="validation")
    axes[0].set_title("Valid-region BCE + Dice")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_global_dice_at_0_5"] for row in records], marker="o")
    axes[1].set_title("Validation global Dice @ 0.5")
    for axis in axes:
        axis.set_xlabel("epoch")
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def _scaler_scale(scaler) -> float | None:
    return float(scaler.get_scale()) if scaler is not None else None


def _stable_resume_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "experiment_identity",
            "configuration_sha256",
            "dataset",
            "seed",
            "model_initialization_sha256",
            "precision_mode",
            "augmentation",
            "optimizer_attempts_per_epoch",
            "maximum_optimizer_attempts",
            "gradient_clip_max_norm",
        )
    }


def run_final_real_baseline(
    configuration: dict[str, Any], *, report_dir: Path, checkpoint_dir: Path, resume: bool
) -> dict[str, Any]:
    validate_final_baseline_configuration(configuration)
    _validate_stabilized_output_identity(
        configuration, report_dir, checkpoint_dir, resume=resume
    )
    precision_mode = str(configuration["precision"]["mode"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stabilized BF16 training refuses CPU fallback")
    if precision_mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16; refusing to begin the experiment")
    configure_reproducibility(int(configuration["seed"]), deterministic=True, warn_only=True)
    training, validation = _build_datasets(configuration)
    train_loader, validation_loader, sampler, loader_generator = _build_loaders(
        configuration, training, validation
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model = UNet(
        input_channels=int(configuration["model"]["input_channels"]),
        output_channels=int(configuration["model"]["output_channels"]),
        base_channels=int(configuration["model"]["base_channels"]),
    )
    initialization_hash = model_state_sha256(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["optimizer"]["learning_rate"]),
        weight_decay=float(configuration["optimizer"]["weight_decay"]),
    )
    scheduler = build_plateau_scheduler(optimizer, configuration)
    criterion = CombinedBCEDiceLoss(
        bce_weight=float(configuration["loss"]["bce_weight"]),
        dice_weight=float(configuration["loss"]["dice_weight"]),
        pos_weight=float(configuration["loss"]["pos_weight"]),
    )
    scaler_enabled = bool(configuration["precision"]["grad_scaler"])
    if precision_mode == "bf16" and scaler_enabled:
        raise ValueError("BF16 configuration must not enable GradScaler")
    scaler = torch.amp.GradScaler("cuda") if scaler_enabled else None
    gradient_clip_max_norm = configuration["precision"]["gradient_clip_max_norm"]
    controller = NumericalStepController(
        optimizer,
        precision_mode=precision_mode,
        scaler=scaler,
        gradient_clip_max_norm=gradient_clip_max_norm,
        automatic_fp32_retry=False,
    )
    stopping = EarlyStopping(
        patience=int(configuration["early_stopping"]["patience"]),
        minimum_delta=float(configuration["early_stopping"]["minimum_delta"]),
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reproducibility_metadata = {
        "experiment_identity": configuration_identity(configuration),
        "git": git_worktree_state(REPO_ROOT),
        "configuration": configuration,
        "configuration_sha256": configuration_fingerprint(configuration),
        "dataset": _dataset_reproducibility_metadata(configuration, training, validation),
        "seed": int(configuration["seed"]),
        "model_initialization_sha256": initialization_hash,
        "precision_mode": precision_mode,
        "grad_scaler_applicable": scaler is not None,
        "augmentation": configuration["augmentation"],
        "optimizer_attempts_per_epoch": len(train_loader),
        "maximum_optimizer_attempts": len(train_loader)
        * int(configuration["training"]["maximum_epochs"]),
        "gradient_clip_max_norm": gradient_clip_max_norm,
        "detector_protocol": "frozen_real_only_v1",
    }
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    start_epoch = 1
    records: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    best = {"epoch": 0, "validation_total_loss": math.inf}
    anomaly_path = report_dir / "numerical_anomalies.json"
    failure_report_path = report_dir / "latest_numerical_failure.json"
    failure_base_context = {
        "path": failure_report_path,
        "checkpoint_paths": {"best": best_path, "last": last_path},
        "repo_root": REPO_ROOT,
    }
    if resume:
        payload = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_configuration=configuration,
            sampler_generator=sampler.generator,
            loader_generator=loader_generator,
            map_location=device,
            numerical_controller=controller,
            early_stopping=stopping,
        )
        start_epoch = int(payload["epoch"]) + 1
        records = list(payload["metric_records"])
        best = dict(payload["best_validation"])
        checkpoint_reproducibility = payload.get("checkpoint_metadata", {}).get("reproducibility")
        if checkpoint_reproducibility is None:
            raise ValueError("Checkpoint lacks stabilized reproducibility metadata")
        if _stable_resume_metadata(checkpoint_reproducibility) != _stable_resume_metadata(
            reproducibility_metadata
        ):
            raise ValueError("Checkpoint reproducibility metadata does not match the current run")
        resume_git_state = reproducibility_metadata["git"]
        reproducibility_metadata["git"] = checkpoint_reproducibility["git"]
        reproducibility_metadata["resume_git_history"] = [
            *checkpoint_reproducibility.get("resume_git_history", []),
            resume_git_state,
        ]
        if payload.get("checkpoint_metadata", {}).get("finalized"):
            raise RuntimeError("This E1 run is already finalized; refusing a second threshold sweep")
        if stopping.stopped:
            start_epoch = int(configuration["training"]["maximum_epochs"]) + 1
        if anomaly_path.is_file():
            anomalies = json.loads(anomaly_path.read_text(encoding="utf-8"))

    atomic_write_json(report_dir / "run_metadata.json", reproducibility_metadata)

    maximum_epochs = int(configuration["training"]["maximum_epochs"])
    for epoch in range(start_epoch, maximum_epochs + 1):
        epoch_start = time.perf_counter()
        training.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scaler_scale_initial = _scaler_scale(scaler)
        (
            train_losses,
            sampled_fraction,
            events,
            epoch_anomalies,
            maximum_pre_clipping_gradient,
            maximum_post_clipping_gradient,
            most_recent_gradient,
            maximum_logit,
        ) = (
            _train_epoch(
                model,
                train_loader,
                criterion,
                controller,
                device,
                failure_context={**failure_base_context, "epoch": epoch},
            )
        )
        validation_losses, outputs, maximum_validation_logit = _validate(
            model,
            validation_loader,
            criterion,
            device,
            precision_mode,
            keep_outputs=True,
            failure_context={**failure_base_context, "epoch": epoch},
            scaler_scale=_scaler_scale(scaler),
            most_recent_gradient_norm=most_recent_gradient,
        )
        metrics, _ = detailed_validation_metrics(
            outputs["probabilities"],
            outputs["targets"],
            outputs["valid_regions"],
            outputs["labels"],
            threshold=0.5,
        )
        scheduler.step(validation_losses["total"])
        should_stop = stopping.step(validation_losses["total"])
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "precision_mode": precision_mode,
            "train_bce": train_losses["bce"],
            "train_dice": train_losses["dice"],
            "train_total_loss": train_losses["total"],
            "validation_bce": validation_losses["bce"],
            "validation_dice": validation_losses["dice"],
            "validation_total_loss": validation_losses["total"],
            "validation_global_dice_at_0_5": metrics["global_dice"],
            "validation_global_iou_at_0_5": metrics["global_iou"],
            "validation_pixel_precision_at_0_5": metrics["pixel_precision"],
            "validation_pixel_recall_at_0_5": metrics["pixel_recall"],
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "sampled_training_defective_fraction": sampled_fraction,
            **{key: events[key] for key in events},
            "grad_scaler_applicable": scaler is not None,
            "grad_scaler_scale_initial": scaler_scale_initial,
            "grad_scaler_scale": _scaler_scale(scaler),
            "gradient_clip_max_norm": gradient_clip_max_norm,
            "maximum_training_gradient_norm": maximum_pre_clipping_gradient,
            "maximum_pre_clipping_gradient_norm": maximum_pre_clipping_gradient,
            "maximum_post_clipping_gradient_norm": maximum_post_clipping_gradient,
            "maximum_training_absolute_logit": maximum_logit,
            "maximum_validation_absolute_logit": maximum_validation_logit,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
        }
        record = {field: record[field] for field in EPOCH_FIELDS}
        records.append(record)
        anomalies.extend({"epoch": epoch, **event} for event in epoch_anomalies)
        improved = validation_losses["total"] < float(best["validation_total_loss"])
        if improved:
            best = {"epoch": epoch, "validation_total_loss": validation_losses["total"]}
        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "configuration": configuration,
            "best_validation": best,
            "sampler_generator": sampler.generator,
            "loader_generator": loader_generator,
            "metric_records": records,
            "numerical_controller": controller,
            "early_stopping_state": stopping.state_dict(),
            "checkpoint_metadata": {
                "kind": "best" if improved else "last",
                "finalized": False,
                "reproducibility": reproducibility_metadata,
            },
        }
        if improved:
            save_training_checkpoint(best_path, **checkpoint_arguments)
        checkpoint_arguments["checkpoint_metadata"] = {
            "kind": "last",
            "finalized": False,
            "reproducibility": reproducibility_metadata,
        }
        save_training_checkpoint(last_path, **checkpoint_arguments)
        write_metric_logs(records, report_dir / "epoch_metrics.csv", report_dir / "epoch_metrics.json")
        anomaly_path.write_text(json.dumps(anomalies, indent=2) + "\n", encoding="utf-8")
        _save_epoch_plot(records, report_dir / "training_curves.png")
        print(
            f"epoch {epoch}/{maximum_epochs}: train={train_losses['total']:.6f}, "
            f"validation={validation_losses['total']:.6f}, dice@0.5={metrics['global_dice']:.6f}, "
            f"updates={events['optimizer_step_executed']}/{events['attempted_batches']}, "
            f"lr={learning_rate:.6g}, next_lr={next_learning_rate:.6g}, precision={precision_mode}",
            flush=True,
        )
        del outputs
        if should_stop:
            break

    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    gate = PostTrainingValidationGate()
    gate.mark_training_complete()
    best_payload = load_training_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_configuration=configuration,
        sampler_generator=sampler.generator,
        loader_generator=loader_generator,
        map_location=device,
        numerical_controller=controller,
        early_stopping=stopping,
    )
    gate.mark_best_checkpoint_loaded()
    best_losses, outputs, _ = _validate(
        model,
        validation_loader,
        criterion,
        device,
        precision_mode,
        keep_outputs=True,
        failure_context={**failure_base_context, "epoch": int(best_payload["epoch"])},
        scaler_scale=_scaler_scale(scaler),
        most_recent_gradient_norm=None,
    )
    metrics_at_half, _ = detailed_validation_metrics(
        outputs["probabilities"], outputs["targets"], outputs["valid_regions"], outputs["labels"], 0.5
    )
    gate.claim_threshold_sweep()
    sweep_rows, best_global, best_defective = validation_threshold_sweep(
        outputs["probabilities"],
        outputs["targets"],
        outputs["valid_regions"],
        outputs["labels"],
        thresholds=threshold_candidates(configuration),
    )
    selected_threshold = float(best_global["threshold"])
    selected_metrics, per_image = detailed_validation_metrics(
        outputs["probabilities"],
        outputs["targets"],
        outputs["valid_regions"],
        outputs["labels"],
        selected_threshold,
    )
    for row, sample_id in zip(per_image, outputs["sample_ids"]):
        row["sample_id"] = sample_id
    _write_rows(sweep_rows, report_dir / "validation_threshold_sweep.csv")
    _write_rows(per_image, report_dir / "validation_per_image_metrics.csv")
    threshold_report = {
        "data_source": "validation only",
        "execution_count": 1,
        "best_checkpoint_reloaded": True,
        "selected_objective": "maximum global Dice",
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "best_global_dice": best_global,
        "best_mean_defective_image_dice": best_defective,
        "thresholds": sweep_rows,
    }
    (report_dir / "validation_threshold_sweep.json").write_text(
        json.dumps(threshold_report, indent=2) + "\n", encoding="utf-8"
    )
    _save_fixed_overlays(
        validation,
        outputs,
        REPO_ROOT / configuration["paths"]["fixed_validation_ids"],
        selected_threshold,
        report_dir / "fixed_validation_overlays.png",
    )
    final_metadata = {
        "finalized": True,
        "selected_validation_threshold": selected_threshold,
        "threshold_selection_objective": "maximum global Dice",
        "threshold_sweep_executions": 1,
        "reproducibility": reproducibility_metadata,
    }
    update_checkpoint_metadata(best_path, final_metadata)
    update_checkpoint_metadata(last_path, final_metadata)
    summary = {
        "status": "PASS - STABILIZED BF16 FINAL REAL-ONLY BASELINE",
        "experiment_identity": configuration_identity(configuration),
        "best_epoch": int(best_payload["epoch"]),
        "epochs_completed": len(records),
        "stopped_early": len(records) < maximum_epochs,
        "best_validation_loss": best_losses["total"],
        "validation_metrics_at_0_5": metrics_at_half,
        "selected_validation_threshold": selected_threshold,
        "selected_validation_metrics": selected_metrics,
        "defective_image_dice_at_selected_threshold": selected_metrics["mean_defective_image_dice"],
        "normal_image_false_positive_rate_at_selected_threshold": selected_metrics[
            "normal_image_false_positive_rate"
        ],
        "official_test_samples_loaded": 0,
        "threshold_sweep_executions": 1,
        "model_parameter_count": count_parameters(model),
        "reproducibility": reproducibility_metadata,
        "numerical_counters": {
            key: sum(int(record[key]) for record in records)
            for key in (
                "attempted_batches",
                "optimizer_step_executed",
                "optimizer_step_skipped",
                "nonfinite_forward_loss",
                "nonfinite_gradient",
                "amp_overflow_scale_drop",
                "fp32_retry_attempted",
                "fp32_retry_executed",
            )
        },
        "checkpoint_best": best_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_last": last_path.relative_to(REPO_ROOT).as_posix(),
        "configuration": configuration,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    controller.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "final_real_baseline_bf16.json"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        configuration = load_final_baseline_configuration(args.config)
        report_dir = REPO_ROOT / configuration["paths"]["report_directory"]
        checkpoint_dir = REPO_ROOT / configuration["paths"]["checkpoint_directory"]
        summary = run_final_real_baseline(
            configuration, report_dir=report_dir, checkpoint_dir=checkpoint_dir, resume=args.resume
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"E1.2 BF16 PASS: best_epoch={summary['best_epoch']}, "
        f"best_validation_loss={summary['best_validation_loss']:.6f}, "
        f"selected_threshold={summary['selected_validation_threshold']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
