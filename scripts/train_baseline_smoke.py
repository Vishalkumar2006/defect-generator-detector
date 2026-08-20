"""Run a bounded, real-only CUDA smoke test; this is not baseline training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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

from defectgen.data.full_image import (  # noqa: E402
    KSDD2FullImageDataset,
    geometry_from_batch,
    restore_to_native,
)
from defectgen.data.sampling import deterministic_weighted_sampler, seed_worker  # noqa: E402
from defectgen.data.splits import load_development_manifest  # noqa: E402
from defectgen.models import UNet, count_parameters  # noqa: E402
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.metrics import (  # noqa: E402
    image_defect_probabilities,
    segmentation_metrics,
    select_validation_threshold,
)
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def _stable_subset(rows, split: str, defective: bool, count: int, seed: int) -> set[str]:
    candidates = [
        row for row in rows if row["development_split"] == split and bool(row["has_defect"]) == defective
    ]
    ordered = sorted(
        candidates,
        key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode("ascii")).hexdigest(),
    )
    if len(ordered) < count:
        raise ValueError(f"Need {count} split={split} defective={defective} rows; found {len(ordered)}")
    return {row["sample_id"] for row in ordered[:count]}


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


def _validation(
    model,
    loader,
    criterion,
    device,
    mixed_precision: bool,
) -> tuple[float, dict[str, float], list[float], list[bool], dict[str, Any]]:
    model.eval()
    losses = []
    logits_batches = []
    target_batches = []
    valid_batches = []
    labels = []
    overlay_payload = None
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_region"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, targets, valid)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            losses.append(float(loss.item()))
            logits_batches.append(logits.float().cpu())
            target_batches.append(targets.cpu())
            valid_batches.append(valid.cpu())
            labels.extend(bool(value) for value in batch["has_defect"].tolist())
            if overlay_payload is None:
                selected = next((index for index, value in enumerate(batch["has_defect"].tolist()) if value), 0)
                overlay_payload = {
                    "batch": batch,
                    "index": selected,
                    "logits": logits[selected].float().cpu(),
                }
    all_logits = torch.cat(logits_batches)
    all_targets = torch.cat(target_batches)
    all_valid = torch.cat(valid_batches)
    label_tensor = torch.tensor(labels, dtype=torch.bool)
    metrics = segmentation_metrics(all_logits, all_targets, all_valid, has_defect=label_tensor)
    probabilities = image_defect_probabilities(all_logits, all_valid).tolist()
    return float(np.mean(losses)), metrics, probabilities, labels, overlay_payload


def _save_overlay(payload, threshold: float, mean, std, output: Path) -> dict[str, Any]:
    batch = payload["batch"]
    index = payload["index"]
    geometry = geometry_from_batch(batch, index)
    normalized = batch["image"][index]
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    image = (normalized * std_tensor + mean_tensor).clamp(0, 1)
    native_image = restore_to_native(image, geometry).permute(1, 2, 0).numpy()
    native_target = restore_to_native(batch["mask"][index], geometry).squeeze(0).numpy().astype(bool)
    native_prediction = (
        torch.sigmoid(restore_to_native(payload["logits"], geometry)).squeeze(0).numpy() >= threshold
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 7), constrained_layout=True)
    axes[0].imshow(native_image)
    axes[0].set_title(f'{batch["sample_id"][index]} native image', fontsize=10)
    truth_overlay = native_image.copy()
    truth_overlay[native_target] = truth_overlay[native_target] * 0.2 + np.array([1.0, 0.0, 0.0]) * 0.8
    axes[1].imshow(truth_overlay)
    axes[1].set_title("ground truth (red)", fontsize=10)
    prediction_overlay = native_image.copy()
    prediction_overlay[native_prediction] = prediction_overlay[native_prediction] * 0.2 + np.array([0.0, 1.0, 0.0]) * 0.8
    axes[2].imshow(prediction_overlay)
    axes[2].set_title(f"non-final prediction (green), t={threshold:.2f}", fontsize=10)
    for axis in axes:
        axis.axis("off")
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)
    return {
        "sample_id": batch["sample_id"][index],
        "restored_shape": list(native_prediction.shape),
        "expected_native_shape": [geometry.original_height, geometry.original_width],
    }


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Baseline CUDA smoke test (non-final)",
        "",
        "This is a bounded plumbing check, not a trained or selected baseline.",
        "",
        f'- Status: **{report["status"]}**',
        f'- Epochs: {report["epochs_completed"]}',
        f'- Physical batch size: {report["physical_batch_size"]}',
        f'- Model parameters: {report["model_parameter_count"]:,}',
        f'- Parameters updated: {report["parameters_updated"]}',
        f'- Finite gradients: {report["finite_gradients"]}',
        f'- Checkpoint round trip: {report["checkpoint_round_trip"]}',
        "",
        "## Losses",
        "",
        f'- Training: {report["training_losses"]}',
        f'- Validation: {report["validation_losses"]}',
        f'- First-batch unweighted BCE + Dice: {report["loss_comparison"]["unweighted_total"]:.6f}',
        f'- First-batch capped-weight BCE + Dice: {report["loss_comparison"]["capped_weight_total"]:.6f}',
        "",
        "## Final smoke validation metrics (non-final)",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["validation_metrics"].items())
    return "\n".join(lines) + "\n"


def run_smoke(config: dict, memory_report: dict) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; smoke test refuses CPU fallback")
    seed = int(config["seed"])
    configure_reproducibility(seed, deterministic=True, warn_only=True)
    device = torch.device("cuda:0")
    manifest = load_development_manifest(REPO_ROOT, REPO_ROOT / config["paths"]["manifest"])
    smoke = config["smoke"]
    training_ids = _stable_subset(manifest, "train", True, smoke["training_samples_per_class"], seed)
    training_ids |= _stable_subset(manifest, "train", False, smoke["training_samples_per_class"], seed)
    validation_ids = _stable_subset(manifest, "validation", True, smoke["validation_defective_samples"], seed)
    validation_ids |= _stable_subset(manifest, "validation", False, smoke["validation_normal_samples"], seed)
    if any(row["development_split"] == "test" and row["sample_id"] in training_ids | validation_ids for row in manifest):
        raise RuntimeError("Official test leakage detected")
    normalization = config["detector_normalization"]
    dataset_arguments = {
        "repo_root": REPO_ROOT,
        "manifest_path": REPO_ROOT / config["paths"]["manifest"],
        "target_size": (config["model"]["input_width"], config["model"]["input_height"]),
        "image_padding_mode": config["model"]["image_padding_mode"],
        "mean": normalization["mean"],
        "standard_deviation": normalization["standard_deviation"],
    }
    train_dataset = KSDD2FullImageDataset(
        development_split="train", sample_ids=training_ids, **dataset_arguments
    )
    validation_dataset = KSDD2FullImageDataset(
        development_split="validation", sample_ids=validation_ids, **dataset_arguments
    )
    physical_batch_size = min(config["training"]["batch_size"], memory_report["successful_physical_batch_size"])
    sampler = deterministic_weighted_sampler(train_dataset.labels, seed=seed, num_samples=len(train_dataset))
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "batch_size": physical_batch_size,
        "num_workers": config["training"]["num_workers"],
        "pin_memory": bool(config["training"]["pin_memory"] and torch.cuda.is_available()),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_arguments)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_arguments)
    model = UNet(base_channels=config["model"]["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    criterion = CombinedBCEDiceLoss(
        config["loss"]["bce_weight"], config["loss"]["dice_weight"], config["loss"]["pos_weight"]
    )
    unweighted = CombinedBCEDiceLoss(config["loss"]["bce_weight"], config["loss"]["dice_weight"], None)
    mixed_precision = bool(config["training"]["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    parameter_before = next(model.parameters()).detach().clone()
    training_losses = []
    validation_losses = []
    sampled_defective = 0
    sampled_total = 0
    finite_gradients = True
    loss_comparison = None
    final_validation = None

    for epoch in range(min(int(config["training"]["epochs"]), 2)):
        model.train()
        batch_losses = []
        for batch in train_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_region"].to(device, non_blocking=True)
            sampled_defective += int(batch["has_defect"].sum().item())
            sampled_total += len(batch["has_defect"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, targets, valid)
                if loss_comparison is None:
                    loss_comparison = {
                        "unweighted_total": float(unweighted(logits, targets, valid).detach().item()),
                        "capped_weight_total": float(loss.detach().item()),
                        "capped_pos_weight": config["loss"]["pos_weight"],
                    }
            if logits.shape != targets.shape or not torch.isfinite(loss):
                raise RuntimeError("Smoke training produced invalid shape or non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradients = finite_gradients and _finite_gradients(model)
            if not finite_gradients:
                raise RuntimeError("Smoke training produced non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.item()))
        training_losses.append(float(np.mean(batch_losses)))
        final_validation = _validation(model, validation_loader, criterion, device, mixed_precision)
        validation_losses.append(final_validation[0])

    parameters_updated = not torch.equal(parameter_before, next(model.parameters()).detach())
    if not parameters_updated:
        raise RuntimeError("Model parameters did not update")
    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_directory"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "smoke_model.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config}, checkpoint_path)
    loaded = UNet(base_channels=config["model"]["base_channels"]).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    loaded.load_state_dict(payload["model"])
    loaded.eval()
    model.eval()
    reference_batch = next(iter(validation_loader))["image"].to(device)
    with torch.no_grad():
        original_output = model(reference_batch)
        loaded_output = loaded(reference_batch)
    checkpoint_round_trip = bool(torch.equal(original_output, loaded_output))
    if not checkpoint_round_trip:
        raise RuntimeError("Checkpoint round-trip output mismatch")

    validation_loss, validation_metrics, probabilities, labels, overlay_payload = final_validation
    threshold, image_metrics = select_validation_threshold(probabilities, labels)
    validation_metrics.update({f"image_{key}": value for key, value in image_metrics.items()})
    report_dir = REPO_ROOT / config["paths"]["report_directory"]
    report_dir.mkdir(parents=True, exist_ok=True)
    overlay = _save_overlay(
        overlay_payload,
        threshold,
        normalization["mean"],
        normalization["standard_deviation"],
        report_dir / "validation_overlay.png",
    )
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    epochs = np.arange(1, len(training_losses) + 1)
    axis.plot(epochs, training_losses, marker="o", label="training")
    axis.plot(epochs, validation_losses, marker="o", label="validation")
    axis.set_xlabel("smoke epoch")
    axis.set_ylabel("combined loss")
    axis.set_title("Non-final smoke losses")
    axis.legend()
    figure.savefig(report_dir / "loss_curve.png", dpi=160, facecolor="white")
    plt.close(figure)
    return {
        "status": "PASS - NON-FINAL SMOKE ONLY",
        "device": torch.cuda.get_device_name(0),
        "epochs_completed": len(training_losses),
        "physical_batch_size": physical_batch_size,
        "mixed_precision": mixed_precision,
        "model_parameter_count": count_parameters(model),
        "training_sample_ids": sorted(training_ids),
        "validation_sample_ids": sorted(validation_ids),
        "official_test_samples_loaded": 0,
        "sampled_training_defective_fraction": sampled_defective / sampled_total,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "loss_comparison": loss_comparison,
        "parameters_updated": parameters_updated,
        "finite_gradients": finite_gradients,
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_round_trip": checkpoint_round_trip,
        "prediction_shape": list(original_output.shape),
        "validation_threshold_selected_without_test": threshold,
        "validation_metrics": validation_metrics,
        "overlay_restoration": overlay,
        "augmentation_enabled": False,
        "notes": [
            "Results are non-final and must not be used to select the final loss or model.",
            "Bit-for-bit reproducibility is not guaranteed across PyTorch, CUDA, driver, or GPU versions.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baseline.json")
    parser.add_argument("--memory-report", type=Path, default=REPO_ROOT / "reports" / "baseline_smoke" / "memory_probe.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        memory_report = json.loads(args.memory_report.read_text(encoding="utf-8"))
        report = run_smoke(config, memory_report)
        report_dir = REPO_ROOT / config["paths"]["report_directory"]
        (report_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (report_dir / "smoke_summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f'Smoke test PASS (non-final): train losses={report["training_losses"]}, validation losses={report["validation_losses"]}')
    print(f'Official test samples loaded: {report["official_test_samples_loaded"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
