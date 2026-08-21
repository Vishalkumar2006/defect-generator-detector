"""Compare frozen baseline checkpoints under FP16, BF16, and FP32 validation forwards.

This is a read-only diagnostic: it creates no optimizer, performs no backward
pass, and constructs only the development-validation dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.data.sampling import seed_worker  # noqa: E402
from defectgen.models import UNet  # noqa: E402
from defectgen.training.failure_diagnostics import (  # noqa: E402
    atomic_write_json,
    inspect_checkpoint_finiteness,
    model_state_sha256,
    nonfinite_components,
    tensor_numerical_summary,
)
from defectgen.training.final_baseline import validate_final_baseline_configuration  # noqa: E402
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.numerics import precision_autocast  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


PRECISION_MODES = ("fp16", "bf16", "fp32")


def _build_validation_dataset(configuration: dict[str, Any]):
    data = configuration["data"]
    normalization = data["detector_normalization"]
    dataset = KSDD2FullImageDataset(
        repo_root=REPO_ROOT,
        development_split=data["validation_split"],
        manifest_path=REPO_ROOT / data["manifest"],
        target_size=(int(data["canvas_width"]), int(data["canvas_height"])),
        image_padding_mode="reflect",
        mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"],
        spatial_transform=None,
    )
    if any(row["development_split"] != "validation" for row in dataset.rows):
        raise RuntimeError("Precision diagnostic dataset contains a non-validation row")
    return dataset


def _build_validation_loader(configuration: dict[str, Any], dataset):
    settings = configuration["training"]
    generator = torch.Generator().manual_seed(int(configuration["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(settings["num_workers"]),
        pin_memory=bool(settings["pin_memory"] and torch.cuda.is_available()),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _loss_values(components: dict[str, torch.Tensor]) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for name, value in components.items():
        number = float(value.detach().item())
        if number != number:
            values[name] = "nan"
        elif number == float("inf"):
            values[name] = "+inf"
        elif number == float("-inf"):
            values[name] = "-inf"
        else:
            values[name] = number
    return values


def diagnose_precision_mode(model, loader, criterion, device: torch.device, precision_mode: str):
    if precision_mode not in PRECISION_MODES:
        raise ValueError(f"Unsupported precision mode: {precision_mode}")
    model.eval()
    maximum_absolute_finite_logit = 0.0
    total_nan_logits = total_positive_inf_logits = total_negative_inf_logits = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_region"].to(device, non_blocking=True)
            with precision_autocast(device.type, precision_mode):
                logits = model(inputs)
            components = criterion.components(logits.float(), targets.float(), valid.float())
            statistics = tensor_numerical_summary(logits)
            maximum_absolute_finite_logit = max(
                maximum_absolute_finite_logit, statistics["maximum_absolute_finite"] or 0.0
            )
            total_nan_logits += int(statistics["nan_count"])
            total_positive_inf_logits += int(statistics["positive_inf_count"])
            total_negative_inf_logits += int(statistics["negative_inf_count"])
            failures = nonfinite_components(logits, components)
            if failures:
                return {
                    "status": "numerical_failure",
                    "precision_mode": precision_mode,
                    "batches_completed": batch_index - 1,
                    "failing_batch_index": batch_index,
                    "failing_sample_ids": [str(value) for value in batch.get("sample_id", [])],
                    "nonfinite_components": failures,
                    "failing_logits": statistics,
                    "loss_components": _loss_values(components),
                    "maximum_absolute_finite_logit": maximum_absolute_finite_logit,
                    "total_nan_logits": total_nan_logits,
                    "total_positive_inf_logits": total_positive_inf_logits,
                    "total_negative_inf_logits": total_negative_inf_logits,
                }
    return {
        "status": "finite",
        "precision_mode": precision_mode,
        "batches_completed": len(loader),
        "failing_batch_index": None,
        "failing_sample_ids": [],
        "nonfinite_components": [],
        "maximum_absolute_finite_logit": maximum_absolute_finite_logit,
        "total_nan_logits": total_nan_logits,
        "total_positive_inf_logits": total_positive_inf_logits,
        "total_negative_inf_logits": total_negative_inf_logits,
    }


def evaluate_precision_modes(model, loader, criterion, device: torch.device, *, on_result=None):
    results: dict[str, dict[str, Any]] = {}
    for mode in PRECISION_MODES:
        if mode == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            results[mode] = {
                "status": "unsupported",
                "precision_mode": mode,
                "reason": "CUDA device does not report BF16 support",
            }
        else:
            try:
                results[mode] = diagnose_precision_mode(model, loader, criterion, device, mode)
            except Exception as error:
                results[mode] = {
                    "status": "invariant_violation",
                    "precision_mode": mode,
                    "error": str(error),
                }
        if on_result is not None:
            on_result(mode, results[mode])
    return results


def _checkpoint_diagnostic(
    label: str,
    checkpoint_path: Path,
    device: torch.device,
    output_directory: Path,
) -> tuple[dict[str, Any], bool]:
    payload, inspection = inspect_checkpoint_finiteness(checkpoint_path)
    report: dict[str, Any] = {"checkpoint": label, "inspection": inspection}
    if not inspection["parameters_usable"]:
        report["status"] = "unusable_nonfinite_parameters"
        atomic_write_json(output_directory / f"{label}.json", report)
        return report, True

    configuration = payload["configuration"]
    validate_final_baseline_configuration(configuration)
    configure_reproducibility(int(configuration["seed"]), deterministic=True, warn_only=True)
    dataset = _build_validation_dataset(configuration)
    loader = _build_validation_loader(configuration, dataset)
    model = UNet(
        input_channels=int(configuration["model"]["input_channels"]),
        output_channels=int(configuration["model"]["output_channels"]),
        base_channels=int(configuration["model"]["base_channels"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    before_hash = model_state_sha256(model)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Fresh diagnostic model unexpectedly has gradients")
    report.update(
        {
            "status": "running",
            "validation_samples": len(dataset),
            "official_test_samples_loaded": 0,
            "optimizer_constructed": False,
            "backward_executed": False,
            "parameter_hash_before": before_hash,
            "precision_results": {},
        }
    )
    atomic_write_json(output_directory / f"{label}.json", report)

    def persist_mode(mode: str, result: dict[str, Any]) -> None:
        report["precision_results"][mode] = result
        atomic_write_json(output_directory / f"{label}.json", report)

    results = evaluate_precision_modes(model, loader, CombinedBCEDiceLoss(
        bce_weight=float(configuration["loss"]["bce_weight"]),
        dice_weight=float(configuration["loss"]["dice_weight"]),
        pos_weight=float(configuration["loss"]["pos_weight"]),
    ), device, on_result=persist_mode)
    after_hash = model_state_sha256(model)
    gradients_created = any(parameter.grad is not None for parameter in model.parameters())
    invariant_violation = before_hash != after_hash or gradients_created or any(
        result["status"] == "invariant_violation" for result in results.values()
    )
    report.update(
        {
            "status": "invariant_violation" if invariant_violation else "complete",
            "parameter_hash_before": before_hash,
            "parameter_hash_after": after_hash,
            "parameter_hash_unchanged": before_hash == after_hash,
            "parameter_gradients_created": gradients_created,
            "precision_results": results,
        }
    )
    atomic_write_json(output_directory / f"{label}.json", report)
    return report, invariant_violation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--last-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "reports" / "final_real_baseline" / "precision_diagnostic",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_directory = args.output_directory.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        print("ERROR: precision diagnostic requires an available CUDA device", file=sys.stderr)
        return 1
    paths = {"best": args.best_checkpoint.resolve(), "last": args.last_checkpoint.resolve()}
    if paths["best"] == paths["last"]:
        print("ERROR: best and last checkpoint paths must be distinct", file=sys.stderr)
        return 1
    if any(not path.is_file() for path in paths.values()):
        print("ERROR: both explicit checkpoint paths must exist", file=sys.stderr)
        return 1
    reports: dict[str, Any] = {}
    violation = False
    configurations = []
    try:
        for label, path in paths.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            configurations.append(payload.get("configuration"))
            report, failed = _checkpoint_diagnostic(label, path, device, output_directory)
            reports[label] = report
            violation = violation or failed
        if configurations[0] != configurations[1]:
            violation = True
            reports["configuration_mismatch"] = True
        summary = {
            "status": "invariant_violation" if violation else "complete",
            "checkpoint_paths": {label: str(path) for label, path in paths.items()},
            "official_test_samples_loaded": 0,
            "reports": {label: f"{label}.json" for label in paths},
        }
        atomic_write_json(output_directory / "summary.json", summary)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        atomic_write_json(output_directory / "summary.json", {"status": "unusable", "error": str(error)})
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 1 if violation else 0


if __name__ == "__main__":
    raise SystemExit(main())
