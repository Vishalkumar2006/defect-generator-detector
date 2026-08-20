"""Manually run a short development-training-only numerical stability probe.

This command intentionally constructs neither validation nor official-test data.
It is diagnostic tooling, not an evaluation or a source of final model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.full_image import KSDD2FullImageDataset  # noqa: E402
from defectgen.data.sampling import deterministic_weighted_sampler, seed_worker  # noqa: E402
from defectgen.models import UNet, count_parameters  # noqa: E402
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.numerics import NumericalStepController  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


@dataclass(frozen=True)
class ProbeProfile:
    precision_mode: str
    learning_rate: float
    batch_size: int
    gradient_clip_max_norm: float | None
    default_initial_scale: float | None


PROFILES = {
    "current_fp16": ProbeProfile("fp16", 0.001, 4, None, None),
    "safer_fp16": ProbeProfile("fp16", 0.0003, 4, 1.0, 1024.0),
    "bf16": ProbeProfile("bf16", 0.0003, 4, 1.0, None),
    "fp32": ProbeProfile("fp32", 0.0003, 2, 1.0, None),
}


def build_training_dataset(base_config: dict[str, Any]) -> KSDD2FullImageDataset:
    """Construct only the development-training dataset."""
    normalization = base_config["detector_normalization"]
    dataset = KSDD2FullImageDataset(
        REPO_ROOT,
        "train",
        REPO_ROOT / base_config["paths"]["manifest"],
        target_size=(base_config["model"]["input_width"], base_config["model"]["input_height"]),
        image_padding_mode=base_config["model"]["image_padding_mode"],
        mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"],
        augmentation=None,
    )
    if len(dataset) != 1981 or any(row["development_split"] != "train" for row in dataset.rows):
        raise RuntimeError("Numerical probe dataset is not exactly the development-training split")
    return dataset


def _build_loader(dataset, profile: ProbeProfile, steps: int, seed: int):
    sampler = deterministic_weighted_sampler(
        dataset.labels,
        seed=seed,
        num_samples=steps * profile.batch_size,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=profile.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_probe(profile_name: str, steps: int, initial_scale: float | None, seed: int) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the numerical stability probe")
    profile = PROFILES[profile_name]
    if profile.precision_mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16; choose another profile")
    if initial_scale is not None and profile.precision_mode != "fp16":
        raise ValueError("--initial-scale is valid only for fp16 profiles")

    base = json.loads((REPO_ROOT / "configs" / "baseline.json").read_text(encoding="utf-8"))
    configure_reproducibility(seed, deterministic=True, warn_only=True)
    dataset = build_training_dataset(base)
    loader = _build_loader(dataset, profile, steps, seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = UNet(base_channels=base["model"]["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        weight_decay=base["training"]["weight_decay"],
    )
    effective_initial_scale = initial_scale if initial_scale is not None else profile.default_initial_scale
    if profile.precision_mode == "fp16":
        scaler_kwargs = {"init_scale": effective_initial_scale} if effective_initial_scale is not None else {}
        scaler = torch.amp.GradScaler("cuda", **scaler_kwargs)
    else:
        scaler = None
    controller = NumericalStepController(
        optimizer,
        precision_mode=profile.precision_mode,
        scaler=scaler,
        gradient_clip_max_norm=profile.gradient_clip_max_norm,
        automatic_fp32_retry=False,
    )
    criterion = CombinedBCEDiceLoss(
        bce_weight=base["loss"]["bce_weight"],
        dice_weight=base["loss"]["dice_weight"],
        pos_weight=5.0,
    )

    report_dir = REPO_ROOT / "reports" / "numerical_stability" / profile_name
    report_dir.mkdir(parents=True, exist_ok=True)
    telemetry_rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    invariant_violations: list[str] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for attempt, batch in enumerate(loader, start=1):
        if attempt > steps:
            break
        result = controller.run_batch(
            model,
            batch["image"].to(device, non_blocking=True),
            batch["mask"].to(device, non_blocking=True),
            batch["valid_region"].to(device, non_blocking=True),
            criterion,
        )
        row = result.to_dict()
        row["sample_ids"] = ";".join(batch["sample_id"])
        telemetry_rows.append(row)
        if result.is_anomaly:
            anomalies.append(row)
        if result.optimizer_updates_this_batch > 1:
            invariant_violations.append(f"attempt {attempt}: more than one optimizer update")
        if not result.is_anomaly and result.optimizer_updates_this_batch != 1:
            invariant_violations.append(f"attempt {attempt}: successful batch did not execute exactly one update")
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    counters = asdict(controller.counters)
    summary = {
        "status": "PASS" if not invariant_violations else "FAIL",
        "profile": profile_name,
        "profile_configuration": asdict(profile),
        "requested_steps": steps,
        "attempted_batches": counters["attempted_batches"],
        "actual_optimizer_updates": counters["optimizer_step_executed"],
        "skipped_optimizer_steps": counters["optimizer_step_skipped"],
        "true_amp_overflows_by_scale_drop": counters["amp_overflow_scale_drop"],
        "nonfinite_forward_losses": counters["nonfinite_forward_loss"],
        "nonfinite_gradients": counters["nonfinite_gradient"],
        "fp32_retry_attempted": counters["fp32_retry_attempted"],
        "fp32_retry_executed": counters["fp32_retry_executed"],
        "automatic_fp32_retry_enabled": False,
        "initial_scale": telemetry_rows[0]["scale_before"] if telemetry_rows else None,
        "final_scale": telemetry_rows[-1]["scale_after"] if telemetry_rows else None,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "runtime_seconds": runtime,
        "training_samples_available": len(dataset),
        "validation_samples_loaded": 0,
        "official_test_samples_loaded": 0,
        "model_parameter_count": count_parameters(model),
        "invariant_violations": invariant_violations,
        "notes": [
            "Development-training data only; validation and official-test datasets were not constructed.",
            "No model checkpoint or evaluation-ready final model was saved.",
            "pos_weight=5 is used only to audit the provisional D1 candidate's numerical behavior.",
        ],
    }
    _write_csv(telemetry_rows, report_dir / "batch_telemetry.csv")
    (report_dir / "anomalies.json").write_text(json.dumps(anomalies, indent=2) + "\n", encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    controller.close()
    if invariant_violations:
        raise RuntimeError("; ".join(invariant_violations))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="current_fp16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--initial-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_probe(args.profile, args.steps, args.initial_scale, args.seed)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
