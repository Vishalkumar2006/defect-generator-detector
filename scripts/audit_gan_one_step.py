"""Run the bounded G1.4 calibration and exactly one optimizer step per GAN model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.training_pairs import (  # noqa: E402
    GANTrainingPairDataset,
    create_internal_gan_split,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)
from defectgen.models.gan import build_gan_models, load_gan_architecture_config  # noqa: E402
from defectgen.training.gan_losses import load_gan_loss_config  # noqa: E402
from defectgen.training.gan_trainer import (  # noqa: E402
    GANOneStepTrainer,
    calibrate_gan_loss_scales,
    collate_gan_training_samples,
    config_as_dict,
    load_gan_trainer_config,
    parameter_hash,
)
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _combination(contacts: dict[str, bool]) -> str:
    active = [side for side in ("top", "bottom", "left", "right") if contacts[side]]
    return "+".join(active) if active else "none"


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "fraction_below_0_5": float((array < 0.5).mean()),
        "fraction_below_0_25": float((array < 0.25).mean()),
        "fraction_below_0_1": float((array < 0.1).mean()),
        "pixel_count": int(len(array)),
    }


def _containment(support: torch.Tensor, valid: torch.Tensor) -> float:
    denominator = int(support.sum())
    if denominator == 0:
        raise RuntimeError("Validity preflight encountered empty generator support")
    return float((support & valid.bool()).sum()) / denominator


def _validity_preflight(batches, *, support_radius: int) -> dict[str, Any]:
    coverage_values: list[float] = []
    by_combination: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"real_valid": [], "fake_valid": [], "joint_valid": []}
    )
    canonical_failures = 0
    for batch in batches:
        canonical = batch.fake_discriminator_mask.bool()
        coverage_values.extend(batch.real_valid_coverage[canonical].tolist())
        halo = canonical
        if support_radius:
            halo = F.max_pool2d(
                canonical.float(),
                kernel_size=2 * support_radius + 1,
                stride=1,
                padding=support_radius,
            ).bool()
        complete_support = halo | (batch.generator_mask > 0)
        joint = batch.real_valid_mask.bool() & batch.fake_valid_mask.bool()
        canonical_failures += int((canonical & ~joint).sum())
        for index, metadata in enumerate(batch.metadata):
            combination = _combination(metadata["target_contact_sides"])
            by_combination[combination]["real_valid"].append(
                _containment(complete_support[index], batch.real_valid_mask[index])
            )
            by_combination[combination]["fake_valid"].append(
                _containment(complete_support[index], batch.fake_valid_mask[index])
            )
            by_combination[combination]["joint_valid"].append(
                _containment(complete_support[index], joint[index])
            )
    return {
        "canonical_pixel_native_coverage": _statistics(coverage_values),
        "canonical_pixels_outside_joint_validity": canonical_failures,
        "generator_support_definition": (
            "union(feathered generator_mask > 0, canonical mask dilated by the "
            f"configured {support_radius}-pixel generator support radius)"
        ),
        "generator_support_containment_by_target_contact_combination": {
            combination: {
                "sample_count": len(values["joint_valid"]),
                **{
                    name: {
                        "minimum": min(fractions),
                        "mean": float(np.mean(fractions)),
                        "maximum": max(fractions),
                    }
                    for name, fractions in values.items()
                },
            }
            for combination, values in sorted(by_combination.items())
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return (
            "# G1.4 one-step GAN mechanics audit\n\n"
            f"- Status: **FAIL**\n- Error: `{report.get('error')}`\n"
        )
    calibration = report["calibration"]
    raw_lines = [
        f"- `{name}`: min {values['minimum']:.8g}, median {values['median']:.8g}, "
        f"max {values['maximum']:.8g}, finite {values['finite_count']}/"
        f"{values['finite_count'] + values['nonfinite_count']}"
        for name, values in calibration["raw_loss_distributions"].items()
    ]
    gradient_lines = [
        f"- `{name}`: median {values['unit_gradient_norm']['median']:.8g}, "
        f"relative to adversarial {values['median_relative_to_adversarial']}, "
        f"zero batches {values['zero_gradient_count']}"
        for name, values in calibration["generator_unit_gradient_scales"].items()
    ]
    isolation = report["parameter_isolation"]
    validity = report["validity_preflight"]["canonical_pixel_native_coverage"]
    return "\n".join(
        [
            "# G1.4 one-step GAN mechanics audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Precision: `{report['precision']['effective']}` on `{report['device']}`",
            f"- Runtime seconds: {report['runtime_seconds']}",
            f"- Peak CUDA memory bytes: {report['peak_cuda_memory_bytes']}",
            f"- Discriminator optimizer steps: {report['discriminator_optimizer_steps']}",
            f"- Generator optimizer steps: {report['generator_optimizer_steps']}",
            f"- Total training batches optimized: {report['total_training_batches_optimized']}",
            f"- D changed only D parameters: {isolation['discriminator_step_changed_only_discriminator']}",
            f"- G changed only G parameters: {isolation['generator_step_changed_only_generator']}",
            f"- Canonical native coverage minimum / median: {validity['minimum']} / {validity['median']}",
            f"- Canonical coverage fraction below 0.5: {validity['fraction_below_0_5']}",
            f"- Canonical defect gradient coverage: {report['generator_step']['canonical_defect_gradient_coverage']}",
            f"- Maximum invalid fake gradient: {report['generator_step']['maximum_invalid_fake_pixel_gradient']}",
            f"- Generator locality after step: {report['generator_step']['generator_locality_after_step']}",
            f"- Validation rows loaded: {report['validation_rows_loaded']}",
            f"- Official-test rows loaded: {report['official_test_rows_loaded']}",
            f"- Monitor optimizer steps: {report['monitor_optimizer_steps']}",
            f"- Materialized training images: {report['materialized_training_images']}",
            "",
            "## Calibration losses",
            "",
            *raw_lines,
            "",
            "## Unit-coefficient generator gradients",
            "",
            *gradient_lines,
            "",
            "## Suggested provisional coefficients",
            "",
            f"`{json.dumps(calibration['suggested_provisional_generator_coefficients'], sort_keys=True)}`",
            "",
            "Suggestions were not written to configuration and were not used for the mechanics step.",
            "",
            "## One-step losses",
            "",
            f"- Discriminator: `{json.dumps(report['discriminator_step']['losses'], sort_keys=True)}`",
            f"- Generator: `{json.dumps(report['generator_step']['losses'], sort_keys=True)}`",
            "",
        ]
    )


def build_audit(config_path: Path) -> dict[str, Any]:
    started = perf_counter()
    trainer_config = load_gan_trainer_config(config_path)
    configure_reproducibility(trainer_config.seed, deterministic=True, warn_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pair_config = load_gan_training_pair_config(
        REPO_ROOT / trainer_config.training_pair_config_path
    )
    architecture_config = load_gan_architecture_config(
        REPO_ROOT / trainer_config.architecture_config_path
    )
    loss_config = load_gan_loss_config(REPO_ROOT / trainer_config.loss_config_path)
    if pair_config.loss_config_path != trainer_config.loss_config_path:
        raise ValueError("G1.4 and G1.3 loss configuration paths disagree")
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    internal = create_internal_gan_split(
        metadata,
        monitor_fraction=pair_config.monitor_fraction,
        seed=pair_config.base_seed,
    )
    internal.assert_disjoint()
    calibration_samples = (
        trainer_config.deterministic_audit_batches * trainer_config.batch_size
    )
    total_train_samples = calibration_samples + trainer_config.batch_size
    train_dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        pair_config,
        split="train",
        internal_split=internal,
        length=total_train_samples,
    )
    monitor_dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        pair_config,
        split="monitor",
        internal_split=internal,
        length=trainer_config.batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_gan_training_samples,
    )
    monitor_loader = DataLoader(
        monitor_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_gan_training_samples,
    )
    print("Materializing deterministic development-training audit batches", flush=True)
    train_batches = list(train_loader)
    monitor_batch = next(iter(monitor_loader))
    calibration_batches = train_batches[: trainer_config.deterministic_audit_batches]
    optimization_batch = train_batches[trainer_config.deterministic_audit_batches]
    validity = _validity_preflight(
        calibration_batches,
        support_radius=architecture_config.support_dilation_radius,
    )
    generator, discriminator = build_gan_models(architecture_config)
    trainer = GANOneStepTrainer(
        generator,
        discriminator,
        trainer_config,
        loss_config,
        device=device,
    )
    initial_hashes = {
        "generator": parameter_hash(trainer.generator),
        "discriminator": parameter_hash(trainer.discriminator),
    }
    print("Running fixed monitor forward pass before optimization", flush=True)
    monitor_before = trainer.monitor_forward(monitor_batch)
    calibration_started = perf_counter()

    def progress(index: int, total: int) -> None:
        elapsed = perf_counter() - calibration_started
        print(
            f"Calibrated deterministic training batch {index}/{total} "
            f"({elapsed:.1f} seconds)",
            flush=True,
        )

    calibration = calibrate_gan_loss_scales(
        trainer, calibration_batches, progress=progress
    )
    calibration_runtime = perf_counter() - calibration_started
    after_calibration_hashes = {
        "generator": parameter_hash(trainer.generator),
        "discriminator": parameter_hash(trainer.discriminator),
    }
    print("Executing the single discriminator optimizer step", flush=True)
    discriminator_result = trainer.discriminator_step(
        optimization_batch, global_step=0
    )
    print("Executing the single generator optimizer step", flush=True)
    generator_result = trainer.generator_step(optimization_batch)
    print("Running fixed monitor forward pass after optimization", flush=True)
    monitor_after = trainer.monitor_forward(monitor_batch)
    runtime = perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    isolation = {
        "calibration_preserved_generator_parameters": (
            initial_hashes["generator"] == after_calibration_hashes["generator"]
        ),
        "calibration_preserved_discriminator_parameters": (
            initial_hashes["discriminator"]
            == after_calibration_hashes["discriminator"]
        ),
        "discriminator_step_changed_only_discriminator": (
            discriminator_result["discriminator_parameters_changed"]
            and not discriminator_result["generator_parameters_changed"]
        ),
        "generator_step_changed_only_generator": (
            generator_result["generator_parameters_changed"]
            and not generator_result["discriminator_parameters_changed"]
        ),
        "parameter_hashes": {
            "initial": initial_hashes,
            "after_calibration": after_calibration_hashes,
            "before_discriminator_step": {
                "generator": discriminator_result["generator_parameter_hash_before"],
                "discriminator": discriminator_result[
                    "discriminator_parameter_hash_before"
                ],
            },
            "after_discriminator_step": {
                "generator": discriminator_result["generator_parameter_hash_after"],
                "discriminator": discriminator_result[
                    "discriminator_parameter_hash_after"
                ],
            },
            "after_generator_step": {
                "generator": generator_result["generator_parameter_hash_after"],
                "discriminator": generator_result[
                    "discriminator_parameter_hash_after"
                ],
            },
        },
    }
    zero_generator_components = sum(
        component["zero_gradient_count"]
        for component in calibration["generator_unit_gradient_scales"].values()
    )
    invariants = {
        "all_calibration_values_finite": all(
            values["nonfinite_count"] == 0
            for values in calibration["raw_loss_distributions"].values()
        ),
        "all_generator_components_have_nonzero_gradient": zero_generator_components == 0,
        "calibration_parameters_unchanged": all(
            value for name, value in isolation.items() if name.startswith("calibration_")
        ),
        "discriminator_step_changes_only_discriminator": isolation[
            "discriminator_step_changed_only_discriminator"
        ],
        "generator_step_changes_only_generator": isolation[
            "generator_step_changed_only_generator"
        ],
        "optimizer_states_finite": (
            discriminator_result["optimizer_state_finite"]
            and generator_result["optimizer_state_finite"]
        ),
        "generator_locality_exact": generator_result[
            "generator_locality_after_step"
        ],
        "canonical_defect_adversarial_gradient_complete": generator_result[
            "canonical_defect_gradient_coverage"
        ]
        == 1.0,
        "invalid_fake_adversarial_gradient_zero": generator_result[
            "maximum_invalid_fake_pixel_gradient"
        ]
        == 0.0,
        "canonical_containment_complete": validity[
            "canonical_pixels_outside_joint_validity"
        ]
        == 0,
        "exactly_one_discriminator_step": trainer.discriminator_optimizer_steps == 1,
        "exactly_one_generator_step": trainer.generator_optimizer_steps == 1,
        "monitor_never_optimized": (
            monitor_before["optimizer_steps"] == monitor_after["optimizer_steps"] == 0
        ),
    }
    return {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "trainer_version": trainer_config.trainer_version,
        "configuration": config_as_dict(trainer_config),
        "device": str(device),
        "precision": {
            "configured_cuda": trainer_config.cuda_precision,
            "configured_cpu": trainer_config.cpu_precision,
            "effective": trainer.precision,
            "autocast_used": trainer.precision != "fp32",
            "grad_scaler_used": trainer.precision == "fp16",
        },
        "validity_preflight": validity,
        "calibration": calibration,
        "discriminator_step": discriminator_result,
        "generator_step": generator_result,
        "monitor_forward_before": monitor_before,
        "monitor_forward_after": monitor_after,
        "parameter_isolation": isolation,
        "invariants": invariants,
        "discriminator_optimizer_steps": trainer.discriminator_optimizer_steps,
        "generator_optimizer_steps": trainer.generator_optimizer_steps,
        "total_training_batches_optimized": 1,
        "monitor_optimizer_steps": 0,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_training_images": 0,
        "model_checkpoint_saved": False,
        "runtime_seconds": runtime,
        "calibration_batches_per_second": (
            trainer_config.deterministic_audit_batches
            / calibration_runtime
        ),
        "optimized_images_per_second": trainer_config.batch_size / runtime,
        "peak_cuda_memory_bytes": peak_memory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_one_step.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_training" / "one_step" / "audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_training" / "one_step" / "audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_audit(args.config)
    except Exception as error:
        report = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "discriminator_optimizer_steps": 0,
            "generator_optimizer_steps": 0,
            "total_training_batches_optimized": 0,
            "monitor_optimizer_steps": 0,
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "materialized_training_images": 0,
            "model_checkpoint_saved": False,
        }
    _atomic_write(args.json_output, json.dumps(report, indent=2) + "\n")
    _atomic_write(args.markdown_output, _markdown(report))
    print(f"G1.4 audit status: {report['status']}", flush=True)
    if report["status"] != "PASS":
        print(report.get("error", "One or more invariants failed"), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
