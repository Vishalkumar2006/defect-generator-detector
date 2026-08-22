"""Run deterministic gated GAN warmup and joint-update schedules."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
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
from defectgen.models import UNet, build_gan_models, load_gan_architecture_config  # noqa: E402
from defectgen.training.gan_losses import load_gan_loss_config  # noqa: E402
from defectgen.training.gan_smoke import (  # noqa: E402
    AtomicJSONLLog,
    DetectorRetentionGate,
    FrozenDetectorEvaluator,
    GANSmokeConfig,
    SmokeCheckpointIdentity,
    SmokeProgress,
    canonical_configuration_hash,
    canonical_state_hash,
    optimizer_state_hash,
    parameter_state_hash,
    load_gan_smoke_config,
    load_smoke_checkpoint,
    module_parameters_are_finite,
    save_smoke_checkpoint,
    select_fixed_monitor_samples,
    select_stratified_monitor_count,
    stage_one_allows_continuation,
    warmup_gate_decision,
)
from defectgen.training.gan_trainer import (  # noqa: E402
    GANOneStepTrainer,
    GANTrainingBatch,
    GANTrainingNumericalError,
    boundary_residual_telemetry,
    collate_gan_training_samples,
    load_gan_trainer_config,
    optimizer_state_is_finite,
    precision_autocast,
)
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _configuration_dict(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Smoke configuration must be a JSON object")
    return values


class DeterministicBatchStream:
    def __init__(
        self,
        dataset: GANTrainingPairDataset,
        config: GANSmokeConfig,
        progress: SmokeProgress,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.progress = progress
        self.iterator = None

    def _loader(self) -> DataLoader:
        self.dataset.set_epoch(self.progress.data_epoch)
        arguments: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": False,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": False,
            "collate_fn": collate_gan_training_samples,
        }
        if self.config.num_workers:
            arguments["prefetch_factor"] = self.config.prefetch_factor
        loader = DataLoader(self.dataset, **arguments)
        iterator = iter(loader)
        for _ in range(self.progress.batch_position):
            next(iterator)
        return iterator

    def next(self) -> tuple[GANTrainingBatch, float]:
        started = perf_counter()
        if self.iterator is None:
            self.iterator = self._loader()
        batch = next(self.iterator)
        self.progress.batch_position += 1
        if self.progress.batch_position >= self.config.batches_per_data_epoch:
            self.progress.data_epoch += 1
            self.progress.batch_position = 0
            self.iterator = None
        return batch, perf_counter() - started


def _load_detector(config: GANSmokeConfig, device: torch.device) -> FrozenDetectorEvaluator:
    values = json.loads((REPO_ROOT / config.detector_config_path).read_text(encoding="utf-8"))
    model_config = values["model"]
    detector = UNet(
        input_channels=model_config["input_channels"],
        output_channels=model_config["output_channels"],
        base_channels=model_config["base_channels"],
    )
    payload = torch.load(
        REPO_ROOT / config.detector_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    detector.load_state_dict(payload["model_state"])
    normalization = values["data"]["detector_normalization"]
    return FrozenDetectorEvaluator(
        detector,
        mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"],
        threshold=config.detector_threshold,
        device=device,
    )


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if "parameter_hash" not in key and key != "batch_ids"
    }


def _active_contact_combination(metadata: dict[str, Any]) -> str:
    contacts = metadata["target_contact_sides"]
    active = [side for side in ("top", "bottom", "left", "right") if contacts[side]]
    return "+".join(active) if active else "none"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _training_gate(
    trainer: GANOneStepTrainer,
    config: GANSmokeConfig,
    discriminator: dict[str, Any],
    generator: dict[str, Any] | None,
) -> str | None:
    def finite_tree(value: Any) -> bool:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(item) for item in value)
        return True

    if not finite_tree(discriminator) or (
        generator is not None and not finite_tree(generator)
    ):
        return "nonfinite_training_metric"
    logit_extrema = [
        abs(discriminator[branch][bound])
        for branch in ("real_logits", "fake_logits")
        for bound in ("minimum", "maximum")
    ]
    if max(logit_extrema) > config.absolute_logit_stop:
        return "absolute_logit_above_limit"
    if not module_parameters_are_finite(trainer.generator) or not module_parameters_are_finite(
        trainer.discriminator
    ):
        return "nonfinite_parameter"
    if not optimizer_state_is_finite(
        trainer.generator_optimizer
    ) or not optimizer_state_is_finite(trainer.discriminator_optimizer):
        return "nonfinite_optimizer_state"
    if generator is None:
        return None
    if generator["exact_outside_support_change"] != 0:
        return "outside_support_change"
    if generator["maximum_invalid_fake_pixel_gradient"] != 0:
        return "invalid_fake_pixel_adversarial_gradient"
    if generator["canonical_defect_gradient_nonfinite_channel_count"] != 0:
        return "nonfinite_canonical_adversarial_gradient"
    canonical_active = generator["canonical_defect_gradient_active_pixel_count"]
    canonical_total = generator["canonical_defect_gradient_total_pixel_count"]
    if canonical_total <= 0 or canonical_active != canonical_total:
        return "incomplete_canonical_adversarial_gradient"
    if (
        generator["output_range_violation_count"]
        > config.output_range_violation_stop_count
    ):
        return "output_range_violation"
    if (
        generator["mean_absolute_residual_inside_support"]
        > config.mean_support_change_stop
    ):
        return "mean_support_change_above_limit"
    if max(
        abs(generator["fake_logits"]["minimum"]),
        abs(generator["fake_logits"]["maximum"]),
    ) > config.absolute_logit_stop:
        return "absolute_generator_logit_above_limit"
    return None


def _semantic_monitor(
    trainer: GANOneStepTrainer,
    evaluator: FrozenDetectorEvaluator,
    panel: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    aggregate: dict[str, list[float]] = {
        branch: []
        for branch in (
            "composite",
            "refined",
            "genuine_real",
        )
    }
    branch_metrics: dict[str, list[dict[str, float]]] = {
        branch: [] for branch in aggregate
    }
    probabilities: dict[str, torch.Tensor] = {}
    generator_training = trainer.generator.training
    trainer.generator.eval()
    try:
        for category, sample in panel.items():
            batch = collate_gan_training_samples([sample]).to(trainer.device)
            with torch.no_grad(), precision_autocast(trainer.device, trainer.precision):
                generated = trainer.generator(batch.composite_image, batch.generator_mask)
            for branch, image, valid in (
                ("composite", batch.composite_image, batch.fake_valid_mask),
                ("refined", generated.refined_image, batch.fake_valid_mask),
                ("genuine_real", batch.real_image, batch.real_valid_mask),
            ):
                metrics, probability = evaluator.metrics(
                    image, batch.fake_discriminator_mask, valid
                )
                branch_metrics[branch].append(metrics)
                if branch == "refined":
                    probabilities[category] = probability[0]
    finally:
        trainer.generator.train(generator_training)
    summarized = {
        branch: {
            name: float(np.mean([item[name] for item in values]))
            for name in values[0]
        }
        for branch, values in branch_metrics.items()
    }
    summarized["refined_vs_composite"] = {
        name: summarized["refined"][name] - summarized["composite"][name]
        for name in summarized["composite"]
    }
    return summarized, probabilities


def _gan_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach().cpu().float().add(1).div(2).clamp(0, 1).permute(1, 2, 0).numpy()
    )


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    result = rgb.copy()
    result[mask] = 0.45 * result[mask] + 0.55 * np.asarray(color)
    return np.clip(result, 0, 1)


def _write_contact_sheet(
    path: Path,
    *,
    trainer: GANOneStepTrainer,
    panel: dict[str, Any],
    evaluator: FrozenDetectorEvaluator,
    title: str,
) -> None:
    rows = len(panel)
    figure, axes = plt.subplots(rows, 6, figsize=(12, 2.2 * rows), squeeze=False)
    generator_training = trainer.generator.training
    trainer.generator.eval()
    try:
        for row, (category, sample) in enumerate(panel.items()):
            batch = collate_gan_training_samples([sample]).to(trainer.device)
            with torch.no_grad(), precision_autocast(trainer.device, trainer.precision):
                generated = trainer.generator(batch.composite_image, batch.generator_mask)
            detector_metrics, probability = evaluator.metrics(
                generated.refined_image,
                batch.fake_discriminator_mask,
                batch.fake_valid_mask,
            )
            real = _gan_rgb(batch.real_image[0])
            composite = _gan_rgb(batch.composite_image[0])
            refined = _gan_rgb(generated.refined_image[0])
            difference = np.abs(refined - composite)
            difference = np.clip(difference * 8, 0, 1)
            mask = batch.fake_discriminator_mask[0, 0].bool().cpu().numpy()
            predicted = probability[0, 0].numpy() >= evaluator.threshold
            panels = (
                (real, "genuine real"),
                (composite, "initial composite"),
                (refined, "current refined"),
                (difference, "|difference| x8"),
                (_overlay(composite, mask, (1, 0, 0)), "generator mask"),
                (_overlay(refined, predicted, (0, 1, 0)), "detector prediction"),
            )
            for column, (image, heading) in enumerate(panels):
                axes[row, column].imshow(image)
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(heading, fontsize=8)
            contacts = _active_contact_combination(sample.metadata)
            axes[row, 0].set_ylabel(
                f"{category}\n{sample.metadata['template_id']}\n{contacts}",
                fontsize=7,
            )
            axes[row, 2].text(
                0.01,
                0.01,
                f"P-in {detector_metrics['mean_probability_inside_mask']:.3f}\n"
                f"Dice {detector_metrics['dice_at_0_5']:.3f}",
                transform=axes[row, 2].transAxes,
                fontsize=6,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
    finally:
        trainer.generator.train(generator_training)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(figure)


def _plot_series(
    records: list[dict[str, Any]],
    report_dir: Path,
    *,
    discriminator_clip_max_norm: float,
) -> None:
    joint = [record for record in records if record["kind"] == "joint"]
    if not joint:
        return
    steps = [record["joint_step"] for record in joint]
    plots = (
        (
            "losses.png",
            {
                "D hinge": [record["discriminator"]["losses"]["total_hinge"] for record in joint],
                "G adversarial": [record["generator"]["losses"]["adversarial"] for record in joint],
                "G total": [record["generator"]["losses"]["total"] for record in joint],
            },
            "Loss",
        ),
        (
            "logits.png",
            {
                "D real mean": [record["discriminator"]["real_logits"]["mean"] for record in joint],
                "D fake mean": [record["discriminator"]["fake_logits"]["mean"] for record in joint],
                "D margin": [record["discriminator"]["real_minus_fake_logit_margin"] for record in joint],
            },
            "Raw logit",
        ),
        (
            "gradient_norms.png",
            {
                "D pre-clip": [record["discriminator"]["gradient_clipping"]["pre_clipping_norm"] for record in joint],
                "G pre-clip": [record["generator"]["gradient_clipping"]["pre_clipping_norm"] for record in joint],
                "clip limit": [discriminator_clip_max_norm for _ in joint],
            },
            "Gradient norm",
        ),
    )
    for filename, series, ylabel in plots:
        figure, axis = plt.subplots(figsize=(7, 4))
        for name, values in series.items():
            axis.plot(steps, values, label=name)
        axis.set_xlabel("Joint step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(report_dir / filename, dpi=120)
        plt.close(figure)
    monitors = [record for record in records if record["kind"] == "monitor"]
    if monitors:
        figure, axis = plt.subplots(figsize=(7, 4))
        monitor_steps = [record["joint_step"] for record in monitors]
        axis.plot(
            monitor_steps,
            [record["semantic"]["refined_vs_composite"]["mean_probability_inside_mask"] for record in monitors],
            label="refined - composite inside probability",
        )
        axis.plot(
            monitor_steps,
            [record["semantic"]["refined_vs_composite"]["dice_at_0_5"] for record in monitors],
            label="refined - composite Dice",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Joint step")
        axis.set_ylabel("Detector metric change")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(report_dir / "detector_retention.png", dpi=120)
        plt.close(figure)


def _link_last(source: Path, last: Path) -> None:
    temporary = last.with_suffix(last.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, last)


def _checkpoint(
    name: str,
    *,
    checkpoint_dir: Path,
    trainer: GANOneStepTrainer,
    progress: SmokeProgress,
    identity: SmokeCheckpointIdentity,
    configuration: dict[str, Any],
) -> None:
    numbered = checkpoint_dir / name
    save_smoke_checkpoint(
        numbered,
        trainer=trainer,
        progress=progress,
        identity=identity,
        configuration=configuration,
    )
    _link_last(numbered, checkpoint_dir / "last.pt")


def _monitor_record(
    *,
    phase: str,
    trainer: GANOneStepTrainer,
    monitor_batch: GANTrainingBatch,
    evaluator: FrozenDetectorEvaluator,
    panel: dict[str, Any],
    progress: SmokeProgress,
    retention: DetectorRetentionGate,
    last_train_margin: float | None,
    config: GANSmokeConfig,
) -> tuple[dict[str, Any], bool, list[str]]:
    before_g = trainer.generator_optimizer_steps
    before_d = trainer.discriminator_optimizer_steps
    logits = trainer.monitor_forward(monitor_batch)
    semantic, _ = _semantic_monitor(trainer, evaluator, panel)
    if (
        trainer.generator_optimizer_steps != before_g
        or trainer.discriminator_optimizer_steps != before_d
    ):
        raise RuntimeError("Monitor evaluation mutated optimizer-step counts")
    gate = retention.update(
        semantic["composite"]["mean_probability_inside_mask"],
        semantic["refined"]["mean_probability_inside_mask"],
    )
    dice_drop = (
        semantic["composite"]["dice_at_0_5"]
        - semantic["refined"]["dice_at_0_5"]
    )
    warnings: list[str] = []
    if gate["warning"]:
        warnings.append("detector_inside_probability_retention_below_70_percent")
    if dice_drop > config.detector_dice_warning_drop:
        warnings.append("detector_dice_decline_above_0_15")
    progress.monitor_evaluations += 1
    record = {
        "kind": "monitor",
        "phase": phase,
        "warmup_step": progress.warmup_steps,
        "joint_step": progress.joint_generator_steps,
        "logits": logits,
        "train_margin": last_train_margin,
        "train_minus_monitor_margin": (
            None
            if last_train_margin is None
            else last_train_margin - logits["real_minus_fake_logit_margin"]
        ),
        "semantic": semantic,
        "retention_gate": gate,
        "dice_drop": dice_drop,
        "warnings": warnings,
        "optimizer_steps": {
            "generator": trainer.generator_optimizer_steps,
            "discriminator": trainer.discriminator_optimizer_steps,
        },
    }
    return record, gate["stop"], warnings


DETECTOR_METRIC_NAMES = (
    "mean_probability_inside_mask",
    "mean_probability_outside_mask",
    "inside_outside_probability_contrast",
    "dice_at_0_5",
    "iou_at_0_5",
    "samples_with_any_predicted_positive_fraction",
)


def _mean_metric_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in DETECTOR_METRIC_NAMES
    }


def _detector_distance_from_genuine_real(
    refined: dict[str, float], genuine_real: dict[str, float]
) -> dict[str, float]:
    differences = np.asarray(
        [refined[name] - genuine_real[name] for name in DETECTOR_METRIC_NAMES],
        dtype=np.float64,
    )
    return {
        "l2": float(np.sqrt(np.square(differences).sum())),
        "mean_absolute": float(np.abs(differences).mean()),
    }


def _stratified_monitor_record(
    *,
    trainer: GANOneStepTrainer,
    evaluator: FrozenDetectorEvaluator,
    panel: dict[str, tuple[Any, ...]],
    joint_step: int,
    batch_size: int,
    boundary_width: int,
) -> dict[str, Any]:
    before_steps = (
        trainer.generator_optimizer_steps,
        trainer.discriminator_optimizer_steps,
    )
    flattened = [
        (category, sample)
        for category, samples in panel.items()
        for sample in samples
    ]
    detector_values = {
        branch: [] for branch in ("composite", "refined", "genuine_real")
    }
    margins: list[float] = []
    boundary_values: list[dict[str, float]] = []
    tanh_saturation: list[float] = []
    locality_maxima: list[float] = []
    output_range_violations = 0
    generator_training = trainer.generator.training
    trainer.generator.eval()
    try:
        for start in range(0, len(flattened), batch_size):
            selected = flattened[start : start + batch_size]
            batch = collate_gan_training_samples(
                [sample for _, sample in selected]
            ).to(trainer.device)
            logits = trainer.monitor_forward(batch)
            margins.append(logits["real_minus_fake_logit_margin"])
            with torch.no_grad(), precision_autocast(trainer.device, trainer.precision):
                generated = trainer.generator(
                    batch.composite_image, batch.generator_mask
                )
            for branch, image, valid in (
                ("composite", batch.composite_image, batch.fake_valid_mask),
                ("refined", generated.refined_image, batch.fake_valid_mask),
                ("genuine_real", batch.real_image, batch.real_valid_mask),
            ):
                metrics, _ = evaluator.metrics(
                    image, batch.fake_discriminator_mask, valid
                )
                detector_values[branch].append(metrics)
            boundary_values.append(
                boundary_residual_telemetry(
                    generated.applied_residual,
                    generated.support_mask,
                    boundary_width=boundary_width,
                )
            )
            support = generated.support_mask.expand_as(generated.raw_residual)
            raw_direction = torch.tanh(generated.raw_residual.detach().float())
            tanh_saturation.append(
                float((raw_direction.abs() >= 0.99)[support].float().mean())
            )
            outside_support = ~support
            change = (
                generated.refined_image.detach().float()
                - batch.composite_image.detach().float()
            ).abs()
            locality_maxima.append(
                float(change[outside_support].max())
                if bool(outside_support.any())
                else 0.0
            )
            output_range_violations += int(
                ((generated.refined_image < -1) | (generated.refined_image > 1)).sum()
            )
    finally:
        trainer.generator.train(generator_training)
    if before_steps != (
        trainer.generator_optimizer_steps,
        trainer.discriminator_optimizer_steps,
    ):
        raise RuntimeError("Stratified monitor audit mutated optimizer-step counts")
    aggregate = {
        branch: _mean_metric_records(values)
        for branch, values in detector_values.items()
    }
    return {
        "kind": "stratified_monitor",
        "joint_step": joint_step,
        "sample_count": len(flattened),
        "category_counts": {
            category: len(samples) for category, samples in panel.items()
        },
        "detector_statistics": aggregate,
        "detector_statistic_distance_from_genuine_real": (
            _detector_distance_from_genuine_real(
                aggregate["refined"], aggregate["genuine_real"]
            )
        ),
        "real_minus_fake_margin": {
            "mean": float(np.mean(margins)),
            "minimum": float(np.min(margins)),
            "maximum": float(np.max(margins)),
        },
        "boundary_residual": {
            name: float(np.mean([value[name] for value in boundary_values]))
            for name in boundary_values[0]
        },
        "tanh_raw_residual_saturation_fraction": float(np.mean(tanh_saturation)),
        "maximum_outside_support_change": max(locality_maxima, default=0.0),
        "output_range_violation_count": output_range_violations,
        "optimizer_steps": {
            "generator": trainer.generator_optimizer_steps,
            "discriminator": trainer.discriminator_optimizer_steps,
        },
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
    }


def _state_hashes(
    trainer: GANOneStepTrainer,
    *,
    generator_state: dict[str, torch.Tensor] | None = None,
    discriminator_state: dict[str, torch.Tensor] | None = None,
    generator_optimizer_state: dict[str, Any] | None = None,
    discriminator_optimizer_state: dict[str, Any] | None = None,
) -> dict[str, str]:
    generator_names = dict(trainer.generator.named_parameters()).keys()
    discriminator_names = dict(trainer.discriminator.named_parameters()).keys()
    return {
        "generator_parameters": parameter_state_hash(
            generator_state or trainer.generator.state_dict(), generator_names
        ),
        "discriminator_parameters": parameter_state_hash(
            discriminator_state or trainer.discriminator.state_dict(),
            discriminator_names,
        ),
        "generator_optimizer": (
            optimizer_state_hash(trainer.generator_optimizer)
            if generator_optimizer_state is None
            else canonical_state_hash(generator_optimizer_state)
        ),
        "discriminator_optimizer": (
            optimizer_state_hash(trainer.discriminator_optimizer)
            if discriminator_optimizer_state is None
            else canonical_state_hash(discriminator_optimizer_state)
        ),
    }


def _rolling_training_statistics(
    records: list[dict[str, Any]], *, window_size: int
) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    joint = [record for record in records if record["kind"] == "joint"]
    windows: list[dict[str, Any]] = []
    for end in range(window_size, len(joint) + 1, window_size):
        selected = joint[end - window_size : end]
        windows.append(
            {
                "start_step": selected[0]["joint_step"],
                "end_step": selected[-1]["joint_step"],
                "discriminator_clipped_fraction": float(
                    np.mean(
                        [
                            record["discriminator"]["gradient_clipping_applied"]
                            for record in selected
                        ]
                    )
                ),
                "generator_clipped_fraction": float(
                    np.mean(
                        [
                            record["generator"]["gradient_clipping_applied"]
                            for record in selected
                        ]
                    )
                ),
                "mean_real_minus_fake_margin": float(
                    np.mean(
                        [
                            record["discriminator"][
                                "real_minus_fake_logit_margin"
                            ]
                            for record in selected
                        ]
                    )
                ),
                "mean_boundary_residual_mass_fraction": float(
                    np.mean(
                        [
                            record["generator"][
                                "boundary_residual_mass_fraction"
                            ]
                            for record in selected
                        ]
                    )
                ),
                "mean_boundary_residual_enrichment": float(
                    np.mean(
                        [
                            record["generator"]["boundary_residual_enrichment"]
                            for record in selected
                        ]
                    )
                ),
                "maximum_tanh_saturation_fraction": max(
                    record["generator"][
                        "tanh_raw_residual_saturation_fraction"
                    ]
                    for record in selected
                ),
            }
        )
    return windows


def run(config_path: Path, *, resume: bool) -> dict[str, Any]:
    started = perf_counter()
    raw_config = _configuration_dict(config_path)
    config = load_gan_smoke_config(config_path)
    sustained = raw_config.get("run_kind") == "g2_1_sustained"
    stratified_count = int(raw_config.get("stratified_monitor_pair_count", 0))
    stratified_steps = {
        int(step) for step in raw_config.get("stratified_monitor_steps", [])
    }
    rolling_window_size = int(raw_config.get("rolling_window_size", 100))
    configure_reproducibility(config.seed, deterministic=True, warn_only=False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GAN training requires explicitly supported CUDA BF16")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    report_dir = REPO_ROOT / config.report_directory
    checkpoint_dir = REPO_ROOT / config.checkpoint_directory
    if not resume and (checkpoint_dir / "last.pt").exists():
        raise FileExistsError("Smoke checkpoint already exists; use --resume or move it")
    pair_config = load_gan_training_pair_config(
        REPO_ROOT / config.training_pair_config_path
    )
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    internal = create_internal_gan_split(
        metadata,
        monitor_fraction=pair_config.monitor_fraction,
        seed=pair_config.base_seed,
    )
    internal.assert_disjoint()
    architecture = load_gan_architecture_config(
        REPO_ROOT / config.architecture_config_path
    )
    loss_config = load_gan_loss_config(REPO_ROOT / config.loss_config_path)
    base_trainer = load_gan_trainer_config(REPO_ROOT / config.trainer_config_path)
    trainer_config = config.trainer_config(base_trainer)
    generator, discriminator = build_gan_models(architecture)
    trainer = GANOneStepTrainer(
        generator, discriminator, trainer_config, loss_config, device=device
    )
    train_dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        pair_config,
        split="train",
        internal_split=internal,
        length=config.batches_per_data_epoch * config.batch_size,
    )
    monitor_dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        pair_config,
        split="monitor",
        internal_split=internal,
        length=config.monitor_panel_scan_limit,
    )
    candidates = []
    panel = None
    for index in range(config.monitor_panel_scan_limit):
        candidates.append(monitor_dataset[index])
        try:
            panel = select_fixed_monitor_samples(candidates)
            break
        except RuntimeError:
            continue
    if panel is None:
        raise RuntimeError("Fixed monitor panel selection failed")
    monitor_ids = tuple(
        f"{category}:{sample.metadata['template_id']}:{sample.metadata['normal_background_sample_id']}"
        for category, sample in panel.items()
    )
    monitor_batch = collate_gan_training_samples(list(panel.values())[: config.batch_size])
    stratified_panel: dict[str, tuple[Any, ...]] = {}
    stratified_ids: tuple[str, ...] = ()
    if stratified_count:
        stratified_panel = select_stratified_monitor_count(
            monitor_dataset,
            total_count=stratified_count,
        )
        stratified_ids = tuple(
            f"{category}:{sample.metadata['template_id']}:"
            f"{sample.metadata['normal_background_sample_id']}"
            for category, samples in stratified_panel.items()
            for sample in samples
        )
    identity = SmokeCheckpointIdentity(
        configuration_sha256=canonical_configuration_hash(raw_config),
        gan_manifest_content_sha256=metadata["gan_manifest_content_sha256"],
        split_sha256=metadata["split_sha256"],
        fixed_monitor_sample_ids=monitor_ids,
        stratified_monitor_sample_ids=stratified_ids,
    )
    progress = SmokeProgress()
    metrics_path = report_dir / "metrics.jsonl"
    existing_records: list[dict[str, Any]] = []
    if resume:
        progress = load_smoke_checkpoint(
            checkpoint_dir / "last.pt",
            trainer=trainer,
            expected_identity=identity,
            expected_configuration=raw_config,
        )
        if metrics_path.exists():
            loaded_records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            monitor_count = 0
            for record in loaded_records:
                if record["kind"] == "warmup" and record["warmup_step"] > progress.warmup_steps:
                    continue
                if record["kind"] == "joint" and record["joint_step"] > progress.joint_generator_steps:
                    continue
                if record["kind"] == "monitor":
                    if monitor_count >= progress.monitor_evaluations:
                        continue
                    monitor_count += 1
                if (
                    record["kind"] in {"stratified_monitor", "replay_verification"}
                    and record["joint_step"] > progress.joint_generator_steps
                ):
                    continue
                existing_records.append(record)
    stream = DeterministicBatchStream(train_dataset, config, progress)
    metric_log = AtomicJSONLLog(metrics_path, existing_records)
    detector = _load_detector(config, device)
    retention = DetectorRetentionGate(
        warning_ratio=config.detector_inside_retention_warning_ratio,
        stop_ratio=config.detector_inside_retention_stop_ratio,
        consecutive_below_stop=progress.detector_retention_consecutive_below_stop,
    )
    utilization_templates: Counter[str] = Counter()
    utilization_backgrounds: Counter[str] = Counter()
    utilization_contacts: Counter[str] = Counter()
    warnings: list[str] = [
        warning
        for record in existing_records
        if record["kind"] == "monitor"
        for warning in record.get("warnings", [])
    ]
    early_stop_reason: str | None = progress.early_stop_reason
    last_train_margin: float | None = None
    visual_paths: list[str] = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((report_dir / "contact_sheets").glob("*.png"))
    ] if (report_dir / "contact_sheets").exists() else []
    support_containment: dict[str, float] = {}
    for category, sample in panel.items():
        canonical = sample.fake_discriminator_mask.bool().unsqueeze(0)
        halo = F.max_pool2d(
            canonical.float(),
            kernel_size=2 * architecture.support_dilation_radius + 1,
            stride=1,
            padding=architecture.support_dilation_radius,
        ).bool()[0]
        complete_support = halo | (sample.generator_mask > 0)
        joint_valid = sample.real_valid_mask.bool() & sample.fake_valid_mask.bool()
        support_containment[category] = float(
            (complete_support & joint_valid).sum() / complete_support.sum()
        )
    if min(support_containment.values()) < 0.95:
        warnings.append("fixed_monitor_support_containment_below_95_percent")
    if float(np.mean(list(support_containment.values()))) < 0.99:
        warnings.append("fixed_monitor_mean_support_containment_below_99_percent")

    for record in existing_records:
        if record["kind"] not in {"warmup", "joint"}:
            continue
        for identity_value in record.get("template_ids", []):
            utilization_templates[identity_value] += 1
        for identity_value in record.get("background_ids", []):
            utilization_backgrounds[identity_value] += 1
        for combination in record.get("contact_combinations", []):
            utilization_contacts[combination] += 1

    def record_monitor(phase: str) -> dict[str, Any]:
        nonlocal early_stop_reason
        record, semantic_stop, monitor_warnings = _monitor_record(
            phase=phase,
            trainer=trainer,
            monitor_batch=monitor_batch,
            evaluator=detector,
            panel=panel,
            progress=progress,
            retention=retention,
            last_train_margin=last_train_margin,
            config=config,
        )
        metric_log.append(record)
        warnings.extend(monitor_warnings)
        progress.detector_retention_consecutive_below_stop = (
            retention.consecutive_below_stop
        )
        if semantic_stop:
            early_stop_reason = "detector_retention_below_50_percent_twice"
        return record

    def write_visual(phase: str, visual_name: str) -> None:
        path = report_dir / "contact_sheets" / f"{visual_name}.png"
        _write_contact_sheet(
            path,
            trainer=trainer,
            panel=panel,
            evaluator=detector,
            title=f"{'G2.1' if sustained else 'G1.5'} {phase}",
        )
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in visual_paths:
            visual_paths.append(relative)

    def record_stratified_monitor() -> dict[str, Any]:
        if not stratified_panel:
            raise RuntimeError("Stratified monitor panel is not configured")
        record = _stratified_monitor_record(
            trainer=trainer,
            evaluator=detector,
            panel=stratified_panel,
            joint_step=progress.joint_generator_steps,
            batch_size=config.batch_size,
            boundary_width=loss_config.boundary_ring_width,
        )
        metric_log.append(record)
        if record["maximum_outside_support_change"] != 0:
            raise RuntimeError("Stratified monitor violated exact generator locality")
        if record["output_range_violation_count"] != 0:
            raise RuntimeError("Stratified monitor found output-range violations")
        return record

    def verify_selected_smoke_replay() -> dict[str, Any]:
        reference_path = REPO_ROOT / raw_config["selected_smoke_step_200_checkpoint"]
        payload = torch.load(reference_path, map_location="cpu", weights_only=False)
        reference = _state_hashes(
            trainer,
            generator_state=payload["generator_state"],
            discriminator_state=payload["discriminator_state"],
            generator_optimizer_state=payload["generator_optimizer_state"],
            discriminator_optimizer_state=payload["discriminator_optimizer_state"],
        )
        actual = _state_hashes(trainer)
        record = {
            "kind": "replay_verification",
            "joint_step": progress.joint_generator_steps,
            "reference_checkpoint": raw_config[
                "selected_smoke_step_200_checkpoint"
            ],
            "reference_hashes": reference,
            "actual_hashes": actual,
            "all_match": reference == actual,
        }
        metric_log.append(record)
        _atomic_write(
            report_dir / "step_0200_replay_verification.json",
            json.dumps(record, indent=2) + "\n",
        )
        return record

    if not resume:
        record_monitor("step_0")
        write_visual("step_0", "step_000")
        if 0 in stratified_steps:
            record_stratified_monitor()

    while progress.warmup_gate_status == "pending" and early_stop_reason is None:
        batch, sampling_time = stream.next()
        _synchronize(device)
        compute_started = perf_counter()
        try:
            result = trainer.discriminator_step(
                batch,
                global_step=trainer.discriminator_optimizer_steps,
                verify_parameter_isolation=False,
            )
        except GANTrainingNumericalError as error:
            early_stop_reason = f"numerical_failure:{error}"
            metric_log.append(
                {
                    "kind": "failure",
                    "phase": "warmup",
                    "warmup_step_attempted": progress.warmup_steps + 1,
                    "error": str(error),
                }
            )
            break
        _synchronize(device)
        compute_time = perf_counter() - compute_started
        progress.warmup_steps += 1
        progress.last_completed_operation = f"warmup_d_{progress.warmup_steps}"
        last_train_margin = result["real_minus_fake_logit_margin"]
        for metadata_item in batch.metadata:
            utilization_templates[metadata_item["template_id"]] += 1
            utilization_backgrounds[metadata_item["normal_background_sample_id"]] += 1
            utilization_contacts[_active_contact_combination(metadata_item)] += 1
        gate_reason = _training_gate(trainer, config, result, None)
        metric_log.append(
            {
                "kind": "warmup",
                "warmup_step": progress.warmup_steps,
                "discriminator": _compact(result),
                "sampling_time_seconds": sampling_time,
                "gpu_computation_time_seconds": compute_time,
                "total_step_time_seconds": sampling_time + compute_time,
                "pairs_per_second": config.batch_size / (sampling_time + compute_time),
                "data_epoch": progress.data_epoch,
                "batch_position": progress.batch_position,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
                "template_ids": [item["template_id"] for item in batch.metadata],
                "background_ids": [
                    item["normal_background_sample_id"] for item in batch.metadata
                ],
                "contact_combinations": [
                    _active_contact_combination(item) for item in batch.metadata
                ],
            }
        )
        if gate_reason:
            early_stop_reason = gate_reason
            break
        if progress.warmup_steps in {5, 10, 15, 20}:
            monitor = record_monitor(f"warmup_{progress.warmup_steps}")
            decision = warmup_gate_decision(
                completed_steps=progress.warmup_steps,
                monitor_margin=monitor["logits"]["real_minus_fake_logit_margin"],
                initial_steps=config.initial_discriminator_warmup_steps,
                maximum_steps=config.maximum_discriminator_warmup_steps,
            )
            if decision == "accepted":
                progress.warmup_gate_status = "accepted"
                if raw_config.get("visualize_after_warmup", True):
                    write_visual(
                        f"accepted_warmup_{progress.warmup_steps}",
                        f"after_warmup_{progress.warmup_steps:03d}",
                    )
                _checkpoint(
                    f"warmup_{progress.warmup_steps:03d}.pt",
                    checkpoint_dir=checkpoint_dir,
                    trainer=trainer,
                    progress=progress,
                    identity=identity,
                    configuration=raw_config,
                )
            elif decision == "failed":
                progress.warmup_gate_status = "failed"
                early_stop_reason = "nonpositive_monitor_discriminator_margin_after_20_warmup_steps"
        if progress.warmup_steps % config.progress_interval == 0:
            print(
                f"Warmup {progress.warmup_steps}: train margin={last_train_margin:.4f}",
                flush=True,
            )

    while (
        progress.warmup_gate_status == "accepted"
        and progress.joint_generator_steps < config.full_smoke_joint_steps
        and early_stop_reason is None
    ):
        batch, sampling_time = stream.next()
        _synchronize(device)
        compute_started = perf_counter()
        try:
            discriminator_result = trainer.discriminator_step(
                batch,
                global_step=trainer.discriminator_optimizer_steps,
                verify_parameter_isolation=False,
            )
        except GANTrainingNumericalError as error:
            early_stop_reason = f"numerical_failure:{error}"
            metric_log.append(
                {
                    "kind": "failure",
                    "phase": "joint_discriminator",
                    "joint_step_attempted": progress.joint_generator_steps + 1,
                    "error": str(error),
                }
            )
            break
        progress.joint_discriminator_steps += 1
        progress.last_completed_operation = (
            f"joint_{progress.joint_generator_steps + 1}_d"
        )
        try:
            generator_result = trainer.generator_step(
                batch, verify_parameter_isolation=False
            )
        except GANTrainingNumericalError as error:
            early_stop_reason = f"numerical_failure:{error}"
            metric_log.append(
                {
                    "kind": "failure",
                    "phase": "joint_generator",
                    "joint_step_attempted": progress.joint_generator_steps + 1,
                    "error": str(error),
                }
            )
            break
        progress.joint_generator_steps += 1
        _synchronize(device)
        compute_time = perf_counter() - compute_started
        joint_step = progress.joint_generator_steps
        progress.last_completed_operation = f"joint_{joint_step}_g"
        last_train_margin = discriminator_result["real_minus_fake_logit_margin"]
        for metadata_item in batch.metadata:
            utilization_templates[metadata_item["template_id"]] += 1
            utilization_backgrounds[metadata_item["normal_background_sample_id"]] += 1
            utilization_contacts[_active_contact_combination(metadata_item)] += 1
        metric_log.append(
            {
                "kind": "joint",
                "joint_step": joint_step,
                "discriminator": _compact(discriminator_result),
                "generator": _compact(generator_result),
                "sampling_time_seconds": sampling_time,
                "gpu_computation_time_seconds": compute_time,
                "total_step_time_seconds": sampling_time + compute_time,
                "pairs_per_second": config.batch_size / (sampling_time + compute_time),
                "data_epoch": progress.data_epoch,
                "batch_position": progress.batch_position,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
                "template_ids": [item["template_id"] for item in batch.metadata],
                "background_ids": [
                    item["normal_background_sample_id"] for item in batch.metadata
                ],
                "contact_combinations": [
                    _active_contact_combination(item) for item in batch.metadata
                ],
            }
        )
        gate_reason = _training_gate(
            trainer, config, discriminator_result, generator_result
        )
        if gate_reason:
            early_stop_reason = gate_reason
            break
        if joint_step % config.monitor_interval == 0:
            record_monitor(f"joint_{joint_step}")
        if joint_step in stratified_steps:
            record_stratified_monitor()
        if joint_step == int(raw_config.get("verify_selected_smoke_at_step", -1)):
            replay = verify_selected_smoke_replay()
            if not replay["all_match"]:
                early_stop_reason = "selected_smoke_step_200_hash_mismatch"
                break
        visual_due = joint_step in config.visual_steps or (
            not raw_config.get("visual_steps_only", False)
            and joint_step % config.visual_interval == 0
        )
        if visual_due:
            write_visual(f"visual_joint_{joint_step}", f"joint_{joint_step:03d}")
        if joint_step == config.micro_smoke_joint_steps:
            progress.stage_one_passed = stage_one_allows_continuation(
                completed_joint_steps=joint_step,
                target=config.micro_smoke_joint_steps,
                early_stop_reason=early_stop_reason,
            )
            _checkpoint(
                f"micro_{joint_step:03d}.pt",
                checkpoint_dir=checkpoint_dir,
                trainer=trainer,
                progress=progress,
                identity=identity,
                configuration=raw_config,
            )
            if not progress.stage_one_passed:
                early_stop_reason = early_stop_reason or "micro_smoke_gate_failed"
                break
            print("20-step micro-smoke gate passed; continuing same state", flush=True)
        if joint_step % config.checkpoint_interval == 0:
            _checkpoint(
                f"joint_{joint_step:03d}.pt",
                checkpoint_dir=checkpoint_dir,
                trainer=trainer,
                progress=progress,
                identity=identity,
                configuration=raw_config,
            )
        if joint_step % config.progress_interval == 0:
            print(
                f"Joint {joint_step}/{config.full_smoke_joint_steps}: "
                f"D={discriminator_result['losses']['total']:.4f} "
                f"G={generator_result['losses']['total']:.4f} "
                f"margin={last_train_margin:.4f}",
                flush=True,
            )

    if progress.joint_generator_steps == config.full_smoke_joint_steps:
        progress.last_completed_operation = (
            f"joint_{config.full_smoke_joint_steps}_complete"
        )
        _checkpoint(
            f"joint_{config.full_smoke_joint_steps:0{4 if sustained else 3}d}.pt",
            checkpoint_dir=checkpoint_dir,
            trainer=trainer,
            progress=progress,
            identity=identity,
            configuration=raw_config,
        )
    elif early_stop_reason is not None:
        progress.early_stop_reason = early_stop_reason
        progress.last_completed_operation = f"stopped:{early_stop_reason}"
        _checkpoint(
            f"stopped_w{progress.warmup_steps:03d}_j{progress.joint_generator_steps:03d}.pt",
            checkpoint_dir=checkpoint_dir,
            trainer=trainer,
            progress=progress,
            identity=identity,
            configuration=raw_config,
        )
    records = metric_log.records
    _plot_series(
        records,
        report_dir,
        discriminator_clip_max_norm=config.discriminator_gradient_clip_max_norm,
    )
    warmup_records = [record for record in records if record["kind"] == "warmup"]
    joint_records = [record for record in records if record["kind"] == "joint"]
    monitor_records = [record for record in records if record["kind"] == "monitor"]
    d_records = warmup_records + joint_records
    d_clip_fraction = float(
        np.mean(
            [record["discriminator"]["gradient_clipping_applied"] for record in d_records]
        )
    ) if d_records else 0.0
    g_clip_fraction = float(
        np.mean(
            [record["generator"]["gradient_clipping_applied"] for record in joint_records]
        )
    ) if joint_records else 0.0
    if d_clip_fraction > 0.5:
        warnings.append("discriminator_gradient_clipping_above_50_percent")
    if g_clip_fraction > 0.5:
        warnings.append("generator_gradient_clipping_above_50_percent")
    low_hinge_run = 0
    for record in d_records:
        if record["discriminator"]["losses"]["total_hinge"] < 0.05:
            low_hinge_run += 1
            if low_hinge_run >= 20:
                warnings.append("discriminator_hinge_below_0_05_for_20_steps")
                break
        else:
            low_hinge_run = 0
    if len(utilization_contacts) > 1:
        counts = list(utilization_contacts.values())
        if max(counts) > 4 * min(counts):
            warnings.append("strongly_unequal_contact_side_sampling")
    if len(joint_records) >= 20 and float(
        np.mean(
            [
                record["generator"]["mean_absolute_residual_inside_support"]
                for record in joint_records[-20:]
            ]
        )
    ) < 1e-4:
        warnings.append("generator_change_approaching_zero")
    if len(monitor_records) >= 2:
        first_monitor = monitor_records[0]
        last_monitor = monitor_records[-1]
        first_train_margin = first_monitor.get("train_margin")
        last_train_margin_value = last_monitor.get("train_margin")
        monitor_growth = (
            last_monitor["logits"]["real_minus_fake_logit_margin"]
            - first_monitor["logits"]["real_minus_fake_logit_margin"]
        )
        if first_train_margin is not None and last_train_margin_value is not None:
            train_growth = last_train_margin_value - first_train_margin
            if train_growth > 0.5 and monitor_growth < train_growth / 2:
                warnings.append("monitor_margin_growth_much_slower_than_training")
    final_monitor = monitor_records[-1] if monitor_records else None
    best_monitor = None if sustained else (
        max(
            monitor_records,
            key=lambda item: item["semantic"]["refined"]["dice_at_0_5"],
        )
        if monitor_records
        else None
    )
    rolling_statistics = _rolling_training_statistics(
        records, window_size=rolling_window_size
    )
    stratified_records = [
        record for record in records if record["kind"] == "stratified_monitor"
    ]
    replay_records = [
        record for record in records if record["kind"] == "replay_verification"
    ]
    runtime = perf_counter() - started
    summary = {
        "status": "PASS" if early_stop_reason is None and progress.joint_generator_steps == config.full_smoke_joint_steps else "STOPPED",
        "smoke_version": config.smoke_version,
        "training_version": raw_config.get("training_version"),
        "configuration_provisional": True,
        "selected_optimizer_and_clipping": {
            "generator_learning_rate": config.generator_learning_rate,
            "discriminator_learning_rate": config.discriminator_learning_rate,
            "generator_gradient_clip_max_norm": config.generator_gradient_clip_max_norm,
            "discriminator_gradient_clip_max_norm": config.discriminator_gradient_clip_max_norm,
        },
        "warmup_steps": progress.warmup_steps,
        "warmup_gate_status": progress.warmup_gate_status,
        "micro_smoke_gate_passed": progress.stage_one_passed,
        "joint_discriminator_steps": progress.joint_discriminator_steps,
        "joint_generator_steps": progress.joint_generator_steps,
        "trainer_discriminator_optimizer_steps": trainer.discriminator_optimizer_steps,
        "trainer_generator_optimizer_steps": trainer.generator_optimizer_steps,
        "monitor_only_evaluations": progress.monitor_evaluations,
        "early_stop": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "final_monitor": final_monitor,
        "best_monitor": best_monitor,
        "checkpoint_selection_policy": (
            "numbered recovery checkpoints only; no best checkpoint selection"
            if sustained
            else "no best GAN checkpoint"
        ),
        "gradient_clipping_fractions": {
            "discriminator": d_clip_fraction,
            "generator": g_clip_fraction,
        },
        "r1": {
            "gamma": config.r1_gamma,
            "interval": config.r1_interval,
            "schedule_convention": "(global_step + 1) % r1_interval == 0",
            "events": sum(
                record["discriminator"]["r1_scheduled"] for record in d_records
            ),
            "scheduled_steps": [
                index + 1
                for index, record in enumerate(d_records)
                if record["discriminator"]["r1_scheduled"]
            ],
            "raw_penalties": [
                record["discriminator"]["losses"]["raw_r1"]
                for record in d_records
                if record["discriminator"]["r1_scheduled"]
            ],
            "scaled_contributions": [
                record["discriminator"]["losses"]["scaled_r1"]
                for record in d_records
                if record["discriminator"]["r1_scheduled"]
            ],
        },
        "locality_violations": sum(
            record["generator"]["exact_outside_support_change"] != 0
            for record in joint_records
        ),
        "invalid_gradient_violations": sum(
            record["generator"]["maximum_invalid_fake_pixel_gradient"] != 0
            for record in joint_records
        ),
        "nonfinite_canonical_gradient_violations": sum(
            record["generator"][
                "canonical_defect_gradient_nonfinite_channel_count"
            ]
            != 0
            for record in joint_records
        ),
        "output_range_violations": sum(
            record["generator"]["output_range_violation_count"]
            for record in joint_records
        ),
        "rolling_training_statistics": rolling_statistics,
        "overall_training_diagnostics": {
            "mean_real_minus_fake_margin": (
                float(
                    np.mean(
                        [
                            record["discriminator"][
                                "real_minus_fake_logit_margin"
                            ]
                            for record in joint_records
                        ]
                    )
                )
                if joint_records
                else None
            ),
            "mean_boundary_residual_mass_fraction": (
                float(
                    np.mean(
                        [
                            record["generator"][
                                "boundary_residual_mass_fraction"
                            ]
                            for record in joint_records
                        ]
                    )
                )
                if joint_records
                else None
            ),
            "mean_boundary_residual_enrichment": (
                float(
                    np.mean(
                        [
                            record["generator"]["boundary_residual_enrichment"]
                            for record in joint_records
                        ]
                    )
                )
                if joint_records
                else None
            ),
            "maximum_directional_cap_saturation_fraction": max(
                (
                    record["generator"][
                        "directional_cap_saturation_fraction"
                    ]
                    for record in joint_records
                ),
                default=0.0,
            ),
            "maximum_tanh_raw_residual_saturation_fraction": max(
                (
                    record["generator"][
                        "tanh_raw_residual_saturation_fraction"
                    ]
                    for record in joint_records
                ),
                default=0.0,
            ),
        },
        "stratified_monitor_audits": stratified_records,
        "selected_smoke_step_200_replay_verification": (
            replay_records[-1] if replay_records else None
        ),
        "template_utilization": dict(sorted(utilization_templates.items())),
        "background_utilization": dict(sorted(utilization_backgrounds.items())),
        "contact_side_counts": dict(sorted(utilization_contacts.items())),
        "fixed_monitor_support_containment": support_containment,
        "fixed_monitor_sample_ids": list(monitor_ids),
        "stratified_monitor_sample_ids": list(stratified_ids),
        "monitor_source_usage": {
            category: {
                "template_id": sample.metadata["template_id"],
                "template_source_sample_id": sample.metadata[
                    "template_source_sample_id"
                ],
                "background_sample_id": sample.metadata[
                    "normal_background_sample_id"
                ],
            }
            for category, sample in panel.items()
        },
        "train_monitor_source_disjoint": not bool(
            internal.train_defect_source_ids & internal.monitor_defect_source_ids
            or internal.train_background_ids & internal.monitor_background_ids
        ),
        "warnings": sorted(set(warnings)),
        "visual_artifacts": visual_paths,
        "requested_visual_steps": list(config.visual_steps),
        "materialized_monitor_sheets": len(visual_paths),
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_training_images": 0,
        "runtime_seconds": runtime,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint_directory": config.checkpoint_directory,
        "best_checkpoint_created": False,
        "checkpoint_resume_contract_verified_by_cpu_test": True,
    }
    _atomic_write(report_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    timing = {
        "warmup_mean_step_seconds": float(np.mean([record["total_step_time_seconds"] for record in warmup_records])) if warmup_records else None,
        "joint_mean_step_seconds": float(np.mean([record["total_step_time_seconds"] for record in joint_records])) if joint_records else None,
        "joint_mean_pairs_per_second": float(np.mean([record["pairs_per_second"] for record in joint_records])) if joint_records else None,
        "templates_used": len(utilization_templates),
        "backgrounds_used": len(utilization_backgrounds),
        "contact_side_counts": dict(sorted(utilization_contacts.items())),
        "data_epoch": progress.data_epoch,
        "batch_position": progress.batch_position,
    }
    _atomic_write(report_dir / "timing_utilization.json", json.dumps(timing, indent=2) + "\n")
    terminal_joint_lines = ["- No joint update completed"]
    if joint_records:
        terminal = joint_records[-1]
        terminal_joint_lines = [
            f"- Joint step: {terminal['joint_step']}",
            f"- Discriminator total hinge: {terminal['discriminator']['losses']['total_hinge']}",
            f"- Generator losses: `{json.dumps(terminal['generator']['losses'], sort_keys=True)}`",
            f"- D/G pre-clip gradient norms: "
            f"{terminal['discriminator']['gradient_clipping']['pre_clipping_norm']} / "
            f"{terminal['generator']['gradient_clipping']['pre_clipping_norm']}",
            f"- Output-range violations: "
            f"{terminal['generator']['output_range_violation_count']}",
            f"- Would-have-clamped fraction (deprecated additive rule): "
            f"{terminal['generator']['would_have_clamped_fraction_old_additive']}",
            f"- Directional-cap / tanh saturation: "
            f"{terminal['generator']['directional_cap_saturation_fraction']} / "
            f"{terminal['generator']['tanh_raw_residual_saturation_fraction']}",
            f"- Mean/max support change: "
            f"{terminal['generator']['mean_absolute_residual_inside_support']} / "
            f"{terminal['generator']['maximum_absolute_residual']}",
            f"- Boundary residual mass/enrichment: "
            f"{terminal['generator']['boundary_residual_mass_fraction']} / "
            f"{terminal['generator']['boundary_residual_enrichment']}",
            f"- Canonical gradient pixel coverage: "
            f"{terminal['generator']['canonical_defect_gradient_active_pixel_count']} / "
            f"{terminal['generator']['canonical_defect_gradient_total_pixel_count']} "
            f"({terminal['generator']['canonical_defect_gradient_coverage']})",
            f"- Canonical gradient active/total/non-finite RGB components: "
            f"{terminal['generator']['canonical_defect_gradient_active_channel_count']} / "
            f"{terminal['generator']['canonical_defect_gradient_total_channel_count']} / "
            f"{terminal['generator']['canonical_defect_gradient_nonfinite_channel_count']}",
            f"- Invalid adversarial gradient / outside-support change: "
            f"{terminal['generator']['maximum_invalid_fake_pixel_gradient']} / "
            f"{terminal['generator']['exact_outside_support_change']}",
        ]
    detector_lines = ["- No monitor evaluation completed"]
    if final_monitor is not None:
        semantic = final_monitor["semantic"]
        detector_lines = [
            f"- Composite/refined/genuine-real inside probability: "
            f"{semantic['composite']['mean_probability_inside_mask']} / "
            f"{semantic['refined']['mean_probability_inside_mask']} / "
            f"{semantic['genuine_real']['mean_probability_inside_mask']}",
            f"- Composite/refined/genuine-real Dice: "
            f"{semantic['composite']['dice_at_0_5']} / "
            f"{semantic['refined']['dice_at_0_5']} / "
            f"{semantic['genuine_real']['dice_at_0_5']}",
            f"- Retention gate: `{json.dumps(final_monitor['retention_gate'], sort_keys=True)}`",
        ]
    markdown = "\n".join(
        [
            (
                "# G2.1 sustained 2,000-update GAN training"
                if sustained
                else "# G1.5 gated 200-step GAN smoke"
            ),
            "",
            f"- Status: **{summary['status']}**",
            f"- Warmup steps/status: {summary['warmup_steps']} / {summary['warmup_gate_status']}",
            f"- 20-step gate passed: {summary['micro_smoke_gate_passed']}",
            f"- Joint D/G steps: {summary['joint_discriminator_steps']} / {summary['joint_generator_steps']}",
            f"- Early stop: {summary['early_stop']} ({summary['early_stop_reason']})",
            f"- D/G clipping fractions: {d_clip_fraction:.4f} / {g_clip_fraction:.4f}",
            f"- R1 events: {summary['r1']['events']} at {summary['r1']['scheduled_steps']}",
            f"- Locality/invalid-gradient violations: {summary['locality_violations']} / {summary['invalid_gradient_violations']}",
            f"- Monitor sheets: {summary['materialized_monitor_sheets']}",
            f"- Runtime seconds: {runtime:.3f}",
            f"- Peak allocated/reserved VRAM: {summary['peak_allocated_vram_bytes']} / {summary['peak_reserved_vram_bytes']}",
            f"- Validation rows: {summary['validation_rows_loaded']}",
            f"- Official-test rows: {summary['official_test_rows_loaded']}",
            f"- Materialized training images: {summary['materialized_training_images']}",
            f"- Stratified monitor audits/pairs: "
            f"{len(summary['stratified_monitor_audits'])} / "
            f"{stratified_count if stratified_count else 0}",
            f"- Step-200 selected-smoke replay match: "
            f"{(summary['selected_smoke_step_200_replay_verification'] or {}).get('all_match')}",
            f"- Best checkpoint created: {summary['best_checkpoint_created']}",
            "",
            "## Terminal joint update",
            "",
            *terminal_joint_lines,
            "",
            "## Frozen-detector retention",
            "",
            *detector_lines,
            "",
            "## Warnings",
            "",
            *(f"- `{warning}`" for warning in summary["warnings"]),
            "",
            (
                "Numbered checkpoints are recovery milestones; no visually or "
                "detector-confidence-selected best checkpoint was created."
                if sustained
                else "All optimization settings remain provisional; this smoke is not final training."
            ),
        ]
    )
    _atomic_write(report_dir / "summary.md", markdown + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_smoke.json",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.config, resume=args.resume)
    print(
        f"G1.5 status={summary['status']} warmup={summary['warmup_steps']} "
        f"joint={summary['joint_generator_steps']} reason={summary['early_stop_reason']}",
        flush=True,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
