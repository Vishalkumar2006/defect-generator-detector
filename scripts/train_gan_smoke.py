"""Run the gated G1.5 discriminator warmup and at most 200 joint GAN steps."""

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
    load_gan_smoke_config,
    load_smoke_checkpoint,
    module_parameters_are_finite,
    save_smoke_checkpoint,
    select_fixed_monitor_samples,
    stage_one_allows_continuation,
    warmup_gate_decision,
)
from defectgen.training.gan_trainer import (  # noqa: E402
    GANOneStepTrainer,
    GANTrainingBatch,
    GANTrainingNumericalError,
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
    if generator["canonical_defect_gradient_coverage"] != 1:
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


def _plot_series(records: list[dict[str, Any]], report_dir: Path) -> None:
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
                "clip limit": [5.0 for _ in joint],
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


def run(config_path: Path, *, resume: bool) -> dict[str, Any]:
    started = perf_counter()
    raw_config = _configuration_dict(config_path)
    config = load_gan_smoke_config(config_path)
    configure_reproducibility(config.seed, deterministic=True, warn_only=False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("G1.5 requires explicitly supported CUDA BF16")
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
    identity = SmokeCheckpointIdentity(
        configuration_sha256=canonical_configuration_hash(raw_config),
        gan_manifest_content_sha256=metadata["gan_manifest_content_sha256"],
        split_sha256=metadata["split_sha256"],
        fixed_monitor_sample_ids=monitor_ids,
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
            title=f"G1.5 {phase}",
        )
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in visual_paths:
            visual_paths.append(relative)

    if not resume:
        record_monitor("step_0")
        write_visual("step_0", "step_000")

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
        visual_due = joint_step in config.visual_steps or joint_step % config.visual_interval == 0
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
        progress.last_completed_operation = "joint_200_complete"
        _checkpoint(
            "joint_200.pt",
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
    _plot_series(records, report_dir)
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
    best_monitor = (
        max(
            monitor_records,
            key=lambda item: item["semantic"]["refined"]["dice_at_0_5"],
        )
        if monitor_records
        else None
    )
    runtime = perf_counter() - started
    summary = {
        "status": "PASS" if early_stop_reason is None and progress.joint_generator_steps == config.full_smoke_joint_steps else "STOPPED",
        "smoke_version": config.smoke_version,
        "configuration_provisional": True,
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
        "template_utilization": dict(sorted(utilization_templates.items())),
        "background_utilization": dict(sorted(utilization_backgrounds.items())),
        "contact_side_counts": dict(sorted(utilization_contacts.items())),
        "fixed_monitor_support_containment": support_containment,
        "fixed_monitor_sample_ids": list(monitor_ids),
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
            f"- Canonical gradient coverage: "
            f"{terminal['generator']['canonical_defect_gradient_coverage']}",
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
            "# G1.5 gated 200-step GAN smoke",
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
            "All optimization settings remain provisional; this smoke is not final training.",
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
