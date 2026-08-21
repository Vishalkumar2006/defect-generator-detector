"""G1.5 bounded smoke configuration, gates, detector evaluation, and resume state."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from defectgen.gan.training_pairs import GANTrainingSample
from defectgen.models import ARCHITECTURE_VERSION, RESIDUAL_SEMANTICS_VERSION
from defectgen.training.engine import capture_random_states, restore_random_states
from defectgen.training.gan_losses import GeneratorLossWeights
from defectgen.training.gan_trainer import (
    GANOneStepTrainer,
    GANOptimizerConfig,
    GANTrainerConfig,
)


SMOKE_VERSION = "g1_5a_gated_gan_smoke_v1"
MONITOR_CATEGORIES = (
    "non-border",
    "single-horizontal-border",
    "single-vertical-border",
    "corner",
    "left+right",
    "small-thin",
    "large",
)


@dataclass(frozen=True)
class GANSmokeConfig:
    smoke_version: str
    trainer_config_path: str
    training_pair_config_path: str
    architecture_config_path: str
    loss_config_path: str
    detector_config_path: str
    detector_checkpoint_path: str
    report_directory: str
    checkpoint_directory: str
    seed: int
    device: str
    precision: str
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    batches_per_data_epoch: int
    generator_learning_rate: float
    discriminator_learning_rate: float
    adam_betas: tuple[float, float]
    weight_decay: float
    discriminator_steps_per_generator_step: int
    generator_gradient_clip_max_norm: float
    discriminator_gradient_clip_max_norm: float
    generator_loss_weights: GeneratorLossWeights
    r1_gamma: float
    r1_interval: int
    initial_discriminator_warmup_steps: int
    maximum_discriminator_warmup_steps: int
    micro_smoke_joint_steps: int
    full_smoke_joint_steps: int
    monitor_interval: int
    visual_interval: int
    checkpoint_interval: int
    progress_interval: int
    visual_steps: tuple[int, ...]
    monitor_panel_scan_limit: int
    detector_threshold: float
    detector_inside_retention_warning_ratio: float
    detector_inside_retention_stop_ratio: float
    detector_dice_warning_drop: float
    absolute_logit_stop: float
    output_range_violation_stop_count: int
    mean_support_change_stop: float
    provisional_configuration: bool

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GANSmokeConfig":
        required = tuple(field.name for field in fields(cls))
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"GAN smoke config is missing: {', '.join(missing)}")
        normalized = dict(values)
        normalized["adam_betas"] = tuple(values["adam_betas"])
        normalized["visual_steps"] = tuple(values["visual_steps"])
        try:
            normalized["generator_loss_weights"] = GeneratorLossWeights(
                **values["generator_loss_weights"]
            )
        except TypeError as error:
            raise ValueError(f"Invalid smoke generator loss weights: {error}") from error
        config = cls(**{name: normalized[name] for name in required})
        config.validate()
        return config

    def validate(self) -> None:
        if self.smoke_version != SMOKE_VERSION:
            raise ValueError(f"smoke_version must be {SMOKE_VERSION!r}")
        if self.provisional_configuration is not True:
            raise ValueError("G1.5 settings must remain explicitly provisional")
        if self.device != "cuda" or self.precision != "bf16":
            raise ValueError("G1.5 real smoke requires CUDA BF16")
        if self.seed != 42 or self.batch_size != 2:
            raise ValueError("G1.5 requires seed 42 and batch size 2")
        if self.num_workers < 0 or self.prefetch_factor <= 0:
            raise ValueError("Invalid DataLoader worker configuration")
        if self.persistent_workers:
            raise ValueError(
                "Persistent workers are incompatible with mutable deterministic set_epoch state"
            )
        if self.batches_per_data_epoch <= 0:
            raise ValueError("batches_per_data_epoch must be positive")
        allowed_discriminator_ablation = {
            (0.00005, 5.0),
            (0.000025, 10.0),
        }
        if self.generator_learning_rate != 0.0001 or (
            self.discriminator_learning_rate,
            self.discriminator_gradient_clip_max_norm,
        ) not in allowed_discriminator_ablation:
            raise ValueError("Unexpected G1.5/G1.6 discriminator ablation settings")
        if self.adam_betas != (0.0, 0.9) or self.weight_decay != 0:
            raise ValueError("Unexpected G1.5 Adam settings")
        if self.discriminator_steps_per_generator_step != 1:
            raise ValueError("G1.5 requires one discriminator step per generator step")
        if self.generator_gradient_clip_max_norm != 5:
            raise ValueError("G1.5/G1.6 requires generator maximum gradient norm 5")
        expected_weights = GeneratorLossWeights(1.0, 1.0, 1.0, 0.1)
        if self.generator_loss_weights != expected_weights:
            raise ValueError("Unexpected G1.5 provisional generator coefficients")
        if self.r1_gamma != 1 or self.r1_interval != 16:
            raise ValueError("G1.5 requires provisional R1 gamma 1 and interval 16")
        if not (
            self.initial_discriminator_warmup_steps == 10
            and self.maximum_discriminator_warmup_steps == 20
            and self.micro_smoke_joint_steps == 20
            and self.full_smoke_joint_steps == 200
        ):
            raise ValueError("Unexpected G1.5 bounded step counts")
        for name in (
            "monitor_interval",
            "visual_interval",
            "checkpoint_interval",
            "progress_interval",
            "monitor_panel_scan_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not {0, 20, 50, 100, 150, 200}.issubset(self.visual_steps):
            raise ValueError("G1.5 visual steps omit a required fixed milestone")
        if not 0 < self.detector_threshold < 1:
            raise ValueError("Detector threshold must be in (0,1)")
        if not (
            0
            < self.detector_inside_retention_stop_ratio
            < self.detector_inside_retention_warning_ratio
            <= 1
        ):
            raise ValueError("Invalid detector retention gates")
        if self.output_range_violation_stop_count != 0:
            raise ValueError("Output-range violation stop count must be zero")
        if self.absolute_logit_stop <= 0 or self.mean_support_change_stop <= 0:
            raise ValueError("Numerical stop thresholds must be positive")

    def trainer_config(self, base: GANTrainerConfig) -> GANTrainerConfig:
        return replace(
            base,
            seed=self.seed,
            batch_size=self.batch_size,
            generator_optimizer=GANOptimizerConfig(
                "Adam",
                self.generator_learning_rate,
                self.adam_betas,
                self.weight_decay,
            ),
            discriminator_optimizer=GANOptimizerConfig(
                "Adam",
                self.discriminator_learning_rate,
                self.adam_betas,
                self.weight_decay,
            ),
            generator_gradient_clip_max_norm=self.generator_gradient_clip_max_norm,
            discriminator_gradient_clip_max_norm=self.discriminator_gradient_clip_max_norm,
            cuda_precision=self.precision,
            r1_gamma=self.r1_gamma,
            r1_interval=self.r1_interval,
            one_step_generator_loss_weights=self.generator_loss_weights,
        )


def load_gan_smoke_config(path: Path | str) -> GANSmokeConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("GAN smoke config must contain a JSON object")
    return GANSmokeConfig.from_dict(values)


def canonical_configuration_hash(values: Mapping[str, Any]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class SmokeProgress:
    warmup_steps: int = 0
    joint_discriminator_steps: int = 0
    joint_generator_steps: int = 0
    monitor_evaluations: int = 0
    data_epoch: int = 0
    batch_position: int = 0
    last_completed_operation: str = "initialized"
    warmup_gate_status: str = "pending"
    stage_one_passed: bool = False
    detector_retention_consecutive_below_stop: int = 0
    early_stop_reason: str | None = None


@dataclass(frozen=True)
class SmokeCheckpointIdentity:
    configuration_sha256: str
    gan_manifest_content_sha256: str
    split_sha256: str
    fixed_monitor_sample_ids: tuple[str, ...]


def save_smoke_checkpoint(
    path: Path,
    *,
    trainer: GANOneStepTrainer,
    progress: SmokeProgress,
    identity: SmokeCheckpointIdentity,
    configuration: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "smoke_version": SMOKE_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "residual_semantics_version": RESIDUAL_SEMANTICS_VERSION,
        "generator_state": trainer.generator.state_dict(),
        "discriminator_state": trainer.discriminator.state_dict(),
        "generator_optimizer_state": trainer.generator_optimizer.state_dict(),
        "discriminator_optimizer_state": trainer.discriminator_optimizer.state_dict(),
        "generator_scaler_state": (
            trainer.generator_scaler.state_dict()
            if trainer.generator_scaler is not None
            else None
        ),
        "discriminator_scaler_state": (
            trainer.discriminator_scaler.state_dict()
            if trainer.discriminator_scaler is not None
            else None
        ),
        "precision_mode": trainer.precision,
        "trainer_optimizer_steps": {
            "generator": trainer.generator_optimizer_steps,
            "discriminator": trainer.discriminator_optimizer_steps,
        },
        "progress": asdict(progress),
        "configuration": dict(configuration),
        "identity": asdict(identity),
        "random_states": capture_random_states(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_smoke_checkpoint(
    path: Path,
    *,
    trainer: GANOneStepTrainer,
    expected_identity: SmokeCheckpointIdentity,
    expected_configuration: Mapping[str, Any],
) -> SmokeProgress:
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload.get("residual_semantics_version") != RESIDUAL_SEMANTICS_VERSION:
        raise ValueError("Smoke checkpoint residual-semantics version is incompatible")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Smoke checkpoint generator architecture version is incompatible")
    if payload.get("smoke_version") != SMOKE_VERSION:
        raise ValueError("Smoke checkpoint version is incompatible")
    if payload.get("configuration") != dict(expected_configuration):
        raise ValueError("Smoke checkpoint configuration is incompatible")
    actual_identity = SmokeCheckpointIdentity(**payload["identity"])
    if actual_identity != expected_identity:
        raise ValueError("Smoke checkpoint manifest, split, or monitor identity is incompatible")
    if payload.get("precision_mode") != trainer.precision:
        raise ValueError("Smoke checkpoint precision mode is incompatible")
    try:
        trainer.generator.load_state_dict(payload["generator_state"])
        trainer.discriminator.load_state_dict(payload["discriminator_state"])
    except RuntimeError as error:
        raise ValueError("Smoke checkpoint model state dictionary is incompatible") from error
    trainer.generator_optimizer.load_state_dict(payload["generator_optimizer_state"])
    trainer.discriminator_optimizer.load_state_dict(
        payload["discriminator_optimizer_state"]
    )
    for scaler, key in (
        (trainer.generator_scaler, "generator_scaler_state"),
        (trainer.discriminator_scaler, "discriminator_scaler_state"),
    ):
        saved = payload.get(key)
        if scaler is None and saved not in (None, {}):
            raise ValueError("Smoke checkpoint unexpectedly contains GradScaler state")
        if scaler is not None:
            if saved is None:
                raise ValueError("Smoke checkpoint lacks required GradScaler state")
            scaler.load_state_dict(saved)
    steps = payload["trainer_optimizer_steps"]
    trainer.generator_optimizer_steps = int(steps["generator"])
    trainer.discriminator_optimizer_steps = int(steps["discriminator"])
    restore_random_states(payload["random_states"])
    return SmokeProgress(**payload["progress"])


def warmup_gate_decision(
    *,
    completed_steps: int,
    monitor_margin: float,
    initial_steps: int = 10,
    maximum_steps: int = 20,
) -> str:
    if completed_steps < initial_steps:
        return "continue"
    if monitor_margin > 0:
        return "accepted"
    if completed_steps < maximum_steps:
        return "continue"
    return "failed"


def stage_one_allows_continuation(
    *, completed_joint_steps: int, target: int, early_stop_reason: str | None
) -> bool:
    return completed_joint_steps >= target and early_stop_reason is None


@dataclass
class DetectorRetentionGate:
    warning_ratio: float = 0.7
    stop_ratio: float = 0.5
    consecutive_below_stop: int = 0

    def update(self, composite_inside: float, refined_inside: float) -> dict[str, Any]:
        ratio = refined_inside / composite_inside if composite_inside > 0 else 1.0
        warning = ratio < self.warning_ratio
        if ratio < self.stop_ratio:
            self.consecutive_below_stop += 1
        else:
            self.consecutive_below_stop = 0
        return {
            "inside_retention_ratio": ratio,
            "warning": warning,
            "consecutive_below_stop": self.consecutive_below_stop,
            "stop": self.consecutive_below_stop >= 2,
        }


class FrozenDetectorEvaluator:
    """Detached semantic evaluator; its output can never train the GAN."""

    def __init__(
        self,
        detector: nn.Module,
        *,
        mean: Sequence[float],
        standard_deviation: Sequence[float],
        threshold: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.detector = detector.to(self.device).eval()
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)[
            None, :, None, None
        ]
        self.standard_deviation = torch.tensor(
            standard_deviation, dtype=torch.float32, device=self.device
        )[None, :, None, None]
        if bool((self.standard_deviation <= 0).any()):
            raise ValueError("Detector standard deviations must be positive")
        self.threshold = float(threshold)

    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        detached = images.detach().to(self.device).float()
        normalized = (detached.add(1).div(2) - self.mean) / self.standard_deviation
        with torch.no_grad():
            logits = self.detector(normalized)
            probabilities = torch.sigmoid(logits.float())
        return probabilities.detach()

    def metrics(
        self,
        images: torch.Tensor,
        canonical_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor]:
        probabilities = self.probabilities(images)
        target = canonical_mask.detach().to(self.device).bool()
        valid = valid_mask.detach().to(self.device).bool()
        inside = target & valid
        outside = ~target & valid
        prediction = probabilities >= self.threshold
        intersection = (prediction & target & valid).flatten(1).sum(dim=1).float()
        predicted = (prediction & valid).flatten(1).sum(dim=1).float()
        target_count = (target & valid).flatten(1).sum(dim=1).float()
        union = ((prediction | target) & valid).flatten(1).sum(dim=1).float()
        dice = torch.where(
            predicted + target_count > 0,
            2 * intersection / (predicted + target_count),
            torch.ones_like(intersection),
        )
        iou = torch.where(
            union > 0, intersection / union, torch.ones_like(intersection)
        )
        inside_mean = float(probabilities[inside].mean())
        outside_mean = float(probabilities[outside].mean())
        metrics = {
            "mean_probability_inside_mask": inside_mean,
            "mean_probability_outside_mask": outside_mean,
            "inside_outside_probability_contrast": inside_mean - outside_mean,
            "dice_at_0_5": float(dice.mean()),
            "iou_at_0_5": float(iou.mean()),
            "samples_with_any_predicted_positive_fraction": float(
                (prediction & valid).flatten(1).any(dim=1).float().mean()
            ),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Frozen detector evaluation produced a non-finite metric")
        return metrics, probabilities.cpu()


def module_parameters_are_finite(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())


def select_fixed_monitor_samples(
    samples: Sequence[GANTrainingSample],
) -> dict[str, GANTrainingSample]:
    selected: dict[str, GANTrainingSample] = {}
    small_score: tuple[float, int] | None = None
    large_score: tuple[int, int] | None = None
    for index, sample in enumerate(samples):
        contacts = sample.metadata["target_contact_sides"]
        horizontal = contacts["top"] or contacts["bottom"]
        vertical = contacts["left"] or contacts["right"]
        if not any(contacts.values()):
            selected.setdefault("non-border", sample)
        elif contacts["left"] and contacts["right"]:
            selected.setdefault("left+right", sample)
        elif horizontal and vertical:
            selected.setdefault("corner", sample)
        elif horizontal:
            selected.setdefault("single-horizontal-border", sample)
        elif vertical:
            selected.setdefault("single-vertical-border", sample)
        coordinates = torch.nonzero(sample.fake_discriminator_mask[0].bool())
        if not len(coordinates):
            continue
        height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
        width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
        positive = int(len(coordinates))
        thinness = min(height, width) + positive / 1_000_000
        if small_score is None or (thinness, index) < small_score:
            small_score = (thinness, index)
            selected["small-thin"] = sample
        if large_score is None or (positive, -index) > large_score:
            large_score = (positive, -index)
            selected["large"] = sample
    missing = [category for category in MONITOR_CATEGORIES if category not in selected]
    if missing:
        raise RuntimeError(
            f"Unable to select fixed monitor categories: {', '.join(missing)}"
        )
    return {category: selected[category] for category in MONITOR_CATEGORIES}


def select_stratified_monitor_samples(
    samples: Iterable[GANTrainingSample], *, per_category: int = 4
) -> dict[str, tuple[GANTrainingSample, ...]]:
    """Select a deterministic, unique monitor panel across geometry strata."""
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    contact_candidates: dict[str, list[tuple[int, GANTrainingSample]]] = {
        category: [] for category in MONITOR_CATEGORIES[:5]
    }
    small_candidates: list[tuple[float, int, int, GANTrainingSample]] = []
    large_candidates: list[tuple[float, int, int, GANTrainingSample]] = []
    morphology_reserve = per_category * (len(MONITOR_CATEGORIES) + 1)
    for index, sample in enumerate(samples):
        contacts = sample.metadata["target_contact_sides"]
        horizontal = contacts["top"] or contacts["bottom"]
        vertical = contacts["left"] or contacts["right"]
        if not any(contacts.values()):
            category = "non-border"
        elif contacts["left"] and contacts["right"]:
            category = "left+right"
        elif horizontal and vertical:
            category = "corner"
        elif horizontal:
            category = "single-horizontal-border"
        else:
            category = "single-vertical-border"
        if len(contact_candidates[category]) < per_category:
            contact_candidates[category].append((index, sample))
        coordinates = torch.nonzero(sample.fake_discriminator_mask[0].bool())
        if not len(coordinates):
            continue
        height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
        width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
        positive = int(len(coordinates))
        morphology_item = (
            min(height, width) + positive / 1_000_000,
            positive,
            index,
            sample,
        )
        small_candidates.append(morphology_item)
        small_candidates.sort(key=lambda item: (item[0], item[2]))
        del small_candidates[morphology_reserve:]
        large_candidates.append(morphology_item)
        large_candidates.sort(key=lambda item: (-item[1], item[2]))
        del large_candidates[morphology_reserve:]

    selected: dict[str, tuple[GANTrainingSample, ...]] = {}
    used: set[str] = set()

    def identity(sample: GANTrainingSample) -> str:
        return (
            f"{sample.metadata['sample_index']}:"
            f"{sample.metadata['template_id']}:"
            f"{sample.metadata['normal_background_sample_id']}"
        )

    for category in MONITOR_CATEGORIES[:5]:
        chosen: list[GANTrainingSample] = []
        for _, sample in contact_candidates[category]:
            sample_id = identity(sample)
            if sample_id in used:
                continue
            chosen.append(sample)
            used.add(sample_id)
            if len(chosen) == per_category:
                break
        if len(chosen) != per_category:
            raise RuntimeError(f"Insufficient monitor samples for {category}")
        selected[category] = tuple(chosen)

    for category, candidates in (
        ("small-thin", small_candidates),
        ("large", large_candidates),
    ):
        chosen = []
        for _, _, _, sample in candidates:
            sample_id = identity(sample)
            if sample_id in used:
                continue
            chosen.append(sample)
            used.add(sample_id)
            if len(chosen) == per_category:
                break
        if len(chosen) != per_category:
            raise RuntimeError(f"Insufficient monitor samples for {category}")
        selected[category] = tuple(chosen)
    return {category: selected[category] for category in MONITOR_CATEGORIES}


class AtomicJSONLLog:
    def __init__(self, path: Path, records: Sequence[dict[str, Any]] | None = None) -> None:
        self.path = Path(path)
        self.records = list(records or [])

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in self.records),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
