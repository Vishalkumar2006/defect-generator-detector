"""Frozen E1 configuration validation and training-control primitives."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


def _get(configuration: dict[str, Any], path: str) -> Any:
    value: Any = configuration
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Final baseline configuration is missing {path}")
        value = value[key]
    return value


FROZEN_VALUES: dict[str, Any] = {
    "seed": 42,
    "model.architecture": "groupnorm_unet",
    "model.input_channels": 3,
    "model.output_channels": 1,
    "model.base_channels": 32,
    "model.downsampling": "learned_stride_2_convolution",
    "model.decoder": "bilinear_resize_then_convolution",
    "model.output": "logits_without_sigmoid",
    "data.training_split": "train",
    "data.validation_split": "validation",
    "data.manifest": "data/metadata/ksdd2_split_seed42.csv",
    "data.official_test_evaluation": False,
    "data.canvas_width": 256,
    "data.canvas_height": 672,
    "data.resize_or_distort": False,
    "data.image_padding_mode": "reflect_with_edge_fallback_for_degenerate_axes",
    "data.mask_padding_mode": "zero",
    "data.loss_and_metrics_region": "valid_original_pixels_only",
    "augmentation.training_only": True,
    "augmentation.horizontal_flip_probability": 0.5,
    "augmentation.vertical_flip_probability": 0.5,
    "augmentation.synchronized_fields": ["image", "mask", "valid_region"],
    "augmentation.mask_interpolation": "none",
    "augmentation.additional_transforms": [],
    "loss.bce_weight": 1.0,
    "loss.dice_weight": 1.0,
    "loss.pos_weight": 5.0,
    "loss.compute_dtype": "float32",
    "optimizer.type": "AdamW",
    "optimizer.learning_rate": 0.001,
    "optimizer.weight_decay": 0.0001,
    "sampler.type": "deterministic_weighted",
    "sampler.target_defective_fraction": 0.5,
    "sampler.replacement": True,
    "precision.mode": "fp16",
    "precision.grad_scaler": True,
    "precision.automatic_fp32_retry": False,
    "precision.gradient_clip_max_norm": None,
    "training.batch_size": 4,
    "training.maximum_epochs": 12,
    "training.num_workers": 0,
    "training.pin_memory": True,
    "training.maximum_optimizer_updates_per_attempt": 1,
    "scheduler.type": "ReduceLROnPlateau",
    "scheduler.monitor": "validation_total_loss",
    "scheduler.mode": "min",
    "scheduler.factor": 0.5,
    "scheduler.patience": 2,
    "scheduler.minimum_learning_rate": 0.00001,
    "early_stopping.monitor": "validation_total_loss",
    "early_stopping.mode": "min",
    "early_stopping.patience": 4,
    "early_stopping.minimum_delta": 0.0,
    "checkpoint_selection.monitor": "validation_total_loss",
    "checkpoint_selection.mode": "min",
    "threshold_sweep.run": "once_after_training_after_best_checkpoint_reload",
    "threshold_sweep.data_source": "validation_only",
    "threshold_sweep.minimum": 0.05,
    "threshold_sweep.maximum": 0.95,
    "threshold_sweep.increment": 0.05,
    "threshold_sweep.training_metric_threshold": 0.5,
}

FROZEN_CONFIGURATION_SHA256 = "1915c823cda5432537d29c6af1c4405260c2f57cd98414b4fa491ed7ab97ee96"


def validate_final_baseline_configuration(configuration: dict[str, Any]) -> None:
    mismatches = [
        f"{path}: expected {expected!r}, got {_get(configuration, path)!r}"
        for path, expected in FROZEN_VALUES.items()
        if _get(configuration, path) != expected
    ]
    means = _get(configuration, "data.detector_normalization.mean")
    deviations = _get(configuration, "data.detector_normalization.standard_deviation")
    if len(means) != 3 or len(deviations) != 3 or any(float(value) <= 0 for value in deviations):
        mismatches.append("data.detector_normalization must contain three means and positive deviations")
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != FROZEN_CONFIGURATION_SHA256:
        mismatches.append("canonical configuration fingerprint differs from frozen E1")
    if mismatches:
        raise ValueError("Frozen final real baseline was modified: " + "; ".join(mismatches))


def load_final_baseline_configuration(path: Path) -> dict[str, Any]:
    configuration = json.loads(path.read_text(encoding="utf-8"))
    validate_final_baseline_configuration(configuration)
    return configuration


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer, configuration: dict[str, Any]
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    settings = configuration["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=settings["mode"],
        factor=float(settings["factor"]),
        patience=int(settings["patience"]),
        min_lr=float(settings["minimum_learning_rate"]),
    )


@dataclass
class EarlyStopping:
    patience: int
    minimum_delta: float = 0.0
    mode: str = "min"
    best: float = math.inf
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.mode != "min":
            raise ValueError("E1 early stopping supports mode='min' only")
        if self.patience <= 0 or self.minimum_delta < 0:
            raise ValueError("patience must be positive and minimum_delta non-negative")

    def step(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError("Early-stopping value must be finite")
        if value < self.best - self.minimum_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["patience"]) != self.patience or float(state["minimum_delta"]) != self.minimum_delta:
            raise ValueError("Early-stopping configuration does not match checkpoint")
        if state["mode"] != self.mode:
            raise ValueError("Early-stopping mode does not match checkpoint")
        self.best = float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])
        self.stopped = bool(state["stopped"])


@dataclass
class PostTrainingValidationGate:
    """Enforce one sweep, only after training completes and best weights reload."""

    training_complete: bool = False
    best_checkpoint_loaded: bool = False
    sweep_executed: bool = False

    def mark_training_complete(self) -> None:
        self.training_complete = True

    def mark_best_checkpoint_loaded(self) -> None:
        if not self.training_complete:
            raise RuntimeError("Cannot load the final best checkpoint before training completes")
        self.best_checkpoint_loaded = True

    def claim_threshold_sweep(self) -> None:
        if not self.training_complete or not self.best_checkpoint_loaded:
            raise RuntimeError("Threshold sweep requires completed training and reloaded best checkpoint")
        if self.sweep_executed:
            raise RuntimeError("Validation threshold sweep may execute only once")
        self.sweep_executed = True


def threshold_candidates(configuration: dict[str, Any]) -> list[float]:
    settings = configuration["threshold_sweep"]
    start = int(round(float(settings["minimum"]) * 100))
    stop = int(round(float(settings["maximum"]) * 100))
    step = int(round(float(settings["increment"]) * 100))
    return [value / 100 for value in range(start, stop + 1, step)]
