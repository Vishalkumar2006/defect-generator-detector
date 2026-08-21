"""Frozen E1 configuration validation and training-control primitives."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


HISTORICAL_FP16_IDENTITY = "historical_final_real_baseline_fp16_seed42"
STABILIZED_BF16_IDENTITY = "final_real_baseline_bf16_seed42"
FROZEN_CONFIGURATION_SHA256 = {
    HISTORICAL_FP16_IDENTITY: "1915c823cda5432537d29c6af1c4405260c2f57cd98414b4fa491ed7ab97ee96",
    STABILIZED_BF16_IDENTITY: "53cfed97ef97abba4f9e083b3023571ff520ef71582638c253f9710d7516c76e",
}


def configuration_identity(configuration: dict[str, Any]) -> str:
    return str(configuration.get("experiment_identity", HISTORICAL_FP16_IDENTITY))


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_final_baseline_configuration(configuration: dict[str, Any]) -> None:
    identity = configuration_identity(configuration)
    expected = FROZEN_CONFIGURATION_SHA256.get(identity)
    if expected is None:
        raise ValueError(f"Unknown final baseline experiment identity: {identity}")
    actual = configuration_fingerprint(configuration)
    if actual != expected:
        raise ValueError(
            f"Frozen final real baseline was modified: {identity} fingerprint {actual} != {expected}"
        )


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
