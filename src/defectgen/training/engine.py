"""Reusable training-state, checkpoint, and metric-log primitives."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_random_states(
    sampler_generator: torch.Generator | None = None,
    loader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler_generator": sampler_generator.get_state() if sampler_generator is not None else None,
        "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
    }


def restore_random_states(
    states: dict[str, Any],
    sampler_generator: torch.Generator | None = None,
    loader_generator: torch.Generator | None = None,
) -> None:
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    # Checkpoints may be loaded with map_location="cuda" for model state, but
    # PyTorch generator state APIs always require CPU ByteTensors.
    torch.set_rng_state(states["torch_cpu"].cpu())
    if torch.cuda.is_available() and states["torch_cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in states["torch_cuda"]])
    if sampler_generator is not None and states.get("sampler_generator") is not None:
        sampler_generator.set_state(states["sampler_generator"].cpu())
    if loader_generator is not None and states.get("loader_generator") is not None:
        loader_generator.set_state(states["loader_generator"].cpu())


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    configuration: dict[str, Any],
    best_validation: dict[str, Any],
    sampler_generator: torch.Generator | None = None,
    loader_generator: torch.Generator | None = None,
    metric_records: list[dict[str, Any]] | None = None,
    numerical_controller: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "numerical_state": numerical_controller.state_dict() if numerical_controller is not None else None,
        "epoch": epoch,
        "configuration": configuration,
        "random_states": capture_random_states(sampler_generator, loader_generator),
        "best_validation": best_validation,
        "metric_records": metric_records or [],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    expected_configuration: dict[str, Any],
    sampler_generator: torch.Generator | None = None,
    loader_generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
    numerical_controller: Any | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload["configuration"] != expected_configuration:
        raise ValueError("Checkpoint configuration does not match the requested experiment")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload["scheduler_state"] is None:
            raise ValueError("Checkpoint lacks scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    elif payload["scheduler_state"] is not None:
        raise ValueError("Checkpoint has scheduler state but current configuration uses none")
    scaler_state = payload.get("scaler_state")
    if scaler is not None:
        if scaler_state is None:
            raise ValueError("Checkpoint lacks GradScaler state")
        scaler.load_state_dict(scaler_state)
    elif scaler_state not in (None, {}):
        raise ValueError("Checkpoint has GradScaler state but current precision mode does not use it")
    numerical_state = payload.get("numerical_state")
    if numerical_controller is not None and numerical_state is not None:
        numerical_controller.load_state_dict(numerical_state)
    restore_random_states(payload["random_states"], sampler_generator, loader_generator)
    return payload


def write_metric_logs(records: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    if not records:
        raise ValueError("Cannot write an empty metric log")
    fields = list(records[0])
    if any(list(record) != fields for record in records):
        raise ValueError("Metric records must have identical ordered fields")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def configurations_differ_only_by_pos_weight(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_copy = json.loads(json.dumps(first))
    second_copy = json.loads(json.dumps(second))
    first_weight = first_copy["loss"].pop("pos_weight")
    second_weight = second_copy["loss"].pop("pos_weight")
    return first_weight != second_weight and first_copy == second_copy
