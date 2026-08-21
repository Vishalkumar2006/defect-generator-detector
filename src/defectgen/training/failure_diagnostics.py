"""Atomic numerical-failure reports and checkpoint finiteness inspection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _json_number(value: Any) -> float | str | None:
    if value is None:
        return None
    number = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number


def tensor_numerical_summary(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {
            "minimum_finite": None,
            "maximum_finite": None,
            "maximum_absolute_finite": None,
            "nan_count": None,
            "positive_inf_count": None,
            "negative_inf_count": None,
        }
    values = tensor.detach().float().cpu()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    return {
        "minimum_finite": float(finite_values.min().item()) if len(finite_values) else None,
        "maximum_finite": float(finite_values.max().item()) if len(finite_values) else None,
        "maximum_absolute_finite": float(finite_values.abs().max().item()) if len(finite_values) else None,
        "nan_count": int(torch.isnan(values).sum().item()),
        "positive_inf_count": int(torch.isposinf(values).sum().item()),
        "negative_inf_count": int(torch.isneginf(values).sum().item()),
    }


def nonfinite_components(
    logits: torch.Tensor | None, components: Mapping[str, torch.Tensor | float] | None
) -> list[str]:
    failures: list[str] = []
    if logits is not None and not bool(torch.isfinite(logits).all()):
        failures.append("logits")
    for name, value in (components or {}).items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if not bool(torch.isfinite(tensor).all()):
            failures.append(name)
    return failures


def git_worktree_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "status": [], "error": str(error)}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_numerical_failure_report(
    path: Path,
    *,
    phase: str,
    epoch: int,
    batch_index: int,
    sample_ids: list[str],
    precision_mode: str,
    logits: torch.Tensor | None,
    loss_components: Mapping[str, torch.Tensor | float] | None,
    scaler_scale: float | None,
    most_recent_gradient_norm: float | None,
    checkpoint_paths: Mapping[str, Path | str],
    repo_root: Path,
    error: str,
    explicit_nonfinite_component: str | None = None,
) -> dict[str, Any]:
    detected = nonfinite_components(logits, loss_components)
    if explicit_nonfinite_component is not None and explicit_nonfinite_component not in detected:
        detected.append(explicit_nonfinite_component)
    report = {
        "phase": phase,
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
        "precision_mode": precision_mode,
        "nonfinite_components": detected,
        "logits": tensor_numerical_summary(logits),
        "loss_components": {
            name: _json_number(value) for name, value in (loss_components or {}).items()
        },
        "grad_scaler_scale": _json_number(scaler_scale),
        "most_recent_gradient_norm": _json_number(most_recent_gradient_norm),
        "checkpoint_paths": {name: str(value) for name, value in checkpoint_paths.items()},
        "git": git_worktree_state(repo_root),
        "error": error,
    }
    atomic_write_json(path, report)
    return report


def inspect_tensor_tree_finiteness(value: Any, prefix: str = "") -> dict[str, Any]:
    summary = {
        "tensor_count": 0,
        "element_count": 0,
        "floating_element_count": 0,
        "nan_count": 0,
        "positive_inf_count": 0,
        "negative_inf_count": 0,
        "nonfinite_paths": [],
    }

    def visit(current: Any, path: str) -> None:
        if isinstance(current, torch.Tensor):
            summary["tensor_count"] += 1
            summary["element_count"] += current.numel()
            if current.is_floating_point() or current.is_complex():
                values = current.detach()
                summary["floating_element_count"] += values.numel()
                nan_count = int(torch.isnan(values).sum().item())
                positive_inf = int(torch.isposinf(values).sum().item())
                negative_inf = int(torch.isneginf(values).sum().item())
                summary["nan_count"] += nan_count
                summary["positive_inf_count"] += positive_inf
                summary["negative_inf_count"] += negative_inf
                if nan_count or positive_inf or negative_inf:
                    summary["nonfinite_paths"].append(path)
            return
        items = current.items() if isinstance(current, dict) else enumerate(current) if isinstance(current, (list, tuple)) else ()
        for key, child in items:
            visit(child, f"{path}.{key}" if path else str(key))

    visit(value, prefix)
    summary["finite"] = not summary["nonfinite_paths"]
    return summary


def inspect_checkpoint_finiteness(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" not in payload:
        raise ValueError(f"Checkpoint lacks model_state: {path}")
    inspection = {
        "path": str(path),
        "epoch": payload.get("epoch"),
        "model": inspect_tensor_tree_finiteness(payload["model_state"], "model_state"),
        "optimizer": inspect_tensor_tree_finiteness(payload.get("optimizer_state", {}), "optimizer_state"),
    }
    inspection["parameters_usable"] = bool(inspection["model"]["finite"])
    return payload, inspection


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
