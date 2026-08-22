"""Run the single selected G2.1 2,000-update sustained GAN training schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.training.gan_smoke import load_gan_smoke_config  # noqa: E402
from scripts.train_gan_smoke import run as run_gan_schedule  # noqa: E402


TRAINING_VERSION = "g2_1_sustained_gan_training_v1"
SCHEDULE_DIFFERENCES = {
    "report_directory",
    "checkpoint_directory",
    "full_smoke_joint_steps",
    "monitor_interval",
    "checkpoint_interval",
    "visual_steps",
}
G2_ONLY_FIELDS = {
    "training_version",
    "run_kind",
    "visual_steps_only",
    "visualize_after_warmup",
    "stratified_monitor_pair_count",
    "stratified_monitor_steps",
    "rolling_window_size",
    "selected_smoke_config_path",
    "selected_smoke_step_200_checkpoint",
    "verify_selected_smoke_at_step",
}


def _read_configuration(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("G2.1 configuration must be a JSON object")
    return values


def validate_g2_1_configuration(
    path: Path,
    *,
    repository_root: Path = REPO_ROOT,
    require_reference_checkpoint: bool = True,
) -> dict[str, Any]:
    values = _read_configuration(path)
    if values.get("training_version") != TRAINING_VERSION:
        raise ValueError(f"training_version must be {TRAINING_VERSION!r}")
    if values.get("run_kind") != "g2_1_sustained":
        raise ValueError("G2.1 run_kind must be g2_1_sustained")
    selected_path = repository_root / values["selected_smoke_config_path"]
    selected = _read_configuration(selected_path)
    common_keys = set(selected) & set(values)
    unexpected = {
        key
        for key in common_keys
        if selected[key] != values[key] and key not in SCHEDULE_DIFFERENCES
    }
    if unexpected:
        raise ValueError(
            "G2.1 changed selected optimizer/data/safety settings: "
            + ", ".join(sorted(unexpected))
        )
    if set(values) - set(selected) != G2_ONLY_FIELDS:
        raise ValueError("G2.1 contains unexpected or missing sustained-run fields")
    expected = {
        "generator_learning_rate": 0.0001,
        "discriminator_learning_rate": 0.000025,
        "generator_gradient_clip_max_norm": 5.0,
        "discriminator_gradient_clip_max_norm": 10.0,
        "full_smoke_joint_steps": 2000,
        "monitor_interval": 100,
        "checkpoint_interval": 100,
        "visual_steps": [0, 100, 250, 500, 1000, 1500, 2000],
        "stratified_monitor_pair_count": 128,
        "stratified_monitor_steps": [0, 500, 1000, 1500, 2000],
        "verify_selected_smoke_at_step": 200,
    }
    mismatched = [key for key, expected_value in expected.items() if values.get(key) != expected_value]
    if mismatched:
        raise ValueError("G2.1 required schedule mismatch: " + ", ".join(mismatched))
    if not values.get("visual_steps_only") or values.get("visualize_after_warmup"):
        raise ValueError("G2.1 must materialize only the seven requested visual sheets")
    selected_checkpoint = repository_root / values[
        "selected_smoke_step_200_checkpoint"
    ]
    if require_reference_checkpoint and not selected_checkpoint.is_file():
        raise FileNotFoundError(
            f"Selected smoke step-200 checkpoint not found: {selected_checkpoint}"
        )
    if values["report_directory"] in {
        selected["report_directory"],
        "reports/gan_training/smoke",
    } or values["checkpoint_directory"] in {
        selected["checkpoint_directory"],
        "checkpoints/gan_smoke",
    }:
        raise ValueError("G2.1 must use report/checkpoint directories separate from both smoke runs")
    load_gan_smoke_config(path)
    return values


def run(config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    validate_g2_1_configuration(config_path)
    return run_gan_schedule(config_path, resume=resume)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_training_2000.json",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.config, resume=args.resume)
    print(
        f"G2.1 status={summary['status']} warmup={summary['warmup_steps']} "
        f"joint={summary['joint_generator_steps']} reason={summary['early_stop_reason']}",
        flush=True,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
