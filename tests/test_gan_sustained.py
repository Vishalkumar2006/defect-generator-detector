from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from defectgen.training.gan_smoke import canonical_state_hash, load_gan_smoke_config
from scripts.train_gan import validate_g2_1_configuration
from scripts.train_gan_smoke import _rolling_training_statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "gan_training_2000.json"


def test_g2_1_configuration_exactly_replays_selected_training_settings() -> None:
    values = validate_g2_1_configuration(
        CONFIG_PATH, require_reference_checkpoint=False
    )
    config = load_gan_smoke_config(CONFIG_PATH)
    assert values["generator_learning_rate"] == 1e-4
    assert values["discriminator_learning_rate"] == 2.5e-5
    assert values["generator_gradient_clip_max_norm"] == 5
    assert values["discriminator_gradient_clip_max_norm"] == 10
    assert config.full_smoke_joint_steps == 2000
    assert config.checkpoint_interval == config.monitor_interval == 100
    assert values["visual_steps"] == [0, 100, 250, 500, 1000, 1500, 2000]
    assert values["stratified_monitor_pair_count"] == 128


def test_g2_1_configuration_rejects_an_optimizer_change(tmp_path: Path) -> None:
    values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    values["generator_learning_rate"] = 2e-4
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer/data/safety"):
        validate_g2_1_configuration(
            path,
            repository_root=REPO_ROOT,
            require_reference_checkpoint=False,
        )


def test_canonical_training_state_hash_is_stable_and_content_sensitive() -> None:
    first = {"state": {1: {"step": torch.tensor(2), "value": torch.tensor([1.0])}}}
    reordered = {"state": {1: {"value": torch.tensor([1.0]), "step": torch.tensor(2)}}}
    changed = {"state": {1: {"step": torch.tensor(2), "value": torch.tensor([2.0])}}}
    assert canonical_state_hash(first) == canonical_state_hash(reordered)
    assert canonical_state_hash(first) != canonical_state_hash(changed)


def test_rolling_statistics_use_exact_nonoverlapping_windows() -> None:
    records = []
    for step in range(1, 5):
        records.append(
            {
                "kind": "joint",
                "joint_step": step,
                "discriminator": {
                    "gradient_clipping_applied": step % 2 == 0,
                    "real_minus_fake_logit_margin": float(step),
                },
                "generator": {
                    "gradient_clipping_applied": False,
                    "boundary_residual_mass_fraction": step / 10,
                    "boundary_residual_enrichment": float(step),
                    "tanh_raw_residual_saturation_fraction": 0.0,
                },
            }
        )
    windows = _rolling_training_statistics(records, window_size=2)
    assert [(item["start_step"], item["end_step"]) for item in windows] == [
        (1, 2),
        (3, 4),
    ]
    assert [item["discriminator_clipped_fraction"] for item in windows] == [
        0.5,
        0.5,
    ]
    assert windows[0]["mean_real_minus_fake_margin"] == 1.5
