from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from defectgen.data.augmentation import SynchronizedRandomFlips
from defectgen.training.engine import (
    load_training_checkpoint,
    save_training_checkpoint,
    update_checkpoint_metadata,
)
from defectgen.training.final_baseline import (
    EarlyStopping,
    PostTrainingValidationGate,
    build_plateau_scheduler,
    load_final_baseline_configuration,
    threshold_candidates,
    validate_final_baseline_configuration,
)
from defectgen.training.losses import CombinedBCEDiceLoss
from defectgen.training.numerics import NumericalStepController
from scripts.train_final_real_baseline import EPOCH_FIELDS, _build_datasets, _train_epoch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "final_real_baseline.json"


def test_frozen_final_configuration_loads_and_has_exact_thresholds():
    configuration = load_final_baseline_configuration(CONFIG_PATH)
    assert configuration["loss"]["pos_weight"] == 5.0
    assert configuration["training"]["maximum_epochs"] == 12
    assert configuration["data"]["official_test_evaluation"] is False
    assert threshold_candidates(configuration) == pytest.approx([value / 100 for value in range(5, 100, 5)])


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "base_channels", 16),
        ("data", "manifest", "data/metadata/changed.csv"),
        ("precision", "automatic_fp32_retry", True),
        ("training", "batch_size", 8),
    ],
)
def test_frozen_configuration_rejects_critical_changes(section, key, value):
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    configuration[section][key] = value
    with pytest.raises(ValueError, match="Frozen final real baseline was modified"):
        validate_final_baseline_configuration(configuration)


def test_synchronized_flips_preserve_alignment_and_binary_mask():
    image = torch.arange(3 * 3 * 4).reshape(3, 3, 4).float()
    mask = torch.zeros(1, 3, 4)
    mask[:, 0, 1] = 1
    valid = torch.zeros_like(mask)
    valid[:, :2, :3] = 1
    transform = SynchronizedRandomFlips(1.0, 1.0, seed=42)
    transformed = transform(image, mask, valid, sample_id="sample", epoch=1)
    assert torch.equal(transformed[0], torch.flip(image, (-1, -2)))
    assert torch.equal(transformed[1], torch.flip(mask, (-1, -2)))
    assert torch.equal(transformed[2], torch.flip(valid, (-1, -2)))
    assert set(torch.unique(transformed[1]).tolist()) <= {0.0, 1.0}


def test_flip_decisions_are_stateless_and_epoch_deterministic():
    tensors = (torch.randn(3, 5, 6), torch.zeros(1, 5, 6), torch.ones(1, 5, 6))
    transform = SynchronizedRandomFlips(seed=73)
    first = transform(*tensors, sample_id="fixed-id", epoch=4)
    second = transform(*tensors, sample_id="fixed-id", epoch=4)
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_validation_dataset_has_no_augmentation_and_official_test_is_not_constructed(monkeypatch):
    requested: list[tuple[str, object]] = []

    class FakeDataset:
        def __init__(self, development_split, spatial_transform, **kwargs):  # noqa: ARG002
            requested.append((development_split, spatial_transform))
            self.rows = [{"development_split": development_split}]

    monkeypatch.setattr("scripts.train_final_real_baseline.KSDD2FullImageDataset", FakeDataset)
    configuration = load_final_baseline_configuration(CONFIG_PATH)
    _build_datasets(configuration)
    assert [split for split, _ in requested] == ["train", "validation"]
    assert requested[0][1] is not None
    assert requested[1][1] is None


def test_plateau_scheduler_uses_frozen_patience_factor_and_floor():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    configuration = load_final_baseline_configuration(CONFIG_PATH)
    scheduler = build_plateau_scheduler(optimizer, configuration)
    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    for _ in range(40):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00001)


def test_early_stopping_requires_four_non_improving_epochs_and_restores():
    stopping = EarlyStopping(patience=4, minimum_delta=0.0)
    assert stopping.step(1.0) is False
    assert [stopping.step(value) for value in (1.0, 1.1, 1.2, 1.3)] == [False, False, False, True]
    restored = EarlyStopping(patience=4, minimum_delta=0.0)
    restored.load_state_dict(stopping.state_dict())
    assert restored.state_dict() == stopping.state_dict()


def test_best_last_checkpoint_restores_scheduler_earlystop_random_and_model(tmp_path):
    configuration = {"data": {"manifest": "fixed.csv", "training_split": "train"}}
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    stopping = EarlyStopping(patience=4)
    stopping.step(0.8)
    scheduler.step(0.8)
    sampler_generator = torch.Generator().manual_seed(11)
    loader_generator = torch.Generator().manual_seed(12)
    expected_state = copy.deepcopy(model.state_dict())
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    arguments = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": None,
        "epoch": 3,
        "configuration": configuration,
        "best_validation": {"epoch": 3, "validation_total_loss": 0.8},
        "sampler_generator": sampler_generator,
        "loader_generator": loader_generator,
        "metric_records": [{"epoch": 3}],
        "early_stopping_state": stopping.state_dict(),
    }
    save_training_checkpoint(best_path, checkpoint_metadata={"kind": "best"}, **arguments)
    save_training_checkpoint(last_path, checkpoint_metadata={"kind": "last"}, **arguments)
    expected_sampler_draw = torch.rand(1, generator=sampler_generator)

    for parameter in model.parameters():
        parameter.data.zero_()
    restored_stopping = EarlyStopping(patience=4)
    restored_sampler = torch.Generator()
    payload = load_training_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_configuration=configuration,
        sampler_generator=restored_sampler,
        loader_generator=torch.Generator(),
        early_stopping=restored_stopping,
    )
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected_state.items())
    assert torch.equal(torch.rand(1, generator=restored_sampler), expected_sampler_draw)
    assert restored_stopping.best == pytest.approx(0.8)
    assert payload["checkpoint_metadata"]["kind"] == "last"
    assert best_path.is_file() and last_path.is_file()


def test_checkpoint_resume_rejects_data_split_or_configuration_change(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "last.pt"
    configuration = {"data": {"manifest": "a.csv", "training_split": "train"}}
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=1,
        configuration=configuration,
        best_validation={"epoch": 1},
    )
    changed = copy.deepcopy(configuration)
    changed["data"]["manifest"] = "b.csv"
    with pytest.raises(ValueError, match="configuration does not match"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            expected_configuration=changed,
        )


def test_post_training_gate_allows_one_sweep_only_after_best_reload():
    gate = PostTrainingValidationGate()
    with pytest.raises(RuntimeError, match="requires completed training"):
        gate.claim_threshold_sweep()
    gate.mark_training_complete()
    with pytest.raises(RuntimeError, match="requires completed training"):
        gate.claim_threshold_sweep()
    gate.mark_best_checkpoint_loaded()
    gate.claim_threshold_sweep()
    with pytest.raises(RuntimeError, match="only once"):
        gate.claim_threshold_sweep()


def test_post_training_metadata_records_selected_threshold(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "best.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=1,
        configuration={"frozen": True},
        best_validation={"epoch": 1},
    )
    update_checkpoint_metadata(path, {"selected_validation_threshold": 0.35, "finalized": True})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["checkpoint_metadata"] == {
        "selected_validation_threshold": 0.35,
        "finalized": True,
    }


def test_synthetic_epoch_emits_required_telemetry_and_one_update_per_attempt():
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    controller = NumericalStepController(optimizer, precision_mode="fp32", automatic_fp32_retry=False)
    batch = {
        "image": torch.randn(2, 1, 4, 4),
        "mask": torch.zeros(2, 1, 4, 4),
        "valid_region": torch.ones(2, 1, 4, 4),
        "has_defect": torch.tensor([False, True]),
    }
    batch["mask"][1, :, 1:3, 1:3] = 1
    (
        losses,
        fraction,
        events,
        anomalies,
        maximum_gradient,
        maximum_post_gradient,
        most_recent_gradient,
        maximum_logit,
    ) = _train_epoch(
        model, [batch], CombinedBCEDiceLoss(pos_weight=5), controller, torch.device("cpu")
    )
    assert set(losses) == {"bce", "dice", "total"}
    assert fraction == 0.5
    assert events["attempted_batches"] == 1
    assert events["optimizer_step_executed"] == 1
    assert events["optimizer_step_skipped"] == 0
    assert events["fp32_retry_attempted"] == events["fp32_retry_executed"] == 0
    assert anomalies == []
    assert maximum_gradient >= 0 and maximum_post_gradient >= 0
    assert most_recent_gradient >= 0 and maximum_logit >= 0
    assert {
        "train_bce",
        "validation_total_loss",
        "grad_scaler_scale",
        "maximum_training_gradient_norm",
        "peak_allocated_vram_bytes",
    } <= set(EPOCH_FIELDS)
    controller.close()
