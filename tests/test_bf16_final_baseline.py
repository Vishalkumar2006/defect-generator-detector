from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import scripts.train_final_real_baseline as final_trainer
from defectgen.training.engine import load_training_checkpoint, save_training_checkpoint
from defectgen.training.final_baseline import (
    EarlyStopping,
    STABILIZED_BF16_IDENTITY,
    build_plateau_scheduler,
    configuration_identity,
    load_final_baseline_configuration,
)
from defectgen.training.losses import CombinedBCEDiceLoss
from defectgen.training.numerics import NumericalStepController


REPO_ROOT = Path(__file__).resolve().parents[1]
BF16_CONFIG_PATH = REPO_ROOT / "configs" / "final_real_baseline_bf16.json"
FP16_CONFIG_PATH = REPO_ROOT / "configs" / "final_real_baseline.json"


def _model_optimizer(learning_rate: float = 0.0003):
    model = torch.nn.Conv2d(1, 1, 1)
    return model, torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)


def _batch():
    inputs = torch.randn(2, 1, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    targets[1, :, 1:3, 1:3] = 1
    return inputs, targets, torch.ones_like(targets)


def test_stabilized_configuration_changes_only_declared_optimization_and_output_identity():
    fp16 = load_final_baseline_configuration(FP16_CONFIG_PATH)
    bf16 = load_final_baseline_configuration(BF16_CONFIG_PATH)
    assert configuration_identity(bf16) == STABILIZED_BF16_IDENTITY
    assert bf16["precision"] == {
        "mode": "bf16",
        "grad_scaler": False,
        "automatic_fp32_retry": False,
        "gradient_clip_max_norm": 1.0,
    }
    assert bf16["optimizer"]["learning_rate"] == 0.0003
    assert bf16["paths"]["report_directory"] == "reports/final_real_baseline_bf16_seed42"
    assert bf16["paths"]["checkpoint_directory"] == "checkpoints/final_real_baseline_bf16_seed42"
    ignored_top_level = {"phase", "experiment_identity", "optimizer", "precision", "paths"}
    for key in fp16:
        if key not in ignored_top_level:
            assert bf16[key] == fp16[key]
    assert bf16["optimizer"]["type"] == fp16["optimizer"]["type"]
    assert bf16["optimizer"]["weight_decay"] == fp16["optimizer"]["weight_decay"]


def test_bf16_controller_uses_no_scaler_and_keeps_loss_float32():
    model, optimizer = _model_optimizer()
    controller = NumericalStepController(
        optimizer,
        precision_mode="bf16",
        scaler=None,
        gradient_clip_max_norm=1.0,
        automatic_fp32_retry=False,
    )
    telemetry = controller.run_batch(model, *_batch(), CombinedBCEDiceLoss(pos_weight=5))
    assert telemetry.loss_dtype == "torch.float32"
    assert telemetry.scale_before is telemetry.scale_after is None
    assert telemetry.optimizer_updates_this_batch == 1
    assert controller.counters.optimizer_step_executed == 1
    assert final_trainer._scaler_scale(None) is None
    assert {
        "grad_scaler_applicable",
        "grad_scaler_scale_initial",
        "grad_scaler_scale",
    } <= set(final_trainer.EPOCH_FIELDS)
    controller.close()


def test_gradient_norms_are_recorded_before_and_after_clipping_at_one():
    class AmplifiedCriterion:
        def components(self, logits, targets, valid_region):  # noqa: ARG002
            total = logits.float().sum() * 1000.0
            return {"bce": total, "dice": total * 0.0, "total": total}

    model, optimizer = _model_optimizer()
    controller = NumericalStepController(
        optimizer, precision_mode="bf16", scaler=None, gradient_clip_max_norm=1.0
    )
    telemetry = controller.run_batch(model, *_batch(), AmplifiedCriterion())
    assert telemetry.pre_clipping_gradient_norm > 1.0
    assert telemetry.post_clipping_gradient_norm <= 1.0001
    assert telemetry.post_clipping_gradient_norm < telemetry.pre_clipping_gradient_norm
    assert telemetry.optimizer_updates_this_batch == 1
    controller.close()


def test_output_identity_refuses_historical_directories_and_nonempty_new_run(monkeypatch, tmp_path):
    configuration = load_final_baseline_configuration(BF16_CONFIG_PATH)
    monkeypatch.setattr(final_trainer, "REPO_ROOT", tmp_path)
    report = tmp_path / configuration["paths"]["report_directory"]
    checkpoint = tmp_path / configuration["paths"]["checkpoint_directory"]
    historical_report = tmp_path / "reports" / "final_real_baseline"
    historical_checkpoint = tmp_path / "checkpoints" / "final_real_baseline"
    historical_report.mkdir(parents=True)
    historical_checkpoint.mkdir(parents=True)
    (historical_report / "preserve.txt").write_text("failed run", encoding="utf-8")
    final_trainer._validate_stabilized_output_identity(
        configuration, report, checkpoint, resume=False
    )
    report.mkdir(parents=True)
    (report / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Fresh-run directory is not empty"):
        final_trainer._validate_stabilized_output_identity(
            configuration, report, checkpoint, resume=False
        )
    with pytest.raises(ValueError, match="do not match"):
        final_trainer._validate_stabilized_output_identity(
            configuration, historical_report, historical_checkpoint, resume=False
        )
    assert (historical_report / "preserve.txt").read_text(encoding="utf-8") == "failed run"


def test_historical_fp16_configuration_is_refused_before_run(monkeypatch, tmp_path):
    configuration = load_final_baseline_configuration(FP16_CONFIG_PATH)
    monkeypatch.setattr(final_trainer, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="historical failed FP16"):
        final_trainer._validate_stabilized_output_identity(
            configuration,
            tmp_path / configuration["paths"]["report_directory"],
            tmp_path / configuration["paths"]["checkpoint_directory"],
            resume=True,
        )


def test_bf16_support_is_checked_before_dataset_construction(monkeypatch, tmp_path):
    configuration = load_final_baseline_configuration(BF16_CONFIG_PATH)
    monkeypatch.setattr(final_trainer, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    dataset_constructed = False

    def forbidden_dataset_build(configuration):  # noqa: ARG001
        nonlocal dataset_constructed
        dataset_constructed = True
        raise AssertionError("dataset construction must not occur")

    monkeypatch.setattr(final_trainer, "_build_datasets", forbidden_dataset_build)
    with pytest.raises(RuntimeError, match="does not support BF16"):
        final_trainer.run_final_real_baseline(
            configuration,
            report_dir=tmp_path / configuration["paths"]["report_directory"],
            checkpoint_dir=tmp_path / configuration["paths"]["checkpoint_directory"],
            resume=False,
        )
    assert dataset_constructed is False


def test_fp16_checkpoint_cannot_resume_into_bf16_configuration(tmp_path):
    fp16 = load_final_baseline_configuration(FP16_CONFIG_PATH)
    bf16 = load_final_baseline_configuration(BF16_CONFIG_PATH)
    model, optimizer = _model_optimizer()
    path = tmp_path / "historical_fp16.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=3,
        configuration=fp16,
        best_validation={"epoch": 2},
    )
    with pytest.raises(ValueError, match="configuration does not match"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            expected_configuration=bf16,
        )


def test_bf16_checkpoint_resume_restores_scheduler_earlystop_and_numerics(tmp_path):
    configuration = load_final_baseline_configuration(BF16_CONFIG_PATH)
    model, optimizer = _model_optimizer()
    scheduler = build_plateau_scheduler(optimizer, configuration)
    stopping = EarlyStopping(patience=4)
    stopping.step(0.75)
    controller = NumericalStepController(
        optimizer, precision_mode="bf16", scaler=None, gradient_clip_max_norm=1.0
    )
    controller.run_batch(model, *_batch(), CombinedBCEDiceLoss(pos_weight=5))
    expected_model = copy.deepcopy(model.state_dict())
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        configuration=configuration,
        best_validation={"epoch": 1, "validation_total_loss": 0.75},
        numerical_controller=controller,
        early_stopping_state=stopping.state_dict(),
    )
    restored_model, restored_optimizer = _model_optimizer()
    restored_scheduler = build_plateau_scheduler(restored_optimizer, configuration)
    restored_stopping = EarlyStopping(patience=4)
    restored_controller = NumericalStepController(
        restored_optimizer, precision_mode="bf16", scaler=None, gradient_clip_max_norm=1.0
    )
    payload = load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        expected_configuration=configuration,
        numerical_controller=restored_controller,
        early_stopping=restored_stopping,
    )
    assert payload["scaler_state"] is None
    assert restored_controller.counters.optimizer_step_executed == 1
    assert restored_controller.gradient_clip_max_norm == 1.0
    assert restored_stopping.best == pytest.approx(0.75)
    assert all(
        torch.equal(restored_model.state_dict()[name], value) for name, value in expected_model.items()
    )
    controller.close()
    restored_controller.close()


def test_bf16_dataset_builder_constructs_train_and_validation_only(monkeypatch):
    requested: list[tuple[str, object]] = []

    class FakeDataset:
        def __init__(self, development_split, spatial_transform, **kwargs):  # noqa: ARG002
            requested.append((development_split, spatial_transform))
            self.rows = [{"development_split": development_split, "sample_id": development_split}]

    monkeypatch.setattr(final_trainer, "KSDD2FullImageDataset", FakeDataset)
    configuration = load_final_baseline_configuration(BF16_CONFIG_PATH)
    final_trainer._build_datasets(configuration)
    assert [split for split, _ in requested] == ["train", "validation"]
    assert requested[0][1] is not None and requested[1][1] is None
