from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import scripts.diagnose_baseline_precision as precision_diagnostic
from defectgen.training.failure_diagnostics import (
    inspect_checkpoint_finiteness,
    model_state_sha256,
    nonfinite_components,
    write_numerical_failure_report,
)
from defectgen.training.final_baseline import load_final_baseline_configuration
from defectgen.training.losses import CombinedBCEDiceLoss
from scripts.train_final_real_baseline import _validate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_batch():
    mask = torch.zeros(2, 1, 4, 4)
    mask[1, :, 1:3, 1:3] = 1
    return {
        "image": torch.randn(2, 1, 4, 4),
        "mask": mask,
        "valid_region": torch.ones_like(mask),
        "has_defect": torch.tensor([False, True]),
        "sample_id": ["synthetic-normal", "synthetic-defect"],
    }


def test_atomic_crash_report_contains_required_numerical_context(tmp_path):
    path = tmp_path / "crash.json"
    report = write_numerical_failure_report(
        path,
        phase="training",
        epoch=4,
        batch_index=7,
        sample_ids=["a", "b"],
        precision_mode="fp16",
        logits=torch.tensor([0.5, float("nan"), float("inf"), float("-inf")]),
        loss_components={"bce": torch.tensor(1.0), "dice": torch.tensor(float("nan")), "total": float("inf")},
        scaler_scale=16384.0,
        most_recent_gradient_norm=float("inf"),
        checkpoint_paths={"best": "best.pt", "last": "last.pt"},
        repo_root=tmp_path,
        error="synthetic failure",
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == report
    assert not path.with_suffix(".json.tmp").exists()
    assert persisted["phase"] == "training"
    assert persisted["nonfinite_components"] == ["logits", "dice", "total"]
    assert persisted["logits"]["nan_count"] == 1
    assert persisted["logits"]["positive_inf_count"] == 1
    assert persisted["logits"]["negative_inf_count"] == 1
    assert persisted["most_recent_gradient_norm"] == "+inf"


def test_component_specific_nonfinite_detection():
    logits = torch.zeros(1, 1, 2, 2)
    components = {
        "bce": torch.tensor(0.5),
        "dice": torch.tensor(float("nan")),
        "total": torch.tensor(float("inf")),
    }
    assert nonfinite_components(logits, components) == ["dice", "total"]
    logits[0, 0, 0, 0] = float("nan")
    assert nonfinite_components(logits, components) == ["logits", "dice", "total"]


def test_validation_failure_report_is_written_before_component_error(tmp_path):
    class NonfiniteModel(torch.nn.Module):
        def forward(self, inputs):
            return inputs[:, :1] * float("nan")

    path = tmp_path / "validation_failure.json"
    with pytest.raises(RuntimeError, match="logits"):
        _validate(
            NonfiniteModel(),
            [_synthetic_batch()],
            CombinedBCEDiceLoss(pos_weight=5),
            torch.device("cpu"),
            "fp32",
            keep_outputs=False,
            failure_context={
                "path": path,
                "epoch": 4,
                "checkpoint_paths": {"best": "best.pt", "last": "last.pt"},
                "repo_root": tmp_path,
            },
            scaler_scale=16384.0,
            most_recent_gradient_norm=9.0,
        )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "validation"
    assert persisted["epoch"] == 4
    assert persisted["batch_index"] == 1
    assert persisted["sample_ids"] == ["synthetic-normal", "synthetic-defect"]
    assert "logits" in persisted["nonfinite_components"]


def test_checkpoint_finiteness_inspection_detects_model_and_optimizer_failures(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": {"finite": torch.ones(2), "bad": torch.tensor([float("nan")])},
            "optimizer_state": {"state": {0: {"exp_avg": torch.tensor([float("inf")])}}},
            "epoch": 3,
        },
        path,
    )
    _, inspection = inspect_checkpoint_finiteness(path)
    assert inspection["model"]["nan_count"] == 1
    assert inspection["optimizer"]["positive_inf_count"] == 1
    assert inspection["model"]["nonfinite_paths"] == ["model_state.bad"]
    assert inspection["parameters_usable"] is False


def test_precision_diagnosis_has_no_optimizer_update_backward_or_parameter_change():
    model = torch.nn.Conv2d(1, 1, 1)
    before = model_state_sha256(model)
    result = precision_diagnostic.diagnose_precision_mode(
        model,
        [_synthetic_batch()],
        CombinedBCEDiceLoss(pos_weight=5),
        torch.device("cpu"),
        "fp32",
    )
    after = model_state_sha256(model)
    assert result["status"] == "finite"
    assert before == after
    assert all(parameter.grad is None for parameter in model.parameters())


def test_precision_diagnostic_constructs_validation_only(monkeypatch):
    requested: list[str] = []

    class FakeDataset:
        def __init__(self, development_split, **kwargs):  # noqa: ARG002
            requested.append(development_split)
            self.rows = [{"development_split": development_split}]

    monkeypatch.setattr(precision_diagnostic, "KSDD2FullImageDataset", FakeDataset)
    configuration = load_final_baseline_configuration(REPO_ROOT / "configs" / "final_real_baseline.json")
    precision_diagnostic._build_validation_dataset(configuration)
    assert requested == ["validation"]


def test_fp16_failure_does_not_prevent_bf16_and_fp32_diagnostics(monkeypatch):
    called: list[str] = []

    def fake_diagnostic(model, loader, criterion, device, mode):  # noqa: ARG001
        called.append(mode)
        return {
            "status": "numerical_failure" if mode == "fp16" else "finite",
            "precision_mode": mode,
        }

    monkeypatch.setattr(precision_diagnostic, "diagnose_precision_mode", fake_diagnostic)
    results = precision_diagnostic.evaluate_precision_modes(
        object(), object(), object(), torch.device("cpu")
    )
    assert called == ["fp16", "bf16", "fp32"]
    assert results["fp16"]["status"] == "numerical_failure"
    assert results["bf16"]["status"] == results["fp32"]["status"] == "finite"
