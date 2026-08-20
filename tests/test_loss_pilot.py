from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from defectgen.models import UNet
from defectgen.training.diagnostics import select_fixed_validation_ids
from defectgen.training.engine import (
    configurations_differ_only_by_pos_weight,
    load_training_checkpoint,
    save_training_checkpoint,
    write_metric_logs,
)
from defectgen.training.losses import CombinedBCEDiceLoss
from defectgen.training.metrics import detailed_validation_metrics, validation_threshold_sweep
from defectgen.training.reproducibility import configure_reproducibility
from scripts.train_baseline import (
    _build_datasets,
    candidate_configuration,
    candidate_directories,
    model_state_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    base = json.loads((REPO_ROOT / "configs" / "baseline.json").read_text(encoding="utf-8"))
    pilot = json.loads((REPO_ROOT / "configs" / "loss_pilot.json").read_text(encoding="utf-8"))
    return base, pilot


def test_full_training_datasets_exclude_official_test(monkeypatch):
    class FakeDataset:
        def __init__(self, development_split, **kwargs):  # noqa: ARG002
            size = 1981 if development_split == "train" else 350
            self.rows = [{"development_split": development_split}] * size

        def __len__(self):
            return len(self.rows)

    monkeypatch.setattr("scripts.train_baseline.KSDD2FullImageDataset", FakeDataset)
    base, pilot = _configs()
    training, validation = _build_datasets(candidate_configuration(base, pilot, 1))
    assert len(training) == 1981
    assert len(validation) == 350
    assert {row["development_split"] for row in training.rows} == {"train"}
    assert {row["development_split"] for row in validation.rows} == {"validation"}


def test_candidate_initialization_is_identical():
    configure_reproducibility(42, deterministic=True, warn_only=True)
    first = model_state_sha256(UNet(base_channels=32))
    configure_reproducibility(42, deterministic=True, warn_only=True)
    second = model_state_sha256(UNet(base_channels=32))
    assert first == second


def test_checkpoint_round_trip_restores_training_and_random_state(tmp_path):
    configure_reproducibility(42, deterministic=True, warn_only=True)
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler_generator = torch.Generator().manual_seed(10)
    loader_generator = torch.Generator().manual_seed(11)
    loss = model(torch.ones(1, 1, 2, 2)).sum()
    loss.backward()
    optimizer.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    configuration = {"loss": {"pos_weight": 5.0}, "seed": 42}
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        epoch=3,
        configuration=configuration,
        best_validation={"epoch": 2, "validation_total_loss": 0.8},
        sampler_generator=sampler_generator,
        loader_generator=loader_generator,
        metric_records=[{"epoch": 3, "loss": 0.9}],
    )
    with torch.no_grad():
        model.weight.zero_()
    payload = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        expected_configuration=configuration,
        sampler_generator=sampler_generator,
        loader_generator=loader_generator,
    )
    assert payload["epoch"] == 3
    assert payload["metric_records"] == [{"epoch": 3, "loss": 0.9}]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())


def test_metric_logs_are_identical_csv_and_json(tmp_path):
    records = [{"epoch": 1, "loss": 1.2}, {"epoch": 2, "loss": 0.7}]
    csv_path, json_path = tmp_path / "metrics.csv", tmp_path / "metrics.json"
    write_metric_logs(records, csv_path, json_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert csv_rows == [{"epoch": "1", "loss": "1.2"}, {"epoch": "2", "loss": "0.7"}]
    assert json.loads(json_path.read_text(encoding="utf-8")) == records


def test_validation_threshold_sweep_has_exact_grid_and_independent_objectives():
    probabilities = torch.tensor([[[[0.9, 0.4]]], [[[0.6, 0.1]]]])
    targets = torch.tensor([[[[1, 0]]], [[[0, 0]]]], dtype=torch.bool)
    valid = torch.ones_like(targets)
    labels = torch.tensor([True, False])
    rows, best_global, best_defective = validation_threshold_sweep(probabilities, targets, valid, labels)
    assert [round(float(row["threshold"]), 2) for row in rows] == [value / 100 for value in range(5, 100, 5)]
    assert best_global in rows
    assert best_defective in rows


def test_empty_normal_masks_have_explicit_false_positive_metrics():
    probabilities = torch.tensor([[[[0.8, 0.2]]], [[[0.7, 0.1]]]])
    targets = torch.tensor([[[[1, 0]]], [[[0, 0]]]], dtype=torch.bool)
    metrics, rows = detailed_validation_metrics(
        probabilities, targets, torch.ones_like(targets), torch.tensor([True, False]), threshold=0.5
    )
    assert rows[1]["false_positive_pixels"] == 1
    assert metrics["normal_image_count"] == 1
    assert metrics["normal_image_false_positive_rate"] == 1.0
    assert metrics["mean_predicted_defect_fraction_normal_images"] == 0.5


def test_fixed_validation_selection_is_deterministic_and_cross_candidate():
    geometry = [
        {"sample_id": f"d{index}", "development_split": "validation", "mask_pixels": pixels, "touches_border": border}
        for index, (pixels, border) in enumerate([(1, False), (5, False), (10, True), (20, False), (100, True)])
    ]
    candidate_rows = {}
    for candidate, offset in (("pw1", 0), ("pw5", 2)):
        rows = [
            {"sample_id": f"d{index}", "has_defect": 1, "predicted_pixels": pixels, "dice": 0.8 - 0.1 * index}
            for index, pixels in enumerate((1, 4, 8, 12, 20))
        ]
        rows += [
            {"sample_id": "n1", "has_defect": 0, "predicted_pixels": 2 + offset, "dice": 0},
            {"sample_id": "n2", "has_defect": 0, "predicted_pixels": 20 + offset, "dice": 0},
        ]
        candidate_rows[candidate] = rows
    first = select_fixed_validation_ids(geometry, candidate_rows)
    second = select_fixed_validation_ids(reversed(geometry), dict(reversed(list(candidate_rows.items()))))
    assert first == second
    assert first["normal_largest_false_positive_area"] == "n2"
    assert first["defective_lowest_dice"] == "d4"


def test_combined_loss_produces_finite_gradients():
    model = torch.nn.Conv2d(1, 1, 1)
    logits = model(torch.randn(2, 1, 4, 4))
    targets = torch.zeros_like(logits)
    targets[0, :, 1:3, 1:3] = 1
    loss = CombinedBCEDiceLoss(pos_weight=20).components(logits, targets, torch.ones_like(targets))["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_candidate_directories_are_isolated():
    _, pilot = _configs()
    paths = [candidate_directories(pilot, weight) for weight in (1, 5, 10, 20)]
    assert len({report for report, _ in paths}) == 4
    assert len({checkpoint for _, checkpoint in paths}) == 4
    assert all(report.name == checkpoint.name for report, checkpoint in paths)


def test_candidate_configurations_differ_only_by_pos_weight():
    base, pilot = _configs()
    configs = [candidate_configuration(base, pilot, weight) for weight in (1, 5, 10, 20)]
    assert all(configurations_differ_only_by_pos_weight(configs[0], config) for config in configs[1:])
