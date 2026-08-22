from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from defectgen.data.augmentation import SynchronizedRandomFlips
from defectgen.training.g2_2_utility import (
    ScheduledMixtureDataset,
    SyntheticDetectorDataset,
    assert_paired_manifests,
    assert_train_only_provenance,
    build_equal_budget_schedule,
    confirmation_decision,
    meaningful_winner,
    paired_source_signature,
    stratified_validation_metrics,
)


def _row(tmp_path: Path, checkpoint: int) -> dict:
    image = np.full((8, 6, 3), 64 + checkpoint // 1000, dtype=np.uint8)
    mask = np.zeros((8, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    valid = np.zeros((8, 6), dtype=np.uint8)
    valid[:, 1:5] = 255
    for name, value in ((f"image-{checkpoint}.png", image), ("mask.png", mask), ("valid.png", valid)):
        Image.fromarray(value).save(tmp_path / name)
    return {
        "sample_id": "paired-0",
        "sample_index": 0,
        "deterministic_sample_seed": 123,
        "official_split": "train",
        "development_split": "train",
        "mask_path": "mask.png",
        "mask_sha256": "mask",
        "valid_region_path": "valid.png",
        "valid_region_sha256": "valid",
        "coarse_image_content_sha256": "coarse",
        "source_provenance": {
            "template": {"sample_id": "train-t", "official_split": "train", "development_split": "train", "template_id": "t0"},
            "background": {"sample_id": "train-b", "official_split": "train", "development_split": "train"},
        },
        "checkpoint_step": checkpoint,
        "checkpoint_path": f"checkpoint-{checkpoint}.pt",
        "checkpoint_sha256": str(checkpoint),
        "image_path": f"image-{checkpoint}.png",
        "image_sha256": f"image-{checkpoint}",
    }


def test_paired_manifest_signature_excludes_only_checkpoint_rendering(tmp_path: Path) -> None:
    first, second = _row(tmp_path, 1000), _row(tmp_path, 1500)
    assert paired_source_signature(first) == paired_source_signature(second)
    assert_paired_manifests([first], [second])
    second["deterministic_sample_seed"] += 1
    with pytest.raises(RuntimeError, match="diverge before GAN rendering"):
        assert_paired_manifests([first], [second])


def test_train_only_provenance_guard_rejects_validation_and_official_test(tmp_path: Path) -> None:
    row = _row(tmp_path, 1000)
    row["source_provenance"]["background"]["development_split"] = "validation"
    with pytest.raises(RuntimeError, match="non-training background"):
        assert_train_only_provenance([row])
    row = _row(tmp_path, 1000)
    row["official_split"] = "test"
    with pytest.raises(RuntimeError, match="forbidden official_split"):
        assert_train_only_provenance([row])


def test_equal_budget_schedule_is_deterministic_and_exactly_25_percent_synthetic() -> None:
    labels = [False, True, False, True]
    first = build_equal_budget_schedule(
        labels, optimizer_updates=8, batch_size=4, seed=42,
        synthetic_fraction=0.25, synthetic_count=5, variant="checkpoint_1000",
    )
    second = build_equal_budget_schedule(
        labels, optimizer_updates=8, batch_size=4, seed=42,
        synthetic_fraction=0.25, synthetic_count=5, variant="checkpoint_1500",
    )
    assert first == second
    assert len(first) == 32
    assert sum(row.source == "synthetic" for row in first) == 8
    assert all(
        sum(row.source == "synthetic" for row in first[offset : offset + 4]) == 1
        for offset in range(0, len(first), 4)
    )


def test_control_shares_three_real_draws_per_batch_with_synthetic_arm() -> None:
    labels = [False, True, False, True, False, True]
    control = build_equal_budget_schedule(
        labels, optimizer_updates=10, batch_size=4, seed=9,
        synthetic_fraction=0.25, synthetic_count=3, variant="real_only",
    )
    augmented = build_equal_budget_schedule(
        labels, optimizer_updates=10, batch_size=4, seed=9,
        synthetic_fraction=0.25, synthetic_count=3, variant="checkpoint_1000",
    )
    for offset in range(0, len(control), 4):
        control_common = [row.source_index for row in control[offset : offset + 4]][:3]
        augmented_common = [row.source_index for row in augmented[offset : offset + 4] if row.source == "real"]
        assert control_common == augmented_common


def test_synthetic_dataset_preserves_invalid_region_and_normalizes(tmp_path: Path) -> None:
    row = _row(tmp_path, 1000)
    dataset = SyntheticDetectorDataset(
        tmp_path, {"rows": [row]}, mean=[0, 0, 0], standard_deviation=[1, 1, 1]
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 6)
    assert not bool((sample["mask"].bool() & ~sample["valid_region"].bool()).any())
    assert sample["has_defect"]


def test_mixed_real_synthetic_batch_has_one_collatable_contract() -> None:
    class Samples(Dataset):
        def __init__(self, *, real: bool):
            self.real = real

        def __len__(self):
            return 1

        def __getitem__(self, index):
            sample = {
                "image": torch.zeros(3, 8, 8),
                "mask": torch.zeros(1, 8, 8),
                "valid_region": torch.ones(1, 8, 8),
                "has_defect": torch.tensor(False),
                "sample_id": "real" if self.real else "synthetic",
            }
            if self.real:
                sample["original_height"] = 8
            return sample

    schedule = build_equal_budget_schedule(
        [False, True], optimizer_updates=1, batch_size=4, seed=42,
        synthetic_fraction=.25, synthetic_count=1, variant="checkpoint_1000",
    )
    dataset = ScheduledMixtureDataset(
        Samples(real=True), Samples(real=False), schedule,
        SynchronizedRandomFlips(0, 0, seed=42),
    )
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert set(batch) == {
        "image", "mask", "valid_region", "has_defect", "sample_id",
        "schedule_source", "optimizer_step",
    }


def test_stratified_metrics_cover_requested_groups() -> None:
    rows = [
        {"has_defect": 1, "true_positive_pixels": 4, "false_positive_pixels": 1, "false_negative_pixels": 2, "predicted_pixels": 5},
        {"has_defect": 1, "true_positive_pixels": 8, "false_positive_pixels": 0, "false_negative_pixels": 0, "predicted_pixels": 8},
    ]
    metrics = stratified_validation_metrics(rows, ["a", "b"], {"a": "size:small", "b": "contact:border"})
    assert metrics["size:small"]["recall"] == pytest.approx(4 / 6)
    assert metrics["contact:border"]["dice"] == 1.0


def test_meaningful_win_rule_is_precommitted_and_multimetric() -> None:
    control = {"overall": {"global_dice": .5, "global_iou": .4, "normal_image_false_positive_rate": .1, "pixel_precision": .6, "pixel_recall": .7}}
    candidate = {"overall": {"global_dice": .52, "global_iou": .41, "normal_image_false_positive_rate": .11, "pixel_precision": .595, "pixel_recall": .695}}
    rules = {
        "minimum_global_dice_gain": .01, "minimum_global_iou_gain": .005,
        "maximum_normal_fpr_regression": .02, "maximum_precision_regression": .01,
        "maximum_recall_regression": .01,
    }
    assert meaningful_winner(candidate, control, rules=rules)[0]
    candidate["overall"]["normal_image_false_positive_rate"] = .13
    assert not meaningful_winner(candidate, control, rules=rules)[0]


def test_confirmation_requires_three_seeds_and_mean_multimetric_gain() -> None:
    rules = {
        "minimum_global_dice_gain": .01, "minimum_global_iou_gain": .005,
        "maximum_normal_fpr_regression": .02, "maximum_precision_regression": .01,
        "maximum_recall_regression": .01,
    }
    row = {
        "global_dice_gain": .02, "global_iou_gain": .01, "normal_fpr_delta": -.01,
        "pixel_precision_delta": 0, "pixel_recall_delta": 0,
    }
    confirmed, summary = confirmation_decision([row, row, {**row, "global_dice_gain": -.001}], rules=rules)
    assert confirmed
    assert summary["positive_dice_seeds"] == 2
    with pytest.raises(ValueError, match="exactly three"):
        confirmation_decision([row, row], rules=rules)
