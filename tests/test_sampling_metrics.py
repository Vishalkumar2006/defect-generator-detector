from __future__ import annotations

import torch

from defectgen.data.sampling import deterministic_weighted_sampler
from defectgen.training.metrics import (
    image_classification_metrics,
    segmentation_metrics,
    select_validation_threshold,
)


def test_weighted_sampler_is_deterministic_and_balanced() -> None:
    labels = [True] * 10 + [False] * 90
    first = list(deterministic_weighted_sampler(labels, seed=42, num_samples=1000))
    second = list(deterministic_weighted_sampler(labels, seed=42, num_samples=1000))
    assert first == second
    defective_fraction = sum(labels[index] for index in first) / len(first)
    assert 0.45 <= defective_fraction <= 0.55


def test_valid_region_metrics_ignore_padding_and_handle_normals() -> None:
    logits = torch.full((1, 1, 8, 8), -10.0)
    targets = torch.zeros_like(logits)
    valid = torch.zeros_like(logits)
    valid[:, :, 2:6, 2:6] = 1
    logits[valid == 0] = 10.0
    metrics = segmentation_metrics(logits, targets, valid)
    assert metrics["pixel_dice"] == 1.0
    assert metrics["pixel_iou"] == 1.0


def test_validation_threshold_selection_is_separate() -> None:
    threshold, metrics = select_validation_threshold([0.1, 0.4, 0.8, 0.9], [False, False, True, True])
    assert 0 < threshold < 1
    assert metrics["f1"] == 1.0
    fixed_test_metrics = image_classification_metrics([0.2, 0.7], [False, True], threshold)
    assert fixed_test_metrics["threshold"] == threshold

