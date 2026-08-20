from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from defectgen.data.full_image import (
    KSDD2FullImageDataset,
    NativeGeometry,
    pad_full_sample,
    restore_to_native,
)
from defectgen.data.patches import Padding


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "metadata" / "ksdd2_split_seed42.csv"


def test_complete_image_padding_zero_labels_and_valid_region() -> None:
    image = np.arange(5 * 3 * 3, dtype=np.uint8).reshape(5, 3, 3)
    mask = np.zeros((5, 3), dtype=np.uint8)
    mask[0, 0] = 1
    padded_image, padded_mask, valid, padding = pad_full_sample(
        image, mask, target_size=(8, 8), image_padding_mode="reflect"
    )
    assert padded_image.shape == (8, 8, 3)
    assert padded_mask.shape == valid.shape == (8, 8)
    assert padded_mask.sum() == 1  # reflected image content never reflects a positive label
    assert valid.sum() == 15
    assert np.count_nonzero(padded_mask[valid == 0]) == 0
    assert padding == Padding(left=2, top=1, right=3, bottom=2)


def test_native_size_restoration() -> None:
    prediction = torch.arange(8 * 8).reshape(1, 8, 8)
    geometry = NativeGeometry(5, 3, Padding(left=2, top=1, right=3, bottom=2))
    restored = restore_to_native(prediction, geometry)
    assert restored.shape == (1, 5, 3)
    assert torch.equal(restored, prediction[:, 1:6, 2:5])


def test_dataset_is_manifest_only_and_split_isolated() -> None:
    if not MANIFEST.is_file():
        pytest.skip("Phase B split manifest is absent")
    training = KSDD2FullImageDataset(REPO_ROOT, "train", MANIFEST)
    validation = KSDD2FullImageDataset(REPO_ROOT, "validation", MANIFEST)
    test = KSDD2FullImageDataset(REPO_ROOT, "test", MANIFEST)
    assert (len(training), len(validation), len(test)) == (1981, 350, 1004)
    assert {row["sample_id"] for row in training.rows}.isdisjoint(
        {row["sample_id"] for row in validation.rows}
    )
    assert all("(copy)" not in row["image_path"].casefold() for row in training.rows + validation.rows + test.rows)


def test_real_dataset_item_has_complete_canvas_and_binary_masks() -> None:
    if not MANIFEST.is_file():
        pytest.skip("Phase B split manifest is absent")
    dataset = KSDD2FullImageDataset(REPO_ROOT, "train", MANIFEST)
    item = dataset[0]
    assert item["image"].shape == (3, 672, 256)
    assert item["mask"].shape == item["valid_region"].shape == (1, 672, 256)
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
    assert torch.count_nonzero(item["mask"] * (1 - item["valid_region"])) == 0

