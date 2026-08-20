from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from defectgen.data.ksdd2 import EXPECTED_COUNTS, index_ksdd2, interpret_binary_mask


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "extracted" / "KolektorSDD2"


@pytest.fixture(scope="session")
def dataset_result():
    if not DATASET_ROOT.is_dir():
        pytest.skip("KSDD2 has not been extracted; run scripts\\extract_ksdd2.py")
    return index_ksdd2(REPO_ROOT, DATASET_ROOT)


def test_binary_mask_interpretation() -> None:
    expected = np.array([[False, True], [False, True]])
    assert np.array_equal(interpret_binary_mask(np.array([[0, 255], [0, 255]], dtype=np.uint8)), expected)
    assert np.array_equal(interpret_binary_mask(np.array([[0, 1], [0, 1]], dtype=np.uint8)), expected)
    with pytest.raises(ValueError, match="more than two values"):
        interpret_binary_mask(np.array([[0, 127, 255]], dtype=np.uint8))


def test_full_dataset_expected_counts(dataset_result) -> None:
    assert dataset_result.counts == EXPECTED_COUNTS
    assert dataset_result.expected_counts_match


def test_index_is_deterministic_and_paths_are_relative(dataset_result) -> None:
    first = dataset_result
    second = index_ksdd2(REPO_ROOT, DATASET_ROOT)
    assert first.rows == second.rows
    for row in first.rows:
        assert not Path(row["image_path"]).is_absolute()
        assert not row["mask_path"] or not Path(row["mask_path"]).is_absolute()


def test_image_mask_dimensions_and_binary_masks_are_valid(dataset_result) -> None:
    assert dataset_result.invalid_pairs == []
    assert all(row["width"] > 0 and row["height"] > 0 for row in dataset_result.rows)


def test_train_test_separation(dataset_result) -> None:
    train_paths = {row["image_path"] for row in dataset_result.rows if row["split"] == "train"}
    test_paths = {row["image_path"] for row in dataset_result.rows if row["split"] == "test"}
    assert train_paths.isdisjoint(test_paths)
    assert not dataset_result.cross_split_duplicates
