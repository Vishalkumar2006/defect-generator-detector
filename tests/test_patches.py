from __future__ import annotations

import numpy as np
import pytest

from defectgen.data.patches import (
    Padding,
    extract_normal_patch,
    extract_positive_patch,
    mask_centered_crop_coordinates,
    reflection_pad,
)


def test_reflection_padding() -> None:
    array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    padded = reflection_pad(array, Padding(left=1, top=1, right=1, bottom=1))
    assert padded.shape == (4, 4)
    assert np.array_equal(padded, np.array([[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]]))


def test_positive_crop_is_binary_synchronized_and_complete_when_possible() -> None:
    image = np.zeros((40, 20, 3), dtype=np.uint8)
    image[2:9, 0:4] = 200
    mask = np.zeros((40, 20), dtype=np.uint8)
    mask[2:9, 0:4] = 255
    patch = extract_positive_patch(image, mask, (24, 20), np.random.default_rng(7), context_margin=3)
    assert patch.image.shape == (20, 24, 3)
    assert patch.mask.shape == (20, 24)
    assert set(np.unique(patch.mask)) <= {False, True}
    assert np.count_nonzero(patch.mask) == np.count_nonzero(mask)
    assert patch.defect_fully_contained
    assert patch.context_fully_contained


def test_oversized_defect_reports_incomplete_but_remains_positive() -> None:
    image = np.zeros((30, 20), dtype=np.uint8)
    mask = np.zeros((30, 20), dtype=np.uint8)
    mask[2:28, 5:15] = 1
    patch = extract_positive_patch(image, mask, (16, 12), np.random.default_rng(3), context_margin=2)
    assert not patch.defect_fully_contained
    assert not patch.context_fully_contained
    assert np.count_nonzero(patch.mask) > 0


def test_positive_crop_rejects_empty_mask() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="zero mask"):
        extract_positive_patch(image, mask, (10, 10), np.random.default_rng(1))


def test_normal_crop_is_zero_and_deterministic() -> None:
    image = np.arange(18 * 10, dtype=np.uint16).reshape(18, 10)
    mask = np.zeros((18, 10), dtype=np.uint8)
    first = extract_normal_patch(image, mask, (16, 12), np.random.default_rng(42))
    second = extract_normal_patch(image, mask, (16, 12), np.random.default_rng(42))
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.coordinates == second.coordinates
    assert first.image.shape == first.mask.shape == (12, 16)
    assert np.count_nonzero(first.mask) == 0


def test_normal_crop_rejects_positive_mask() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1
    with pytest.raises(ValueError, match="completely zero"):
        extract_normal_patch(image, mask, (10, 10), np.random.default_rng(1))


def test_mask_centered_coordinates_are_deterministic() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:15, 12:18] = True
    first = mask_centered_crop_coordinates(mask, (16, 12), np.random.default_rng(99), 2)
    second = mask_centered_crop_coordinates(mask, (16, 12), np.random.default_rng(99), 2)
    assert first == second
    assert first[2:] == (True, True)

