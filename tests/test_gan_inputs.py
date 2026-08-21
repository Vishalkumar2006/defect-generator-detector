from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from defectgen.gan.dataset import OnlineGANInputDataset
from defectgen.gan.geometry import (
    ComponentWindow,
    connected_components,
    extract_native_window,
    plan_component_windows,
)
from defectgen.gan.manifest import assert_gan_training_rows, build_gan_input_metadata
from defectgen.gan.normalization import binary_mask_tensor, gan_rgb_to_uint8, rgb_to_gan
from defectgen.gan.pipeline import REQUIRED_PROVENANCE_FIELDS, construct_coarse_gan_input


PATCH_WIDTH = 256
PATCH_HEIGHT = 512


def _training_row(sample_id: str, has_defect: bool) -> dict:
    return {
        "sample_id": sample_id,
        "official_split": "train",
        "development_split": "train",
        "image_path": f"{sample_id}.png",
        "mask_path": f"{sample_id}_mask.png" if has_defect else "",
        "has_defect": has_defect,
        "image_sha256": sample_id * 8,
    }


def _transform_settings(**overrides) -> dict:
    settings = {
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5,
        "minimum_scale": 0.9,
        "maximum_scale": 1.1,
        "minimum_retained_area_fraction": 0.70,
        "placement_attempts": 128,
        "feather_radius": 3,
    }
    settings.update(overrides)
    return settings


def _colour_settings(enabled: bool = False) -> dict:
    return {
        "enabled": enabled,
        "boundary_radius": 4,
        "minimum_gain": 0.8,
        "maximum_gain": 1.2,
        "maximum_absolute_offset": 0.15,
    }


def _provenance_base(width: int, height: int) -> dict:
    return {
        "normal_background_sample_id": "normal",
        "source_defect_sample_id": "defect",
        "connected_component_id": 0,
        "source_mask_bounding_box": {"x_min": 8, "y_min": 9, "x_max": 15, "y_max": 18},
        "source_window_coordinates": {"top": 0, "left": 0, "width": width, "height": height},
        "target_window_coordinates": {"top": 0, "left": 0, "width": width, "height": height},
        "partial_component": False,
        "coverage_fraction": 1.0,
        "touches_native_border": False,
        "minimum_positive_pixels": 4,
        "manifest_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "pipeline_version": "test_v1",
    }


def test_split_leakage_guard_rejects_validation_and_official_test() -> None:
    for official, development in (("train", "validation"), ("test", "test")):
        row = _training_row("forbidden", True)
        row["official_split"] = official
        row["development_split"] = development
        with pytest.raises(ValueError, match="split leakage"):
            assert_gan_training_rows([row])


def test_exact_native_window_geometry_and_padding_validity() -> None:
    image = np.arange(10 * 8 * 3, dtype=np.uint8).reshape(10, 8, 3)
    mask = np.zeros((10, 8), dtype=bool)
    mask[8:, 6:] = True
    patch, patch_mask, valid = extract_native_window(image, mask, (7, 5, 5, 6))
    assert patch.shape == (6, 5, 3)
    assert patch_mask.shape == valid.shape == (6, 5)
    assert np.array_equal(patch[:3, :3], image[7:10, 5:8])
    assert valid.sum() == 9
    assert not valid[3:, :].any() and not valid[:, 3:].any()
    assert not patch_mask[~valid].any()


def test_long_component_uses_overlapping_partial_windows_with_full_coverage() -> None:
    mask = np.zeros((600, 720), dtype=bool)
    mask[280:300, 20:700] = True
    component = connected_components(mask)[0]
    plan = plan_component_windows(
        component,
        mask.shape,
        patch_size=(PATCH_WIDTH, PATCH_HEIGHT),
        context_margin=24,
        overlap_fraction=0.5,
        minimum_positive_pixels=8,
        minimum_component_coverage=0.05,
    )
    assert len(plan.windows) > 1
    assert all(window.partial_component for window in plan.windows)
    covered = np.zeros_like(mask)
    for window in plan.windows:
        covered[
            window.top : window.top + window.height,
            window.left : window.left + window.width,
        ] |= component.mask[
            window.top : window.top + window.height,
            window.left : window.left + window.width,
        ]
        assert window.coverage_fraction >= 0.05
    assert np.array_equal(covered, component.mask)


def test_border_touching_and_minimum_positive_pixel_rules() -> None:
    border_mask = np.zeros((PATCH_HEIGHT, 400), dtype=bool)
    border_mask[0:4, 0:5] = True
    border = connected_components(border_mask)[0]
    plan = plan_component_windows(border, border_mask.shape, minimum_positive_pixels=8)
    assert plan.windows and all(window.touches_native_border for window in plan.windows)

    tiny_mask = np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=bool)
    tiny_mask[20:22, 20:22] = True
    tiny = connected_components(tiny_mask)[0]
    rejected = plan_component_windows(tiny, tiny_mask.shape, minimum_positive_pixels=8)
    assert not rejected.windows
    assert rejected.rejected_reasons == ("component_below_minimum_positive_pixels",)


def test_gan_rgb_range_round_trip_and_binary_mask_contract() -> None:
    image = np.array([[[0, 127, 255], [10, 20, 30]]], dtype=np.uint8)
    normalized = rgb_to_gan(image)
    assert normalized.dtype == torch.float32
    assert normalized.min() >= -1 and normalized.max() <= 1
    assert torch.equal(gan_rgb_to_uint8(normalized), torch.from_numpy(image))
    mask = binary_mask_tensor(np.array([[0, 1]], dtype=np.uint8))
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError, match="binary"):
        binary_mask_tensor(np.array([[0.0, 0.5]], dtype=np.float32))


def test_transform_preserves_binary_alignment_and_background_outside_support() -> None:
    height, width = 80, 64
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[:] = [20, 30, 40]
    mask = np.zeros((height, width), dtype=bool)
    mask[9:19, 8:16] = True
    source[mask] = [240, 20, 10]
    background = np.full_like(source, 110)
    sample = construct_coarse_gan_input(
        source,
        mask,
        background,
        np.ones(mask.shape, dtype=bool),
        seed=19,
        transform_settings=_transform_settings(minimum_scale=1.1, maximum_scale=1.1),
        colour_settings=_colour_settings(),
        provenance_base=_provenance_base(width, height),
    )
    condition = sample["conditioning_mask"].bool()
    support = sample["support_mask"].bool()
    assert set(sample["conditioning_mask"].unique().tolist()) <= {0.0, 1.0}
    assert condition.sum() >= 4
    assert torch.all(support[condition])
    transformed_rgb = gan_rgb_to_uint8(sample["source_template"])
    assert bool(
        (transformed_rgb[..., 0][condition[0]] > transformed_rgb[..., 1][condition[0]]).all()
    )
    assert torch.equal(
        sample["coarse_composite"].expand_as(sample["normal_background"])[~support.expand(3, -1, -1)],
        sample["normal_background"][~support.expand(3, -1, -1)],
    )
    assert bool((sample["difference_from_background"][:, condition[0]] > 0).any())
    assert REQUIRED_PROVENANCE_FIELDS <= set(sample["provenance"])


def _dataset_fixture() -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    defect = _training_row("defect", True)
    normal = _training_row("normal", False)
    image = np.full((PATCH_HEIGHT, PATCH_WIDTH, 3), 35, dtype=np.uint8)
    mask = np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=bool)
    mask[220:240, 100:120] = True
    image[mask] = [230, 60, 30]
    component = connected_components(mask)[0]
    window = ComponentWindow(0, 0, PATCH_WIDTH, PATCH_HEIGHT, False, 1.0, 400, False)
    template = {
        **defect,
        "component_id": 0,
        "window_index": 0,
        "source_mask_bounding_box": {"x_min": 100, "y_min": 220, "x_max": 119, "y_max": 239},
        "source_window_coordinates": window.to_dict(),
        "partial_component": False,
        "coverage_fraction": 1.0,
        "positive_pixels": component.positive_pixels,
        "touches_native_border": False,
    }
    normal_image = np.full((PATCH_HEIGHT, PATCH_WIDTH, 3), 125, dtype=np.uint8)
    metadata = {
        "pipeline_version": "test_online_v1",
        "seed": 42,
        "patch": {
            "width": PATCH_WIDTH,
            "height": PATCH_HEIGHT,
            "minimum_positive_pixels": 8,
            "minimum_normal_valid_fraction": 1.0,
        },
        "transform": _transform_settings(),
        "colour_matching": _colour_settings(),
        "manifest_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "templates": [template],
        "normal_backgrounds": [{**normal, "available_window_count": 1}],
        "data_boundary": {
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "validation_predictions_loaded": 0,
        },
        "materialized_image_files": 0,
    }
    arrays = {
        "defect": (image, mask),
        "normal": (normal_image, np.zeros(mask.shape, dtype=bool)),
    }
    return metadata, arrays


def test_online_dataset_is_exactly_deterministic_and_uses_training_normals_only(tmp_path: Path) -> None:
    metadata, arrays = _dataset_fixture()
    loaded: list[dict] = []

    def loader(row: dict) -> tuple[np.ndarray, np.ndarray]:
        loaded.append(row)
        return arrays[row["sample_id"]]

    first = OnlineGANInputDataset(metadata, tmp_path, base_seed=42, length=1, sample_loader=loader)[0]
    second = OnlineGANInputDataset(metadata, tmp_path, base_seed=42, length=1, sample_loader=loader)[0]
    assert first["normal_background"].shape == (3, PATCH_HEIGHT, PATCH_WIDTH)
    assert first["conditioning_mask"].shape == (1, PATCH_HEIGHT, PATCH_WIDTH)
    assert first["provenance"] == second["provenance"]
    for field in (
        "normal_background",
        "source_template",
        "conditioning_mask",
        "feathered_support",
        "valid_region",
        "coarse_composite",
    ):
        assert torch.equal(first[field], second[field])
    assert all(row["official_split"] == row["development_split"] == "train" for row in loaded)
    assert all(not row["has_defect"] for row in loaded if row["sample_id"] == "normal")
    assert list(tmp_path.iterdir()) == []


def test_different_dataset_seeds_change_generated_sample() -> None:
    metadata, arrays = _dataset_fixture()
    loader = lambda row: arrays[row["sample_id"]]
    first = OnlineGANInputDataset(metadata, Path("."), base_seed=1, length=1, sample_loader=loader)[0]
    second = OnlineGANInputDataset(metadata, Path("."), base_seed=2, length=1, sample_loader=loader)[0]
    assert first["provenance"]["generated_sample_seed"] != second["provenance"]["generated_sample_seed"]
    assert not torch.equal(first["conditioning_mask"], second["conditioning_mask"])


def test_metadata_builder_ignores_forbidden_row_files_and_materializes_nothing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    normal = np.full((PATCH_HEIGHT, PATCH_WIDTH, 3), 100, dtype=np.uint8)
    defect = normal.copy()
    mask = np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=np.uint8)
    mask[200:210, 100:112] = 255
    Image.fromarray(normal).save(data / "normal.png")
    Image.fromarray(defect).save(data / "defect.png")
    Image.fromarray(mask).save(data / "defect_mask.png")
    manifest = data / "split.csv"
    columns = [
        "sample_id",
        "official_split",
        "development_split",
        "image_path",
        "mask_path",
        "has_defect",
        "image_sha256",
    ]
    rows = [
        {**_training_row("normal", False), "image_path": "data/normal.png"},
        {
            **_training_row("defect", True),
            "image_path": "data/defect.png",
            "mask_path": "data/defect_mask.png",
        },
        {
            **_training_row("validation", True),
            "development_split": "validation",
            "image_path": "missing_validation_image.png",
            "mask_path": "missing_validation_mask.png",
        },
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    configuration = {
        "pipeline_version": "test_manifest_v1",
        "seed": 42,
        "patch": {
            "width": PATCH_WIDTH,
            "height": PATCH_HEIGHT,
            "context_margin": 24,
            "overlap_fraction": 0.5,
            "minimum_positive_pixels": 8,
            "minimum_component_coverage": 0.05,
            "minimum_normal_valid_fraction": 0.9,
        },
        "template_transform": _transform_settings(),
        "colour_matching": _colour_settings(),
        "data": {"development_manifest": "data/split.csv"},
    }
    metadata, summary = build_gan_input_metadata(tmp_path, configuration)
    assert len(metadata["templates"]) == len(metadata["normal_backgrounds"]) == 1
    assert summary["validation_rows_loaded"] == summary["official_test_rows_loaded"] == 0
    assert metadata["materialized_image_files"] == summary["materialized_image_files"] == 0
    assert not (tmp_path / "reports").exists()
