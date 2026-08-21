from __future__ import annotations

import copy
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from defectgen.gan.geometry import ContactSides, ComponentWindow, connected_components
from defectgen.gan.manifest import gan_manifest_content_hash
from defectgen.gan.training_pairs import (
    GANInternalSplit,
    GANTrainingPairConfig,
    GANTrainingPairDataset,
    GANTrainingSample,
    create_internal_gan_split,
)


WIDTH = 256
HEIGHT = 512
REPO_ROOT = Path(__file__).resolve().parents[1]


class ArrayLoader:
    def __init__(self, arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.arrays = arrays

    def __call__(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        return self.arrays[row["sample_id"]]


def _row(sample_id: str, has_defect: bool) -> dict:
    return {
        "sample_id": sample_id,
        "official_split": "train",
        "development_split": "train",
        "image_path": f"{sample_id}.png",
        "mask_path": f"{sample_id}_mask.png" if has_defect else "",
        "has_defect": has_defect,
        "image_sha256": (sample_id * 64)[:64],
    }


def _template(row: dict, mask: np.ndarray, component_id: int) -> dict:
    component = connected_components(mask)[component_id]
    box = component.bounding_box
    window = ComponentWindow(
        top=0,
        left=0,
        width=WIDTH,
        height=HEIGHT,
        partial_component=False,
        coverage_fraction=1.0,
        positive_pixels=component.positive_pixels,
        source_contact_sides=component.contact_sides,
    )
    return {
        **row,
        "component_id": component_id,
        "window_index": 0,
        "source_mask_bounding_box": {
            "x_min": box.x_min,
            "y_min": box.y_min,
            "x_max": box.x_max,
            "y_max": box.y_max,
        },
        "source_window_coordinates": window.to_dict(),
        "partial_component": False,
        "coverage_fraction": 1.0,
        "positive_pixels": component.positive_pixels,
        "source_contact_sides": component.contact_sides.to_dict(),
        "touches_native_border": component.contact_sides.any,
    }


def _fixture() -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    templates: list[dict] = []
    masks: dict[str, np.ndarray] = {}
    non_border = np.zeros((HEIGHT, WIDTH), dtype=bool)
    non_border[120:130, 90:100] = True
    non_border[300:308, 160:170] = True
    masks["defect-a"] = non_border
    left = np.zeros_like(non_border)
    left[220:232, :8] = True
    masks["defect-b"] = left
    corner = np.zeros_like(non_border)
    corner[:10, :12] = True
    masks["defect-c"] = corner
    opposite = np.zeros_like(non_border)
    opposite[380:386, :] = True
    masks["defect-d"] = opposite
    for source_index, (sample_id, mask) in enumerate(masks.items()):
        image = np.full((HEIGHT, WIDTH, 3), 30 + 20 * source_index, dtype=np.uint8)
        image[mask] = [230, 40 + source_index, 20]
        arrays[sample_id] = (image, mask)
        row = _row(sample_id, True)
        for component_id in range(len(connected_components(mask))):
            templates.append(_template(row, mask, component_id))
    normals: list[dict] = []
    for index in range(8):
        sample_id = f"normal-{index}"
        image = np.full((HEIGHT, WIDTH, 3), 100 + index * 5, dtype=np.uint8)
        arrays[sample_id] = (image, np.zeros((HEIGHT, WIDTH), dtype=bool))
        normals.append(
            {
                **_row(sample_id, False),
                "native_width": WIDTH,
                "native_height": HEIGHT,
                "achievable_valid_fraction": 1.0,
                "available_window_count": 1,
            }
        )
    metadata = {
        "pipeline_version": "test_g1_3",
        "seed": 42,
        "patch": {
            "width": WIDTH,
            "height": HEIGHT,
            "minimum_positive_pixels": 8,
            "minimum_normal_valid_fraction": 1.0,
        },
        "transform": {
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "minimum_scale": 0.9,
            "maximum_scale": 1.1,
            "minimum_retained_area_fraction": 0.70,
            "feather_radius": 3,
            "non_border_native_margin": 4,
        },
        "colour_matching": {
            "enabled": False,
            "boundary_radius": 4,
            "minimum_gain": 0.8,
            "maximum_gain": 1.2,
            "maximum_absolute_offset": 0.15,
        },
        "sampling": {
            "border_fraction_mode": "empirical",
            "border_fraction": None,
            "feasible_transform_selection": "indexed_continuous_intervals",
        },
        "source_manifest_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "templates": templates,
        "normal_backgrounds": normals,
        "data_boundary": {
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "validation_predictions_loaded": 0,
        },
        "materialized_image_files": 0,
    }
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    return metadata, arrays


def _config(**overrides) -> GANTrainingPairConfig:
    values = {
        "data_bridge_version": "g1_3_gan_training_pairs_v1",
        "manifest_path": "reports/gan_inputs/manifest.json",
        "loss_config_path": "configs/gan_losses.json",
        "base_seed": 42,
        "monitor_fraction": 0.25,
        "image_height": HEIGHT,
        "image_width": WIDTH,
        "normalization_range": (-1, 1),
        "discriminator_mask_threshold": 0.5,
        "deterministic_monitoring": True,
        "audit_sample_count": 8,
    }
    values.update(overrides)
    return GANTrainingPairConfig(**values)


def _dataset(split: str = "train", length: int = 8):
    metadata, arrays = _fixture()
    internal = create_internal_gan_split(metadata, monitor_fraction=0.25, seed=42)
    dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        _config(),
        split=split,
        internal_split=internal,
        sample_loader=ArrayLoader(arrays),
        length=length,
    )
    return dataset, metadata, arrays, internal


def test_training_sample_contract_shapes_dtypes_ranges_and_sources() -> None:
    dataset, metadata, _, internal = _dataset()
    sample = dataset[0]
    assert {field.name for field in fields(GANTrainingSample)} == {
        "composite_image",
        "generator_mask",
        "transformed_defect_alpha",
        "fake_discriminator_mask",
        "real_image",
        "real_discriminator_mask",
        "fake_valid_mask",
        "real_valid_mask",
        "real_valid_coverage",
        "metadata",
    }
    for image in (sample.composite_image, sample.real_image):
        assert image.shape == (3, HEIGHT, WIDTH) and image.dtype == torch.float32
        assert torch.isfinite(image).all() and image.min() >= -1 and image.max() <= 1
    for mask in (
        sample.generator_mask,
        sample.transformed_defect_alpha,
        sample.fake_discriminator_mask,
        sample.real_discriminator_mask,
        sample.fake_valid_mask,
        sample.real_valid_mask,
        sample.real_valid_coverage,
    ):
        assert mask.shape == (1, HEIGHT, WIDTH) and mask.dtype == torch.float32
        assert torch.isfinite(mask).all() and mask.min() >= 0 and mask.max() <= 1
    assert set(sample.fake_discriminator_mask.unique().tolist()) <= {0.0, 1.0}
    assert bool(((sample.generator_mask > 0) & (sample.generator_mask < 1)).any())
    assert torch.equal(sample.fake_discriminator_mask, sample.real_discriminator_mask)
    assert bool(
        (
            sample.transformed_defect_alpha
            <= sample.real_valid_coverage + 1e-6
        ).all()
    )
    assert not bool(
        (
            sample.real_discriminator_mask.bool()
            & ~sample.real_valid_mask.bool()
        ).any()
    )
    assert sample.fake_discriminator_mask.any()
    assert sample.metadata["normal_background_sample_id"] in internal.train_background_ids
    assert sample.metadata["template_source_sample_id"] in internal.train_defect_source_ids
    assert any(
        row["sample_id"] == sample.metadata["template_source_sample_id"]
        and row["has_defect"]
        for row in metadata["templates"]
    )
    assert sample.metadata["development_split"] == sample.metadata["official_split"] == "train"
    assert sample.metadata["normal_background_sample_id"] in {
        row["sample_id"] for row in metadata["normal_backgrounds"]
    }


def test_real_and_fake_use_one_bit_exact_transform_record() -> None:
    dataset, _, _, _ = _dataset()
    sample = dataset[1]
    assert sample.metadata["real_transform"] == sample.metadata["fake_transform"]
    assert sample.metadata["real_transform"] == sample.metadata["transform"]
    assert sample.metadata["deterministic_sample_seed"] >= 0
    assert not bool(
        ((sample.generator_mask > 0) & ~sample.fake_valid_mask.bool()).any()
    )


def test_grouped_internal_split_is_disjoint_and_keeps_components_together() -> None:
    _, metadata, _, split = _dataset()
    split.assert_disjoint()
    assert not split.train_defect_source_ids & split.monitor_defect_source_ids
    assert not split.train_background_ids & split.monitor_background_ids
    assignment: dict[str, set[str]] = {}
    for name, indices in (
        ("train", split.train_template_indices),
        ("monitor", split.monitor_template_indices),
    ):
        for index in indices:
            assignment.setdefault(metadata["templates"][index]["sample_id"], set()).add(name)
    assert all(len(splits) == 1 for splits in assignment.values())
    assert split.representation_warnings


def test_forbidden_validation_or_official_test_rows_are_rejected() -> None:
    metadata, arrays = _fixture()
    metadata["templates"][0]["development_split"] = "validation"
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    with pytest.raises(ValueError, match="split leakage"):
        GANTrainingPairDataset(
            metadata,
            REPO_ROOT,
            _config(),
            split="train",
            sample_loader=ArrayLoader(arrays),
        )
    metadata, arrays = _fixture()
    metadata["normal_backgrounds"][0]["official_split"] = "test"
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    with pytest.raises(ValueError, match="split leakage"):
        GANTrainingPairDataset(
            metadata,
            REPO_ROOT,
            _config(),
            split="train",
            sample_loader=ArrayLoader(arrays),
        )


def _assert_samples_equal(first: GANTrainingSample, second: GANTrainingSample) -> None:
    for name in (
        "composite_image",
        "generator_mask",
        "transformed_defect_alpha",
        "fake_discriminator_mask",
        "real_image",
        "real_discriminator_mask",
        "fake_valid_mask",
        "real_valid_mask",
        "real_valid_coverage",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.metadata == second.metadata


def test_seed_split_epoch_index_replay_epoch_change_and_monitor_stability() -> None:
    train, metadata, arrays, internal = _dataset("train", length=2)
    replay = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        _config(),
        split="train",
        internal_split=internal,
        sample_loader=ArrayLoader(arrays),
        length=2,
    )
    _assert_samples_equal(train[0], replay[0])
    epoch_zero = train[0]
    train.set_epoch(1)
    epoch_one = train[0]
    assert epoch_zero.metadata["deterministic_sample_seed"] != epoch_one.metadata[
        "deterministic_sample_seed"
    ]
    assert not torch.equal(epoch_zero.composite_image, epoch_one.composite_image)
    reproducible = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        _config(),
        split="train",
        internal_split=internal,
        sample_loader=ArrayLoader(arrays),
        length=2,
    )
    reproducible.set_epoch(1)
    _assert_samples_equal(epoch_one, reproducible[0])
    monitor = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        _config(),
        split="monitor",
        internal_split=internal,
        sample_loader=ArrayLoader(arrays),
        length=1,
    )
    fixed = monitor[0]
    monitor.set_epoch(99)
    _assert_samples_equal(fixed, monitor[0])


def test_item_retrieval_does_not_change_global_rng_state() -> None:
    dataset, _, _, _ = _dataset(length=1)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    dataset[0]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_dataloader_worker_count_does_not_change_samples() -> None:
    dataset, _, _, _ = _dataset(length=2)
    sequential = list(DataLoader(dataset, batch_size=None, num_workers=0))
    worker = list(DataLoader(dataset, batch_size=None, num_workers=1))
    assert len(sequential) == len(worker) == 2
    for first, second in zip(sequential, worker):
        _assert_samples_equal(first, second)


@pytest.mark.parametrize(
    "source_id",
    ["defect-b", "defect-c", "defect-d"],
)
def test_border_corner_and_left_right_sources_remain_valid(source_id: str) -> None:
    metadata, arrays = _fixture()
    template_indices = tuple(
        index
        for index, template in enumerate(metadata["templates"])
        if template["sample_id"] == source_id
    )
    all_template_sources = frozenset(
        metadata["templates"][index]["sample_id"] for index in template_indices
    )
    normal_indices = tuple(range(len(metadata["normal_backgrounds"])))
    custom = GANInternalSplit(
        train_template_indices=template_indices,
        monitor_template_indices=(),
        train_normal_indices=normal_indices,
        monitor_normal_indices=(),
        train_defect_source_ids=all_template_sources,
        monitor_defect_source_ids=frozenset(),
        train_background_ids=frozenset(
            row["sample_id"] for row in metadata["normal_backgrounds"]
        ),
        monitor_background_ids=frozenset(),
        representation_warnings=(),
    )
    dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        _config(),
        split="train",
        internal_split=custom,
        sample_loader=ArrayLoader(arrays),
        length=1,
    )
    sample = dataset[0]
    assert sample.generator_mask.any() and sample.fake_discriminator_mask.any()
    assert torch.equal(sample.fake_discriminator_mask, sample.real_discriminator_mask)
    assert sample.metadata["template_source_sample_id"] == source_id
    bright_real_defect = sample.real_image[0] > 0
    canonical = sample.real_discriminator_mask[0].bool()
    assert bool(bright_real_defect[canonical].all())


def test_threshold_must_match_g1_2_configuration() -> None:
    metadata, arrays = _fixture()
    with pytest.raises(ValueError, match="disagrees with G1.2"):
        GANTrainingPairDataset(
            metadata,
            REPO_ROOT,
            _config(discriminator_mask_threshold=0.6),
            split="train",
            sample_loader=ArrayLoader(arrays),
        )
