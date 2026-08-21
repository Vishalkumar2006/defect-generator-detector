from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from defectgen.gan.compatibility import (
    GANPlacementCompatibilityIndex,
    measure_transformed_template_geometry,
)
from defectgen.gan.dataset import OnlineGANInputDataset
from defectgen.gan.geometry import (
    ContactSides,
    ComponentWindow,
    connected_components,
    extract_native_window,
    plan_component_windows,
)
from defectgen.gan.manifest import (
    achievable_valid_fraction,
    assert_gan_training_rows,
    build_gan_input_metadata,
    gan_manifest_content_hash,
    source_metadata_hashes,
)
from defectgen.gan.normalization import binary_mask_tensor, gan_rgb_to_uint8, rgb_to_gan
from defectgen.gan.pipeline import REQUIRED_PROVENANCE_FIELDS, construct_coarse_gan_input
from defectgen.gan.dataset import GANSamplingFailure, select_target_window
from defectgen.gan.sampling_audit import build_sampling_audit_summary
from defectgen.gan.visualization import select_visualization_members, summarize_placements


PATCH_WIDTH = 256
PATCH_HEIGHT = 512
REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "feather_radius": 3,
        "non_border_native_margin": 4,
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
        "source_contact_sides": ContactSides().to_dict(),
        "target_window_native_contact_sides": ContactSides(True, True, True, True).to_dict(),
        "minimum_positive_pixels": 4,
        "source_manifest_sha256": "a" * 64,
        "gan_manifest_content_sha256": "e" * 64,
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
    assert np.array_equal(patch[1:4, 1:4], image[7:10, 5:8])
    assert valid.sum() == 9
    assert not valid[:1, :].any() and not valid[4:, :].any()
    assert not valid[:, :1].any() and not valid[:, 4:].any()
    assert not patch_mask[~valid].any()


def test_narrow_native_normal_remains_eligible_without_counting_padding() -> None:
    fraction = achievable_valid_fraction((602, 184), (PATCH_WIDTH, PATCH_HEIGHT))
    assert fraction == 0.71875
    assert fraction >= 0.71875
    assert achievable_valid_fraction((512, 100), (PATCH_WIDTH, PATCH_HEIGHT)) < 0.71875


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


@pytest.mark.parametrize(
    ("mask_slice", "expected"),
    [
        ((slice(0, 3), slice(10, 16)), ContactSides(top=True)),
        ((slice(29, 32), slice(10, 16)), ContactSides(bottom=True)),
        ((slice(10, 16), slice(0, 3)), ContactSides(left=True)),
        ((slice(10, 16), slice(29, 32)), ContactSides(right=True)),
        ((slice(0, 3), slice(0, 3)), ContactSides(top=True, left=True)),
    ],
)
def test_connected_components_record_explicit_contact_sides(mask_slice, expected) -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[mask_slice] = True
    component = connected_components(mask)[0]
    assert component.contact_sides == expected
    assert component.touches_native_border == expected.any


def test_contact_sides_swap_under_flips_and_preserve_multiple_contacts() -> None:
    contacts = ContactSides(top=True, left=True)
    assert contacts.transformed(horizontal_flip=True, vertical_flip=False) == ContactSides(
        top=True, right=True
    )
    assert contacts.transformed(horizontal_flip=False, vertical_flip=True) == ContactSides(
        bottom=True, left=True
    )
    assert contacts.transformed(horizontal_flip=True, vertical_flip=True) == ContactSides(
        bottom=True, right=True
    )


def _construct_contact_sample(
    contacts: ContactSides,
    *,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    scale: float = 1.0,
    valid_width: int = 64,
) -> dict:
    height = width = 64
    mask = np.zeros((height, width), dtype=bool)
    top = 0 if contacts.top else (58 if contacts.bottom else 24)
    left = 0 if contacts.left else (58 if contacts.right else 24)
    mask[top : top + 6, left : left + 6] = True
    source = np.full((height, width, 3), 30, dtype=np.uint8)
    source[mask] = [230, 40, 20]
    background = np.full_like(source, 110)
    valid = np.zeros((height, width), dtype=bool)
    valid[:, :valid_width] = True
    provenance = _provenance_base(width, height)
    provenance["source_contact_sides"] = contacts.to_dict()
    provenance["touches_native_border"] = contacts.any
    provenance["target_window_native_contact_sides"] = ContactSides(
        top=True, bottom=True, left=True, right=True
    ).to_dict()
    return construct_coarse_gan_input(
        source,
        mask,
        background,
        valid,
        seed=42,
        transform_settings=_transform_settings(
            horizontal_flip_probability=float(horizontal_flip),
            vertical_flip_probability=float(vertical_flip),
            minimum_scale=scale,
            maximum_scale=scale,
            feather_radius=2,
        ),
        colour_settings=_colour_settings(),
        provenance_base=provenance,
    )


@pytest.mark.parametrize(
    "contacts",
    [
        ContactSides(top=True),
        ContactSides(bottom=True),
        ContactSides(left=True),
        ContactSides(right=True),
        ContactSides(top=True, left=True),
    ],
)
def test_border_templates_preserve_target_contacts_and_never_move_to_interior(contacts) -> None:
    sample = _construct_contact_sample(contacts, scale=1.1)
    assert sample["provenance"]["target_contact_sides"] == contacts.to_dict()
    assert sample["provenance"]["transformed_source_contact_sides"] == contacts.to_dict()
    assert sample["placement_diagnostics"]["accidental_contact_violations"] == 0


def test_flipped_border_template_contacts_the_swapped_target_side() -> None:
    sample = _construct_contact_sample(
        ContactSides(top=True, left=True), horizontal_flip=True, vertical_flip=True
    )
    expected = ContactSides(bottom=True, right=True).to_dict()
    assert sample["provenance"]["transformed_source_contact_sides"] == expected
    assert sample["provenance"]["target_contact_sides"] == expected


def test_target_window_selection_contains_required_native_edges() -> None:
    rng = np.random.default_rng(42)
    window, sides = select_target_window(
        (700, 300), (PATCH_WIDTH, PATCH_HEIGHT), ContactSides(top=True, left=True), rng
    )
    assert window[:2] == (0, 0)
    assert sides.top and sides.left and not sides.bottom and not sides.right
    window, sides = select_target_window(
        (700, 300), (PATCH_WIDTH, PATCH_HEIGHT), ContactSides(bottom=True, right=True), rng
    )
    assert window[:2] == (188, 44)
    assert sides.bottom and sides.right and not sides.top and not sides.left
    with pytest.raises(ValueError, match="top_and_bottom"):
        select_target_window(
            (700, 300),
            (PATCH_WIDTH, PATCH_HEIGHT),
            ContactSides(top=True, bottom=True),
            rng,
        )


def test_non_border_template_retains_configured_native_margin() -> None:
    sample = _construct_contact_sample(ContactSides())
    coordinates = torch.nonzero(sample["conditioning_mask"][0], as_tuple=False)
    assert int(coordinates[:, 0].min()) >= 4
    assert int(coordinates[:, 1].min()) >= 4
    assert int(coordinates[:, 0].max()) <= 64 - 4 - 1
    assert int(coordinates[:, 1].max()) <= 64 - 4 - 1
    assert sample["provenance"]["target_contact_sides"] == ContactSides().to_dict()


def test_incompatible_border_template_is_rejected_instead_of_moved() -> None:
    height = width = 64
    mask = np.zeros((height, width), dtype=bool)
    mask[30:34, :] = True
    source = np.full((height, width, 3), 50, dtype=np.uint8)
    background = np.full_like(source, 100)
    valid = np.zeros((height, width), dtype=bool)
    valid[:, :40] = True
    provenance = _provenance_base(width, height)
    contacts = ContactSides(left=True, right=True)
    provenance["source_contact_sides"] = contacts.to_dict()
    provenance["touches_native_border"] = True
    with pytest.raises(ValueError, match="incompatible_border_placement"):
        construct_coarse_gan_input(
            source,
            mask,
            background,
            valid,
            seed=42,
            transform_settings=_transform_settings(
                horizontal_flip_probability=0.0,
                vertical_flip_probability=0.0,
                minimum_scale=1.0,
                maximum_scale=1.0,
            ),
            colour_settings=_colour_settings(),
            provenance_base=provenance,
        )


def _indexed_background(sample_id: str, height: int, width: int) -> dict:
    return {
        **_training_row(sample_id, False),
        "native_height": height,
        "native_width": width,
        "achievable_valid_fraction": min(1.0, height / 64) * min(1.0, width / 64),
        "available_window_count": 1,
    }


def _measured_geometry(mask: np.ndarray, contacts: ContactSides, **transform_overrides):
    settings = _transform_settings(
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        minimum_scale=1.0,
        maximum_scale=1.0,
        feather_radius=2,
    )
    settings.update(transform_overrides)
    return measure_transformed_template_geometry(
        mask,
        contacts,
        seed=42,
        transform_settings=settings,
        colour_settings=_colour_settings(),
        minimum_positive_pixels=4,
    )


@pytest.mark.parametrize(
    "contacts",
    [
        ContactSides(top=True),
        ContactSides(bottom=True),
        ContactSides(left=True),
        ContactSides(right=True),
        ContactSides(top=True, left=True),
        ContactSides(bottom=True, right=True),
    ],
)
def test_compatibility_index_handles_sides_and_corners(contacts) -> None:
    mask = np.zeros((64, 64), dtype=bool)
    top = 0 if contacts.top else (58 if contacts.bottom else 24)
    left = 0 if contacts.left else (58 if contacts.right else 24)
    mask[top : top + 6, left : left + 6] = True
    index = GANPlacementCompatibilityIndex(
        [
            _indexed_background("exact", 64, 64),
            _indexed_background("larger", 80, 80),
            _indexed_background("narrow", 64, 40),
        ],
        patch_size=(64, 64),
        non_border_margin=4,
        feather_radius=2,
    )
    pool = index.query(_measured_geometry(mask, contacts))
    assert pool.background_indices
    assert pool.candidates_examined == 3


def test_compatibility_index_handles_opposite_and_multiple_side_contacts() -> None:
    horizontal_mask = np.zeros((64, 64), dtype=bool)
    horizontal_mask[30:34, :] = True
    vertical_mask = np.zeros((64, 64), dtype=bool)
    vertical_mask[:, 30:34] = True
    index = GANPlacementCompatibilityIndex(
        [
            _indexed_background("exact", 64, 64),
            _indexed_background("wide", 64, 80),
            _indexed_background("tall", 80, 64),
            _indexed_background("narrow", 64, 40),
        ],
        patch_size=(64, 64),
        non_border_margin=4,
        feather_radius=2,
    )
    horizontal = index.query(
        _measured_geometry(horizontal_mask, ContactSides(left=True, right=True))
    )
    vertical = index.query(
        _measured_geometry(vertical_mask, ContactSides(top=True, bottom=True))
    )
    assert horizontal.background_indices == (0, 2)
    assert vertical.background_indices == (0, 1, 3)
    all_sides = np.ones((64, 64), dtype=bool)
    pool = index.query(
        _measured_geometry(
            all_sides, ContactSides(top=True, bottom=True, left=True, right=True)
        )
    )
    assert pool.background_indices == (0,)


def test_compatibility_index_uses_post_flip_sides_and_narrow_geometry() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:30, 0:8] = True
    geometry = _measured_geometry(
        mask, ContactSides(left=True), horizontal_flip_probability=1.0
    )
    assert geometry.transformed_contact_sides == ContactSides(right=True)
    index = GANPlacementCompatibilityIndex(
        [_indexed_background("narrow", 64, 40)],
        patch_size=(64, 64),
        non_border_margin=4,
        feather_radius=2,
    )
    assert index.query(geometry).background_indices == (0,)


def _centered_narrow_contact_patch(
    contacts: ContactSides,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((64, 40, 3), 80, dtype=np.uint8)
    mask = np.zeros((64, 40), dtype=bool)
    if contacts.left and contacts.right:
        top = 0 if contacts.top else (58 if contacts.bottom else 29)
        mask[top : top + 6, :] = True
    else:
        top = 0 if contacts.top else (58 if contacts.bottom else 29)
        left = 0 if contacts.left else (34 if contacts.right else 17)
        mask[top : top + 6, left : left + 6] = True
    image[mask] = [220, 30, 20]
    return extract_native_window(image, mask, (0, 0, 64, 64))


@pytest.mark.parametrize(
    ("source_contacts", "horizontal_flip", "expected_contacts"),
    [
        (ContactSides(left=True), False, ContactSides(left=True)),
        (ContactSides(right=True), False, ContactSides(right=True)),
        (ContactSides(right=True), True, ContactSides(left=True)),
        (ContactSides(top=True, left=True), False, ContactSides(top=True, left=True)),
        (ContactSides(top=True, right=True), False, ContactSides(top=True, right=True)),
        (ContactSides(bottom=True, left=True), False, ContactSides(bottom=True, left=True)),
        (ContactSides(bottom=True, right=True), False, ContactSides(bottom=True, right=True)),
        (ContactSides(left=True, right=True), True, ContactSides(left=True, right=True)),
    ],
)
def test_centered_narrow_valid_edges_are_symmetric_compatibility_targets(
    source_contacts, horizontal_flip, expected_contacts
) -> None:
    _, mask, valid = _centered_narrow_contact_patch(source_contacts)
    assert not valid[:, :12].any() and valid[:, 12:52].all() and not valid[:, 52:].any()
    geometry = _measured_geometry(
        mask,
        source_contacts,
        horizontal_flip_probability=float(horizontal_flip),
    )
    assert geometry.transformed_contact_sides == expected_contacts
    index = GANPlacementCompatibilityIndex(
        [_indexed_background("narrow", 64, 40)],
        patch_size=(64, 64),
        non_border_margin=4,
        feather_radius=2,
    )
    assert index.query(geometry).background_indices == (0,)
    symmetry = index.horizontal_symmetry_audit(geometry)
    assert symmetry["availability_symmetric"]
    assert symmetry["pool_size_difference"] == 0


def test_feasible_scale_index_keeps_left_right_contacts_without_impossible_retries() -> None:
    _, mask, _ = _centered_narrow_contact_patch(ContactSides(left=True, right=True))
    index = GANPlacementCompatibilityIndex(
        [_indexed_background("narrow", 64, 40)],
        patch_size=(64, 64),
        non_border_margin=4,
        feather_radius=2,
    )
    feasible = index.feasible_transformations(
        mask,
        ContactSides(left=True, right=True),
        seed=91,
        transform_settings=_transform_settings(
            minimum_scale=0.8,
            maximum_scale=1.2,
            feather_radius=2,
        ),
        colour_settings=_colour_settings(),
        minimum_positive_pixels=4,
    )
    assert feasible.candidates
    assert feasible.transform_states_excluded > 0
    assert all(candidate.geometry.mask_bounding_box.width == 40 for candidate in feasible.candidates)
    assert all(candidate.pool.background_indices == (0,) for candidate in feasible.candidates)


@pytest.mark.parametrize(
    ("contacts", "horizontal_flip"),
    [
        (ContactSides(left=True), False),
        (ContactSides(right=True), False),
        (ContactSides(right=True), True),
        (ContactSides(top=True, left=True), False),
        (ContactSides(bottom=True, right=True), True),
        (ContactSides(left=True, right=True), False),
    ],
)
def test_pipeline_contacts_centered_native_edges_not_tensor_edges(
    contacts, horizontal_flip
) -> None:
    source, mask, _ = _centered_narrow_contact_patch(contacts)
    background_native = np.full((64, 40, 3), 120, dtype=np.uint8)
    background, _, valid = extract_native_window(
        background_native, np.zeros((64, 40), dtype=bool), (0, 0, 64, 64)
    )
    provenance = _provenance_base(64, 64)
    provenance["source_contact_sides"] = contacts.to_dict()
    provenance["touches_native_border"] = True
    sample = construct_coarse_gan_input(
        source,
        mask,
        background,
        valid,
        seed=17,
        transform_settings=_transform_settings(
            horizontal_flip_probability=float(horizontal_flip),
            vertical_flip_probability=0.0,
            minimum_scale=1.0,
            maximum_scale=1.0,
        ),
        colour_settings=_colour_settings(),
        provenance_base=provenance,
        transform_parameters={
            "horizontal_flip": horizontal_flip,
            "vertical_flip": False,
            "scale": 1.0,
        },
    )
    expected = contacts.transformed(
        horizontal_flip=horizontal_flip, vertical_flip=False
    )
    assert sample["provenance"]["target_contact_sides"] == expected.to_dict()
    condition = sample["conditioning_mask"][0].bool().numpy()
    box = np.argwhere(condition)
    if expected.left:
        assert int(box[:, 1].min()) == 12
    if expected.right:
        assert int(box[:, 1].max()) == 51
    assert not condition[:, :12].any() and not condition[:, 52:].any()
    assert sample["placement_diagnostics"]["support_pixels_outside_valid_region"] == 0


def test_compatibility_index_reports_empty_pool_without_runtime_pairing() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, :] = True
    index = GANPlacementCompatibilityIndex(
        [_indexed_background("narrow", 64, 40), _indexed_background("wide", 64, 80)],
        patch_size=(64, 64),
        non_border_margin=4,
        feather_radius=2,
    )
    pool = index.query(_measured_geometry(mask, ContactSides(left=True, right=True)))
    assert pool.background_indices == ()
    assert pool.candidates_excluded == pool.candidates_examined == 2
    assert sum(pool.exclusion_reasons.values()) == 2


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
    window = ComponentWindow(
        0, 0, PATCH_WIDTH, PATCH_HEIGHT, False, 1.0, 400, ContactSides()
    )
    template = {
        **defect,
        "component_id": 0,
        "window_index": 0,
        "source_mask_bounding_box": {"x_min": 100, "y_min": 220, "x_max": 119, "y_max": 239},
        "source_window_coordinates": window.to_dict(),
        "partial_component": False,
        "coverage_fraction": 1.0,
        "positive_pixels": component.positive_pixels,
        "source_contact_sides": ContactSides().to_dict(),
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
        "source_manifest_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "templates": [template],
        "normal_backgrounds": [
            {
                **normal,
                "native_width": PATCH_WIDTH,
                "native_height": PATCH_HEIGHT,
                "achievable_valid_fraction": 1.0,
                "available_window_count": 1,
            }
        ],
        "data_boundary": {
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "validation_predictions_loaded": 0,
        },
        "materialized_image_files": 0,
    }
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
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


def test_narrow_background_propagates_valid_region_and_blocks_padded_columns() -> None:
    metadata, arrays = _dataset_fixture()
    native_width = 184
    arrays["normal"] = (
        arrays["normal"][0][:, :native_width].copy(),
        arrays["normal"][1][:, :native_width].copy(),
    )
    metadata["patch"]["minimum_normal_valid_fraction"] = 0.71875
    metadata["normal_backgrounds"][0]["native_width"] = native_width
    metadata["normal_backgrounds"][0]["achievable_valid_fraction"] = 0.71875
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    loader = lambda row: arrays[row["sample_id"]]
    sample = OnlineGANInputDataset(
        metadata, Path("."), base_seed=42, length=1, sample_loader=loader
    )[0]
    valid = sample["valid_region"].bool()
    condition = sample["conditioning_mask"].bool()
    support = sample["support_mask"].bool()
    feather = sample["feathered_support"] > 0
    valid_left = (PATCH_WIDTH - native_width) // 2
    valid_right = valid_left + native_width
    assert bool(valid[:, :, valid_left:valid_right].all())
    assert not bool(valid[:, :, :valid_left].any())
    assert not bool(valid[:, :, valid_right:].any())
    assert not bool(condition[~valid].any())
    assert not bool(support[~valid].any())
    assert not bool(feather[~valid].any())
    outside_support = ~support.expand(3, -1, -1)
    assert torch.equal(
        sample["coarse_composite"][outside_support],
        sample["normal_background"][outside_support],
    )


def test_selected_threshold_includes_at_least_95_percent_of_audited_normals() -> None:
    audit_path = REPO_ROOT / "reports" / "gan_input_design" / "normal_valid_fraction_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    threshold = float(audit["selected_minimum_valid_fraction"])
    histogram = audit["native_width"]["histogram"]
    fractions = [
        min(1.0, int(width) / PATCH_WIDTH)
        for width, count in histogram.items()
        for _ in range(int(count))
    ]
    accepted = sum(fraction >= threshold for fraction in fractions)
    assert len(fractions) == audit["normal_training_images"] == 1772
    assert accepted == audit["selected_threshold_expected_accepted"]
    assert accepted / len(fractions) >= 0.95


def test_canonical_gan_manifest_hash_is_stable_and_content_sensitive() -> None:
    metadata, _ = _dataset_fixture()
    expected = metadata["gan_manifest_content_sha256"]
    assert gan_manifest_content_hash(metadata) == expected
    assert gan_manifest_content_hash(copy.deepcopy(metadata)) == expected
    changed = copy.deepcopy(metadata)
    changed["pipeline_version"] = "meaningfully_changed"
    assert gan_manifest_content_hash(changed) != expected
    changed = copy.deepcopy(metadata)
    changed["normal_backgrounds"][0]["available_window_count"] = 99
    assert gan_manifest_content_hash(changed) != expected
    self_hash_only = copy.deepcopy(metadata)
    self_hash_only["gan_manifest_content_sha256"] = "0" * 64
    assert gan_manifest_content_hash(self_hash_only) == expected


def test_online_dataset_rejects_manifest_content_hash_mismatch() -> None:
    metadata, _ = _dataset_fixture()
    metadata["transform"]["maximum_scale"] = 1.09
    with pytest.raises(ValueError, match="content hash mismatch"):
        OnlineGANInputDataset(metadata, Path("."), length=1, sample_loader=lambda row: None)


def test_source_manifest_hash_has_explicit_name_and_split_definition() -> None:
    rows = [_training_row("normal", False), _training_row("defect", True)]
    source_hash, split_hash = source_metadata_hashes(rows)
    metadata, _ = _dataset_fixture()
    assert "source_manifest_sha256" in metadata and "manifest_sha256" not in metadata
    assert len(source_hash) == len(split_hash) == 64
    changed_path = copy.deepcopy(rows)
    changed_path[0]["image_path"] = "different.png"
    changed_source, unchanged_split = source_metadata_hashes(changed_path)
    assert changed_source != source_hash
    assert unchanged_split == split_hash
    changed_label = copy.deepcopy(rows)
    changed_label[0]["has_defect"] = True
    _, changed_split = source_metadata_hashes(changed_label)
    assert changed_split != split_hash


def test_category_aware_visualization_selection() -> None:
    metadata, _ = _dataset_fixture()
    base = metadata["templates"][0]
    border = copy.deepcopy(base)
    border["source_contact_sides"] = ContactSides(left=True).to_dict()
    border["touches_native_border"] = True
    border["positive_pixels"] = 20
    border["source_mask_bounding_box"] = {
        "x_min": 0,
        "y_min": 10,
        "x_max": 1,
        "y_max": 19,
    }
    large = copy.deepcopy(base)
    large["positive_pixels"] = 10_000
    large["source_mask_bounding_box"] = {
        "x_min": 10,
        "y_min": 10,
        "x_max": 209,
        "y_max": 309,
    }
    metadata["templates"] = [base, border, large]
    narrow = copy.deepcopy(metadata["normal_backgrounds"][0])
    narrow["achievable_valid_fraction"] = 0.71875
    wide = copy.deepcopy(narrow)
    wide["sample_id"] = "wide-normal"
    wide["achievable_valid_fraction"] = 0.94
    metadata["normal_backgrounds"] = [narrow, wide]
    assert select_visualization_members(metadata, "border")["template_indices"] == [1]
    assert select_visualization_members(metadata, "non-border")["template_indices"] == [0, 2]
    assert 1 in select_visualization_members(metadata, "small-thin")["template_indices"]
    assert 2 in select_visualization_members(metadata, "large")["template_indices"]
    assert select_visualization_members(metadata, "narrow-background")["normal_indices"] == [0]


def test_placement_accounting_includes_successes_and_compatibility_rejections() -> None:
    sample = {
        "placement_diagnostics": {
            "successful_target_contact_sides": ContactSides(left=True).to_dict(),
            "non_border_placement": False,
            "accidental_contact_violations": 0,
            "support_pixels_outside_valid_region": 0,
            "compatibility_candidates_examined": 10,
            "compatibility_candidates_excluded": 7,
            "compatibility_exclusion_reasons": {"incompatible_border_placement": 7},
            "actual_transform_placement_retries": 0,
            "actual_placement_retries": 0,
            "empty_compatibility_pools": 0,
            "attempts_per_successful_sample": 1,
            "successful_side_combination": "left",
            "template_identity": "defect:0:0",
            "background_identity": "normal",
        }
    }
    accounting = summarize_placements(
        [sample], ["target_window_missing_required_native_top_edge"]
    )
    assert accounting["successful_placements_by_target_contact_side"]["left"] == 1
    assert accounting["candidates_excluded_by_compatibility_index"] == 7
    assert accounting["compatibility_index_exclusions_by_reason"] == {
        "incompatible_border_placement": 7
    }
    assert accounting["terminal_visualization_search_failures"] == 1
    assert accounting["actual_placement_retries"] == 0
    assert accounting["non_border_placements"] == 0
    assert accounting["accidental_contact_violations"] == 0
    assert accounting["support_pixels_outside_valid_region"] == 0


def test_different_dataset_seeds_change_generated_sample() -> None:
    metadata, arrays = _dataset_fixture()
    loader = lambda row: arrays[row["sample_id"]]
    first = OnlineGANInputDataset(metadata, Path("."), base_seed=1, length=1, sample_loader=loader)[0]
    second = OnlineGANInputDataset(metadata, Path("."), base_seed=2, length=1, sample_loader=loader)[0]
    assert first["provenance"]["generated_sample_seed"] != second["provenance"]["generated_sample_seed"]
    assert not torch.equal(first["conditioning_mask"], second["conditioning_mask"])


def test_online_sampling_uses_explicit_configurable_border_distribution() -> None:
    metadata, _ = _dataset_fixture()
    border = copy.deepcopy(metadata["templates"][0])
    border["sample_id"] = "border-defect"
    border["source_contact_sides"] = ContactSides(left=True).to_dict()
    border["touches_native_border"] = True
    border["source_window_coordinates"]["source_contact_sides"] = ContactSides(
        left=True
    ).to_dict()
    border["source_window_coordinates"]["touches_native_border"] = True
    metadata["templates"].append(border)
    metadata["sampling"] = {
        "border_fraction_mode": "fixed",
        "border_fraction": 1.0,
        "maximum_transform_attempts": 2,
    }
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    dataset = OnlineGANInputDataset(metadata, Path("."), length=1, sample_loader=lambda row: None)
    selected, selected_class, target = dataset._select_template(np.random.default_rng(42))
    assert selected is border
    assert selected_class == "border" and target == 1.0
    metadata["sampling"]["border_fraction"] = 0.0
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    dataset = OnlineGANInputDataset(metadata, Path("."), length=1, sample_loader=lambda row: None)
    _, selected_class, target = dataset._select_template(np.random.default_rng(42))
    assert selected_class == "non-border" and target == 0.0


def test_online_dataset_reports_empty_compatibility_pool_without_background_attempts() -> None:
    metadata, arrays = _dataset_fixture()
    mask = np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=bool)
    mask[250:254, :] = True
    image = np.full((PATCH_HEIGHT, PATCH_WIDTH, 3), 30, dtype=np.uint8)
    image[mask] = [230, 30, 20]
    arrays["defect"] = (image, mask)
    arrays["normal"] = (
        arrays["normal"][0][:, :184].copy(),
        arrays["normal"][1][:, :184].copy(),
    )
    contacts = ContactSides(left=True, right=True)
    template = metadata["templates"][0]
    template["source_contact_sides"] = contacts.to_dict()
    template["touches_native_border"] = True
    template["positive_pixels"] = int(mask.sum())
    template["source_mask_bounding_box"] = {
        "x_min": 0,
        "y_min": 250,
        "x_max": PATCH_WIDTH - 1,
        "y_max": 253,
    }
    template["source_window_coordinates"]["positive_pixels"] = int(mask.sum())
    template["source_window_coordinates"]["source_contact_sides"] = contacts.to_dict()
    template["source_window_coordinates"]["touches_native_border"] = True
    metadata["normal_backgrounds"][0]["native_width"] = 184
    metadata["normal_backgrounds"][0]["achievable_valid_fraction"] = 0.71875
    metadata["patch"]["minimum_normal_valid_fraction"] = 0.71875
    metadata["sampling"] = {
        "border_fraction_mode": "empirical",
        "border_fraction": None,
        "maximum_transform_attempts": 2,
    }
    metadata["gan_manifest_content_sha256"] = gan_manifest_content_hash(metadata)
    loaded: list[str] = []

    def loader(row: dict):
        loaded.append(row["sample_id"])
        return arrays[row["sample_id"]]

    dataset = OnlineGANInputDataset(metadata, Path("."), length=1, sample_loader=loader)
    with pytest.raises(GANSamplingFailure) as captured:
        dataset[0]
    assert captured.value.accounting["empty_compatibility_pools"] == 0
    assert captured.value.accounting["transform_states_excluded"] > 0
    assert captured.value.accounting["compatibility_candidates_excluded"] > 0
    assert "left+right" in captured.value.accounting["failure_side_combinations"]
    assert loaded == ["defect"]


def test_sampling_audit_summary_reports_efficiency_utilization_and_drift() -> None:
    metadata, arrays = _dataset_fixture()
    loader = lambda row: arrays[row["sample_id"]]
    sample = OnlineGANInputDataset(
        metadata, Path("."), base_seed=42, length=1, sample_loader=loader
    )[0]
    summary = build_sampling_audit_summary(
        metadata=metadata,
        requested_samples=1,
        samples=[sample],
        failures=[],
        elapsed_seconds=0.5,
    )
    assert summary["success_rate"] == 1.0
    assert summary["attempts_per_requested_sample"]["maximum"] >= 1
    assert summary["actual_placement_retries"] == 0
    assert summary["template_utilization"]["unique_used"] == 1
    assert summary["background_utilization"]["unique_used"] == 1
    assert summary["support_pixels_outside_valid_region"] == 0
    assert summary["accidental_contact_violations"] == 0
    assert set(summary["target_side_symmetry"]["expected_counts"]) == {
        "top",
        "bottom",
        "left",
        "right",
    }
    assert summary["target_side_symmetry"]["horizontal_counterpart_comparisons"] == 1
    assert summary["target_side_symmetry"]["availability_asymmetries"] == 0
    assert summary["sampled_templates_with_no_feasible_transform_or_pool"] == []


def test_metadata_compatibility_audit_is_training_only_and_symmetric() -> None:
    metadata, arrays = _dataset_fixture()
    loader = lambda row: arrays[row["sample_id"]]
    dataset = OnlineGANInputDataset(metadata, Path("."), length=1, sample_loader=loader)
    audit = dataset.audit_metadata_compatibility()
    assert audit["templates_audited"] == 1
    assert audit["horizontal_counterpart_comparisons"] > 0
    assert audit["horizontal_availability_asymmetries"] == []
    assert audit["templates_with_no_feasible_transform_or_pool"] == []
    assert audit["validation_rows_loaded"] == audit["official_test_rows_loaded"] == 0
    assert audit["materialized_generated_images"] == 0


def test_metadata_builder_ignores_forbidden_row_files_and_materializes_nothing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    normal = np.full((PATCH_HEIGHT, 184, 3), 100, dtype=np.uint8)
    rejected_normal = np.full((PATCH_HEIGHT, 100, 3), 100, dtype=np.uint8)
    defect = np.full((PATCH_HEIGHT, PATCH_WIDTH, 3), 100, dtype=np.uint8)
    mask = np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=np.uint8)
    mask[200:210, 100:112] = 255
    mask[10, 10] = 255
    Image.fromarray(normal).save(data / "normal.png")
    Image.fromarray(rejected_normal).save(data / "rejected_normal.png")
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
            **_training_row("rejected-normal", False),
            "image_path": "data/rejected_normal.png",
        },
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
            "minimum_normal_valid_fraction": 0.71875,
        },
        "template_transform": _transform_settings(),
        "colour_matching": _colour_settings(),
        "sampling": {
            "border_fraction_mode": "empirical",
            "border_fraction": None,
            "maximum_transform_attempts": 4,
        },
        "data": {"development_manifest": "data/split.csv"},
    }
    metadata, summary = build_gan_input_metadata(tmp_path, configuration)
    assert len(metadata["templates"]) == len(metadata["normal_backgrounds"]) == 1
    assert len(metadata["rejected_defect_components"]) == 1
    assert len(metadata["rejected_normal_backgrounds"]) == 1
    assert "rejected" not in metadata
    assert "source_manifest_sha256" in metadata and "manifest_sha256" not in metadata
    assert metadata["gan_manifest_content_sha256"] == gan_manifest_content_hash(metadata)
    assert summary["total_defective_training_images"] == 1
    assert summary["connected_components_found"] == 2
    assert summary["accepted_defect_components"] == 1
    assert summary["rejected_defect_components"] == 1
    assert summary["defect_rejection_reasons"] == {
        "component_below_minimum_positive_pixels": 1
    }
    assert summary["total_normal_training_images"] == 2
    assert summary["accepted_normal_background_images"] == 1
    assert summary["rejected_normal_background_images"] == 1
    assert summary["normal_rejection_reasons"] == {
        "normal_below_minimum_valid_fraction": 1
    }
    assert summary["templates_by_source_contact_side"]["none"] == 1
    assert summary["validation_rows_loaded"] == summary["official_test_rows_loaded"] == 0
    assert metadata["materialized_image_files"] == summary["materialized_image_files"] == 0
    assert not (tmp_path / "reports").exists()
