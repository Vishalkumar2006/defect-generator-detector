"""Deterministic metadata index for physically compatible GAN placements."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from defectgen.data.geometry import BoundingBox, bounding_box

from .geometry import ContactSides
from .pipeline import sample_transform_parameters


@dataclass(frozen=True)
class TransformedTemplateGeometry:
    horizontal_flip: bool
    vertical_flip: bool
    scale: float
    transformed_contact_sides: ContactSides
    crop_width: int
    crop_height: int
    scaled_width: int
    scaled_height: int
    mask_bounding_box: BoundingBox
    positive_pixels: int
    retained_area_fraction: float

    @property
    def side_combination(self) -> str:
        active = [
            side
            for side in ("top", "bottom", "left", "right")
            if getattr(self.transformed_contact_sides, side)
        ]
        return "+".join(active) if active else "none"


@dataclass(frozen=True)
class CompatibilityPool:
    background_indices: tuple[int, ...]
    candidates_examined: int
    candidates_excluded: int
    exclusion_reasons: dict[str, int]


def measure_transformed_template_geometry(
    source_mask: np.ndarray,
    source_contact_sides: ContactSides,
    *,
    seed: int,
    transform_settings: dict[str, Any],
    colour_settings: dict[str, Any],
    minimum_positive_pixels: int,
) -> TransformedTemplateGeometry:
    """Reproduce mask transformation geometry without loading a background image."""
    source_mask = source_mask.astype(bool, copy=False)
    box = bounding_box(source_mask)
    if box is None:
        raise ValueError("A source defect template cannot have an empty mask")
    parameters = sample_transform_parameters(seed, transform_settings)
    context = max(
        int(transform_settings["feather_radius"]), int(colour_settings["boundary_radius"])
    )
    height, width = source_mask.shape
    crop_left = max(0, box.x_min - context)
    crop_right = min(width, box.x_max + context + 1)
    crop_top = max(0, box.y_min - context)
    crop_bottom = min(height, box.y_max + context + 1)
    mask_crop = torch.from_numpy(
        source_mask[crop_top:crop_bottom, crop_left:crop_right]
    ).float()
    scaled_height = max(1, int(round(mask_crop.shape[0] * parameters["scale"])))
    scaled_width = max(1, int(round(mask_crop.shape[1] * parameters["scale"])))
    resized_mask = F.interpolate(
        mask_crop[None, None], size=(scaled_height, scaled_width), mode="nearest"
    )[0, 0].bool()
    dimensions: list[int] = []
    if parameters["horizontal_flip"]:
        dimensions.append(1)
    if parameters["vertical_flip"]:
        dimensions.append(0)
    if dimensions:
        resized_mask = torch.flip(resized_mask, dimensions)
    positive_pixels = int(resized_mask.sum())
    expected_area = max(1.0, float(source_mask.sum()) * parameters["scale"] ** 2)
    retained_fraction = positive_pixels / expected_area
    if positive_pixels < minimum_positive_pixels:
        raise ValueError("transformation_below_minimum_positive_pixels")
    if retained_fraction < float(transform_settings["minimum_retained_area_fraction"]):
        raise ValueError("transformation_removed_too_much_defect_area")
    if scaled_height > height or scaled_width > width:
        raise ValueError("transformed_template_exceeds_patch")
    resized_box = bounding_box(resized_mask.numpy())
    assert resized_box is not None
    return TransformedTemplateGeometry(
        horizontal_flip=parameters["horizontal_flip"],
        vertical_flip=parameters["vertical_flip"],
        scale=parameters["scale"],
        transformed_contact_sides=source_contact_sides.transformed(
            horizontal_flip=parameters["horizontal_flip"],
            vertical_flip=parameters["vertical_flip"],
        ),
        crop_width=mask_crop.shape[1],
        crop_height=mask_crop.shape[0],
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        mask_bounding_box=resized_box,
        positive_pixels=positive_pixels,
        retained_area_fraction=retained_fraction,
    )


def _axis_bounds(
    *,
    mask_minimum: int,
    mask_maximum: int,
    valid_maximum: int,
    patch_extent: int,
    content_extent: int,
    contact_minimum: bool,
    contact_maximum: bool,
    margin: int,
) -> tuple[int, int] | None:
    lower = max(0, -mask_minimum)
    upper = min(patch_extent - content_extent, valid_maximum - mask_maximum)
    if contact_minimum:
        upper = min(upper, -mask_minimum)
    else:
        lower = max(lower, margin - mask_minimum)
    if contact_maximum:
        lower = max(lower, valid_maximum - mask_maximum)
        upper = min(upper, valid_maximum - mask_maximum)
    else:
        upper = min(upper, valid_maximum - margin - mask_maximum)
    return None if lower > upper else (lower, upper)


class GANPlacementCompatibilityIndex:
    """Group backgrounds by native geometry and return exact compatible pools."""

    def __init__(
        self,
        normal_backgrounds: list[dict[str, Any]],
        *,
        patch_size: tuple[int, int],
        non_border_margin: int,
        feather_radius: int,
    ) -> None:
        self.normal_backgrounds = normal_backgrounds
        self.patch_size = patch_size
        self.non_border_margin = int(non_border_margin)
        self.feather_radius = int(feather_radius)
        if self.non_border_margin < self.feather_radius:
            raise ValueError("non_border_native_margin_must_cover_feather_radius")
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, background in enumerate(normal_backgrounds):
            grouped[(int(background["native_height"]), int(background["native_width"]))].append(
                index
            )
        self._groups = {geometry: tuple(indices) for geometry, indices in grouped.items()}

    def _group_rejection_reason(
        self, native_shape: tuple[int, int], geometry: TransformedTemplateGeometry
    ) -> str | None:
        native_height, native_width = native_shape
        patch_width, patch_height = self.patch_size
        contacts = geometry.transformed_contact_sides
        if contacts.top and contacts.bottom and native_height > patch_height:
            return "target_window_cannot_contain_top_and_bottom_native_edges"
        if contacts.left and contacts.right and native_width > patch_width:
            return "target_window_cannot_contain_left_and_right_native_edges"
        valid_width = min(native_width, patch_width)
        valid_height = min(native_height, patch_height)
        if valid_width <= 0 or valid_height <= 0:
            return "empty_native_valid_region"
        horizontal = _axis_bounds(
            mask_minimum=geometry.mask_bounding_box.x_min,
            mask_maximum=geometry.mask_bounding_box.x_max,
            valid_maximum=valid_width - 1,
            patch_extent=patch_width,
            content_extent=geometry.scaled_width,
            contact_minimum=contacts.left,
            contact_maximum=contacts.right,
            margin=self.non_border_margin,
        )
        vertical = _axis_bounds(
            mask_minimum=geometry.mask_bounding_box.y_min,
            mask_maximum=geometry.mask_bounding_box.y_max,
            valid_maximum=valid_height - 1,
            patch_extent=patch_height,
            content_extent=geometry.scaled_height,
            contact_minimum=contacts.top,
            contact_maximum=contacts.bottom,
            margin=self.non_border_margin,
        )
        if horizontal is None or vertical is None:
            return (
                "incompatible_border_placement"
                if contacts.any
                else "no_non_border_placement_with_required_margin"
            )
        return None

    def query(self, geometry: TransformedTemplateGeometry) -> CompatibilityPool:
        compatible: list[int] = []
        exclusions: Counter[str] = Counter()
        for native_shape, indices in sorted(self._groups.items()):
            reason = self._group_rejection_reason(native_shape, geometry)
            if reason is None:
                compatible.extend(indices)
            else:
                exclusions[reason] += len(indices)
        compatible.sort()
        total = len(self.normal_backgrounds)
        return CompatibilityPool(
            background_indices=tuple(compatible),
            candidates_examined=total,
            candidates_excluded=total - len(compatible),
            exclusion_reasons=dict(sorted(exclusions.items())),
        )
