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


@dataclass(frozen=True)
class FeasibleTransformCandidate:
    """A continuous scale interval with invariant transformed mask geometry."""

    horizontal_flip: bool
    vertical_flip: bool
    minimum_scale: float
    maximum_scale: float
    selection_weight: float
    geometry: TransformedTemplateGeometry
    pool: CompatibilityPool


@dataclass(frozen=True)
class FeasibleTransformationPool:
    candidates: tuple[FeasibleTransformCandidate, ...]
    transform_states_examined: int
    transform_states_excluded: int
    candidates_examined: int
    candidates_excluded: int
    exclusion_reasons: dict[str, int]
    empty_pool_side_combinations: dict[str, int]


def measure_transformed_template_geometry(
    source_mask: np.ndarray,
    source_contact_sides: ContactSides,
    *,
    seed: int,
    transform_settings: dict[str, Any],
    colour_settings: dict[str, Any],
    minimum_positive_pixels: int,
    transform_parameters: dict[str, Any] | None = None,
    enforce_retained_area: bool = True,
) -> TransformedTemplateGeometry:
    """Reproduce mask transformation geometry without loading a background image."""
    source_mask = source_mask.astype(bool, copy=False)
    box = bounding_box(source_mask)
    if box is None:
        raise ValueError("A source defect template cannot have an empty mask")
    parameters = (
        sample_transform_parameters(seed, transform_settings)
        if transform_parameters is None
        else transform_parameters
    )
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
    if enforce_retained_area and retained_fraction < float(
        transform_settings["minimum_retained_area_fraction"]
    ):
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
    valid_minimum: int,
    valid_maximum: int,
    patch_extent: int,
    content_extent: int,
    contact_minimum: bool,
    contact_maximum: bool,
    margin: int,
) -> tuple[int, int] | None:
    lower = max(0, valid_minimum - mask_minimum)
    upper = min(patch_extent - content_extent, valid_maximum - mask_maximum)
    if contact_minimum:
        lower = max(lower, valid_minimum - mask_minimum)
        upper = min(upper, valid_minimum - mask_minimum)
    else:
        lower = max(lower, valid_minimum + margin - mask_minimum)
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
        valid_left = (patch_width - valid_width) // 2
        valid_top = (patch_height - valid_height) // 2
        horizontal = _axis_bounds(
            mask_minimum=geometry.mask_bounding_box.x_min,
            mask_maximum=geometry.mask_bounding_box.x_max,
            valid_minimum=valid_left,
            valid_maximum=valid_left + valid_width - 1,
            patch_extent=patch_width,
            content_extent=geometry.scaled_width,
            contact_minimum=contacts.left,
            contact_maximum=contacts.right,
            margin=self.non_border_margin,
        )
        vertical = _axis_bounds(
            mask_minimum=geometry.mask_bounding_box.y_min,
            mask_maximum=geometry.mask_bounding_box.y_max,
            valid_minimum=valid_top,
            valid_maximum=valid_top + valid_height - 1,
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

    def horizontal_symmetry_audit(
        self, geometry: TransformedTemplateGeometry
    ) -> dict[str, Any]:
        """Compare a state with the exact horizontally mirrored state."""
        box = geometry.mask_bounding_box
        mirrored = TransformedTemplateGeometry(
            horizontal_flip=not geometry.horizontal_flip,
            vertical_flip=geometry.vertical_flip,
            scale=geometry.scale,
            transformed_contact_sides=geometry.transformed_contact_sides.transformed(
                horizontal_flip=True, vertical_flip=False
            ),
            crop_width=geometry.crop_width,
            crop_height=geometry.crop_height,
            scaled_width=geometry.scaled_width,
            scaled_height=geometry.scaled_height,
            mask_bounding_box=BoundingBox(
                x_min=geometry.scaled_width - 1 - box.x_max,
                y_min=box.y_min,
                x_max=geometry.scaled_width - 1 - box.x_min,
                y_max=box.y_max,
            ),
            positive_pixels=geometry.positive_pixels,
            retained_area_fraction=geometry.retained_area_fraction,
        )
        original_pool = self.query(geometry)
        mirrored_pool = self.query(mirrored)
        return {
            "original_side_combination": geometry.side_combination,
            "mirrored_side_combination": mirrored.side_combination,
            "original_compatible_backgrounds": len(original_pool.background_indices),
            "mirrored_compatible_backgrounds": len(mirrored_pool.background_indices),
            "pool_size_difference": len(original_pool.background_indices)
            - len(mirrored_pool.background_indices),
            "availability_symmetric": bool(original_pool.background_indices)
            == bool(mirrored_pool.background_indices),
        }

    @staticmethod
    def _flip_options(probability: float) -> tuple[tuple[bool, float], ...]:
        if probability <= 0:
            return ((False, 1.0),)
        if probability >= 1:
            return ((True, 1.0),)
        return ((False, 1.0 - probability), (True, probability))

    @staticmethod
    def _scale_intervals(
        crop_width: int,
        crop_height: int,
        minimum_scale: float,
        maximum_scale: float,
    ) -> tuple[tuple[float, float], ...]:
        if minimum_scale == maximum_scale:
            return ((minimum_scale, maximum_scale),)
        boundaries = {minimum_scale, maximum_scale}
        for extent in (crop_width, crop_height):
            first = max(0, int(np.floor(minimum_scale * extent - 0.5)))
            last = int(np.ceil(maximum_scale * extent - 0.5))
            for output_extent in range(first, last + 1):
                boundary = (output_extent + 0.5) / extent
                if minimum_scale < boundary < maximum_scale:
                    boundaries.add(boundary)
        ordered = sorted(boundaries)
        return tuple(zip(ordered[:-1], ordered[1:]))

    def feasible_transformations(
        self,
        source_mask: np.ndarray,
        source_contact_sides: ContactSides,
        *,
        seed: int,
        transform_settings: dict[str, Any],
        colour_settings: dict[str, Any],
        minimum_positive_pixels: int,
    ) -> FeasibleTransformationPool:
        """Index all feasible flip/scale states before online selection.

        Scale intervals are split wherever rounded tensor dimensions change. A
        candidate is therefore sampled only from a continuous interval whose
        transformed mask geometry and compatible background pool are known.
        """
        source_mask = source_mask.astype(bool, copy=False)
        box = bounding_box(source_mask)
        if box is None:
            raise ValueError("A source defect template cannot have an empty mask")
        context = max(
            int(transform_settings["feather_radius"]),
            int(colour_settings["boundary_radius"]),
        )
        crop_width = min(source_mask.shape[1], box.x_max + context + 1) - max(
            0, box.x_min - context
        )
        crop_height = min(source_mask.shape[0], box.y_max + context + 1) - max(
            0, box.y_min - context
        )
        minimum_scale = float(transform_settings["minimum_scale"])
        maximum_scale = float(transform_settings["maximum_scale"])
        intervals = self._scale_intervals(
            crop_width, crop_height, minimum_scale, maximum_scale
        )
        horizontal_options = self._flip_options(
            float(transform_settings["horizontal_flip_probability"])
        )
        vertical_options = self._flip_options(
            float(transform_settings["vertical_flip_probability"])
        )
        candidates: list[FeasibleTransformCandidate] = []
        exclusions: Counter[str] = Counter()
        empty_sides: Counter[str] = Counter()
        states_examined = 0
        indexed_examined = 0
        indexed_excluded = 0
        scale_span = maximum_scale - minimum_scale
        for horizontal_flip, horizontal_weight in horizontal_options:
            for vertical_flip, vertical_weight in vertical_options:
                for lower, upper in intervals:
                    states_examined += 1
                    representative = lower if lower == upper else (lower + upper) / 2
                    parameters = {
                        "horizontal_flip": horizontal_flip,
                        "vertical_flip": vertical_flip,
                        "scale": representative,
                    }
                    try:
                        geometry = measure_transformed_template_geometry(
                            source_mask,
                            source_contact_sides,
                            seed=seed,
                            transform_settings=transform_settings,
                            colour_settings=colour_settings,
                            minimum_positive_pixels=minimum_positive_pixels,
                            transform_parameters=parameters,
                            enforce_retained_area=False,
                        )
                    except ValueError as error:
                        exclusions[str(error)] += 1
                        continue
                    # The nearest-neighbour mask is constant in this interval,
                    # while retained fraction decreases monotonically with scale.
                    retained_limit = np.sqrt(
                        geometry.positive_pixels
                        / (
                            max(1, int(source_mask.sum()))
                            * float(transform_settings["minimum_retained_area_fraction"])
                        )
                    )
                    feasible_upper = min(upper, float(retained_limit))
                    if lower != upper and feasible_upper <= lower:
                        exclusions["transformation_removed_too_much_defect_area"] += 1
                        continue
                    pool = self.query(geometry)
                    indexed_examined += pool.candidates_examined
                    indexed_excluded += pool.candidates_excluded
                    exclusions.update(pool.exclusion_reasons)
                    if not pool.background_indices:
                        empty_sides[geometry.side_combination] += 1
                        continue
                    interval_weight = (
                        1.0 if scale_span == 0 else max(0.0, feasible_upper - lower) / scale_span
                    )
                    weight = horizontal_weight * vertical_weight * interval_weight
                    if weight <= 0:
                        continue
                    candidates.append(
                        FeasibleTransformCandidate(
                            horizontal_flip=horizontal_flip,
                            vertical_flip=vertical_flip,
                            minimum_scale=lower,
                            maximum_scale=feasible_upper,
                            selection_weight=weight,
                            geometry=geometry,
                            pool=pool,
                        )
                    )
        return FeasibleTransformationPool(
            candidates=tuple(candidates),
            transform_states_examined=states_examined,
            transform_states_excluded=states_examined - len(candidates),
            candidates_examined=indexed_examined,
            candidates_excluded=indexed_excluded,
            exclusion_reasons=dict(sorted(exclusions.items())),
            empty_pool_side_combinations=dict(sorted(empty_sides.items())),
        )
