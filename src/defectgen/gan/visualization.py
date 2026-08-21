"""Metadata-only category selection and placement accounting for GAN contact sheets."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


CATEGORIES = ("all", "border", "non-border", "small-thin", "large", "narrow-background")


def _template_dimensions(template: dict[str, Any]) -> tuple[int, int]:
    box = template["source_mask_bounding_box"]
    return int(box["x_max"]) - int(box["x_min"]) + 1, int(box["y_max"]) - int(box["y_min"]) + 1


def select_visualization_members(metadata: dict[str, Any], category: str) -> dict[str, list[int]]:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown visualization category: {category}")
    templates = metadata["templates"]
    normals = metadata["normal_backgrounds"]
    template_indices = list(range(len(templates)))
    normal_indices = list(range(len(normals)))
    if category == "border":
        template_indices = [
            index for index, item in enumerate(templates) if any(item["source_contact_sides"].values())
        ]
    elif category == "non-border":
        template_indices = [
            index for index, item in enumerate(templates) if not any(item["source_contact_sides"].values())
        ]
    elif category in {"small-thin", "large"}:
        pixels = np.asarray([int(item["positive_pixels"]) for item in templates])
        widths, heights = zip(*(_template_dimensions(item) for item in templates))
        minimum_extents = np.minimum(widths, heights)
        maximum_extents = np.maximum(widths, heights)
        if category == "small-thin":
            pixel_limit = float(np.quantile(pixels, 0.25, method="linear"))
            thin_limit = float(np.quantile(minimum_extents, 0.25, method="linear"))
            template_indices = [
                index
                for index, item in enumerate(templates)
                if int(item["positive_pixels"]) <= pixel_limit
                or min(_template_dimensions(item)) <= thin_limit
            ]
        else:
            pixel_limit = float(np.quantile(pixels, 0.75, method="linear"))
            extent_limit = float(np.quantile(maximum_extents, 0.75, method="linear"))
            template_indices = [
                index
                for index, item in enumerate(templates)
                if int(item["positive_pixels"]) >= pixel_limit
                or max(_template_dimensions(item)) >= extent_limit
            ]
    elif category == "narrow-background":
        fractions = np.asarray(
            [float(item["achievable_valid_fraction"]) for item in normals], dtype=float
        )
        limit = float(np.quantile(fractions, 0.25, method="linear"))
        normal_indices = [
            index
            for index, item in enumerate(normals)
            if float(item["achievable_valid_fraction"]) <= limit
        ]
    if not template_indices or not normal_indices:
        raise ValueError(f"Visualization category has no eligible records: {category}")
    return {"template_indices": template_indices, "normal_indices": normal_indices}


def summarize_placements(
    samples: Iterable[dict[str, Any]], rejection_reasons: Iterable[str] = ()
) -> dict[str, Any]:
    sample_list = list(samples)
    target_counts = Counter({side: 0 for side in ("top", "bottom", "left", "right")})
    non_border = 0
    accidental = 0
    support_outside = 0
    rejections = Counter(rejection_reasons)
    for sample in sample_list:
        diagnostic = sample["placement_diagnostics"]
        for side, active in diagnostic["successful_target_contact_sides"].items():
            target_counts[side] += int(bool(active))
        non_border += int(diagnostic["non_border_placement"])
        accidental += int(diagnostic["accidental_contact_violations"])
        support_outside += int(diagnostic["support_pixels_outside_valid_region"])
        rejections.update(
            diagnostic.get("candidate_background_rejection_reasons_before_success", [])
        )
    return {
        "successful_placements": len(sample_list),
        "successful_placements_by_target_contact_side": dict(target_counts),
        "incompatible_border_placement_rejections": sum(
            count
            for reason, count in rejections.items()
            if "border" in reason
            or "native_" in reason
            or "target_window_cannot_contain" in reason
        ),
        "rejection_reasons": dict(sorted(rejections.items())),
        "non_border_placements": non_border,
        "accidental_contact_violations": accidental,
        "support_pixels_outside_valid_region": support_outside,
    }
