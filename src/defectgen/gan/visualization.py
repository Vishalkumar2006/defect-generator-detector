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
    terminal_failures = Counter(rejection_reasons)
    indexed_exclusions: Counter[str] = Counter()
    candidates_examined = 0
    candidates_excluded = 0
    transform_retries = 0
    placement_retries = 0
    empty_pools = 0
    attempts: list[int] = []
    side_combinations: Counter[str] = Counter()
    template_utilization: Counter[str] = Counter()
    background_utilization: Counter[str] = Counter()
    for sample in sample_list:
        diagnostic = sample["placement_diagnostics"]
        for side, active in diagnostic["successful_target_contact_sides"].items():
            target_counts[side] += int(bool(active))
        non_border += int(diagnostic["non_border_placement"])
        accidental += int(diagnostic["accidental_contact_violations"])
        support_outside += int(diagnostic["support_pixels_outside_valid_region"])
        candidates_examined += int(diagnostic.get("compatibility_candidates_examined", 0))
        candidates_excluded += int(diagnostic.get("compatibility_candidates_excluded", 0))
        indexed_exclusions.update(diagnostic.get("compatibility_exclusion_reasons", {}))
        transform_retries += int(diagnostic.get("actual_transform_placement_retries", 0))
        placement_retries += int(diagnostic.get("actual_placement_retries", 0))
        empty_pools += int(diagnostic.get("empty_compatibility_pools", 0))
        attempts.append(int(diagnostic.get("attempts_per_successful_sample", 1)))
        side_combinations[diagnostic.get("successful_side_combination", "none")] += 1
        template_utilization[diagnostic.get("template_identity", "unknown")] += 1
        background_utilization[diagnostic.get("background_identity", "unknown")] += 1
    attempt_array = np.asarray(attempts or [0], dtype=float)
    return {
        "successful_placements": len(sample_list),
        "successful_placements_by_target_contact_side": dict(target_counts),
        "successful_placements_by_side_combination": dict(sorted(side_combinations.items())),
        "candidates_examined_by_compatibility_index": candidates_examined,
        "candidates_excluded_by_compatibility_index": candidates_excluded,
        "compatibility_index_exclusions_by_reason": dict(sorted(indexed_exclusions.items())),
        "actual_transform_placement_retries": transform_retries,
        "actual_placement_retries": placement_retries,
        "empty_compatibility_pools": empty_pools,
        "terminal_visualization_search_failures": sum(terminal_failures.values()),
        "terminal_visualization_failure_reasons": dict(sorted(terminal_failures.items())),
        "attempts_per_successful_sample": {
            "mean": float(attempt_array.mean()),
            "p95": float(np.quantile(attempt_array, 0.95, method="linear")),
            "p99": float(np.quantile(attempt_array, 0.99, method="linear")),
            "maximum": int(attempt_array.max()),
        },
        "non_border_placements": non_border,
        "accidental_contact_violations": accidental,
        "support_pixels_outside_valid_region": support_outside,
        "unique_templates_used": len(template_utilization),
        "unique_backgrounds_used": len(background_utilization),
        "template_utilization": dict(sorted(template_utilization.items())),
        "background_utilization": dict(sorted(background_utilization.items())),
    }
