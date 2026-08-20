"""Deterministic validation-example selection for cross-candidate diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _pick_ranked(rows: list[dict[str, Any]], key, used: set[str]) -> str:
    for row in sorted(rows, key=key):
        if row["sample_id"] not in used:
            used.add(row["sample_id"])
            return row["sample_id"]
    # Very small synthetic fixtures may not contain enough unique examples.
    return sorted(rows, key=key)[0]["sample_id"]


def select_fixed_validation_ids(
    geometry_rows: Iterable[Mapping[str, Any]],
    candidate_per_image_rows: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, str]:
    """Select the same six validation IDs for every candidate.

    Defect-size examples come only from ground truth. Failure examples use the
    mean result across candidates, avoiding candidate-specific cherry-picking.
    """
    defective = [
        dict(row)
        for row in geometry_rows
        if row["development_split"] == "validation" and float(row["mask_pixels"]) > 0
    ]
    if not defective:
        raise ValueError("No defective validation geometry rows supplied")
    defective.sort(key=lambda row: (float(row["mask_pixels"]), row["sample_id"]))
    used: set[str] = set()
    tiny = _pick_ranked(defective, lambda row: (float(row["mask_pixels"]), row["sample_id"]), used)
    median_pixels = float(defective[(len(defective) - 1) // 2]["mask_pixels"])
    median = _pick_ranked(
        defective,
        lambda row: (abs(float(row["mask_pixels"]) - median_pixels), row["sample_id"]),
        used,
    )
    large = _pick_ranked(defective, lambda row: (-float(row["mask_pixels"]), row["sample_id"]), used)
    border_rows = [row for row in defective if _as_bool(row["touches_border"])]
    if not border_rows:
        raise ValueError("No border-touching validation defect supplied")
    border = _pick_ranked(
        border_rows,
        lambda row: (abs(float(row["mask_pixels"]) - median_pixels), row["sample_id"]),
        used,
    )

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, bool] = {}
    for rows in candidate_per_image_rows.values():
        for row in rows:
            sample_id = str(row["sample_id"])
            labels[sample_id] = bool(int(row["has_defect"]))
            values[sample_id]["predicted_pixels"].append(float(row["predicted_pixels"]))
            values[sample_id]["dice"].append(float(row["dice"]))
    if not values:
        raise ValueError("No candidate per-image metrics supplied")
    normal_ids = [sample_id for sample_id, label in labels.items() if not label]
    defect_ids = [sample_id for sample_id, label in labels.items() if label]
    if not normal_ids or not defect_ids:
        raise ValueError("Candidate metrics must include normal and defective validation images")
    largest_normal_fp = min(
        normal_ids,
        key=lambda sample_id: (-mean(values[sample_id]["predicted_pixels"]), sample_id),
    )
    lowest_defect_dice = min(
        defect_ids,
        key=lambda sample_id: (mean(values[sample_id]["dice"]), sample_id),
    )
    return {
        "tiny_defect": tiny,
        "median_defect": median,
        "large_defect": large,
        "border_touching_defect": border,
        "normal_largest_false_positive_area": largest_normal_fp,
        "defective_lowest_dice": lowest_defect_dice,
    }
