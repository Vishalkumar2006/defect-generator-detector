"""Metadata-only GAN input manifest construction with hard train-split guards."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from defectgen.data.ksdd2 import interpret_binary_mask

from .geometry import connected_components, plan_component_windows


REQUIRED_COLUMNS = {
    "sample_id",
    "official_split",
    "development_split",
    "image_path",
    "mask_path",
    "has_defect",
    "image_sha256",
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def assert_gan_training_rows(rows: list[dict[str, Any]]) -> None:
    leaked = [
        str(row.get("sample_id", "<unknown>"))
        for row in rows
        if row.get("development_split") != "train" or row.get("official_split") != "train"
    ]
    if leaked:
        raise ValueError(f"GAN pipeline split leakage detected: {leaked[:8]}")


def load_gan_training_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Read only development-training row content from the split CSV."""
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"GAN source manifest is missing columns: {sorted(missing)}")
        rows = [
            {
                **raw,
                "has_defect": _parse_bool(raw["has_defect"]),
            }
            for raw in reader
            if raw["development_split"] == "train"
        ]
    assert_gan_training_rows(rows)
    if not rows or {bool(row["has_defect"]) for row in rows} != {False, True}:
        raise ValueError("GAN training rows must contain normal and defective development-training samples")
    return rows


def training_manifest_hashes(rows: list[dict[str, Any]]) -> tuple[str, str]:
    assert_gan_training_rows(rows)
    canonical_rows = [
        {
            key: row[key]
            for key in (
                "sample_id",
                "official_split",
                "development_split",
                "image_path",
                "mask_path",
                "has_defect",
                "image_sha256",
            )
        }
        for row in rows
    ]
    manifest_bytes = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    split_bytes = json.dumps(
        [(row["sample_id"], bool(row["has_defect"])) for row in rows], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(manifest_bytes).hexdigest(), hashlib.sha256(split_bytes).hexdigest()


def _load_pair(repo_root: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(repo_root / row["image_path"]) as source:
        image = np.asarray(source.convert("RGB"))
    if row["mask_path"]:
        with Image.open(repo_root / row["mask_path"]) as source:
            mask = interpret_binary_mask(np.asarray(source))
    else:
        mask = np.zeros(image.shape[:2], dtype=bool)
    if mask.shape != image.shape[:2] or bool(mask.any()) != bool(row["has_defect"]):
        raise ValueError(f"Image/mask label mismatch for {row['sample_id']}")
    return image, mask


def _distribution(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": min(values),
        "median": float(np.median(values)),
        "maximum": max(values),
    }


def _valid_fraction_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            name: None
            for name in ("minimum", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "maximum")
        }
    fractions = np.asarray(values, dtype=float)
    return {
        "minimum": float(fractions.min()),
        "p01": float(np.quantile(fractions, 0.01, method="linear")),
        "p05": float(np.quantile(fractions, 0.05, method="linear")),
        "p10": float(np.quantile(fractions, 0.10, method="linear")),
        "p25": float(np.quantile(fractions, 0.25, method="linear")),
        "median": float(np.quantile(fractions, 0.50, method="linear")),
        "p75": float(np.quantile(fractions, 0.75, method="linear")),
        "p90": float(np.quantile(fractions, 0.90, method="linear")),
        "p95": float(np.quantile(fractions, 0.95, method="linear")),
        "p99": float(np.quantile(fractions, 0.99, method="linear")),
        "maximum": float(fractions.max()),
    }


def achievable_valid_fraction(
    image_shape: tuple[int, int], patch_size: tuple[int, int]
) -> float:
    """Maximum native-pixel fraction in one fixed-size patch without resizing."""
    height, width = image_shape
    patch_width, patch_height = patch_size
    if min(height, width, patch_width, patch_height) <= 0:
        raise ValueError("Image and patch dimensions must be positive")
    return min(1.0, height / patch_height) * min(1.0, width / patch_width)


def _normal_window_count(shape: tuple[int, int], patch_size: tuple[int, int], overlap: float) -> int:
    height, width = shape
    patch_width, patch_height = patch_size
    def starts(length: int, patch: int) -> list[int]:
        if length <= patch:
            return [0]
        step = max(1, int(round(patch * (1 - overlap))))
        result = list(range(0, length - patch + 1, step))
        if result[-1] != length - patch:
            result.append(length - patch)
        return result

    x_starts = starts(width, patch_width)
    y_starts = starts(height, patch_height)
    return len(x_starts) * len(y_starts)


def build_gan_input_metadata(repo_root: Path, configuration: dict[str, Any]):
    manifest_path = repo_root / configuration["data"]["development_manifest"]
    rows = load_gan_training_rows(manifest_path)
    manifest_sha256, split_sha256 = training_manifest_hashes(rows)
    patch = configuration["patch"]
    patch_size = (int(patch["width"]), int(patch["height"]))
    templates: list[dict[str, Any]] = []
    normals: list[dict[str, Any]] = []
    rejected_defect_components: list[dict[str, Any]] = []
    rejected_normal_backgrounds: list[dict[str, Any]] = []
    accepted_component_keys: set[tuple[str, int]] = set()
    component_dimensions: list[tuple[int, int]] = []
    components_requiring_overlapping_windows = 0
    normal_valid_fractions: list[float] = []
    total_defective_training_images = sum(bool(row["has_defect"]) for row in rows)
    total_normal_training_images = len(rows) - total_defective_training_images
    for row in rows:
        image, mask = _load_pair(repo_root, row)
        if not row["has_defect"]:
            valid_fraction = achievable_valid_fraction(image.shape[:2], patch_size)
            normal_valid_fractions.append(valid_fraction)
            if valid_fraction < float(patch["minimum_normal_valid_fraction"]):
                rejected_normal_backgrounds.append(
                    {
                        "sample_id": row["sample_id"],
                        "reason": "normal_below_minimum_valid_fraction",
                        "native_width": image.shape[1],
                        "native_height": image.shape[0],
                        "achievable_valid_fraction": valid_fraction,
                    }
                )
                continue
            normals.append(
                {
                    **row,
                    "native_width": image.shape[1],
                    "native_height": image.shape[0],
                    "achievable_valid_fraction": valid_fraction,
                    "available_window_count": _normal_window_count(
                        image.shape[:2], patch_size, float(patch["overlap_fraction"])
                    ),
                }
            )
            continue
        for component in connected_components(mask):
            box = component.bounding_box
            component_dimensions.append((box.width, box.height))
            plan = plan_component_windows(
                component,
                mask.shape,
                patch_size=patch_size,
                context_margin=int(patch["context_margin"]),
                overlap_fraction=float(patch["overlap_fraction"]),
                minimum_positive_pixels=int(patch["minimum_positive_pixels"]),
                minimum_component_coverage=float(patch["minimum_component_coverage"]),
            )
            if not plan.windows:
                rejected_defect_components.append(
                    {
                        "sample_id": row["sample_id"],
                        "component_id": component.component_id,
                        "component_width": box.width,
                        "component_height": box.height,
                        "positive_pixels": component.positive_pixels,
                        "reasons": list(plan.rejected_reasons),
                    }
                )
                continue
            accepted_component_keys.add((row["sample_id"], component.component_id))
            if len(plan.windows) > 1:
                components_requiring_overlapping_windows += 1
            for window_index, window in enumerate(plan.windows):
                templates.append(
                    {
                        **row,
                        "component_id": component.component_id,
                        "window_index": window_index,
                        "source_mask_bounding_box": {
                            "x_min": box.x_min,
                            "y_min": box.y_min,
                            "x_max": box.x_max,
                            "y_max": box.y_max,
                        },
                        "source_window_coordinates": window.to_dict(),
                        "partial_component": window.partial_component,
                        "coverage_fraction": window.coverage_fraction,
                        "positive_pixels": window.positive_pixels,
                        "touches_native_border": window.touches_native_border,
                    }
                )
    if not templates or not normals:
        raise ValueError("GAN input metadata requires usable defect templates and normal backgrounds")
    metadata = {
        "pipeline_version": configuration["pipeline_version"],
        "seed": int(configuration["seed"]),
        "patch": patch,
        "transform": configuration["template_transform"],
        "colour_matching": configuration["colour_matching"],
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "templates": templates,
        "normal_backgrounds": normals,
        "rejected_defect_components": rejected_defect_components,
        "rejected_normal_backgrounds": rejected_normal_backgrounds,
        "data_boundary": {
            "development_training_rows": len(rows),
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "validation_predictions_loaded": 0,
        },
        "materialized_image_files": 0,
    }
    positive_pixels = [item["positive_pixels"] for item in templates]
    coverage = [item["coverage_fraction"] for item in templates]
    defect_reason_counts: dict[str, int] = {}
    for item in rejected_defect_components:
        for reason in item["reasons"]:
            defect_reason_counts[reason] = defect_reason_counts.get(reason, 0) + 1
    normal_reason_counts: dict[str, int] = {}
    for item in rejected_normal_backgrounds:
        reason = item["reason"]
        normal_reason_counts[reason] = normal_reason_counts.get(reason, 0) + 1
    accepted_normal_fraction = len(normals) / total_normal_training_images
    summary = {
        "pipeline_version": configuration["pipeline_version"],
        "total_defective_training_images": total_defective_training_images,
        "connected_components_found": len(component_dimensions),
        "accepted_defect_components": len(accepted_component_keys),
        "rejected_defect_components": len(rejected_defect_components),
        "defect_rejection_reasons": defect_reason_counts,
        "components_requiring_overlapping_windows": components_requiring_overlapping_windows,
        "template_windows": len(templates),
        "full_template_windows": sum(not item["partial_component"] for item in templates),
        "partial_template_windows": sum(item["partial_component"] for item in templates),
        "border_touching_template_windows": sum(
            item["touches_native_border"] for item in templates
        ),
        "maximum_component_width": max(width for width, _ in component_dimensions),
        "maximum_component_height": max(height for _, height in component_dimensions),
        "total_normal_training_images": total_normal_training_images,
        "accepted_normal_background_images": len(normals),
        "rejected_normal_background_images": len(rejected_normal_backgrounds),
        "normal_rejection_reasons": normal_reason_counts,
        "normal_background_patch_availability": sum(
            item["available_window_count"] for item in normals
        ),
        "minimum_normal_valid_fraction": float(patch["minimum_normal_valid_fraction"]),
        "normal_background_inclusion_fraction": accepted_normal_fraction,
        "normal_valid_fraction_distribution": _valid_fraction_distribution(
            normal_valid_fractions
        ),
        "positive_pixel_distribution": _distribution(positive_pixels),
        "component_coverage_distribution": _distribution(coverage),
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "validation_predictions_loaded": 0,
        "materialized_image_files": 0,
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }
    return metadata, summary
