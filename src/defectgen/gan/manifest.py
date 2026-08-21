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
    rejected: list[dict[str, Any]] = []
    for row in rows:
        image, mask = _load_pair(repo_root, row)
        if not row["has_defect"]:
            valid_fraction = min(1.0, image.shape[0] / patch_size[1]) * min(
                1.0, image.shape[1] / patch_size[0]
            )
            if valid_fraction < float(patch["minimum_normal_valid_fraction"]):
                rejected.append(
                    {"sample_id": row["sample_id"], "reason": "normal_below_minimum_valid_fraction"}
                )
                continue
            normals.append(
                {
                    **row,
                    "native_width": image.shape[1],
                    "native_height": image.shape[0],
                    "minimum_valid_fraction": valid_fraction,
                    "available_window_count": _normal_window_count(
                        image.shape[:2], patch_size, float(patch["overlap_fraction"])
                    ),
                }
            )
            continue
        for component in connected_components(mask):
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
                rejected.append(
                    {
                        "sample_id": row["sample_id"],
                        "component_id": component.component_id,
                        "reasons": list(plan.rejected_reasons),
                    }
                )
                continue
            box = component.bounding_box
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
        "rejected": rejected,
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
    reason_counts: dict[str, int] = {}
    for item in rejected:
        for reason in item.get("reasons", [item.get("reason", "unknown")]):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "pipeline_version": configuration["pipeline_version"],
        "usable_training_defect_components": len(
            {(item["sample_id"], item["component_id"]) for item in templates}
        ),
        "template_windows": len(templates),
        "full_templates": sum(not item["partial_component"] for item in templates),
        "partial_templates": sum(item["partial_component"] for item in templates),
        "border_touching_templates": sum(item["touches_native_border"] for item in templates),
        "rejected_templates": len(rejected),
        "rejection_reasons": reason_counts,
        "normal_background_images": len(normals),
        "normal_background_patch_availability": sum(
            item["available_window_count"] for item in normals
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
