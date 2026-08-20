"""Measure KSDD2 defect geometry and evaluate candidate patch dimensions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.geometry import (  # noqa: E402
    CONTEXT_MARGINS,
    PATCH_CANDIDATES,
    bbox_fits_patch,
    bounding_box,
    connected_component_count,
    touches_border,
)
from defectgen.data.ksdd2 import interpret_binary_mask  # noqa: E402
from defectgen.data.splits import load_development_manifest  # noqa: E402


QUANTILES = (("minimum", 0.0), ("median", 0.5), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("maximum", 1.0))
STAT_FIELDS = ("bbox_width", "bbox_height", "mask_pixels", "mask_fraction")


def _fit_column(patch_size: tuple[int, int], context: int) -> str:
    return f"fits_{patch_size[0]}x{patch_size[1]}_context{context}"


def measure_geometry(repo_root: Path, manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sample in manifest_rows:
        if not sample["has_defect"]:
            continue
        if not sample["mask_path"]:
            raise ValueError(f"Defective sample has no physical mask: {sample['sample_id']}")
        with Image.open(repo_root / sample["mask_path"]) as image:
            mask = interpret_binary_mask(np.asarray(image))
        box = bounding_box(mask)
        if box is None:
            raise ValueError(f"Defective manifest row has an empty mask: {sample['sample_id']}")
        area = int(np.count_nonzero(mask))
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "official_split": sample["official_split"],
            "development_split": sample["development_split"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
            "image_width": int(mask.shape[1]),
            "image_height": int(mask.shape[0]),
            "bbox_x_min": box.x_min,
            "bbox_y_min": box.y_min,
            "bbox_x_max": box.x_max,
            "bbox_y_max": box.y_max,
            "bbox_width": box.width,
            "bbox_height": box.height,
            "mask_pixels": area,
            "mask_fraction": area / mask.size,
            "connected_components": connected_component_count(mask),
            "touches_border": touches_border(mask),
        }
        for candidate in PATCH_CANDIDATES:
            for margin in CONTEXT_MARGINS:
                row[_fit_column(candidate, margin)] = bbox_fits_patch(box, candidate, margin)
        results.append(row)
    return results


def _statistics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    statistics: dict[str, dict[str, float]] = {}
    for field in STAT_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        statistics[field] = {
            name: float(np.quantile(values, quantile, method="linear")) for name, quantile in QUANTILES
        }
    return statistics


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    development_train = [row for row in rows if row["development_split"] == "train"]
    fit_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for candidate in PATCH_CANDIDATES:
        label = f"{candidate[0]}x{candidate[1]}"
        fit_summary[label] = {}
        for margin in CONTEXT_MARGINS:
            count = sum(bool(row[_fit_column(candidate, margin)]) for row in development_train)
            total = len(development_train)
            fit_summary[label][str(margin)] = {
                "count": count,
                "total": total,
                "percentage": 100.0 * count / total if total else 0.0,
            }

    recommendation = None
    for candidate in PATCH_CANDIDATES:
        label = f"{candidate[0]}x{candidate[1]}"
        if fit_summary[label]["32"]["percentage"] >= 95.0:
            recommendation = label
            break
    unfittable = []
    if recommendation is not None:
        width, height = (int(value) for value in recommendation.split("x"))
        column = _fit_column((width, height), 32)
        unfittable = [row["sample_id"] for row in development_train if not row[column]]

    return {
        "status": "PASS",
        "defective_images": {
            "all": len(rows),
            "development_train": len(development_train),
            "validation": sum(row["development_split"] == "validation" for row in rows),
            "test": sum(row["development_split"] == "test" for row in rows),
        },
        "statistics_all_defective_images": _statistics(rows),
        "border_touching": {
            "all_count": sum(bool(row["touches_border"]) for row in rows),
            "development_train_count": sum(bool(row["touches_border"]) for row in development_train),
        },
        "candidate_fit_development_train": fit_summary,
        "recommendation": {
            "patch_size": recommendation,
            "criterion": "smallest candidate containing at least 95% of development-training defects with 32 pixels of context",
            "unfittable_sample_ids": unfittable,
        },
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# KSDD2 defect bounding-box analysis",
        "",
        f'**Status: {summary["status"]}**',
        "",
        "All coordinates are pixel coordinates with inclusive maxima. Connected components use 8-connectivity.",
        "",
        "## Defect counts",
        "",
    ]
    for scope, count in summary["defective_images"].items():
        lines.append(f"- {scope}: {count}")
    lines += ["", "## Geometry quantiles (all defective images)", "", "| Measure | Min | Median | P75 | P90 | P95 | P99 | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for field, values in summary["statistics_all_defective_images"].items():
        lines.append(
            f'| {field} | {values["minimum"]:.6g} | {values["median"]:.6g} | {values["p75"]:.6g} | '
            f'{values["p90"]:.6g} | {values["p95"]:.6g} | {values["p99"]:.6g} | {values["maximum"]:.6g} |'
        )
    border = summary["border_touching"]
    lines += [
        "",
        "## Border contact",
        "",
        f'- All defective images: {border["all_count"]}',
        f'- Development-training defective images: {border["development_train_count"]}',
        "",
        "## Complete-defect fit in development training",
        "",
        "Reflection padding may supply image background near an image edge, but cannot make an oversized defect count as fitting.",
        "",
        "| Patch width × height | Context | Fits | Total | Percentage |",
        "|---|---:|---:|---:|---:|",
    ]
    for patch, contexts in summary["candidate_fit_development_train"].items():
        for context, values in contexts.items():
            lines.append(f'| {patch} | {context} px | {values["count"]} | {values["total"]} | {values["percentage"]:.2f}% |')
    recommendation = summary["recommendation"]
    lines += ["", "## Recommendation", ""]
    if recommendation["patch_size"] is None:
        lines.append(f'No candidate meets the rule: {recommendation["criterion"]}.')
    else:
        lines.append(f'Recommended patch: **{recommendation["patch_size"]}** — it is the {recommendation["criterion"]}.')
        lines.append(f'Development-training samples that do not fit with 32-pixel context: {len(recommendation["unfittable_sample_ids"])}.')
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "metadata" / "ksdd2_split_seed42.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "preprocessing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_development_manifest(REPO_ROOT, args.manifest)
        rows = measure_geometry(REPO_ROOT, manifest)
        summary = build_summary(rows)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise ValueError("No defective rows found in development manifest")
        with (output_dir / "bbox_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "bbox_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output_dir / "bbox_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Geometry analysis PASS: {len(rows)} defective images -> {output_dir}")
    recommendation = summary["recommendation"]["patch_size"] or "none (no candidate met the 95% rule)"
    print(f"Recommended patch: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

