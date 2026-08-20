"""Audit KSDD2 and write a manifest plus machine/human-readable summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.ksdd2 import EXPECTED_COUNTS, dimension_distribution, index_ksdd2  # noqa: E402


def _area_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    values = np.asarray([float(row["defect_fraction"]) for row in rows], dtype=float)
    positive = values[values > 0]

    def describe(data: np.ndarray) -> dict[str, float | int]:
        if data.size == 0:
            return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "count": int(data.size),
            "min": float(data.min()),
            "mean": float(data.mean()),
            "median": float(np.median(data)),
            "max": float(data.max()),
        }

    return {"all_images": describe(values), "defective_images": describe(positive)}


def build_summary(result) -> dict[str, object]:
    return {
        "status": "PASS" if result.passed else "FAIL",
        "counts": result.counts,
        "expected_counts": EXPECTED_COUNTS,
        "expected_counts_match": result.expected_counts_match,
        "total_images": len(result.rows),
        "total_physical_masks": sum(bool(row["mask_path"]) for row in result.rows),
        "dimension_distribution": dimension_distribution(result.rows),
        "mask_area_fraction_statistics": _area_stats(result.rows),
        "invalid_pairs": result.invalid_pairs,
        "cross_split_duplicate_images": result.cross_split_duplicates,
        "ignored_nonconforming_files": result.ignored_files,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def summary_markdown(summary: dict[str, object]) -> str:
    counts = summary["counts"]
    expected = summary["expected_counts"]
    lines = [
        "# KSDD2 Data Audit",
        "",
        f'**Final status: {summary["status"]}**',
        "",
        "## Counts",
        "",
        "| Split | Observed defective | Expected defective | Observed normal | Expected normal | Observed total | Expected total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "test", "total"):
        lines.append(
            f'| {split} | {counts[split]["defective"]} | {expected[split]["defective"]} | '
            f'{counts[split]["normal"]} | {expected[split]["normal"]} | '
            f'{counts[split]["total"]} | {expected[split]["total"]} |'
        )
    lines += [
        "",
        f'- Indexed images: {summary["total_images"]}',
        f'- Physical masks: {summary["total_physical_masks"]}',
        f'- Expected counts match: {summary["expected_counts_match"]}',
        "",
        "## Image dimensions",
        "",
        "| Width x height | Count |",
        "|---|---:|",
    ]
    for dimensions, count in summary["dimension_distribution"].items():
        lines.append(f"| {dimensions} | {count} |")
    lines += ["", "## Mask-area fraction statistics", ""]
    for group, values in summary["mask_area_fraction_statistics"].items():
        lines.append(
            f'- {group}: count={values["count"]}, min={values["min"]:.8f}, '
            f'mean={values["mean"]:.8f}, median={values["median"]:.8f}, max={values["max"]:.8f}'
        )
    lines += [
        "",
        "## Pair and duplicate validation",
        "",
        f'- Missing or invalid pairs: {len(summary["invalid_pairs"])}',
        f'- Exact train/test image duplicates: {len(summary["cross_split_duplicate_images"])}',
        f'- Nonconforming files ignored: {len(summary["ignored_nonconforming_files"])}',
    ]
    if summary["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in summary["warnings"]]
    if summary["errors"]:
        lines += ["", "## Errors", ""] + [f"- {error}" for error in summary["errors"]]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "data_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = index_ksdd2(REPO_ROOT, args.dataset_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "split", "sample_id", "image_path", "mask_path", "has_defect", "width", "height",
        "positive_mask_pixels", "defect_fraction",
    ]
    pd.DataFrame(result.rows, columns=columns).to_csv(output_dir / "manifest.csv", index=False)
    summary = build_summary(result)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")

    print(f'Audit {summary["status"]}: {len(result.rows):,} images; reports written to {output_dir}')
    for split in ("train", "test"):
        count = result.counts[split]
        print(f'{split}: {count["defective"]} defective, {count["normal"]} normal')
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

