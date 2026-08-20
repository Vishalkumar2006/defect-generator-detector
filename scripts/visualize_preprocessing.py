"""Create deterministic visual checks for KSDD2 geometry and patch extraction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.geometry import PATCH_CANDIDATES, bounding_box  # noqa: E402
from defectgen.data.ksdd2 import interpret_binary_mask  # noqa: E402
from defectgen.data.patches import extract_normal_patch, extract_positive_patch  # noqa: E402
from defectgen.data.splits import load_development_manifest  # noqa: E402


SEED = 42
CONTEXT_MARGIN = 32


def _read_geometry(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("bbox_width", "bbox_height", "mask_pixels", "mask_fraction"):
            row[field] = float(row[field])
        row["touches_border"] = row["touches_border"].casefold() == "true"
    return rows


def _load_pair(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(REPO_ROOT / row["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(REPO_ROOT / row["mask_path"]) as image:
        mask = interpret_binary_mask(np.asarray(image))
    return rgb, mask


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    halo = np.asarray(mask_image.filter(ImageFilter.MaxFilter(9))) > 0
    result[halo & ~mask] = result[halo & ~mask] * 0.45 + np.array([255, 230, 0]) * 0.55
    result[mask] = result[mask] * 0.10 + np.array([255, 0, 0]) * 0.90
    return np.clip(result, 0, 255).astype(np.uint8)


def _show_with_box(axis, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    axis.imshow(_overlay(image, mask))
    box = bounding_box(mask)
    if box is not None:
        axis.add_patch(
            Rectangle(
                (box.x_min - 0.5, box.y_min - 0.5),
                box.width,
                box.height,
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
            )
        )
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _closest(rows: list[dict[str, Any]], field: str, target: float) -> dict[str, Any]:
    return min(rows, key=lambda row: (abs(float(row[field]) - target), row["sample_id"]))


def _select_defect_examples(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (float(row["mask_fraction"]), row["sample_id"]))
    values = np.asarray([float(row["mask_fraction"]) for row in ordered])
    selections = [
        ("smallest mask fraction", ordered[0]),
        ("median mask fraction", _closest(ordered, "mask_fraction", float(np.quantile(values, 0.50)))),
        ("90th-percentile mask fraction", _closest(ordered, "mask_fraction", float(np.quantile(values, 0.90)))),
        ("largest mask fraction", ordered[-1]),
    ]
    border_rows = [row for row in ordered if row["touches_border"]]
    if border_rows:
        border = border_rows[len(border_rows) // 2]
        selections.append(("border-touching defect", border))
    return selections


def _defect_sheet(
    examples: list[tuple[str, dict[str, Any]]], patch_size: tuple[int, int], output: Path
) -> None:
    figure, axes = plt.subplots(len(examples), 4, figsize=(13, 4.8 * len(examples)), constrained_layout=True)
    figure.suptitle(
        f"KSDD2 preprocessing checks — selected patch {patch_size[0]}×{patch_size[1]}, context {CONTEXT_MARGIN}px",
        fontsize=15,
    )
    for index, (description, row) in enumerate(examples):
        image, mask = _load_pair(row)
        patch = extract_positive_patch(
            image, mask, patch_size, np.random.default_rng(SEED + index), context_margin=CONTEXT_MARGIN
        )
        _show_with_box(
            axes[index, 0], image, mask, f'{description}\n{row["sample_id"]} — complete native image'
        )
        _show_with_box(
            axes[index, 1],
            patch.image,
            patch.mask,
            f"selected patch\ncontained={patch.defect_fully_contained}, context={patch.context_fully_contained}",
        )
        axes[index, 2].imshow(patch.mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[index, 2].set_title(f'binary mask — {int(np.count_nonzero(patch.mask))} pixels', fontsize=9)
        axes[index, 2].axis("off")
        axes[index, 3].imshow(patch.image)
        axes[index, 3].set_title(
            f'raw synchronized crop\nsource origin ({patch.coordinates.source_left}, {patch.coordinates.source_top})',
            fontsize=9,
        )
        axes[index, 3].axis("off")
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)


def _candidate_sheet(row: dict[str, Any], output: Path) -> None:
    image, mask = _load_pair(row)
    figure, axes = plt.subplots(1, len(PATCH_CANDIDATES), figsize=(12, 6), constrained_layout=True)
    figure.suptitle(f'Candidate patches with {CONTEXT_MARGIN}px context — {row["sample_id"]}', fontsize=15)
    for index, (axis, candidate) in enumerate(zip(axes, PATCH_CANDIDATES)):
        patch = extract_positive_patch(
            image, mask, candidate, np.random.default_rng(SEED + 100 + index), context_margin=CONTEXT_MARGIN
        )
        _show_with_box(
            axis,
            patch.image,
            patch.mask,
            f"{candidate[0]}×{candidate[1]}\ncontained={patch.defect_fully_contained}, context={patch.context_fully_contained}",
        )
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def _normal_sheet(manifest: list[dict[str, Any]], patch_size: tuple[int, int], output: Path) -> None:
    candidates = [
        row
        for row in manifest
        if row["development_split"] == "train" and not row["has_defect"] and row["mask_path"]
    ]
    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(len(candidates), size=8, replace=False))
    figure, axes = plt.subplots(2, 4, figsize=(12, 9), constrained_layout=True)
    figure.suptitle(f"Development-training normal patches — {patch_size[0]}×{patch_size[1]}", fontsize=15)
    for axis, candidate_index in zip(axes.flat, indices):
        row = candidates[int(candidate_index)]
        image, mask = _load_pair(row)
        patch = extract_normal_patch(image, mask, patch_size, rng)
        axis.imshow(patch.image)
        axis.set_title(f'{row["sample_id"]}\nmask pixels={int(np.count_nonzero(patch.mask))}', fontsize=9)
        axis.axis("off")
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "metadata" / "ksdd2_split_seed42.csv")
    parser.add_argument("--geometry", type=Path, default=REPO_ROOT / "reports" / "preprocessing" / "bbox_statistics.csv")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "reports" / "preprocessing" / "bbox_summary.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "preprocessing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_development_manifest(REPO_ROOT, args.manifest)
        geometry = _read_geometry(args.geometry)
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        recommendation = summary["recommendation"]["patch_size"]
        if recommendation:
            patch_size = tuple(int(value) for value in recommendation.split("x"))
            patch_note = "geometry recommendation"
        else:
            patch_size = PATCH_CANDIDATES[-1]
            patch_note = "largest candidate used for visualization only; no candidate met the recommendation rule"
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        examples = _select_defect_examples(geometry)
        outputs = {
            "defect": output_dir / "defect_patch_examples.png",
            "candidates": output_dir / "candidate_patch_comparison.png",
            "normal": output_dir / "normal_patch_examples.png",
        }
        _defect_sheet(examples, patch_size, outputs["defect"])
        p90_height = float(np.quantile([row["bbox_height"] for row in geometry], 0.90))
        comparison_row = _closest(geometry, "bbox_height", p90_height)
        _candidate_sheet(comparison_row, outputs["candidates"])
        _normal_sheet(manifest, patch_size, outputs["normal"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for output in outputs.values():
        print(f"Saved {output}")
    print(f"Visualisation seed: {SEED}; selected patch: {patch_size[0]}x{patch_size[1]} ({patch_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
