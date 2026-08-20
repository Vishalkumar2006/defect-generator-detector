"""Calculate full-image canvas, training-only RGB statistics, and class imbalance."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.ksdd2 import interpret_binary_mask  # noqa: E402
from defectgen.data.splits import load_development_manifest  # noqa: E402


def _round_up(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def analyze(rows, multiple: int = 32) -> dict[str, object]:
    dimensions: list[tuple[int, int]] = []
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    training_pixels = 0
    positive_pixels = 0

    for row in rows:
        with Image.open(REPO_ROOT / row["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        height, width = rgb.shape[:2]
        dimensions.append((width, height))
        if row["development_split"] != "train":
            continue
        channel_sum += rgb.sum(axis=(0, 1))
        channel_squared_sum += np.square(rgb).sum(axis=(0, 1))
        training_pixels += height * width
        if not row["mask_path"]:
            if row["has_defect"]:
                raise ValueError(f"Defective training sample lacks a mask: {row['sample_id']}")
            continue
        with Image.open(REPO_ROOT / row["mask_path"]) as image:
            mask = interpret_binary_mask(np.asarray(image))
        if mask.shape != (height, width):
            raise ValueError(f"Image/mask dimensions disagree for {row['sample_id']}")
        mask_pixels = int(np.count_nonzero(mask))
        if bool(mask_pixels) != bool(row["has_defect"]):
            raise ValueError(f"Manifest label disagrees with mask content for {row['sample_id']}")
        positive_pixels += mask_pixels

    if not dimensions or not training_pixels:
        raise ValueError("No manifest samples or development-training pixels were available")
    maximum_width = max(width for width, _ in dimensions)
    maximum_height = max(height for _, height in dimensions)
    model_width = _round_up(maximum_width, multiple)
    model_height = _round_up(maximum_height, multiple)
    mean = channel_sum / training_pixels
    variance = np.maximum(channel_squared_sum / training_pixels - np.square(mean), 0.0)
    standard_deviation = np.sqrt(variance)
    negative_pixels = training_pixels - positive_pixels
    return {
        "status": "PASS",
        "source_split": "development_train only for channel and pixel statistics",
        "sample_counts": {
            "all": len(rows),
            "development_train": sum(row["development_split"] == "train" for row in rows),
        },
        "native_dimensions": {
            "maximum_width": maximum_width,
            "maximum_height": maximum_height,
            "minimum_width": min(width for width, _ in dimensions),
            "minimum_height": min(height for _, height in dimensions),
        },
        "model_canvas": {
            "width": model_width,
            "height": model_height,
            "multiple": multiple,
        },
        "detector_rgb_statistics_0_to_1": {
            "mean": mean.tolist(),
            "standard_deviation": standard_deviation.tolist(),
            "pixel_count": training_pixels,
        },
        "training_pixel_balance": {
            "positive_pixels": positive_pixels,
            "negative_pixels": negative_pixels,
            "total_valid_pixels": training_pixels,
            "negative_to_positive_ratio": negative_pixels / positive_pixels,
            "positive_to_negative_ratio": positive_pixels / negative_pixels,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "metadata" / "ksdd2_split_seed42.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "preprocessing" / "full_image_statistics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = load_development_manifest(REPO_ROOT, args.manifest)
        report = analyze(rows)
        canvas = report["model_canvas"]
        if canvas != {"width": 256, "height": 672, "multiple": 32}:
            raise ValueError(f"Unexpected KSDD2 model canvas: {canvas}; expected exact 256x672 at multiple 32")
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f'Full-image analysis PASS: native max {report["native_dimensions"]["maximum_width"]}x{report["native_dimensions"]["maximum_height"]}; canvas {canvas["width"]}x{canvas["height"]}')
    print(f'RGB mean: {report["detector_rgb_statistics_0_to_1"]["mean"]}')
    print(f'RGB std: {report["detector_rgb_statistics_0_to_1"]["standard_deviation"]}')
    print(f'Negative:positive valid-pixel ratio: {report["training_pixel_balance"]["negative_to_positive_ratio"]:.6f}:1')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

