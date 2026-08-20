"""Create deterministic, non-interactive KSDD2 visual verification sheets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.ksdd2 import index_ksdd2, interpret_binary_mask  # noqa: E402

SEED = 1729
SAMPLE_COUNT = 8


def _select(rows, split: str, defective: bool, rng: np.random.Generator, count: int = SAMPLE_COUNT):
    candidates = [row for row in rows if row["split"] == split and bool(row["has_defect"]) == defective]
    if len(candidates) < count:
        raise ValueError(f"Need {count} {split} defective={defective} samples, found {len(candidates)}")
    if defective:
        # Always exercise the smallest annotation; the yellow dilation halo
        # keeps it visible without changing the mask itself.
        smallest = min(candidates, key=lambda row: (row["positive_mask_pixels"], row["sample_id"]))
        remaining = [row for row in candidates if row is not smallest]
        indices = np.sort(rng.choice(len(remaining), size=count - 1, replace=False))
        return [smallest, *(remaining[int(index)] for index in indices)]
    indices = np.sort(rng.choice(len(candidates), size=count, replace=False))
    return [candidates[int(index)] for index in indices]


def _load_rgb(row):
    with Image.open(REPO_ROOT / row["image_path"]) as image:
        return np.asarray(image.convert("RGB"))


def _load_mask(row):
    if not row["mask_path"]:
        return np.zeros((row["height"], row["width"]), dtype=bool)
    with Image.open(REPO_ROOT / row["mask_path"]) as mask:
        return interpret_binary_mask(np.asarray(mask))


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    halo = np.asarray(mask_image.filter(ImageFilter.MaxFilter(9))) > 0
    halo_only = halo & ~mask
    result[halo_only] = result[halo_only] * 0.55 + np.array([255, 220, 0]) * 0.45
    result[mask] = result[mask] * 0.15 + np.array([255, 0, 0]) * 0.85
    return np.clip(result, 0, 255).astype(np.uint8)


def _save_sheet(rows, output: Path, title: str, mode: str = "image") -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
    figure.suptitle(title, fontsize=15)
    for axis, row in zip(axes.flat, rows):
        image = _load_rgb(row)
        mask = _load_mask(row)
        if mode == "mask":
            axis.imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        elif mode == "overlay":
            axis.imshow(_overlay(image, mask))
        else:
            axis.imshow(image)
        axis.set_title(f'{row["sample_id"]}\nmask pixels: {row["positive_mask_pixels"]}', fontsize=9)
        axis.axis("off")
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "data_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = index_ksdd2(REPO_ROOT, args.dataset_root)
    if not result.passed:
        print("ERROR: Dataset audit must pass before visualisation.", file=sys.stderr)
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    normal_train = _select(result.rows, "train", False, rng)
    defective_train = _select(result.rows, "train", True, rng)
    defective_test = _select(result.rows, "test", True, rng)

    outputs = [
        (normal_train, output_dir / "train_normal_contact_sheet.png", "KSDD2: representative normal training images", "image"),
        (defective_train, output_dir / "train_defective_contact_sheet.png", "KSDD2: representative defective training images", "image"),
        (defective_test, output_dir / "test_defective_contact_sheet.png", "KSDD2: representative defective test images", "image"),
        (defective_train, output_dir / "mask_contact_sheet.png", "KSDD2: training defect masks", "mask"),
        (defective_train, output_dir / "overlay_contact_sheet.png", "KSDD2: image-mask overlays (red defect, yellow halo)", "overlay"),
    ]
    for rows, output, title, mode in outputs:
        _save_sheet(rows, output, title, mode)
        print(f"Saved {output}")
    print(f"Visualisation seed: {SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
