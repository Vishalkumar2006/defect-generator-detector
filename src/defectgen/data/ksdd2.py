"""Deterministic indexing and validation for KolektorSDD2 (KSDD2)."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


EXPECTED_COUNTS = {
    "train": {"defective": 246, "normal": 2085, "total": 2331},
    "test": {"defective": 110, "normal": 894, "total": 1004},
    "total": {"defective": 356, "normal": 2979, "total": 3335},
}
IMAGE_PATTERN = re.compile(r"^(?P<id>\d+)\.png$", re.IGNORECASE)
MASK_PATTERN = re.compile(r"^(?P<id>\d+)_GT\.png$", re.IGNORECASE)
SPLITS = ("train", "test")


@dataclass
class IndexResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    invalid_pairs: list[dict[str, str]] = field(default_factory=list)
    cross_split_duplicates: list[dict[str, str]] = field(default_factory=list)
    ignored_files: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    expected_counts_match: bool = False

    @property
    def passed(self) -> bool:
        return not self.errors and not self.invalid_pairs and not self.cross_split_duplicates and self.expected_counts_match


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image), image.size


def interpret_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask, rejecting ambiguous multi-level masks."""
    if mask.ndim == 3:
        if mask.shape[2] not in (3, 4):
            raise ValueError(f"unsupported mask shape {mask.shape}")
        rgb = mask[..., :3]
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
            raise ValueError("mask RGB channels disagree")
        mask = rgb[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"unsupported mask shape {mask.shape}")
    values = np.unique(mask)
    if len(values) > 2:
        preview = ", ".join(map(str, values[:8]))
        raise ValueError(f"mask has more than two values ({preview})")
    if len(values) == 2 and values[0] != 0:
        raise ValueError(f"binary mask background is not zero: {values.tolist()}")
    if len(values) == 1 and values[0] != 0:
        raise ValueError(f"constant non-zero mask is ambiguous: {values.tolist()}")
    return mask != 0


def _natural_id(value: str) -> tuple[int, str]:
    return int(value), value


def _count_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        defective = sum(bool(row["has_defect"]) for row in selected)
        counts[split] = {"defective": defective, "normal": len(selected) - defective, "total": len(selected)}
    defective = sum(bool(row["has_defect"]) for row in rows)
    counts["total"] = {"defective": defective, "normal": len(rows) - defective, "total": len(rows)}
    return counts


def index_ksdd2(repo_root: Path, dataset_root: Path | None = None) -> IndexResult:
    """Index KSDD2 in a stable order and collect all validation findings."""
    repo_root = repo_root.resolve()
    dataset_root = (dataset_root or repo_root / "data" / "extracted" / "KolektorSDD2").resolve()
    result = IndexResult()
    if not dataset_root.is_dir():
        result.errors.append(f"Dataset directory not found: {dataset_root}")
        result.counts = _count_rows([])
        return result

    image_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            result.errors.append(f"Official split directory is missing: {split_dir}")
            continue

        images: dict[str, Path] = {}
        masks: dict[str, Path] = {}
        for path in sorted(split_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.name.startswith("."):
                continue
            image_match = IMAGE_PATTERN.fullmatch(path.name)
            mask_match = MASK_PATTERN.fullmatch(path.name)
            if image_match:
                images[image_match.group("id")] = path
            elif mask_match:
                masks[mask_match.group("id")] = path
            else:
                relative = _relative(path, repo_root)
                result.ignored_files.append(relative)

        for orphan_id in sorted(masks.keys() - images.keys(), key=_natural_id):
            detail = {"split": split, "sample_id": f"{split}-{orphan_id}", "issue": "mask has no image"}
            result.invalid_pairs.append(detail)

        for numeric_id in sorted(images, key=_natural_id):
            image_path = images[numeric_id]
            mask_path = masks.get(numeric_id)
            sample_id = f"{split}-{numeric_id}"
            image_rel = _relative(image_path, repo_root)
            mask_rel = _relative(mask_path, repo_root) if mask_path else ""
            if sample_id in seen_ids or image_rel in seen_paths:
                result.errors.append(f"Duplicate manifest row: {sample_id} ({image_rel})")
                continue
            seen_ids.add(sample_id)
            seen_paths.add(image_rel)

            try:
                _, image_size = _read_image(image_path)
            except (OSError, UnidentifiedImageError) as exc:
                result.invalid_pairs.append({"split": split, "sample_id": sample_id, "issue": f"invalid image: {exc}"})
                continue

            positive_pixels = 0
            mask_valid = True
            if mask_path is not None:
                try:
                    mask_array, mask_size = _read_image(mask_path)
                    if image_size != mask_size:
                        raise ValueError(f"dimension mismatch: image {image_size}, mask {mask_size}")
                    binary_mask = interpret_binary_mask(mask_array)
                    positive_pixels = int(np.count_nonzero(binary_mask))
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    mask_valid = False
                    result.invalid_pairs.append({"split": split, "sample_id": sample_id, "issue": str(exc)})

            width, height = image_size
            row = {
                "split": split,
                "sample_id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "has_defect": bool(positive_pixels) if mask_valid else False,
                "width": width,
                "height": height,
                "positive_mask_pixels": positive_pixels,
                "defect_fraction": positive_pixels / (width * height),
            }
            result.rows.append(row)
            try:
                image_hashes[_sha256(image_path)].append((split, image_rel))
            except OSError as exc:
                result.errors.append(f"Could not hash {image_rel}: {exc}")

    for digest, occurrences in sorted(image_hashes.items()):
        splits = {item[0] for item in occurrences}
        if len(splits) > 1:
            train_paths = [path for split, path in occurrences if split == "train"]
            test_paths = [path for split, path in occurrences if split == "test"]
            for train_path in train_paths:
                for test_path in test_paths:
                    result.cross_split_duplicates.append(
                        {"sha256": digest, "train_path": train_path, "test_path": test_path}
                    )

    if result.ignored_files:
        result.warnings.append(
            f"Ignored {len(result.ignored_files)} file(s) that do not match the numeric KSDD2 image/mask convention."
        )
        # Explain exact duplicate copies without allowing them into the official manifest.
        for ignored in result.ignored_files:
            ignored_path = repo_root / Path(ignored)
            match = re.match(r"^(\d+)(?:_GT)? \(copy\)\.png$", ignored_path.name, re.IGNORECASE)
            if match:
                original_name = ignored_path.name.replace(" (copy)", "")
                original = ignored_path.with_name(original_name)
                if original.is_file() and _sha256(ignored_path) == _sha256(original):
                    result.warnings.append(f"Exact duplicate copy ignored: {ignored} duplicates {_relative(original, repo_root)}")

    result.counts = _count_rows(result.rows)
    result.expected_counts_match = result.counts == EXPECTED_COUNTS
    if not result.expected_counts_match:
        for scope in (*SPLITS, "total"):
            if result.counts.get(scope) != EXPECTED_COUNTS[scope]:
                result.errors.append(
                    f"Count mismatch for {scope}: observed={result.counts.get(scope)}, expected={EXPECTED_COUNTS[scope]}"
                )
    if result.cross_split_duplicates:
        result.errors.append(f"Found {len(result.cross_split_duplicates)} exact image duplicate(s) across train and test")
    return result


def dimension_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f'{row["width"]}x{row["height"]}' for row in rows).items()))

