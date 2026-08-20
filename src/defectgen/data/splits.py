"""Manifest-only, deterministic KSDD2 development split creation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .ksdd2 import EXPECTED_COUNTS


SEED = 42
VALIDATION_FRACTION = 0.15
REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "image_path",
    "mask_path",
    "has_defect",
}
OUTPUT_COLUMNS = (
    "sample_id",
    "official_split",
    "development_split",
    "image_path",
    "mask_path",
    "has_defect",
    "image_sha256",
)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value in manifest: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_relative_existing(repo_root: Path, relative: str, kind: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"{kind} path must be repository-relative: {relative}")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{kind} path escapes the repository: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Referenced {kind} does not exist: {relative}")
    return resolved


def load_validated_manifest(repo_root: Path, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    """Load the audited CSV; never discover samples from the image directories."""
    repo_root = repo_root.resolve()
    manifest_path = (manifest_path or repo_root / "reports" / "data_audit" / "manifest.csv").resolve()
    summary_path = manifest_path.with_name("summary.json")
    try:
        audit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"A readable audit summary is required beside the manifest: {summary_path}") from exc
    if audit_summary.get("status") != "PASS":
        raise ValueError(f"Refusing to use a manifest whose audit status is not PASS: {summary_path}")

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        raw_rows = list(reader)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        sample_id = raw["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID in audited manifest: {sample_id}")
        seen.add(sample_id)
        if raw["split"] not in {"train", "test"}:
            raise ValueError(f"Invalid official split for {sample_id}: {raw['split']}")
        if "(copy)" in raw["image_path"].casefold() or "(copy)" in raw["mask_path"].casefold():
            raise ValueError(f"Nonconforming duplicate-copy file entered the audited manifest: {sample_id}")
        image = _assert_relative_existing(repo_root, raw["image_path"], "image")
        if raw["mask_path"]:
            _assert_relative_existing(repo_root, raw["mask_path"], "mask")
        rows.append(
            {
                "sample_id": sample_id,
                "official_split": raw["split"],
                "image_path": Path(raw["image_path"]).as_posix(),
                "mask_path": Path(raw["mask_path"]).as_posix() if raw["mask_path"] else "",
                "has_defect": _parse_bool(raw["has_defect"]),
                "image_sha256": _sha256(image),
            }
        )
    _validate_official_counts(rows)
    return rows


def _validate_official_counts(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    observed: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        selected = [row for row in rows if row["official_split"] == split]
        defective = sum(bool(row["has_defect"]) for row in selected)
        observed[split] = {"defective": defective, "normal": len(selected) - defective, "total": len(selected)}
    defective = sum(bool(row["has_defect"]) for row in rows)
    observed["total"] = {"defective": defective, "normal": len(rows) - defective, "total": len(rows)}
    if observed != EXPECTED_COUNTS:
        differences = [
            f"{scope}: observed={observed[scope]}, expected={EXPECTED_COUNTS[scope]}"
            for scope in ("train", "test", "total")
            if observed[scope] != EXPECTED_COUNTS[scope]
        ]
        raise ValueError("Official audited counts differ; refusing to split. " + "; ".join(differences))


def _choose_validation_hashes(groups: dict[str, list[dict[str, Any]]], target: int, seed: int) -> set[str]:
    """Choose whole hash groups with a deterministic subset sum nearest target."""
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode("ascii")).hexdigest(),
    )
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for index, (_, members) in enumerate(ordered):
        weight = len(members)
        for previous in sorted(tuple(reachable), reverse=True):
            total = previous + weight
            if total not in reachable:
                reachable[total] = (previous, index)
    chosen_total = min(reachable, key=lambda total: (abs(total - target), total > target, total))
    chosen_indices: set[int] = set()
    cursor = chosen_total
    while cursor:
        parent = reachable[cursor]
        if parent is None:
            break
        cursor, index = parent
        chosen_indices.add(index)
    return {digest for index, (digest, _) in enumerate(ordered) if index in chosen_indices}


def assign_development_splits(
    rows: list[dict[str, Any]], seed: int = SEED, validation_fraction: float = VALIDATION_FRACTION
) -> list[dict[str, Any]]:
    """Assign official train rows to train/validation, preserving hash groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    _validate_official_counts(rows)
    result = [dict(row) for row in rows]
    training = [row for row in result if row["official_split"] == "train"]
    labels_by_hash: dict[str, set[bool]] = defaultdict(set)
    for row in training:
        labels_by_hash[row["image_sha256"]].add(bool(row["has_defect"]))
    inconsistent = [digest for digest, labels in labels_by_hash.items() if len(labels) > 1]
    if inconsistent:
        raise ValueError(
            f"Exact duplicate training content has inconsistent labels ({len(inconsistent)} hash group(s)); refusing to split"
        )

    for label in (False, True):
        labelled = [row for row in training if bool(row["has_defect"]) == label]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in labelled:
            groups[row["image_sha256"]].append(row)
        target = int(len(labelled) * validation_fraction + 0.5)
        validation_hashes = _choose_validation_hashes(groups, target, seed + int(label))
        for row in labelled:
            row["development_split"] = "validation" if row["image_sha256"] in validation_hashes else "train"

    for row in result:
        if row["official_split"] == "test":
            row["development_split"] = "test"
    validate_development_split(result)
    return result


def validate_development_split(rows: list[dict[str, Any]]) -> None:
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("A sample ID appears in more than one development split")
    if any(row.get("development_split") not in {"train", "validation", "test"} for row in rows):
        raise ValueError("Development manifest contains an invalid split name")
    if any(row["official_split"] == "test" and row.get("development_split") != "test" for row in rows):
        raise ValueError("Official test samples were not preserved")
    if any(row["official_split"] == "train" and row.get("development_split") == "test" for row in rows):
        raise ValueError("An official training sample entered the development test split")
    validation = [row for row in rows if row.get("development_split") == "validation"]
    if not {bool(row["has_defect"]) for row in validation} == {False, True}:
        raise ValueError("Validation must contain both normal and defective samples")
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["official_split"] == "train":
            hash_splits[row["image_sha256"]].add(row["development_split"])
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("An exact duplicate hash was divided between development train and validation")


def development_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["development_split"] == split]
        defective = sum(bool(row["has_defect"]) for row in selected)
        counts[split] = {"defective": defective, "normal": len(selected) - defective, "total": len(selected)}
    return counts


def serialize_split_csv(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})
    return stream.getvalue()


def load_development_manifest(repo_root: Path, path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate generated development metadata without image discovery."""
    repo_root = repo_root.resolve()
    path = (path or repo_root / "data" / "metadata" / "ksdd2_split_seed42.csv").resolve()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Development manifest is missing columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            if "(copy)" in raw["image_path"].casefold() or "(copy)" in raw["mask_path"].casefold():
                raise ValueError(f"Duplicate-copy path in development manifest: {raw['sample_id']}")
            _assert_relative_existing(repo_root, raw["image_path"], "image")
            if raw["mask_path"]:
                _assert_relative_existing(repo_root, raw["mask_path"], "mask")
            rows.append(
                {
                    "sample_id": raw["sample_id"],
                    "official_split": raw["official_split"],
                    "development_split": raw["development_split"],
                    "image_path": Path(raw["image_path"]).as_posix(),
                    "mask_path": Path(raw["mask_path"]).as_posix() if raw["mask_path"] else "",
                    "has_defect": _parse_bool(raw["has_defect"]),
                    "image_sha256": raw["image_sha256"],
                }
            )
    _validate_official_counts(rows)
    validate_development_split(rows)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample ID in development manifest")
    return rows
