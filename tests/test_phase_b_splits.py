from __future__ import annotations

from pathlib import Path

import pytest

import defectgen.data.splits as split_module
from defectgen.data.splits import (
    assign_development_splits,
    development_counts,
    load_development_manifest,
    load_validated_manifest,
    serialize_split_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MANIFEST = REPO_ROOT / "reports" / "data_audit" / "manifest.csv"
SPLIT_MANIFEST = REPO_ROOT / "data" / "metadata" / "ksdd2_split_seed42.csv"


@pytest.fixture(scope="module")
def audited_rows():
    if not AUDIT_MANIFEST.is_file():
        pytest.skip("Validated KSDD2 audit manifest is absent")
    return load_validated_manifest(REPO_ROOT, AUDIT_MANIFEST)


@pytest.fixture(scope="module")
def split_rows():
    if not SPLIT_MANIFEST.is_file():
        pytest.skip("Development split metadata is absent; run scripts\\create_development_split.py")
    return load_development_manifest(REPO_ROOT, SPLIT_MANIFEST)


def test_development_split_is_reproducible(audited_rows) -> None:
    first = assign_development_splits(audited_rows, seed=42)
    second = assign_development_splits(audited_rows, seed=42)
    assert serialize_split_csv(first) == serialize_split_csv(second)


def test_split_counts_and_exact_disjointness(split_rows) -> None:
    assert development_counts(split_rows) == {
        "train": {"defective": 209, "normal": 1772, "total": 1981},
        "validation": {"defective": 37, "normal": 313, "total": 350},
        "test": {"defective": 110, "normal": 894, "total": 1004},
    }
    ids_by_split = {
        name: {row["sample_id"] for row in split_rows if row["development_split"] == name}
        for name in ("train", "validation", "test")
    }
    assert ids_by_split["train"].isdisjoint(ids_by_split["validation"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["validation"].isdisjoint(ids_by_split["test"])


def test_official_test_is_preserved(split_rows) -> None:
    official_test = {row["sample_id"] for row in split_rows if row["official_split"] == "test"}
    development_test = {row["sample_id"] for row in split_rows if row["development_split"] == "test"}
    assert official_test == development_test
    assert len(official_test) == 1004


def test_duplicate_copies_are_excluded_and_discovery_is_manifest_only(audited_rows, split_rows) -> None:
    audited_ids = {row["sample_id"] for row in audited_rows}
    split_ids = {row["sample_id"] for row in split_rows}
    assert split_ids == audited_ids
    assert len(split_ids) == 3335
    assert all("(copy)" not in row["image_path"].casefold() for row in split_rows)
    assert all("(copy)" not in row["mask_path"].casefold() for row in split_rows)


def test_manifest_loader_does_not_discover_unlisted_images(tmp_path: Path, monkeypatch) -> None:
    report_dir = tmp_path / "reports" / "data_audit"
    image_dir = tmp_path / "data" / "extracted" / "KolektorSDD2" / "train"
    report_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (report_dir / "summary.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (report_dir / "manifest.csv").write_text(
        "sample_id,split,image_path,mask_path,has_defect\n"
        "train-listed,train,data/extracted/KolektorSDD2/train/listed.png,,False\n",
        encoding="utf-8",
    )
    (image_dir / "listed.png").write_bytes(b"listed")
    (image_dir / "unlisted.png").write_bytes(b"must not be discovered")
    monkeypatch.setattr(split_module, "_validate_official_counts", lambda rows: None)
    rows = load_validated_manifest(tmp_path, report_dir / "manifest.csv")
    assert [row["sample_id"] for row in rows] == ["train-listed"]


def test_generated_metadata_paths_are_relative(split_rows) -> None:
    assert all(not Path(row["image_path"]).is_absolute() for row in split_rows)
    assert all(not row["mask_path"] or not Path(row["mask_path"]).is_absolute() for row in split_rows)
