from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.extract_ksdd2 import ExtractionError, extract_archive


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/absolute.txt", "C:/drive.txt", "folder/../../escape.txt", "..\\escape.txt"])
def test_rejects_path_traversal(tmp_path: Path, unsafe_name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    _make_zip(archive, {unsafe_name: b"bad"})
    destination = tmp_path / "output"
    with pytest.raises(ExtractionError, match="Unsafe ZIP member"):
        extract_archive(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_extraction_is_idempotent_but_never_overwrites(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _make_zip(archive, {"train/100.png": b"image", "train/100_GT.png": b"mask"})
    destination = tmp_path / "output"
    first = extract_archive(archive, destination)
    second = extract_archive(archive, destination)
    assert first == (2, 9, False)
    assert second == (2, 9, True)
    assert (destination / "train" / "100.png").read_bytes() == b"image"


def test_refuses_unverified_existing_destination(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _make_zip(archive, {"file.txt": b"new"})
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "file.txt").write_bytes(b"existing")
    with pytest.raises(ExtractionError, match="not a verified complete extraction"):
        extract_archive(archive, destination)
    assert (destination / "file.txt").read_bytes() == b"existing"

