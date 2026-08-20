"""Safely and idempotently extract the KSDD2 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "raw" / "KolektorSDD2.zip"
DEFAULT_DESTINATION = REPO_ROOT / "data" / "extracted" / "KolektorSDD2"
MARKER_NAME = ".ksdd2_extraction.json"


class ExtractionError(RuntimeError):
    """Raised when extraction cannot be completed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_path(name: str) -> Path:
    """Convert a ZIP member name to a safe, relative platform path."""
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ExtractionError(f"Unsafe ZIP member path: {name!r}")
    if pure.parts and (":" in pure.parts[0] or pure.parts[0].startswith("~")):
        raise ExtractionError(f"Unsafe ZIP member path: {name!r}")
    return Path(*pure.parts)


def _is_symlink(external_attr: int) -> bool:
    mode = external_attr >> 16
    return stat.S_ISLNK(mode)


def _marker_is_valid(destination: Path, archive: Path) -> bool:
    marker_path = destination / MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if marker.get("archive_size") != archive.stat().st_size:
        return False
    if marker.get("archive_sha256") != _sha256(archive):
        return False
    files = [p for p in destination.rglob("*") if p.is_file() and p.name != MARKER_NAME]
    return len(files) == marker.get("file_count")


def extract_archive(archive: Path, destination: Path) -> tuple[int, int, bool]:
    """Extract *archive* without traversal or overwrite.

    Returns ``(file_count, byte_count, already_extracted)``.
    """
    archive = archive.resolve()
    destination = destination.resolve()
    if not archive.is_file():
        raise ExtractionError(f"Archive not found: {archive}")

    if destination.exists():
        if destination.is_dir() and _marker_is_valid(destination, archive):
            marker = json.loads((destination / MARKER_NAME).read_text(encoding="utf-8"))
            return int(marker["file_count"]), int(marker["uncompressed_bytes"]), True
        raise ExtractionError(
            f"Destination already exists but is not a verified complete extraction: {destination}. "
            "Move it aside and run the command again; it will not be overwritten."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-extract-", dir=destination.parent))
    try:
        with ZipFile(archive) as zip_file:
            bad_member = zip_file.testzip()
            if bad_member is not None:
                raise ExtractionError(f"Corrupt ZIP member: {bad_member}")

            file_count = 0
            byte_count = 0
            for info in zip_file.infolist():
                relative = _safe_member_path(info.filename)
                if _is_symlink(info.external_attr):
                    raise ExtractionError(f"Symbolic links are not allowed in the archive: {info.filename!r}")
                target = staging / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ExtractionError(f"Duplicate ZIP destination: {info.filename!r}")
                with zip_file.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                file_count += 1
                byte_count += info.file_size

        marker = {
            "archive_name": archive.name,
            "archive_size": archive.stat().st_size,
            "archive_sha256": _sha256(archive),
            "file_count": file_count,
            "uncompressed_bytes": byte_count,
        }
        (staging / MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, destination)
        return file_count, byte_count, False
    except (BadZipFile, EOFError) as exc:
        raise ExtractionError(f"Archive is not a valid complete ZIP file: {archive}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count, byte_count, existed = extract_archive(args.archive, args.destination)
    except (ExtractionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    action = "Verified existing extraction" if existed else "Extracted"
    print(f"{action}: {count:,} files ({byte_count:,} bytes) -> {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

