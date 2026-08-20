"""Create the deterministic, stratified KSDD2 development split manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.splits import (  # noqa: E402
    SEED,
    assign_development_splits,
    development_counts,
    load_validated_manifest,
    serialize_split_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "reports" / "data_audit" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "metadata" / "ksdd2_split_seed42.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audited = load_validated_manifest(REPO_ROOT, args.manifest)
        rows = assign_development_splits(audited, seed=SEED)
        content = serialize_split_csv(rows)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists() or output.read_text(encoding="utf-8") != content:
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8", newline="")
            temporary.replace(output)
        counts = development_counts(rows)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Development split written to {output}")
    print(f"Seed: {SEED}; validation fraction: 15% (stratified by has_defect)")
    for split in ("train", "validation", "test"):
        count = counts[split]
        print(f'{split}: {count["total"]} total ({count["defective"]} defective, {count["normal"]} normal)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

