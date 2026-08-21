"""Build training-only metadata for deterministic online GAN inputs.

This creates metadata and summaries only. It never materializes generated image
patches and never reads validation/test images or predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.manifest import build_gan_input_metadata  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary_markdown(summary: dict[str, Any]) -> str:
    rejection_lines = [
        f"- `{reason}`: {count}" for reason, count in sorted(summary["rejection_reasons"].items())
    ] or ["- None"]
    return "\n".join(
        [
            "# GAN input manifest summary",
            "",
            f"- Pipeline version: `{summary['pipeline_version']}`",
            f"- Usable training defect components: {summary['usable_training_defect_components']}",
            f"- Template windows: {summary['template_windows']} "
            f"({summary['full_templates']} full, {summary['partial_templates']} partial)",
            f"- Border-touching templates: {summary['border_touching_templates']}",
            f"- Normal training images: {summary['normal_background_images']}",
            f"- Available normal windows: {summary['normal_background_patch_availability']}",
            f"- Validation rows/images loaded: {summary['validation_rows_loaded']}",
            f"- Official-test rows/images loaded: {summary['official_test_rows_loaded']}",
            f"- Validation predictions loaded: {summary['validation_predictions_loaded']}",
            f"- Materialized generated images: {summary['materialized_image_files']}",
            f"- Training-manifest SHA-256: `{summary['manifest_sha256']}`",
            f"- Training-split SHA-256: `{summary['split_sha256']}`",
            "",
            "## Rejections",
            "",
            *rejection_lines,
            "",
            "Generated inputs remain online and deterministic from seed plus manifest hashes.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gan_inputs.json")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    metadata, summary = build_gan_input_metadata(REPO_ROOT, configuration)
    outputs = configuration["outputs"]
    manifest_path = REPO_ROOT / outputs["manifest"]
    summary_path = REPO_ROOT / outputs["summary"]
    markdown_path = REPO_ROOT / outputs["summary_markdown"]
    _write_json(manifest_path, metadata)
    _write_json(summary_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    print(f"Wrote metadata-only GAN manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    print("Validation/test images and predictions loaded: 0")
    print("Materialized generated image files: 0")


if __name__ == "__main__":
    main()
