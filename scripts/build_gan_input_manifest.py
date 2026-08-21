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
    defect_rejection_lines = [
        f"- `{reason}`: {count}"
        for reason, count in sorted(summary["defect_rejection_reasons"].items())
    ] or ["- None"]
    normal_rejection_lines = [
        f"- `{reason}`: {count}"
        for reason, count in sorted(summary["normal_rejection_reasons"].items())
    ] or ["- None"]
    contact_lines = [
        f"- `{side}`: {count}"
        for side, count in summary["templates_by_source_contact_side"].items()
    ]
    valid_distribution = summary["normal_valid_fraction_distribution"]
    return "\n".join(
        [
            "# GAN input manifest summary",
            "",
            f"- Pipeline version: `{summary['pipeline_version']}`",
            "",
            "## Defect templates",
            "",
            f"- Total defective training images: {summary['total_defective_training_images']}",
            f"- Connected components found: {summary['connected_components_found']}",
            f"- Accepted defect components: {summary['accepted_defect_components']}",
            f"- Rejected defect components: {summary['rejected_defect_components']}",
            f"- Components requiring overlapping windows: "
            f"{summary['components_requiring_overlapping_windows']}",
            f"- Template windows: {summary['template_windows']} "
            f"({summary['full_template_windows']} full, "
            f"{summary['partial_template_windows']} partial)",
            f"- Border-touching template windows: {summary['border_touching_template_windows']}",
            "",
            "### Templates by source contact side",
            "",
            *contact_lines,
            f"- Empirical border-template fraction: "
            f"{summary['empirical_border_template_fraction']:.8f}",
            f"- Border sampling mode: `{summary['sampling_border_fraction_mode']}`",
            f"- Configured fixed border fraction: {summary['configured_border_fraction']}",
            f"- Maximum component width/height: {summary['maximum_component_width']} / "
            f"{summary['maximum_component_height']}",
            "",
            "### Defect rejection reasons",
            "",
            *defect_rejection_lines,
            "",
            "## Normal backgrounds",
            "",
            f"- Total normal training images: {summary['total_normal_training_images']}",
            f"- Accepted normal-background images: {summary['accepted_normal_background_images']}",
            f"- Rejected normal-background images: {summary['rejected_normal_background_images']}",
            f"- Normal-background inclusion: "
            f"{100 * summary['normal_background_inclusion_fraction']:.4f}%",
            f"- Minimum valid fraction: {summary['minimum_normal_valid_fraction']:.8f}",
            f"- Achievable valid-fraction min/median/p95/max: "
            f"{valid_distribution['minimum']:.8f} / {valid_distribution['median']:.8f} / "
            f"{valid_distribution['p95']:.8f} / {valid_distribution['maximum']:.8f}",
            f"- Normal-background patch availability: "
            f"{summary['normal_background_patch_availability']}",
            "",
            "### Normal-background rejection reasons",
            "",
            *normal_rejection_lines,
            "",
            "## Isolation and reproducibility",
            "",
            f"- Validation rows/images loaded: {summary['validation_rows_loaded']}",
            f"- Official-test rows/images loaded: {summary['official_test_rows_loaded']}",
            f"- Validation predictions loaded: {summary['validation_predictions_loaded']}",
            f"- Materialized generated images: {summary['materialized_image_files']}",
            f"- Source-manifest SHA-256: `{summary['source_manifest_sha256']}`",
            f"- Training-split SHA-256: `{summary['split_sha256']}`",
            f"- GAN-manifest content SHA-256: `{summary['gan_manifest_content_sha256']}`",
            "",
            "Generated inputs remain online and deterministic from seed plus manifest hashes.",
            "Compatibility-index exclusions, actual retries, successful side combinations,",
            "utilization, accidental contacts, and support validity are emitted by visualization",
            "accounting and the bounded sampling-audit command.",
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
