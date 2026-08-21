"""Audit deterministic training-only GAN sampling without materializing images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.dataset import GANSamplingFailure, OnlineGANInputDataset  # noqa: E402
from defectgen.gan.sampling_audit import build_sampling_audit_summary  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _markdown(summary: dict[str, Any]) -> str:
    attempts = summary["attempts_per_requested_sample"]
    border = summary["border_distribution"]
    failure_lines = [
        f"- `{reason}`: {count}" for reason, count in summary["failures_by_reason"].items()
    ] or ["- None"]
    side_lines = [
        f"- `{combination}`: {count}"
        for combination, count in summary["successful_placements_by_side_combination"].items()
    ] or ["- None"]
    return "\n".join(
        [
            "# GAN online-sampling audit",
            "",
            f"- Pipeline: `{summary['pipeline_version']}`",
            f"- Requested/successful/failed: {summary['requested_samples']} / "
            f"{summary['successful_samples']} / {summary['failed_samples']}",
            f"- Success rate: {100 * summary['success_rate']:.4f}%",
            f"- Runtime: {summary['runtime_seconds']:.4f} seconds",
            f"- Samples/second: {summary['samples_per_second']}",
            f"- Attempts mean/P95/P99/max: {attempts['mean']:.4f} / "
            f"{attempts['p95']:.4f} / {attempts['p99']:.4f} / {attempts['maximum']}",
            f"- Candidates excluded by compatibility index: "
            f"{summary['candidates_excluded_by_compatibility_index']}",
            f"- Actual transform/placement retries: "
            f"{summary['actual_transform_placement_retries']}",
            f"- Actual placement retries after indexing: {summary['actual_placement_retries']}",
            f"- Empty compatibility pools: {summary['empty_compatibility_pools']}",
            f"- Template utilization: {summary['template_utilization']['unique_used']} / "
            f"{summary['template_utilization']['available']}",
            f"- Background utilization: {summary['background_utilization']['unique_used']} / "
            f"{summary['background_utilization']['available']}",
            f"- Border fraction empirical/target/observed/drift: "
            f"{border['empirical_template_fraction']:.6f} / {border['target_fraction']:.6f} / "
            f"{border['observed_success_fraction']:.6f} / {border['absolute_drift']:.6f}",
            f"- Accidental contact violations: {summary['accidental_contact_violations']}",
            f"- Support pixels outside valid: {summary['support_pixels_outside_valid_region']}",
            f"- Materialized generated images: {summary['materialized_generated_images']}",
            f"- Validation rows loaded: {summary['validation_rows_loaded']}",
            f"- Official-test rows loaded: {summary['official_test_rows_loaded']}",
            "",
            "## Failures by reason",
            "",
            *failure_lines,
            "",
            "## Successful placements by side combination",
            "",
            *side_lines,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "reports" / "gan_inputs" / "manifest.json"
    )
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
    outputs = metadata.get("configuration", {}).get("outputs", {})
    json_output = args.json_output or REPO_ROOT / outputs.get(
        "sampling_audit_json", "reports/gan_inputs/sampling_audit.json"
    )
    markdown_output = args.markdown_output or REPO_ROOT / outputs.get(
        "sampling_audit_markdown", "reports/gan_inputs/sampling_audit.md"
    )
    dataset = OnlineGANInputDataset(
        metadata, REPO_ROOT, base_seed=args.seed, length=args.samples
    )
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = perf_counter()
    for index in range(args.samples):
        try:
            samples.append(dataset[index])
        except GANSamplingFailure as error:
            failures.append(
                {"sample_index": index, "reason": str(error), "accounting": error.accounting}
            )
        except ValueError as error:
            failures.append({"sample_index": index, "reason": str(error), "accounting": {}})
    elapsed = perf_counter() - started
    summary = build_sampling_audit_summary(
        metadata=metadata,
        requested_samples=args.samples,
        samples=samples,
        failures=failures,
        elapsed_seconds=elapsed,
    )
    _atomic_write(json_output, json.dumps(summary, indent=2) + "\n")
    _atomic_write(markdown_output, _markdown(summary))
    print(json.dumps(summary, indent=2))
    print(f"Wrote JSON audit: {json_output}")
    print(f"Wrote Markdown audit: {markdown_output}")
    print("Materialized generated image files: 0")


if __name__ == "__main__":
    main()
