"""Pure aggregation for deterministic online GAN sampling audits."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .visualization import summarize_placements


def build_sampling_audit_summary(
    *,
    metadata: dict[str, Any],
    requested_samples: int,
    samples: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    placement = summarize_placements(samples, [failure["reason"] for failure in failures])
    failure_reasons = Counter(failure["reason"] for failure in failures)
    failure_sides: Counter[str] = Counter()
    failed_candidates_examined = 0
    failed_candidates_excluded = 0
    failed_empty_pools = 0
    failed_placement_retries = 0
    attempts = [
        int(sample["placement_diagnostics"].get("attempts_per_successful_sample", 1))
        for sample in samples
    ]
    for failure in failures:
        accounting = failure.get("accounting", {})
        failure_sides.update(accounting.get("failure_side_combinations", {}))
        failed_candidates_examined += int(accounting.get("compatibility_candidates_examined", 0))
        failed_candidates_excluded += int(accounting.get("compatibility_candidates_excluded", 0))
        failed_empty_pools += int(accounting.get("empty_compatibility_pools", 0))
        failed_placement_retries += int(accounting.get("actual_placement_retries", 0))
        attempts.append(int(accounting.get("attempts", 1)))
    attempt_array = np.asarray(attempts or [0], dtype=float)
    templates = metadata["templates"]
    empirical_border_fraction = sum(
        any(template["source_contact_sides"].values()) for template in templates
    ) / len(templates)
    sampling = metadata.get("sampling", {"border_fraction_mode": "empirical"})
    target_border_fraction = (
        empirical_border_fraction
        if sampling["border_fraction_mode"] == "empirical"
        else float(sampling["border_fraction"])
    )
    border_successes = sum(
        sample["placement_diagnostics"].get("selected_template_class") == "border"
        for sample in samples
    )
    observed_border_fraction = border_successes / len(samples) if samples else 0.0
    successful_templates = {
        sample["placement_diagnostics"]["template_identity"] for sample in samples
    }
    successful_backgrounds = {
        sample["placement_diagnostics"]["background_identity"] for sample in samples
    }
    success_count = len(samples)
    return {
        "pipeline_version": metadata["pipeline_version"],
        "requested_samples": requested_samples,
        "successful_samples": success_count,
        "failed_samples": len(failures),
        "success_rate": success_count / requested_samples if requested_samples else 0.0,
        "runtime_seconds": elapsed_seconds,
        "samples_per_second": success_count / elapsed_seconds if elapsed_seconds > 0 else None,
        "attempts_per_requested_sample": {
            "mean": float(attempt_array.mean()),
            "p95": float(np.quantile(attempt_array, 0.95, method="linear")),
            "p99": float(np.quantile(attempt_array, 0.99, method="linear")),
            "maximum": int(attempt_array.max()),
        },
        "failures_by_reason": dict(sorted(failure_reasons.items())),
        "failures_by_side_combination": dict(sorted(failure_sides.items())),
        "candidates_examined_by_compatibility_index": placement[
            "candidates_examined_by_compatibility_index"
        ]
        + failed_candidates_examined,
        "candidates_excluded_by_compatibility_index": placement[
            "candidates_excluded_by_compatibility_index"
        ]
        + failed_candidates_excluded,
        "empty_compatibility_pools": placement["empty_compatibility_pools"]
        + failed_empty_pools,
        "actual_transform_placement_retries": placement[
            "actual_transform_placement_retries"
        ]
        + sum(
            int(failure.get("accounting", {}).get("actual_transform_placement_retries", 0))
            for failure in failures
        ),
        "actual_placement_retries": placement["actual_placement_retries"]
        + failed_placement_retries,
        "successful_placements_by_target_contact_side": placement[
            "successful_placements_by_target_contact_side"
        ],
        "successful_placements_by_side_combination": placement[
            "successful_placements_by_side_combination"
        ],
        "template_utilization": {
            "unique_used": len(successful_templates),
            "available": len(templates),
            "fraction": len(successful_templates) / len(templates),
            "counts": placement["template_utilization"],
        },
        "background_utilization": {
            "unique_used": len(successful_backgrounds),
            "available": len(metadata["normal_backgrounds"]),
            "fraction": len(successful_backgrounds) / len(metadata["normal_backgrounds"]),
            "counts": placement["background_utilization"],
        },
        "border_distribution": {
            "mode": sampling["border_fraction_mode"],
            "empirical_template_fraction": empirical_border_fraction,
            "target_fraction": target_border_fraction,
            "observed_success_fraction": observed_border_fraction,
            "absolute_drift": abs(observed_border_fraction - target_border_fraction),
        },
        "accidental_contact_violations": placement["accidental_contact_violations"],
        "support_pixels_outside_valid_region": placement[
            "support_pixels_outside_valid_region"
        ],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "split_sha256": metadata["split_sha256"],
        "gan_manifest_content_sha256": metadata["gan_manifest_content_sha256"],
        "materialized_generated_images": 0,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
    }
