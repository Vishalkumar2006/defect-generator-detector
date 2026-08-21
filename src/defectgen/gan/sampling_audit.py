"""Pure aggregation for deterministic online GAN sampling audits."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .visualization import summarize_placements


def _expected_target_side_counts(
    metadata: dict[str, Any], requested_samples: int
) -> dict[str, float]:
    templates = metadata["templates"]
    border = [row for row in templates if any(row["source_contact_sides"].values())]
    non_border = [row for row in templates if not any(row["source_contact_sides"].values())]
    sampling = metadata.get("sampling", {"border_fraction_mode": "empirical"})
    if not border:
        border_fraction = 0.0
    elif not non_border:
        border_fraction = 1.0
    else:
        border_fraction = (
            len(border) / len(templates)
            if sampling["border_fraction_mode"] == "empirical"
            else float(sampling["border_fraction"])
        )
    horizontal_probability = float(metadata["transform"]["horizontal_flip_probability"])
    vertical_probability = float(metadata["transform"]["vertical_flip_probability"])

    def conditional(rows: list[dict[str, Any]], side: str) -> float:
        if not rows:
            return 0.0
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[side]
        flip_probability = (
            horizontal_probability if side in {"left", "right"} else vertical_probability
        )
        return sum(
            (1 - flip_probability) * bool(row["source_contact_sides"][side])
            + flip_probability * bool(row["source_contact_sides"][opposite])
            for row in rows
        ) / len(rows)

    return {
        side: requested_samples
        * (
            border_fraction * conditional(border, side)
            + (1 - border_fraction) * conditional(non_border, side)
        )
        for side in ("top", "bottom", "left", "right")
    }


def build_sampling_audit_summary(
    *,
    metadata: dict[str, Any],
    requested_samples: int,
    samples: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    elapsed_seconds: float,
    metadata_compatibility_audit: dict[str, Any] | None = None,
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
    observed_side_counts = placement["successful_placements_by_target_contact_side"]
    symmetry_rows = [
        sample["placement_diagnostics"]["horizontal_symmetry_audit"]
        for sample in samples
        if "horizontal_symmetry_audit" in sample["placement_diagnostics"]
    ]
    sampled_infeasible_templates = sorted(
        {
            failure.get("accounting", {}).get("template_identity")
            for failure in failures
            if failure.get("accounting", {}).get("template_identity")
            and failure["reason"] == "no_feasible_transformation_or_compatibility_pool"
        }
    )
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
        "target_side_symmetry": {
            "expected_counts": _expected_target_side_counts(metadata, requested_samples),
            "observed_counts": observed_side_counts,
            "horizontal_counterpart_comparisons": len(symmetry_rows),
            "availability_asymmetries": sum(
                not row["availability_symmetric"] for row in symmetry_rows
            ),
            "maximum_absolute_pool_size_difference": max(
                (abs(int(row["pool_size_difference"])) for row in symmetry_rows),
                default=0,
            ),
        },
        "sampled_templates_with_no_feasible_transform_or_pool": sampled_infeasible_templates,
        "metadata_compatibility_audit": metadata_compatibility_audit,
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
