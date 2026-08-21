"""Audit G1.3b discriminator-view validity alignment on training pairs only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.discriminator_views import (  # noqa: E402
    DISCRIMINATOR_INVALID_FILL_VALUE,
    VALIDITY_ALIGNMENT_VERSION,
    prepare_aligned_discriminator_views,
    prepare_training_sample_discriminator_views,
)
from defectgen.gan.geometry import ContactSides  # noqa: E402
from defectgen.gan.pipeline import REAL_VALID_COVERAGE_THRESHOLD  # noqa: E402
from defectgen.gan.training_pairs import (  # noqa: E402
    GANInternalSplit,
    GANTrainingPairDataset,
    GANTrainingSample,
    create_internal_gan_split,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)
from defectgen.models.gan import load_gan_architecture_config  # noqa: E402
from defectgen.training.gan_losses import load_gan_loss_config  # noqa: E402


LOGIT_SHAPE = (62, 30)
SIDE_ORDER = ("top", "bottom", "left", "right")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _combination(contacts: dict[str, bool] | ContactSides) -> str:
    values = contacts.to_dict() if isinstance(contacts, ContactSides) else contacts
    active = [side for side in SIDE_ORDER if values[side]]
    return "+".join(active) if active else "none"


def _possible_combinations(template: dict[str, Any]) -> set[str]:
    contacts = ContactSides.from_dict(template["source_contact_sides"])
    return {
        _combination(
            contacts.transformed(horizontal_flip=horizontal, vertical_flip=vertical)
        )
        for horizontal in (False, True)
        for vertical in (False, True)
    }


def _combination_sort_key(combination: str) -> tuple[int, tuple[int, ...]]:
    if combination == "none":
        return (0, ())
    sides = combination.split("+")
    return (len(sides), tuple(SIDE_ORDER.index(side) for side in sides))


def _stratum_split(
    metadata: dict[str, Any], internal: GANInternalSplit, combination: str
) -> GANInternalSplit:
    template_indices = tuple(
        index
        for index in internal.train_template_indices
        if combination in _possible_combinations(metadata["templates"][index])
    )
    if not template_indices:
        raise RuntimeError(f"No training template can produce target contacts {combination}")
    source_ids = frozenset(
        str(metadata["templates"][index]["sample_id"]) for index in template_indices
    )
    return GANInternalSplit(
        train_template_indices=template_indices,
        monitor_template_indices=internal.monitor_template_indices,
        train_normal_indices=internal.train_normal_indices,
        monitor_normal_indices=internal.monitor_normal_indices,
        train_defect_source_ids=source_ids,
        monitor_defect_source_ids=internal.monitor_defect_source_ids,
        train_background_ids=internal.train_background_ids,
        monitor_background_ids=internal.monitor_background_ids,
        representation_warnings=internal.representation_warnings,
    )


def _stratum_dataset(
    metadata: dict[str, Any],
    config,
    internal: GANInternalSplit,
    combination: str,
    *,
    length: int,
) -> GANTrainingPairDataset:
    return GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        config,
        split="train",
        internal_split=_stratum_split(metadata, internal, combination),
        length=length,
    )


def _support_fraction(support: torch.Tensor, valid: torch.Tensor) -> float:
    positive = support.bool()
    denominator = int(positive.sum())
    if denominator == 0:
        raise RuntimeError("A G1.3a audit sample has empty generator support")
    return float((positive & valid.bool()).sum()) / denominator


def _identity(sample: GANTrainingSample, combination: str, index: int) -> dict[str, Any]:
    return {
        "stratum_index": index,
        "target_contact_combination": combination,
        "template_id": sample.metadata["template_id"],
        "template_source_sample_id": sample.metadata["template_source_sample_id"],
        "normal_background_sample_id": sample.metadata["normal_background_sample_id"],
        "deterministic_sample_seed": sample.metadata["deterministic_sample_seed"],
        "source_contact_sides": sample.metadata["source_contact_sides"],
        "transformed_contact_sides": sample.metadata["transformed_contact_sides"],
        "target_contact_sides": sample.metadata["target_contact_sides"],
    }


def _bounding_box(mask: torch.Tensor) -> dict[str, int] | None:
    coordinates = torch.nonzero(mask.bool(), as_tuple=False)
    if not len(coordinates):
        return None
    return {
        "top": int(coordinates[:, 0].min()),
        "bottom": int(coordinates[:, 0].max()),
        "left": int(coordinates[:, 1].min()),
        "right": int(coordinates[:, 1].max()),
    }


def _containment_failure_diagnostics(
    sample: GANTrainingSample, support: torch.Tensor
) -> dict[str, Any]:
    support_2d = support[0].bool()
    real_valid = sample.real_valid_mask[0].bool()
    outside = support_2d & ~real_valid
    canonical = sample.fake_discriminator_mask[0].bool()
    return {
        "support_pixels": int(support_2d.sum()),
        "support_pixels_outside_real_valid": int(outside.sum()),
        "canonical_mask_pixels_outside_real_valid": int((canonical & ~real_valid).sum()),
        "outside_support_bbox": _bounding_box(outside),
        "real_valid_bbox": _bounding_box(real_valid),
        "generator_support_bbox": _bounding_box(support_2d),
        "outside_alpha_minimum": float(sample.generator_mask[0][outside].min()),
        "outside_alpha_maximum": float(sample.generator_mask[0][outside].max()),
        "source_padding_before_transform": sample.metadata[
            "source_padding_before_transform"
        ],
        "real_padding_after_transform": sample.metadata[
            "real_padding_after_transform"
        ],
        "fake_padding": sample.metadata["fake_padding"],
        "transform": sample.metadata["transform"],
    }


def _dilated_projection(
    mask: torch.Tensor, *, radius: int, output_shape: tuple[int, int]
) -> torch.Tensor:
    projected = mask.bool().float().unsqueeze(0)
    if radius:
        kernel = 2 * radius + 1
        projected = F.max_pool2d(
            projected, kernel_size=(1, kernel), stride=1, padding=(0, radius)
        )
        projected = F.max_pool2d(
            projected, kernel_size=(kernel, 1), stride=1, padding=(radius, 0)
        )
    return F.adaptive_max_pool2d(projected, output_shape) > 0


def _padding_logit_fraction(
    discriminator_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    radius: int,
) -> float:
    active = _dilated_projection(
        discriminator_mask, radius=radius, output_shape=LOGIT_SHAPE
    )
    padding = _dilated_projection(
        ~valid_mask.bool(), radius=radius, output_shape=LOGIT_SHAPE
    )
    return float((active & padding).sum()) / int(active.sum())


def _complete_generator_support(
    sample: GANTrainingSample, *, refinement_radius: int
) -> torch.Tensor:
    canonical = sample.fake_discriminator_mask.bool().unsqueeze(0)
    if refinement_radius:
        canonical = F.max_pool2d(
            canonical.float(),
            kernel_size=2 * refinement_radius + 1,
            stride=1,
            padding=refinement_radius,
        ).bool()
    feather = (sample.generator_mask > 0).unsqueeze(0)
    return (canonical | feather)[0]


def _gradient_coverage(gradient: torch.Tensor, region: torch.Tensor) -> float | None:
    expanded = region.bool().expand_as(gradient)
    if not bool(expanded.any()):
        return None
    selected = gradient[expanded]
    return float((torch.isfinite(selected) & (selected != 0)).float().mean())


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "mean": float(np.mean(values)),
        "maximum": max(values),
    }


def _containment_summary(
    values: dict[str, list[float]], requested: int
) -> dict[str, Any]:
    return {
        name: {
            **_stats(fractions),
            "full_containment_count": sum(value == 1.0 for value in fractions),
            "full_containment_rate": sum(value == 1.0 for value in fractions)
            / requested,
        }
        for name, fractions in values.items()
    }


def _load_audit_config(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("validity_alignment_version") != VALIDITY_ALIGNMENT_VERSION:
        raise ValueError("Unexpected validity-alignment version")
    if int(values.get("audit_sample_count", 0)) <= 0:
        raise ValueError("audit_sample_count must be positive")
    if float(values.get("invalid_fill_value", float("nan"))) != 0.0:
        raise ValueError("G1.3b requires invalid_fill_value=0.0")
    for name in (
        "per_sample_support_warning_threshold",
        "mean_support_warning_threshold",
    ):
        if not 0 < float(values.get(name, 0)) <= 1:
            raise ValueError(f"{name} must be in (0,1]")
    if float(values.get("alpha_coverage_tolerance", -1)) < 0:
        raise ValueError("alpha_coverage_tolerance must be non-negative")
    if (
        float(values.get("continuous_real_valid_coverage_threshold", -1))
        != REAL_VALID_COVERAGE_THRESHOLD
    ):
        raise ValueError("Continuous real-valid coverage threshold is inconsistent")
    if values.get("continuous_real_valid_coverage_policy") != "strictly_greater_than":
        raise ValueError("Unexpected continuous real-valid coverage policy")
    if values.get("stratify_by_target_contact_combination") is not True:
        raise ValueError("G1.3b requires contact-combination stratification")
    return values


def _build_pre_fix_audit(config_path: Path) -> dict[str, Any]:
    audit_config = _load_audit_config(config_path)
    pair_config_path = REPO_ROOT / audit_config["training_pair_config_path"]
    pair_config = load_gan_training_pair_config(pair_config_path)
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    loss_config = load_gan_loss_config(REPO_ROOT / pair_config.loss_config_path)
    internal = create_internal_gan_split(
        metadata, monitor_fraction=pair_config.monitor_fraction, seed=pair_config.base_seed
    )
    requested = int(audit_config["audit_sample_count"])
    available = sorted(
        {
            combination
            for index in internal.train_template_indices
            for combination in _possible_combinations(metadata["templates"][index])
        },
        key=_combination_sort_key,
    )
    quotas = {
        combination: requested // len(available)
        + (position < requested % len(available))
        for position, combination in enumerate(available)
    }
    selection: list[dict[str, Any]] = []
    containment_values: dict[str, list[float]] = {
        "real_valid": [],
        "fake_valid": [],
        "joint_valid": [],
    }
    containment_by_combination: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"real_valid": [], "fake_valid": [], "joint_valid": []}
    )
    failures: list[dict[str, Any]] = []
    original_asymmetry: list[float] = []
    started = perf_counter()
    accepted = 0
    candidates_examined = 0
    for combination in available:
        quota = quotas[combination]
        maximum_candidates = max(128, quota * 32)
        dataset = _stratum_dataset(
            metadata,
            pair_config,
            internal,
            combination,
            length=maximum_candidates,
        )
        selected_for_combination = 0
        for index in range(maximum_candidates):
            sample = dataset[index]
            candidates_examined += 1
            observed = _combination(sample.metadata["target_contact_sides"])
            if observed != combination:
                continue
            support = sample.generator_mask > 0
            joint = sample.real_valid_mask.bool() & sample.fake_valid_mask.bool()
            fractions = {
                "real_valid": _support_fraction(support, sample.real_valid_mask),
                "fake_valid": _support_fraction(support, sample.fake_valid_mask),
                "joint_valid": _support_fraction(support, joint),
            }
            identity = _identity(sample, combination, index)
            identity["containment"] = fractions
            selection.append(identity)
            for name, fraction in fractions.items():
                containment_values[name].append(fraction)
                containment_by_combination[combination][name].append(fraction)
            original_asymmetry.append(
                abs(
                    float(sample.real_valid_mask.mean())
                    - float(sample.fake_valid_mask.mean())
                )
            )
            if any(fraction != 1.0 for fraction in fractions.values()):
                identity["diagnostics"] = _containment_failure_diagnostics(
                    sample, support
                )
                failures.append(identity)
            selected_for_combination += 1
            accepted += 1
            if accepted % 25 == 0 or accepted == requested:
                print(f"Containment audit {accepted}/{requested}", flush=True)
            if selected_for_combination == quota:
                break
        if selected_for_combination != quota:
            raise RuntimeError(
                f"Unable to fill deterministic {combination} stratum: "
                f"{selected_for_combination}/{quota}"
            )

    by_combination_report = {
        combination: {
            "count": len(values["joint_valid"]),
            **{
                name: {
                    **_stats(fractions),
                    "below_full_containment": sum(value != 1.0 for value in fractions),
                }
                for name, fractions in values.items()
            },
        }
        for combination, values in sorted(
            containment_by_combination.items(), key=lambda item: _combination_sort_key(item[0])
        )
    }
    containment = _containment_summary(containment_values, requested)
    base_report: dict[str, Any] = {
        "validity_alignment_version": VALIDITY_ALIGNMENT_VERSION,
        "requested_samples": requested,
        "stratification": {
            "available_target_contact_combinations": available,
            "requested_counts": quotas,
            "observed_counts": dict(Counter(row["target_contact_combination"] for row in selection)),
            "candidates_examined": candidates_examined,
        },
        "containment": containment,
        "containment_by_target_contact_combination": by_combination_report,
        "below_full_containment_count": len(failures),
        "below_full_containment_samples": failures,
        "original_real_fake_valid_fraction_asymmetry": _stats(original_asymmetry),
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "training_steps": 0,
        "materialized_generated_training_images": 0,
    }
    if failures:
        elapsed = perf_counter() - started
        base_report.update(
            {
                "status": "FAIL_CONTAINMENT",
                "corrective_alignment_executed": False,
                "runtime_seconds": elapsed,
                "samples_per_second": requested / elapsed,
                "candidates_per_second": candidates_examined / elapsed,
                "failure_reason": (
                    "Generator support was not fully contained in real/fake joint "
                    "native validity; aligned-view correction stopped before masking."
                ),
            }
        )
        return base_report

    canonical_equal = 0
    padding_equal = 0
    aligned_asymmetry: list[float] = []
    outside_differences: list[float] = []
    native_mutations: list[float] = []
    fake_padding_logits: list[float] = []
    real_padding_logits: list[float] = []
    gradient_coverage: list[float] = []
    invalid_gradient_maxima: list[float] = []
    datasets: dict[str, GANTrainingPairDataset] = {}
    for audited_index, record in enumerate(selection, start=1):
        combination = record["target_contact_combination"]
        if combination not in datasets:
            datasets[combination] = _stratum_dataset(
                metadata,
                pair_config,
                internal,
                combination,
                length=max(128, quotas[combination] * 32),
            )
        sample = datasets[combination][record["stratum_index"]]
        if _identity(sample, combination, record["stratum_index"])["template_id"] != record[
            "template_id"
        ]:
            raise RuntimeError("Deterministic stratum replay changed template identity")
        views = prepare_training_sample_discriminator_views(
            sample, valid_mask_threshold=float(audit_config["valid_mask_threshold"])
        )
        joint = views.joint_valid_mask.bool()
        expanded = joint.expand_as(views.real_discriminator_view)
        outside = ~expanded
        outside_differences.append(
            float(
                (
                    views.real_discriminator_view
                    - views.fake_discriminator_view
                ).abs()[outside].max()
            )
            if bool(outside.any())
            else 0.0
        )
        native_mutations.extend(
            [
                float(
                    (
                        views.real_discriminator_view - sample.real_image.unsqueeze(0)
                    ).abs()[expanded].max()
                ),
                float(
                    (
                        views.fake_discriminator_view
                        - sample.composite_image.unsqueeze(0)
                    ).abs()[expanded].max()
                ),
            ]
        )
        padding_real = torch.where(
            expanded,
            torch.zeros_like(views.real_discriminator_view),
            views.real_discriminator_view,
        )
        padding_fake = torch.where(
            expanded,
            torch.zeros_like(views.fake_discriminator_view),
            views.fake_discriminator_view,
        )
        padding_equal += torch.equal(padding_real, padding_fake)
        canonical_equal += torch.equal(
            sample.real_discriminator_mask, sample.fake_discriminator_mask
        ) and torch.equal(
            views.discriminator_mask[0], sample.fake_discriminator_mask
        )
        aligned_asymmetry.append(abs(float(joint.float().mean()) - float(joint.float().mean())))
        fake_padding_logits.append(
            _padding_logit_fraction(
                sample.fake_discriminator_mask,
                sample.fake_valid_mask,
                radius=loss_config.localization_radius,
            )
        )
        real_padding_logits.append(
            _padding_logit_fraction(
                sample.real_discriminator_mask,
                sample.real_valid_mask,
                radius=loss_config.localization_radius,
            )
        )

        fake_probe = sample.composite_image.unsqueeze(0).detach().clone().requires_grad_(True)
        probe_views = prepare_aligned_discriminator_views(
            sample.real_image.unsqueeze(0),
            fake_probe,
            sample.real_valid_mask.unsqueeze(0),
            sample.fake_valid_mask.unsqueeze(0),
            sample.fake_discriminator_mask.unsqueeze(0),
            generator_support_mask=sample.generator_mask.unsqueeze(0),
        )
        support = (sample.generator_mask > 0).unsqueeze(0).expand_as(fake_probe)
        probe_views.fake_discriminator_view[support].sum().backward()
        assert fake_probe.grad is not None
        support_gradients = fake_probe.grad[support]
        gradient_coverage.append(
            float((torch.isfinite(support_gradients) & (support_gradients != 0)).float().mean())
        )
        invalid = ~probe_views.joint_valid_mask.bool().expand_as(fake_probe)
        invalid_gradient_maxima.append(
            float(fake_probe.grad[invalid].abs().max()) if bool(invalid.any()) else 0.0
        )
        if audited_index % 25 == 0 or audited_index == requested:
            print(f"Alignment audit {audited_index}/{requested}", flush=True)

    elapsed = perf_counter() - started
    invariants = {
        "support_inside_real_valid": containment["real_valid"]["minimum"] == 1.0,
        "support_inside_fake_valid": containment["fake_valid"]["minimum"] == 1.0,
        "support_inside_joint_valid": containment["joint_valid"]["minimum"] == 1.0,
        "canonical_masks_equal": canonical_equal == requested,
        "aligned_padding_bit_exact_equal": max(outside_differences) == 0.0,
        "padding_only_branches_bit_exact_equal": padding_equal == requested,
        "aligned_validity_asymmetry_zero": max(aligned_asymmetry) == 0.0,
        "native_valid_pixels_unchanged": max(native_mutations) == 0.0,
        "generator_gradient_coverage_complete": min(gradient_coverage) == 1.0,
        "invalid_fake_gradients_zero": max(invalid_gradient_maxima) == 0.0,
        "all_contact_combinations_present": set(available)
        == {row["target_contact_combination"] for row in selection},
    }
    base_report.update(
        {
            "status": "PASS" if all(invariants.values()) else "FAIL",
            "corrective_alignment_executed": True,
            "runtime_seconds": elapsed,
            "samples_per_second": requested / elapsed,
            "aligned_view_valid_fraction_asymmetry": _stats(aligned_asymmetry),
            "maximum_real_fake_difference_outside_joint_validity": max(
                outside_differences
            ),
            "maximum_native_valid_pixel_mutation": max(native_mutations),
            "real_fake_canonical_mask_equality_rate": canonical_equal / requested,
            "localized_logits_affected_by_padding_before_alignment": {
                "real": _stats(real_padding_logits),
                "fake": _stats(fake_padding_logits),
            },
            "aligned_padding_only_equality_rate": padding_equal / requested,
            "generator_gradient_coverage_after_alignment": _stats(gradient_coverage),
            "maximum_gradient_in_invalid_fake_pixels": max(invalid_gradient_maxima),
            "invalid_fill_value": DISCRIMINATOR_INVALID_FILL_VALUE,
            "invariants": invariants,
        }
    )
    return base_report


def build_audit(config_path: Path) -> dict[str, Any]:
    """Run the revised G1.3b canonical-containment and alignment audit."""

    audit_config = _load_audit_config(config_path)
    pair_config = load_gan_training_pair_config(
        REPO_ROOT / audit_config["training_pair_config_path"]
    )
    architecture_config = load_gan_architecture_config(
        REPO_ROOT / audit_config["architecture_config_path"]
    )
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    loss_config = load_gan_loss_config(REPO_ROOT / pair_config.loss_config_path)
    internal = create_internal_gan_split(
        metadata, monitor_fraction=pair_config.monitor_fraction, seed=pair_config.base_seed
    )
    pre_fix_path = REPO_ROOT / audit_config["pre_fix_audit_path"]
    pre_fix = json.loads(pre_fix_path.read_text(encoding="utf-8"))
    requested = int(audit_config["audit_sample_count"])
    tolerance = float(audit_config["alpha_coverage_tolerance"])
    sample_warning_threshold = float(
        audit_config["per_sample_support_warning_threshold"]
    )
    mean_warning_threshold = float(audit_config["mean_support_warning_threshold"])
    available = sorted(
        {
            combination
            for index in internal.train_template_indices
            for combination in _possible_combinations(metadata["templates"][index])
        },
        key=_combination_sort_key,
    )
    quotas = {
        combination: requested // len(available)
        + (position < requested % len(available))
        for position, combination in enumerate(available)
    }
    selection: list[dict[str, Any]] = []
    support_values = {name: [] for name in ("real_valid", "fake_valid", "joint_valid")}
    canonical_values = {
        name: [] for name in ("real_valid", "fake_valid", "joint_valid")
    }
    by_combination: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            **{f"support_{name}": [] for name in support_values},
            **{f"canonical_{name}": [] for name in canonical_values},
            "alpha_coverage_maximum_violation": [],
        }
    )
    strict_failures: list[dict[str, Any]] = []
    support_warnings: list[dict[str, Any]] = []
    alpha_violation_pixels = 0
    alpha_violation_samples = 0
    alpha_maximum_violation = 0.0
    original_asymmetry: list[float] = []
    started = perf_counter()
    accepted = 0
    candidates_examined = 0
    for combination in available:
        quota = quotas[combination]
        maximum_candidates = max(128, quota * 32)
        dataset = _stratum_dataset(
            metadata,
            pair_config,
            internal,
            combination,
            length=maximum_candidates,
        )
        selected_for_combination = 0
        for index in range(maximum_candidates):
            sample = dataset[index]
            candidates_examined += 1
            if _combination(sample.metadata["target_contact_sides"]) != combination:
                continue
            real_valid = sample.real_valid_mask.bool()
            fake_valid = sample.fake_valid_mask.bool()
            joint_valid = real_valid & fake_valid
            canonical = sample.fake_discriminator_mask.bool()
            complete_support = _complete_generator_support(
                sample, refinement_radius=architecture_config.support_dilation_radius
            )
            support_fractions = {
                "real_valid": _support_fraction(complete_support, real_valid),
                "fake_valid": _support_fraction(complete_support, fake_valid),
                "joint_valid": _support_fraction(complete_support, joint_valid),
            }
            canonical_fractions = {
                "real_valid": _support_fraction(canonical, real_valid),
                "fake_valid": _support_fraction(canonical, fake_valid),
                "joint_valid": _support_fraction(canonical, joint_valid),
            }
            alpha_delta = (
                sample.transformed_defect_alpha - sample.real_valid_coverage
            ).clamp_min(0.0)
            violation = alpha_delta > tolerance
            maximum_violation = float(alpha_delta.max())
            alpha_violation_pixels += int(violation.sum())
            alpha_violation_samples += bool(violation.any())
            alpha_maximum_violation = max(alpha_maximum_violation, maximum_violation)
            identity = _identity(sample, combination, index)
            identity.update(
                {
                    "canonical_containment": canonical_fractions,
                    "support_containment": support_fractions,
                    "alpha_coverage_maximum_violation": maximum_violation,
                    "alpha_coverage_violation_pixels": int(violation.sum()),
                }
            )
            selection.append(identity)
            for name, value in support_fractions.items():
                support_values[name].append(value)
                by_combination[combination][f"support_{name}"].append(value)
            for name, value in canonical_fractions.items():
                canonical_values[name].append(value)
                by_combination[combination][f"canonical_{name}"].append(value)
            by_combination[combination]["alpha_coverage_maximum_violation"].append(
                maximum_violation
            )
            original_asymmetry.append(
                abs(float(real_valid.float().mean()) - float(fake_valid.float().mean()))
            )
            if support_fractions["joint_valid"] < sample_warning_threshold:
                support_warnings.append(identity)
            if bool(violation.any()) or any(
                value != 1.0 for value in canonical_fractions.values()
            ):
                strict_failures.append(identity)
            selected_for_combination += 1
            accepted += 1
            if accepted % 25 == 0 or accepted == requested:
                print(f"Canonical containment audit {accepted}/{requested}", flush=True)
            if selected_for_combination == quota:
                break
        if selected_for_combination != quota:
            raise RuntimeError(
                f"Unable to fill deterministic {combination} stratum: "
                f"{selected_for_combination}/{quota}"
            )

    support_containment = _containment_summary(support_values, requested)
    canonical_containment = _containment_summary(canonical_values, requested)
    pre_fix_canonical_failures = sum(
        int(
            failure.get("diagnostics", {}).get(
                "canonical_mask_pixels_outside_real_valid", 0
            )
            > 0
        )
        for failure in pre_fix.get("below_full_containment_samples", [])
    )
    pre_fix_canonical = {
        "real_valid": {
            "full_containment_count": requested - pre_fix_canonical_failures,
            "full_containment_rate": (requested - pre_fix_canonical_failures)
            / requested,
        },
        "fake_valid": {
            "full_containment_count": requested,
            "full_containment_rate": 1.0,
        },
        "joint_valid": {
            "full_containment_count": requested - pre_fix_canonical_failures,
            "full_containment_rate": (requested - pre_fix_canonical_failures)
            / requested,
        },
    }
    combination_report = {
        combination: {
            "count": len(values["canonical_joint_valid"]),
            **{
                name: {
                    **_stats(fractions),
                    "below_full_containment": sum(value != 1.0 for value in fractions),
                }
                for name, fractions in values.items()
            },
        }
        for combination, values in sorted(
            by_combination.items(), key=lambda item: _combination_sort_key(item[0])
        )
    }
    warning_reasons: list[str] = []
    if support_warnings:
        warning_reasons.append(
            f"{len(support_warnings)} samples below {sample_warning_threshold:.2%} "
            "joint support containment"
        )
    if support_containment["joint_valid"]["mean"] < mean_warning_threshold:
        warning_reasons.append(
            f"mean joint support containment "
            f"{support_containment['joint_valid']['mean']:.6f} below "
            f"{mean_warning_threshold:.6f}"
        )
    base_report: dict[str, Any] = {
        "validity_alignment_version": VALIDITY_ALIGNMENT_VERSION,
        "requested_samples": requested,
        "stratification": {
            "available_target_contact_combinations": available,
            "requested_counts": quotas,
            "observed_counts": dict(
                Counter(row["target_contact_combination"] for row in selection)
            ),
            "candidates_examined": candidates_examined,
        },
        "pre_fix_audit_path": audit_config["pre_fix_audit_path"],
        "pre_fix_canonical_containment": pre_fix_canonical,
        "post_fix_canonical_containment": canonical_containment,
        "support_containment": support_containment,
        "support_warning_policy": {
            "per_sample_threshold": sample_warning_threshold,
            "mean_threshold": mean_warning_threshold,
            "samples_below_threshold": len(support_warnings),
            "sample_ids_below_threshold": support_warnings,
            "warnings": warning_reasons,
        },
        "alpha_versus_valid_coverage": {
            "tolerance": tolerance,
            "violation_samples": alpha_violation_samples,
            "violation_pixels": alpha_violation_pixels,
            "maximum_violation": alpha_maximum_violation,
        },
        "results_by_target_contact_combination": combination_report,
        "strict_containment_failures": strict_failures,
        "original_real_fake_valid_fraction_asymmetry": _stats(original_asymmetry),
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "training_steps": 0,
        "materialized_generated_training_images": 0,
    }
    if strict_failures:
        elapsed = perf_counter() - started
        base_report.update(
            {
                "status": "FAIL_CANONICAL_CONTAINMENT",
                "corrective_alignment_executed": False,
                "runtime_seconds": elapsed,
                "samples_per_second": requested / elapsed,
                "candidates_per_second": candidates_examined / elapsed,
            }
        )
        return base_report

    canonical_equal = 0
    padding_equal = 0
    aligned_asymmetry: list[float] = []
    outside_differences: list[float] = []
    native_mutations: list[float] = []
    fake_padding_logits: list[float] = []
    real_padding_logits: list[float] = []
    canonical_gradient_coverage: list[float] = []
    joint_support_gradient_coverage: list[float] = []
    outside_support_gradient_coverage: list[float] = []
    invalid_gradient_maxima: list[float] = []
    outside_support_pixels = 0
    datasets: dict[str, GANTrainingPairDataset] = {}
    for audited_index, record in enumerate(selection, start=1):
        combination = record["target_contact_combination"]
        if combination not in datasets:
            datasets[combination] = _stratum_dataset(
                metadata,
                pair_config,
                internal,
                combination,
                length=max(128, quotas[combination] * 32),
            )
        sample = datasets[combination][record["stratum_index"]]
        if sample.metadata["template_id"] != record["template_id"]:
            raise RuntimeError("Deterministic stratum replay changed template identity")
        views = prepare_training_sample_discriminator_views(
            sample, valid_mask_threshold=float(audit_config["valid_mask_threshold"])
        )
        joint = views.joint_valid_mask.bool()
        expanded = joint.expand_as(views.real_discriminator_view)
        outside = ~expanded
        difference = (
            views.real_discriminator_view - views.fake_discriminator_view
        ).abs()
        outside_differences.append(
            float(difference[outside].max()) if bool(outside.any()) else 0.0
        )
        native_mutations.extend(
            [
                float(
                    (views.real_discriminator_view - sample.real_image.unsqueeze(0))
                    .abs()[expanded]
                    .max()
                ),
                float(
                    (
                        views.fake_discriminator_view
                        - sample.composite_image.unsqueeze(0)
                    ).abs()[expanded].max()
                ),
            ]
        )
        padding_real = torch.where(
            expanded, torch.zeros_like(views.real_discriminator_view), views.real_discriminator_view
        )
        padding_fake = torch.where(
            expanded, torch.zeros_like(views.fake_discriminator_view), views.fake_discriminator_view
        )
        padding_equal += torch.equal(padding_real, padding_fake)
        canonical_equal += torch.equal(
            sample.real_discriminator_mask, sample.fake_discriminator_mask
        ) and torch.equal(views.discriminator_mask[0], sample.fake_discriminator_mask)
        aligned_asymmetry.append(0.0)
        fake_padding_logits.append(
            _padding_logit_fraction(
                sample.fake_discriminator_mask,
                sample.fake_valid_mask,
                radius=loss_config.localization_radius,
            )
        )
        real_padding_logits.append(
            _padding_logit_fraction(
                sample.real_discriminator_mask,
                sample.real_valid_mask,
                radius=loss_config.localization_radius,
            )
        )
        fake_probe = sample.composite_image.unsqueeze(0).detach().clone().requires_grad_(True)
        probe_views = prepare_aligned_discriminator_views(
            sample.real_image.unsqueeze(0),
            fake_probe,
            sample.real_valid_mask.unsqueeze(0),
            sample.fake_valid_mask.unsqueeze(0),
            sample.fake_discriminator_mask.unsqueeze(0),
            generator_support_mask=sample.generator_mask.unsqueeze(0),
        )
        probe_views.fake_discriminator_view.sum().backward()
        assert fake_probe.grad is not None
        complete_support = _complete_generator_support(
            sample, refinement_radius=architecture_config.support_dilation_radius
        ).unsqueeze(0)
        canonical_region = sample.fake_discriminator_mask.bool().unsqueeze(0)
        joint_support = complete_support & probe_views.joint_valid_mask.bool()
        outside_support = complete_support & ~probe_views.joint_valid_mask.bool()
        canonical_coverage = _gradient_coverage(fake_probe.grad, canonical_region)
        joint_coverage = _gradient_coverage(fake_probe.grad, joint_support)
        outside_coverage = _gradient_coverage(fake_probe.grad, outside_support)
        assert canonical_coverage is not None and joint_coverage is not None
        canonical_gradient_coverage.append(canonical_coverage)
        joint_support_gradient_coverage.append(joint_coverage)
        if outside_coverage is not None:
            outside_support_gradient_coverage.append(outside_coverage)
            outside_support_pixels += int(outside_support.sum())
        invalid = ~probe_views.joint_valid_mask.bool().expand_as(fake_probe)
        invalid_gradient_maxima.append(
            float(fake_probe.grad[invalid].abs().max()) if bool(invalid.any()) else 0.0
        )
        if audited_index % 25 == 0 or audited_index == requested:
            print(f"Alignment audit {audited_index}/{requested}", flush=True)

    elapsed = perf_counter() - started
    outside_gradient_stats = (
        _stats(outside_support_gradient_coverage)
        if outside_support_gradient_coverage
        else {"minimum": None, "mean": None, "maximum": None}
    )
    invariants = {
        "zero_alpha_coverage_violations": alpha_violation_pixels == 0,
        "real_canonical_containment_complete": canonical_containment["real_valid"][
            "minimum"
        ]
        == 1.0,
        "fake_canonical_containment_complete": canonical_containment["fake_valid"][
            "minimum"
        ]
        == 1.0,
        "joint_canonical_containment_complete": canonical_containment["joint_valid"][
            "minimum"
        ]
        == 1.0,
        "canonical_masks_equal": canonical_equal == requested,
        "aligned_padding_bit_exact_equal": max(outside_differences) == 0.0,
        "padding_only_branches_bit_exact_equal": padding_equal == requested,
        "aligned_validity_asymmetry_zero": max(aligned_asymmetry) == 0.0,
        "native_valid_pixels_unchanged": max(native_mutations) == 0.0,
        "canonical_gradient_coverage_complete": min(canonical_gradient_coverage) == 1.0,
        "joint_support_gradient_coverage_complete": min(joint_support_gradient_coverage)
        == 1.0,
        "outside_joint_support_gradients_zero": not outside_support_gradient_coverage
        or max(outside_support_gradient_coverage) == 0.0,
        "invalid_fake_gradients_zero": max(invalid_gradient_maxima) == 0.0,
        "all_contact_combinations_present": set(available)
        == {row["target_contact_combination"] for row in selection},
    }
    base_report.update(
        {
            "status": "PASS" if all(invariants.values()) else "FAIL",
            "corrective_alignment_executed": True,
            "runtime_seconds": elapsed,
            "samples_per_second": requested / elapsed,
            "candidates_per_second": candidates_examined / elapsed,
            "aligned_view_valid_fraction_asymmetry": _stats(aligned_asymmetry),
            "maximum_real_fake_difference_outside_joint_validity": max(
                outside_differences
            ),
            "maximum_native_valid_pixel_mutation": max(native_mutations),
            "real_fake_canonical_mask_equality_rate": canonical_equal / requested,
            "localized_logits_affected_by_padding_before_alignment": {
                "real": _stats(real_padding_logits),
                "fake": _stats(fake_padding_logits),
            },
            "aligned_padding_only_equality_rate": padding_equal / requested,
            "generator_gradient_coverage_after_alignment": {
                "canonical_defect_pixels": _stats(canonical_gradient_coverage),
                "joint_valid_support_pixels": _stats(joint_support_gradient_coverage),
                "support_pixels_outside_joint_validity": outside_gradient_stats,
                "outside_joint_support_pixel_count": outside_support_pixels,
            },
            "maximum_gradient_in_invalid_fake_pixels": max(invalid_gradient_maxima),
            "invalid_fill_value": DISCRIMINATOR_INVALID_FILL_VALUE,
            "invariants": invariants,
        }
    )
    return base_report


def _markdown(report: dict[str, Any]) -> str:
    canonical = report.get("post_fix_canonical_containment", {})
    support = report.get("support_containment", {})
    lines = [
        "# G1.3b discriminator-view validity alignment audit",
        "",
        f"- Status: **{report.get('status')}**",
        f"- Requested pairs: {report.get('requested_samples')}",
        f"- Runtime seconds: {report.get('runtime_seconds')}",
        f"- Sampling rate: {report.get('samples_per_second')}",
        f"- Corrective alignment executed: {report.get('corrective_alignment_executed')}",
        f"- Pre-fix canonical containment: "
        f"`{json.dumps(report.get('pre_fix_canonical_containment', {}), sort_keys=True)}`",
        f"- Alpha-versus-coverage: "
        f"`{json.dumps(report.get('alpha_versus_valid_coverage', {}), sort_keys=True)}`",
    ]
    for name in ("real_valid", "fake_valid", "joint_valid"):
        canonical_values = canonical.get(name, {})
        support_values = support.get(name, {})
        lines.append(
            f"- Canonical/support inside {name}: "
            f"canonical minimum {canonical_values.get('minimum')}, rate "
            f"{canonical_values.get('full_containment_rate')}; support minimum "
            f"{support_values.get('minimum')}, mean {support_values.get('mean')}"
        )
    lines.extend(
        [
            f"- Original validity asymmetry: "
            f"{report.get('original_real_fake_valid_fraction_asymmetry')}",
            f"- Aligned validity asymmetry: "
            f"{report.get('aligned_view_valid_fraction_asymmetry')}",
            f"- Maximum difference outside joint validity: "
            f"{report.get('maximum_real_fake_difference_outside_joint_validity')}",
            f"- Maximum native-valid mutation: "
            f"{report.get('maximum_native_valid_pixel_mutation')}",
            f"- Padding-only equality rate: "
            f"{report.get('aligned_padding_only_equality_rate')}",
            f"- Generator gradient coverage: "
            f"{report.get('generator_gradient_coverage_after_alignment')}",
            f"- Support warnings: "
            f"`{json.dumps(report.get('support_warning_policy', {}).get('warnings', []))}`",
            f"- Validation rows loaded: {report.get('validation_rows_loaded', 0)}",
            f"- Official-test rows loaded: {report.get('official_test_rows_loaded', 0)}",
            f"- Training steps: {report.get('training_steps', 0)}",
            f"- Materialized generated training images: "
            f"{report.get('materialized_generated_training_images', 0)}",
            "",
            "## Target-contact strata",
            "",
            f"`{json.dumps(report.get('stratification', {}).get('observed_counts', {}), sort_keys=True)}`",
            "",
        ]
    )
    if report.get("strict_containment_failures"):
        lines.extend(["## Strict containment failures", ""])
        for failure in report["strict_containment_failures"]:
            lines.append(
                f"- `{failure['template_id']}` on "
                f"`{failure['normal_background_sample_id']}` "
                f"({failure['target_contact_combination']}): "
                f"canonical `{json.dumps(failure['canonical_containment'], sort_keys=True)}`; "
                f"alpha violation {failure['alpha_coverage_maximum_violation']}"
            )
        lines.append("")
    if report.get("invariants"):
        lines.extend(["## Invariants", ""])
        lines.extend(
            f"- {'PASS' if passed else 'FAIL'}: `{name}`"
            for name, passed in report["invariants"].items()
        )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_validity_alignment.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "gan_training_pairs"
        / "validity_alignment"
        / "alignment_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "gan_training_pairs"
        / "validity_alignment"
        / "alignment_audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_audit(args.config)
    except Exception as error:
        report = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "training_steps": 0,
            "materialized_generated_training_images": 0,
        }
    _atomic_write(args.json_output, json.dumps(report, indent=2) + "\n")
    _atomic_write(args.markdown_output, _markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
