"""Audit G1.5a identity initialization and range-aware residual semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.discriminator_views import prepare_aligned_discriminator_views  # noqa: E402
from defectgen.gan.training_pairs import (  # noqa: E402
    GANTrainingPairDataset,
    create_internal_gan_split,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)
from defectgen.models import (  # noqa: E402
    ARCHITECTURE_VERSION,
    RESIDUAL_SEMANTICS_VERSION,
    build_gan_models,
    load_gan_architecture_config,
)
from defectgen.training.gan_losses import (  # noqa: E402
    boundary_seam_loss,
    load_gan_loss_config,
    localized_generator_adversarial_loss,
    masked_total_variation_loss,
    patch_logit_localization_weights,
    support_normalized_change_loss,
)
from defectgen.training.gan_trainer import collate_gan_training_samples  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hashes() -> dict[str, str]:
    directory = REPO_ROOT / "checkpoints" / "gan_smoke"
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in sorted(directory.glob("*.pt"))
    }


def _legacy_checkpoint_metadata() -> dict[str, Any] | None:
    path = REPO_ROOT / "checkpoints" / "gan_smoke" / "stopped_w010_j001.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "smoke_version": payload.get("smoke_version"),
        "architecture_version": payload.get("architecture_version"),
        "residual_semantics_version": payload.get("residual_semantics_version"),
    }


def _gradient_coverage(parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
    selected = list(parameters)
    total_elements = sum(parameter.numel() for parameter in selected)
    finite_elements = 0
    nonzero_elements = 0
    tensors_with_nonzero = 0
    for parameter in selected:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite_elements += int(torch.isfinite(gradient).sum())
        nonzero = torch.isfinite(gradient) & (gradient != 0)
        nonzero_elements += int(nonzero.sum())
        tensors_with_nonzero += int(bool(nonzero.any()))
    return {
        "parameter_tensors": len(selected),
        "parameter_elements": total_elements,
        "tensors_with_finite_nonzero_gradient": tensors_with_nonzero,
        "finite_element_fraction": finite_elements / total_elements,
        "nonzero_element_fraction": nonzero_elements / total_elements,
    }


def _output_metrics(generator, generated, composite: torch.Tensor) -> dict[str, Any]:
    composite = composite.detach()
    support = generated.support_mask.expand_as(composite)
    refined = generated.refined_image.detach().float()
    difference = (refined - composite.float()).abs()
    applied = generated.applied_residual.detach().float()
    raw = generated.raw_residual.detach().float()
    direction = torch.tanh(raw)
    cap = torch.full_like(composite.float(), generator.residual_scale)
    positive_cap = torch.minimum(cap, 1.0 - composite.float())
    negative_cap = torch.minimum(cap, composite.float() + 1.0)
    directional_cap = torch.where(direction >= 0, positive_cap, negative_cap)
    cap_pixels = support & (directional_cap > 0)
    outside = ~support
    old_candidate = composite.float() + generator.residual_scale * direction
    return {
        "exact_identity": torch.equal(generated.refined_image.detach(), composite),
        "mean_actual_support_change": float(difference[support].mean()),
        "maximum_actual_support_change": float(difference[support].max()),
        "raw_residual_mean_absolute": float(raw.abs()[support].mean()),
        "raw_residual_maximum_absolute": float(raw.abs()[support].max()),
        "applied_residual_mean_absolute": float(applied.abs()[support].mean()),
        "applied_residual_maximum_absolute": float(applied.abs()[support].max()),
        "maximum_outside_support_difference": float(
            difference[outside].max() if bool(outside.any()) else 0.0
        ),
        "output_range_violation_count": int(
            ((refined < -1.0) | (refined > 1.0)).sum()
        ),
        "would_have_clamped_fraction_old_additive": float(
            ((old_candidate < -1.0) | (old_candidate > 1.0))[support]
            .float()
            .mean()
        ),
        "directional_cap_saturation_fraction": float(
            (applied.abs() >= 0.99 * directional_cap)[cap_pixels].float().mean()
            if bool(cap_pixels.any())
            else 0.0
        ),
        "tanh_raw_residual_saturation_fraction": float(
            (direction.abs() >= 0.99)[support].float().mean()
        ),
    }


def _regularizers(generated, composite: torch.Tensor, boundary_width: int) -> dict[str, float]:
    return {
        "change": float(
            support_normalized_change_loss(
                generated.refined_image.float(),
                composite.float(),
                generated.support_mask,
            ).detach()
        ),
        "seam": float(
            boundary_seam_loss(
                generated.refined_image.float(),
                composite.float(),
                generated.support_mask,
                boundary_width=boundary_width,
            ).detach()
        ),
        "total_variation": float(
            masked_total_variation_loss(
                generated.applied_residual.float(), generated.support_mask
            ).detach()
        ),
    }


def _adversarial_loss(discriminator, generated, batch, loss_config):
    aligned = prepare_aligned_discriminator_views(
        batch.real_image,
        generated.refined_image,
        batch.real_valid_mask,
        batch.fake_valid_mask,
        batch.fake_discriminator_mask,
        generator_support_mask=generated.support_mask.float(),
        discriminator_mask_threshold=loss_config.canonical_mask_threshold,
    )
    logits = discriminator(aligned.fake_discriminator_view, aligned.discriminator_mask)
    weights = patch_logit_localization_weights(
        aligned.discriminator_mask,
        logits.float(),
        localization_radius=loss_config.localization_radius,
        mask_threshold=loss_config.canonical_mask_threshold,
    )
    return localized_generator_adversarial_loss(logits.float(), weights).total


def build_audit(
    architecture_path: Path,
    training_pair_path: Path,
    *,
    head_learning_rate: float,
) -> dict[str, Any]:
    started = perf_counter()
    configure_reproducibility(42, deterministic=True, warn_only=False)
    checkpoint_hashes_before = _checkpoint_hashes()
    legacy_checkpoint_metadata = _legacy_checkpoint_metadata()
    architecture = load_gan_architecture_config(architecture_path)
    pair_config = load_gan_training_pair_config(training_pair_path)
    loss_config = load_gan_loss_config(REPO_ROOT / pair_config.loss_config_path)
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    internal = create_internal_gan_split(
        metadata,
        monitor_fraction=pair_config.monitor_fraction,
        seed=pair_config.base_seed,
    )
    internal.assert_disjoint()
    dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        pair_config,
        split="train",
        internal_split=internal,
        length=2,
    )
    batch = collate_gan_training_samples([dataset[0], dataset[1]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    generator, discriminator = build_gan_models(architecture)
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    synthetic = torch.linspace(
        -0.999,
        0.999,
        architecture.image_height * architecture.image_width,
        device=device,
    ).reshape(1, 1, architecture.image_height, architecture.image_width).expand(1, 3, -1, -1)
    synthetic_mask = torch.zeros(
        1, 1, architecture.image_height, architecture.image_width, device=device
    )
    synthetic_mask[
        :,
        :,
        architecture.image_height // 2 - 8 : architecture.image_height // 2 + 8,
        architecture.image_width // 2 - 12 : architecture.image_width // 2 + 12,
    ] = 1
    with torch.no_grad():
        synthetic_initial = generator(synthetic, synthetic_mask)
        real_initial = generator(batch.composite_image, batch.generator_mask)
    initial = {
        "synthetic": _output_metrics(generator, synthetic_initial, synthetic),
        "real_training_batch": _output_metrics(
            generator, real_initial, batch.composite_image
        ),
        "regularizers": _regularizers(
            real_initial, batch.composite_image, loss_config.boundary_ring_width
        ),
    }

    generator.zero_grad(set_to_none=True)
    first = generator(batch.composite_image, batch.generator_mask)
    first_loss = _adversarial_loss(discriminator, first, batch, loss_config)
    first_loss.backward()
    named_parameters = list(generator.named_parameters())
    head_parameters = [
        parameter for name, parameter in named_parameters if name.startswith("output_head.")
    ]
    earlier_parameters = [
        parameter for name, parameter in named_parameters if not name.startswith("output_head.")
    ]
    first_gradients = {
        "output_head": _gradient_coverage(head_parameters),
        "earlier_layers": _gradient_coverage(earlier_parameters),
    }
    optimizer = torch.optim.SGD(head_parameters, lr=head_learning_rate)
    optimizer.step()

    generator.zero_grad(set_to_none=True)
    after_step = generator(batch.composite_image, batch.generator_mask)
    after_step_metrics = _output_metrics(generator, after_step, batch.composite_image)
    after_step_regularizers = _regularizers(
        after_step, batch.composite_image, loss_config.boundary_ring_width
    )
    second_loss = _adversarial_loss(discriminator, after_step, batch, loss_config)
    second_loss.backward()
    second_gradients = {
        "output_head": _gradient_coverage(head_parameters),
        "earlier_layers": _gradient_coverage(earlier_parameters),
    }
    checkpoint_hashes_after = _checkpoint_hashes()

    invariants = {
        "synthetic_initial_identity": initial["synthetic"]["exact_identity"],
        "real_initial_identity": initial["real_training_batch"]["exact_identity"],
        "initial_raw_residual_zero": initial["real_training_batch"][
            "raw_residual_maximum_absolute"
        ] == 0,
        "initial_applied_residual_zero": initial["real_training_batch"][
            "applied_residual_maximum_absolute"
        ] == 0,
        "initial_regularizers_zero": all(value == 0 for value in initial["regularizers"].values()),
        "initial_output_range_violations_zero": initial["real_training_batch"][
            "output_range_violation_count"
        ] == 0,
        "first_backward_reaches_output_head": first_gradients["output_head"][
            "tensors_with_finite_nonzero_gradient"
        ] > 0,
        "first_backward_earlier_layers_are_staged_zero": first_gradients[
            "earlier_layers"
        ]["tensors_with_finite_nonzero_gradient"] == 0,
        "one_step_output_range_violations_zero": after_step_metrics[
            "output_range_violation_count"
        ] == 0,
        "one_step_residual_nonzero": after_step_metrics[
            "maximum_actual_support_change"
        ] > 0,
        "second_backward_reaches_earlier_layers": second_gradients["earlier_layers"][
            "tensors_with_finite_nonzero_gradient"
        ] > 0,
        "outside_support_remains_exact": after_step_metrics[
            "maximum_outside_support_difference"
        ] == 0,
        "failed_checkpoints_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "preserved_checkpoint_lacks_new_residual_semantics": (
            legacy_checkpoint_metadata is None
            or legacy_checkpoint_metadata["residual_semantics_version"] is None
        ),
    }
    return {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "architecture_version": ARCHITECTURE_VERSION,
        "residual_semantics_version": RESIDUAL_SEMANTICS_VERSION,
        "equation": {
            "direction": "r = tanh(raw)",
            "positive_cap": "min(d, 1 - x)",
            "negative_cap": "min(d, x + 1)",
            "delta": "where(r >= 0, r * positive_cap, r * negative_cap)",
            "refined": "where(support, x + delta, x)",
            "maximum_absolute_delta": architecture.residual_scale,
            "hard_clamp": False,
        },
        "initialization": initial,
        "first_adversarial_backward": {
            "loss": float(first_loss.detach()),
            "gradient_coverage": first_gradients,
        },
        "after_one_controlled_output_head_step": {
            "optimizer": "SGD",
            "learning_rate": head_learning_rate,
            "metrics": after_step_metrics,
            "regularizers": after_step_regularizers,
        },
        "second_adversarial_backward": {
            "loss": float(second_loss.detach()),
            "gradient_coverage": second_gradients,
        },
        "failed_g1_5_comparison": {
            "previous_untrained_mean_support_change_approximate": 0.119,
            "previous_pre_clamp_violation_fraction": 0.07349,
            "corrected_initial_mean_support_change": initial["real_training_batch"][
                "mean_actual_support_change"
            ],
            "corrected_initial_output_range_violations": initial[
                "real_training_batch"
            ]["output_range_violation_count"],
        },
        "checkpoint_compatibility": {
            "old_checkpoint_required_residual_semantics_version": RESIDUAL_SEMANTICS_VERSION,
            "old_checkpoints_are_resume_compatible": False,
            "observed_preserved_checkpoint_metadata": legacy_checkpoint_metadata,
            "checkpoint_hashes_before": checkpoint_hashes_before,
            "checkpoint_hashes_after": checkpoint_hashes_after,
            "unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        },
        "training_batch_sample_ids": [item["sample_index"] for item in batch.metadata],
        "device": str(device),
        "runtime_seconds": perf_counter() - started,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_generated_images": 0,
        "controlled_generator_optimizer_steps": 1,
        "discriminator_optimizer_steps": 0,
        "invariants": invariants,
    }


def _markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return f"# G1.5a identity/range corrective audit\n\n- Status: **FAIL**\n- Error: `{report.get('error')}`\n"
    initial = report["initialization"]["real_training_batch"]
    after = report["after_one_controlled_output_head_step"]["metrics"]
    first = report["first_adversarial_backward"]["gradient_coverage"]
    second = report["second_adversarial_backward"]["gradient_coverage"]
    return "\n".join(
        [
            "# G1.5a identity/range corrective audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Architecture: `{report['architecture_version']}`",
            f"- Residual semantics: `{report['residual_semantics_version']}`",
            f"- Initial mean/max support change: {initial['mean_actual_support_change']} / {initial['maximum_actual_support_change']}",
            f"- Initial raw/applied maximum: {initial['raw_residual_maximum_absolute']} / {initial['applied_residual_maximum_absolute']}",
            f"- Initial output-range violations: {initial['output_range_violation_count']}",
            f"- After-step mean/max support change: {after['mean_actual_support_change']} / {after['maximum_actual_support_change']}",
            f"- After-step output-range violations: {after['output_range_violation_count']}",
            f"- After-step directional-cap / tanh saturation: {after['directional_cap_saturation_fraction']} / {after['tanh_raw_residual_saturation_fraction']}",
            f"- First backward head / earlier nonzero tensors: {first['output_head']['tensors_with_finite_nonzero_gradient']} / {first['earlier_layers']['tensors_with_finite_nonzero_gradient']}",
            f"- Second backward head / earlier nonzero tensors: {second['output_head']['tensors_with_finite_nonzero_gradient']} / {second['earlier_layers']['tensors_with_finite_nonzero_gradient']}",
            f"- Failed checkpoints unchanged: {report['checkpoint_compatibility']['unchanged']}",
            f"- Validation / official-test rows loaded: {report['validation_rows_loaded']} / {report['official_test_rows_loaded']}",
            f"- Materialized generated images: {report['materialized_generated_images']}",
            "",
            "## Invariants",
            "",
            *[
                f"- {'PASS' if value else 'FAIL'}: `{name}`"
                for name, value in report["invariants"].items()
            ],
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture-config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_architecture.json",
    )
    parser.add_argument(
        "--training-pair-config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_training_pairs.json",
    )
    parser.add_argument("--head-learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "gan_architecture"
        / "identity_range_correction"
        / "audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "gan_architecture"
        / "identity_range_correction"
        / "audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.head_learning_rate <= 0:
        raise ValueError("head-learning-rate must be positive")
    try:
        report = build_audit(
            args.architecture_config,
            args.training_pair_config,
            head_learning_rate=args.head_learning_rate,
        )
    except Exception as error:
        report = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "materialized_generated_images": 0,
        }
    _atomic_write(args.json_output, json.dumps(report, indent=2) + "\n")
    _atomic_write(args.markdown_output, _markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
