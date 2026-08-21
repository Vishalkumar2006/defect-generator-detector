"""Audit G1.2 localized GAN objectives with deterministic synthetic tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.models import build_gan_models, load_gan_architecture_config  # noqa: E402
from defectgen.training.gan_losses import (  # noqa: E402
    aggregate_generator_losses,
    boundary_seam_loss,
    canonicalize_discriminator_mask,
    load_gan_loss_config,
    localized_discriminator_hinge_loss,
    localized_generator_adversarial_loss,
    localized_r1_gradient_penalty,
    masked_total_variation_loss,
    patch_logit_localization_weights,
    support_normalized_change_loss,
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _synthetic_image(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(-0.7, 0.7, height).view(1, 1, height, 1).expand(1, 1, height, width)
    x = torch.linspace(-0.6, 0.6, width).view(1, 1, 1, width).expand(1, 1, height, width)
    return torch.cat((x, y, (x + y) / 2), dim=1).contiguous()


def _mask_cases(height: int, width: int) -> dict[str, torch.Tensor]:
    cases: dict[str, torch.Tensor] = {}
    central = torch.zeros(1, 1, height, width)
    central[:, :, height // 2 - 4 : height // 2 + 4, width // 2 - 8 : width // 2 + 8] = 1
    cases["central"] = central
    border = torch.zeros_like(central)
    border[:, :, height // 2 - 5 : height // 2 + 5, :6] = 1
    cases["border_left"] = border
    corner = torch.zeros_like(central)
    corner[:, :, :8, :8] = 1
    cases["corner_top_left"] = corner
    opposite = torch.zeros_like(central)
    opposite[:, :, height // 2 - 3 : height // 2 + 3, :] = 1
    cases["left_right"] = opposite
    return cases


def _gradient_coverage(model: torch.nn.Module) -> dict[str, int | float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters]
    total_elements = sum(parameter.numel() for parameter in parameters)
    finite_elements = sum(
        int(torch.isfinite(gradient).sum())
        for gradient in gradients
        if gradient is not None
    )
    nonzero_elements = sum(
        int((gradient != 0).sum()) for gradient in gradients if gradient is not None
    )
    tensors_with_gradients = sum(gradient is not None for gradient in gradients)
    return {
        "parameter_tensors_with_gradients": tensors_with_gradients,
        "total_trainable_parameter_tensors": len(parameters),
        "parameter_tensor_coverage_fraction": tensors_with_gradients / len(parameters),
        "finite_gradient_elements": finite_elements,
        "nonzero_gradient_elements": nonzero_elements,
        "total_trainable_parameter_elements": total_elements,
        "finite_gradient_element_fraction": finite_elements / total_elements,
        "nonzero_gradient_element_fraction": nonzero_elements / total_elements,
    }


def build_audit(architecture_config_path: Path, loss_config_path: Path) -> dict[str, Any]:
    torch.manual_seed(42)
    architecture = load_gan_architecture_config(architecture_config_path)
    loss_config = load_gan_loss_config(loss_config_path)
    generator, discriminator = build_gan_models(architecture)
    generator.eval()
    discriminator.eval()
    composite = _synthetic_image(architecture.image_height, architecture.image_width)
    real_image = (composite * 0.9).contiguous()
    case_reports: dict[str, dict[str, Any]] = {}
    zero_weight_invariance = True
    canonical_masks_identical = True
    all_case_losses_finite = True

    with torch.no_grad():
        for name, binary_mask in _mask_cases(
            architecture.image_height, architecture.image_width
        ).items():
            real_fractional_mask = binary_mask * 0.75
            synthetic_fractional_mask = binary_mask.clone()
            real_mask = canonicalize_discriminator_mask(
                real_fractional_mask, threshold=loss_config.canonical_mask_threshold
            )
            fake_mask = canonicalize_discriminator_mask(
                synthetic_fractional_mask, threshold=loss_config.canonical_mask_threshold
            )
            canonical_masks_identical &= torch.equal(real_mask, fake_mask)
            generated = generator(composite, synthetic_fractional_mask)
            real_logits = discriminator(real_image, real_mask)
            fake_logits = discriminator(generated.refined_image, fake_mask)
            real_weights = patch_logit_localization_weights(
                real_mask,
                real_logits,
                localization_radius=loss_config.localization_radius,
                mask_threshold=loss_config.canonical_mask_threshold,
            )
            fake_weights = patch_logit_localization_weights(
                fake_mask,
                fake_logits,
                localization_radius=loss_config.localization_radius,
                mask_threshold=loss_config.canonical_mask_threshold,
            )
            discriminator_components = localized_discriminator_hinge_loss(
                real_logits, fake_logits, real_weights, fake_weights
            )
            adversarial = localized_generator_adversarial_loss(fake_logits, fake_weights)
            change = support_normalized_change_loss(
                generated.refined_image, composite, generated.support_mask
            )
            boundary = boundary_seam_loss(
                generated.refined_image,
                composite,
                generated.support_mask,
                boundary_width=loss_config.boundary_ring_width,
            )
            total_variation = masked_total_variation_loss(
                generated.raw_residual, generated.support_mask
            )
            aggregated = aggregate_generator_losses(
                adversarial=adversarial.total,
                change=change,
                boundary=boundary,
                total_variation=total_variation,
                weights=loss_config.generator_loss_weights,
            )
            inactive = fake_weights == 0
            unchanged = adversarial.total
            if bool(inactive.any()):
                modified_logits = fake_logits.clone()
                modified_logits[inactive] = 10000
                modified = localized_generator_adversarial_loss(
                    modified_logits, fake_weights
                ).total
                zero_weight_invariance &= torch.equal(unchanged, modified)
            components = {
                "discriminator_real_hinge": float(discriminator_components.real),
                "discriminator_fake_hinge": float(discriminator_components.fake),
                "discriminator_total": float(discriminator_components.total),
                "generator_adversarial": float(adversarial.total),
                "support_normalized_change": float(change),
                "boundary_seam": float(boundary),
                "masked_total_variation": float(total_variation),
                "provisional_aggregated_generator": float(aggregated.total),
            }
            all_case_losses_finite &= all(
                torch.isfinite(torch.tensor(value)).item() for value in components.values()
            )
            active = int(fake_weights.sum())
            case_reports[name] = {
                "active_patch_logits": active,
                "total_patch_logits": fake_weights.numel(),
                "active_patch_logit_fraction": active / fake_weights.numel(),
                "loss_components": components,
            }

    central_mask = _mask_cases(
        architecture.image_height, architecture.image_width
    )["central"]
    canonical_central = canonicalize_discriminator_mask(
        central_mask, threshold=loss_config.canonical_mask_threshold
    )

    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    generated_for_discriminator = generator(composite, central_mask)
    real_logits = discriminator(real_image, canonical_central)
    fake_logits = discriminator(
        generated_for_discriminator.refined_image.detach(), canonical_central
    )
    real_weights = patch_logit_localization_weights(
        canonical_central,
        real_logits,
        localization_radius=loss_config.localization_radius,
    )
    fake_weights = patch_logit_localization_weights(
        canonical_central,
        fake_logits,
        localization_radius=loss_config.localization_radius,
    )
    discriminator_loss = localized_discriminator_hinge_loss(
        real_logits, fake_logits, real_weights, fake_weights
    ).total
    discriminator_loss.backward()
    discriminator_coverage = _gradient_coverage(discriminator)
    discriminator_step_generator_gradients = sum(
        parameter.grad is not None for parameter in generator.parameters()
    )

    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    generated_for_generator = generator(composite, central_mask)
    fake_logits_for_generator = discriminator(
        generated_for_generator.refined_image, canonical_central
    )
    generator_weights = patch_logit_localization_weights(
        canonical_central,
        fake_logits_for_generator,
        localization_radius=loss_config.localization_radius,
    )
    adversarial = localized_generator_adversarial_loss(
        fake_logits_for_generator, generator_weights
    ).total
    change = support_normalized_change_loss(
        generated_for_generator.refined_image,
        composite,
        generated_for_generator.support_mask,
    )
    boundary = boundary_seam_loss(
        generated_for_generator.refined_image,
        composite,
        generated_for_generator.support_mask,
        boundary_width=loss_config.boundary_ring_width,
    )
    total_variation = masked_total_variation_loss(
        generated_for_generator.raw_residual,
        generated_for_generator.support_mask,
    )
    generator_loss = aggregate_generator_losses(
        adversarial=adversarial,
        change=change,
        boundary=boundary,
        total_variation=total_variation,
        weights=loss_config.generator_loss_weights,
    ).total
    generator_loss.backward()
    generator_coverage = _gradient_coverage(generator)
    generator_step_discriminator_gradients = sum(
        parameter.grad is not None for parameter in discriminator.parameters()
    )

    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    discriminator.zero_grad(set_to_none=True)
    r1_image = real_image.detach().requires_grad_(True)
    r1 = localized_r1_gradient_penalty(
        discriminator,
        r1_image,
        canonical_central,
        localization_radius=loss_config.localization_radius,
        mask_threshold=loss_config.canonical_mask_threshold,
    )
    r1.backward()
    r1_gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in discriminator.parameters()
    )

    invariants = {
        "canonical_real_fake_masks_identical": canonical_masks_identical,
        "all_case_losses_finite": all_case_losses_finite,
        "zero_weight_logits_have_zero_influence": zero_weight_invariance,
        "discriminator_step_has_no_generator_gradients": (
            discriminator_step_generator_gradients == 0
        ),
        "generator_step_has_no_discriminator_parameter_gradients": (
            generator_step_discriminator_gradients == 0
        ),
        "generator_gradients_finite": (
            generator_coverage["finite_gradient_element_fraction"] == 1.0
        ),
        "generator_has_nonzero_gradients": (
            generator_coverage["nonzero_gradient_elements"] > 0
        ),
        "discriminator_gradients_finite": (
            discriminator_coverage["finite_gradient_element_fraction"] == 1.0
        ),
        "discriminator_has_nonzero_gradients": (
            discriminator_coverage["nonzero_gradient_elements"] > 0
        ),
        "r1_finite": bool(torch.isfinite(r1)) and r1_gradients_finite,
    }
    return {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "loss_version": loss_config.loss_version,
        "architecture_version": architecture.architecture_version,
        "input_shape": [1, 3, architecture.image_height, architecture.image_width],
        "canonical_mask_policy": {
            "operator": "greater_than_or_equal",
            "threshold": loss_config.canonical_mask_threshold,
        },
        "localization_radius_input_pixels": loss_config.localization_radius,
        "boundary_ring_width_input_pixels": loss_config.boundary_ring_width,
        "aggregation_weights_provisional": loss_config.aggregation_weights_provisional,
        "provisional_generator_loss_weights": {
            "adversarial": loss_config.generator_loss_weights.adversarial,
            "change": loss_config.generator_loss_weights.change,
            "boundary": loss_config.generator_loss_weights.boundary,
            "total_variation": loss_config.generator_loss_weights.total_variation,
        },
        "mask_cases": case_reports,
        "localized_r1_unscaled": float(r1.detach()),
        "generator_gradient_coverage": generator_coverage,
        "discriminator_gradient_coverage": discriminator_coverage,
        "invariants": invariants,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_generated_images": 0,
        "training_steps": 0,
        "paired_target_reconstruction_loss": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    case_lines: list[str] = []
    for name, case in report.get("mask_cases", {}).items():
        case_lines.extend(
            [
                f"### {name}",
                "",
                f"- Active logits: {case['active_patch_logits']} / "
                f"{case['total_patch_logits']} "
                f"({case['active_patch_logit_fraction']:.6f})",
                *[
                    f"- `{component}`: {value:.9g}"
                    for component, value in case["loss_components"].items()
                ],
                "",
            ]
        )
    invariant_lines = [
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in report.get("invariants", {}).items()
    ]
    generator = report.get("generator_gradient_coverage", {})
    discriminator = report.get("discriminator_gradient_coverage", {})
    return "\n".join(
        [
            "# G1.2 localized GAN loss audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Loss version: `{report.get('loss_version', 'unknown')}`",
            f"- Input shape: `{report.get('input_shape')}`",
            f"- Localized R1, unscaled: {report.get('localized_r1_unscaled')}",
            f"- Generator gradient tensors: "
            f"{generator.get('parameter_tensors_with_gradients')} / "
            f"{generator.get('total_trainable_parameter_tensors')}",
            f"- Generator finite/non-zero gradient elements: "
            f"{generator.get('finite_gradient_elements')} / "
            f"{generator.get('nonzero_gradient_elements')} / "
            f"{generator.get('total_trainable_parameter_elements')}",
            f"- Discriminator gradient tensors: "
            f"{discriminator.get('parameter_tensors_with_gradients')} / "
            f"{discriminator.get('total_trainable_parameter_tensors')}",
            f"- Discriminator finite/non-zero gradient elements: "
            f"{discriminator.get('finite_gradient_elements')} / "
            f"{discriminator.get('nonzero_gradient_elements')} / "
            f"{discriminator.get('total_trainable_parameter_elements')}",
            f"- Validation rows loaded: {report.get('validation_rows_loaded', 0)}",
            f"- Official-test rows loaded: {report.get('official_test_rows_loaded', 0)}",
            f"- Training steps: {report.get('training_steps', 0)}",
            "",
            "## Invariants",
            "",
            *invariant_lines,
            "",
            "## Mask cases and unweighted components",
            "",
            *case_lines,
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
        "--loss-config", type=Path, default=REPO_ROOT / "configs" / "gan_losses.json"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_losses" / "loss_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_losses" / "loss_audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_audit(args.architecture_config, args.loss_config)
    except Exception as error:
        report = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "materialized_generated_images": 0,
            "training_steps": 0,
        }
    _atomic_write(args.json_output, json.dumps(report, indent=2) + "\n")
    _atomic_write(args.markdown_output, _markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
