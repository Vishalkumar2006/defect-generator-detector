"""Run a synthetic, architecture-only audit of the G1.1 conditional GAN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.models import (  # noqa: E402
    build_gan_models,
    count_parameters,
    load_gan_architecture_config,
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _maximum_change_outside_support(
    refined: torch.Tensor, original: torch.Tensor, support: torch.Tensor
) -> float:
    outside = ~support.expand_as(original)
    if not bool(outside.any()):
        return 0.0
    return float((refined - original).abs()[outside].max().item())


def build_audit(config_path: Path) -> dict[str, Any]:
    torch.manual_seed(42)
    config = load_gan_architecture_config(config_path)
    generator, discriminator = build_gan_models(config)
    generator.train()
    discriminator.train()
    shape = (1, 3, config.image_height, config.image_width)
    composite = torch.zeros(shape, dtype=torch.float32)
    defect_mask = torch.zeros((1, 1, config.image_height, config.image_width))
    centre_y = config.image_height // 2
    centre_x = config.image_width // 2
    defect_mask[:, :, centre_y - 3 : centre_y + 3, centre_x - 6 : centre_x + 6] = 1

    generated = generator(composite, defect_mask)
    logits = discriminator(generated.refined_image, defect_mask)
    combined_scalar = generated.refined_image.square().mean() + logits.square().mean()
    combined_scalar.backward()
    named_generator_gradients = [
        (name, parameter.grad)
        for name, parameter in generator.named_parameters()
        if parameter.grad is not None
    ]
    generator_gradients = [gradient for _, gradient in named_generator_gradients]
    discriminator_gradients = [
        parameter.grad for parameter in discriminator.parameters() if parameter.grad is not None
    ]

    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        zero_mask = torch.zeros_like(defect_mask)
        zero_result = generator(composite, zero_mask)
        border_mask = torch.zeros_like(defect_mask)
        border_mask[:, :, :2, :] = 1
        border_mask[:, :, -2:, :] = 1
        border_mask[:, :, :, :2] = 1
        border_mask[:, :, :, -2:] = 1
        border_result = generator(composite, border_mask)

    outside_change = _maximum_change_outside_support(
        generated.refined_image.detach(), composite, generated.support_mask
    )
    border_outside_change = _maximum_change_outside_support(
        border_result.refined_image, composite, border_result.support_mask
    )
    zero_change = float((zero_result.refined_image - composite).abs().max().item())
    forward_finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            generated.refined_image,
            generated.raw_residual,
            generated.applied_residual,
            logits,
            border_result.refined_image,
        )
    )
    backward_finite = (
        bool(generator_gradients)
        and bool(discriminator_gradients)
        and all(bool(torch.isfinite(gradient).all()) for gradient in generator_gradients)
        and all(bool(torch.isfinite(gradient).all()) for gradient in discriminator_gradients)
    )
    nonzero_generator_gradients = sum(
        bool((gradient != 0).any()) for gradient in generator_gradients
    )
    nonzero_head_gradients = sum(
        name.startswith("output_head.") and bool((gradient != 0).any())
        for name, gradient in named_generator_gradients
    )
    nonzero_earlier_gradients = sum(
        not name.startswith("output_head.") and bool((gradient != 0).any())
        for name, gradient in named_generator_gradients
    )
    invariants = {
        "production_output_shape_matches": list(generated.refined_image.shape) == list(shape),
        "discriminator_logits_nonempty": logits.numel() > 0,
        "output_in_normalized_range": bool(
            (generated.refined_image >= -1).all()
            and (generated.refined_image <= 1).all()
        ),
        "maximum_change_outside_support_is_zero": outside_change == 0.0,
        "border_change_outside_support_is_zero": border_outside_change == 0.0,
        "zero_mask_change_is_zero": zero_change == 0.0,
        "nonzero_mask_initial_output_is_exact_identity": torch.equal(
            generated.refined_image.detach(), composite
        ),
        "initial_raw_residual_is_zero": int(generated.raw_residual.count_nonzero()) == 0,
        "initial_applied_residual_is_zero": int(
            generated.applied_residual.count_nonzero()
        ) == 0,
        "forward_tensors_finite": forward_finite,
        "backward_gradients_finite": backward_finite,
        "initial_adversarial_gradient_reaches_output_head": nonzero_head_gradients > 0,
        "earlier_generator_gradients_are_staged_zero": nonzero_earlier_gradients == 0,
    }
    return {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "architecture_version": config.architecture_version,
        "input_shape": list(shape),
        "generator_trainable_parameters": count_parameters(generator),
        "discriminator_trainable_parameters": count_parameters(discriminator),
        "discriminator_logit_shape": list(logits.shape),
        "maximum_absolute_change_outside_support": outside_change,
        "border_maximum_absolute_change_outside_support": border_outside_change,
        "zero_mask_maximum_absolute_change": zero_change,
        "generator_parameters_with_nonzero_gradients": nonzero_generator_gradients,
        "output_head_parameters_with_nonzero_gradients": nonzero_head_gradients,
        "earlier_parameters_with_nonzero_gradients": nonzero_earlier_gradients,
        "invariants": invariants,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_generated_images": 0,
        "training_steps": 0,
    }


def _markdown(report: dict[str, Any]) -> str:
    invariant_lines = [
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in report.get("invariants", {}).items()
    ]
    return "\n".join(
        [
            "# G1.1 GAN architecture audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Architecture: `{report.get('architecture_version', 'unknown')}`",
            f"- Input shape: `{report.get('input_shape')}`",
            f"- Generator trainable parameters: "
            f"{report.get('generator_trainable_parameters')}",
            f"- Discriminator trainable parameters: "
            f"{report.get('discriminator_trainable_parameters')}",
            f"- Discriminator logit shape: `{report.get('discriminator_logit_shape')}`",
            f"- Maximum change outside support: "
            f"{report.get('maximum_absolute_change_outside_support')}",
            f"- Border maximum change outside support: "
            f"{report.get('border_maximum_absolute_change_outside_support')}",
            f"- Zero-mask maximum change: "
            f"{report.get('zero_mask_maximum_absolute_change')}",
            f"- Validation rows loaded: {report.get('validation_rows_loaded', 0)}",
            f"- Official-test rows loaded: {report.get('official_test_rows_loaded', 0)}",
            f"- Training steps: {report.get('training_steps', 0)}",
            "",
            "## Invariants",
            "",
            *invariant_lines,
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "gan_architecture.json"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_architecture" / "architecture_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_architecture" / "architecture_audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_audit(args.config)
    except Exception as error:  # Preserve a machine-readable failed architecture audit.
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
