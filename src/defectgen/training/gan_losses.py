"""Localized, independently testable objectives for mask-conditioned GAN refinement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F


LOSS_VERSION = "g1_2_localized_gan_losses_v1"


@dataclass(frozen=True)
class GeneratorLossWeights:
    """Provisional coefficients pending a training-smoke relative-scale audit."""

    adversarial: float = 1.0
    change: float = 1.0
    boundary: float = 1.0
    total_variation: float = 1.0

    def validate(self) -> None:
        values = (self.adversarial, self.change, self.boundary, self.total_variation)
        if any(value < 0 for value in values) or not any(value > 0 for value in values):
            raise ValueError("Generator loss weights must be non-negative and not all zero")


@dataclass(frozen=True)
class GANLossConfig:
    loss_version: str
    canonical_mask_threshold: float
    localization_radius: int
    boundary_ring_width: int
    aggregation_weights_provisional: bool
    generator_loss_weights: GeneratorLossWeights

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GANLossConfig":
        required = (
            "loss_version",
            "canonical_mask_threshold",
            "localization_radius",
            "boundary_ring_width",
            "aggregation_weights_provisional",
            "generator_loss_weights",
        )
        missing = [field for field in required if field not in values]
        if missing:
            raise ValueError(f"GAN loss config is missing: {', '.join(missing)}")
        raw_weights = values["generator_loss_weights"]
        if not isinstance(raw_weights, dict):
            raise ValueError("generator_loss_weights must be a JSON object")
        try:
            weights = GeneratorLossWeights(**raw_weights)
        except TypeError as error:
            raise ValueError(f"Invalid generator_loss_weights: {error}") from error
        config = cls(
            loss_version=values["loss_version"],
            canonical_mask_threshold=values["canonical_mask_threshold"],
            localization_radius=values["localization_radius"],
            boundary_ring_width=values["boundary_ring_width"],
            aggregation_weights_provisional=values["aggregation_weights_provisional"],
            generator_loss_weights=weights,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.loss_version != LOSS_VERSION:
            raise ValueError(f"loss_version must be {LOSS_VERSION!r}")
        if not 0 < float(self.canonical_mask_threshold) <= 1:
            raise ValueError("canonical_mask_threshold must be in (0, 1]")
        if not isinstance(self.localization_radius, int) or self.localization_radius < 0:
            raise ValueError("localization_radius must be a non-negative integer")
        if not isinstance(self.boundary_ring_width, int) or self.boundary_ring_width <= 0:
            raise ValueError("boundary_ring_width must be a positive integer")
        if self.aggregation_weights_provisional is not True:
            raise ValueError("G1.2 aggregation weights must remain marked provisional")
        self.generator_loss_weights.validate()


def load_gan_loss_config(path: Path | str) -> GANLossConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("GAN loss config must contain a JSON object")
    return GANLossConfig.from_dict(values)


def _validate_mask(mask: torch.Tensor, *, name: str = "mask") -> None:
    if mask.ndim != 4:
        raise ValueError(f"{name} must have rank 4 [B,1,H,W]")
    if mask.shape[1] != 1:
        raise ValueError(f"{name} must have exactly 1 channel")
    if mask.shape[0] <= 0 or mask.shape[-2] <= 0 or mask.shape[-1] <= 0:
        raise ValueError(f"{name} batch and spatial dimensions must be positive")
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError(f"{name} must be bool or floating point")
    if mask.dtype != torch.bool:
        if not bool(torch.isfinite(mask).all()):
            raise ValueError(f"{name} must contain only finite values")
        if bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ValueError(f"{name} values must be in [0, 1]")


def canonicalize_discriminator_mask(
    mask: torch.Tensor, *, threshold: float = 0.5
) -> torch.Tensor:
    """Return float32 binary conditioning using the inclusive ``>= threshold`` policy."""
    _validate_mask(mask, name="discriminator mask")
    if not 0 < float(threshold) <= 1:
        raise ValueError("canonical mask threshold must be in (0, 1]")
    if mask.dtype == torch.bool:
        return mask.to(dtype=torch.float32)
    return (mask >= threshold).to(dtype=torch.float32)


def patch_logit_localization_weights(
    defect_mask: torch.Tensor,
    logits: torch.Tensor,
    *,
    localization_radius: int = 35,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Dilate a canonical mask and max-project it onto the PatchGAN logit grid."""
    canonical = canonicalize_discriminator_mask(defect_mask, threshold=mask_threshold)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,h,w]")
    if logits.shape[0] != canonical.shape[0]:
        raise ValueError("mask and logits batch dimensions must match")
    if logits.shape[-2] <= 0 or logits.shape[-1] <= 0:
        raise ValueError("logit spatial dimensions must be positive")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values")
    if not isinstance(localization_radius, int) or localization_radius < 0:
        raise ValueError("localization_radius must be a non-negative integer")
    if localization_radius:
        canonical = F.max_pool2d(
            canonical,
            kernel_size=2 * localization_radius + 1,
            stride=1,
            padding=localization_radius,
        )
    projected = F.adaptive_max_pool2d(canonical, logits.shape[-2:])
    weights = (projected > 0).to(device=logits.device, dtype=logits.dtype)
    active_by_sample = weights.flatten(1).sum(dim=1)
    if bool((active_by_sample == 0).any()):
        raise ValueError("Every sample must have at least one active localization weight")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise RuntimeError("Localized PatchGAN weights must be finite and non-negative")
    return weights


def localized_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    if not values.is_floating_point() or not weights.is_floating_point():
        raise ValueError("values and weights must be floating point")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("values and weights must contain only finite values")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        raise ValueError("Localized weighted mean requires positive total weight")
    return (values * weights).sum() / denominator


@dataclass(frozen=True)
class DiscriminatorHingeLossComponents:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


def localized_discriminator_hinge_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    real_weights: torch.Tensor,
    fake_weights: torch.Tensor,
) -> DiscriminatorHingeLossComponents:
    """Localized raw-logit hinge loss; callers must discriminate ``fake.detach()``."""
    real = localized_weighted_mean(F.relu(1 - real_logits), real_weights)
    fake = localized_weighted_mean(F.relu(1 + fake_logits), fake_weights)
    return DiscriminatorHingeLossComponents(total=real + fake, real=real, fake=fake)


@dataclass(frozen=True)
class GeneratorAdversarialLossComponents:
    total: torch.Tensor
    adversarial: torch.Tensor


def localized_generator_adversarial_loss(
    fake_logits: torch.Tensor, fake_weights: torch.Tensor
) -> GeneratorAdversarialLossComponents:
    adversarial = -localized_weighted_mean(fake_logits, fake_weights)
    return GeneratorAdversarialLossComponents(
        total=adversarial, adversarial=adversarial
    )


def _validate_image_pair(
    refined_image: torch.Tensor, composite_image: torch.Tensor, support_mask: torch.Tensor
) -> torch.Tensor:
    if refined_image.ndim != 4 or refined_image.shape[1] != 3:
        raise ValueError("refined_image must have shape [B,3,H,W]")
    if refined_image.shape != composite_image.shape:
        raise ValueError("refined_image and composite_image must have identical shapes")
    if not refined_image.is_floating_point() or not composite_image.is_floating_point():
        raise ValueError("refined_image and composite_image must be floating point")
    if not bool(torch.isfinite(refined_image).all()) or not bool(
        torch.isfinite(composite_image).all()
    ):
        raise ValueError("Images must contain only finite values")
    support = canonicalize_discriminator_mask(support_mask, threshold=0.5).bool()
    if support.shape[0] != refined_image.shape[0] or support.shape[-2:] != refined_image.shape[-2:]:
        raise ValueError("support_mask must match image batch and spatial dimensions")
    return support.to(device=refined_image.device)


def support_normalized_change_loss(
    refined_image: torch.Tensor,
    composite_image: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Magnitude regularizer inside support, not a paired-target reconstruction loss."""
    support = _validate_image_pair(refined_image, composite_image, support_mask)
    expanded = support.expand_as(refined_image)
    numerator = (refined_image - composite_image).abs()[expanded].sum()
    denominator = expanded.sum()
    if int(denominator) == 0:
        return (refined_image - composite_image).sum() * 0
    return numerator / denominator


def inner_boundary_ring(support_mask: torch.Tensor, *, width: int = 3) -> torch.Tensor:
    canonical = canonicalize_discriminator_mask(support_mask, threshold=0.5).bool()
    if not isinstance(width, int) or width <= 0:
        raise ValueError("Boundary-ring width must be a positive integer")
    nearby_inactive = F.max_pool2d(
        (~canonical).float(), kernel_size=2 * width + 1, stride=1, padding=width
    ).bool()
    return canonical & nearby_inactive


def boundary_seam_loss(
    refined_image: torch.Tensor,
    composite_image: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    boundary_width: int = 3,
) -> torch.Tensor:
    support = _validate_image_pair(refined_image, composite_image, support_mask)
    ring = inner_boundary_ring(support, width=boundary_width).to(refined_image.device)
    expanded = ring.expand_as(refined_image)
    difference = (refined_image - composite_image).abs()
    if int(expanded.sum()) == 0:
        return difference.sum() * 0
    return difference[expanded].sum() / expanded.sum()


def masked_total_variation_loss(
    residual: torch.Tensor, support_mask: torch.Tensor
) -> torch.Tensor:
    if residual.ndim != 4 or residual.shape[1] != 3 or not residual.is_floating_point():
        raise ValueError("residual must be floating [B,3,H,W]")
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("residual must contain only finite values")
    support = canonicalize_discriminator_mask(support_mask, threshold=0.5).bool()
    if support.shape[0] != residual.shape[0] or support.shape[-2:] != residual.shape[-2:]:
        raise ValueError("support_mask must match residual batch and spatial dimensions")
    support = support.to(residual.device)
    horizontal_pairs = support[:, :, :, :-1] & support[:, :, :, 1:]
    vertical_pairs = support[:, :, :-1, :] & support[:, :, 1:, :]
    horizontal = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs()
    vertical = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs()
    horizontal_weight = horizontal_pairs.expand_as(horizontal)
    vertical_weight = vertical_pairs.expand_as(vertical)
    numerator = (horizontal * horizontal_weight).sum() + (vertical * vertical_weight).sum()
    denominator = horizontal_weight.sum() + vertical_weight.sum()
    if int(denominator) == 0:
        return residual.sum() * 0
    return numerator / denominator


@dataclass(frozen=True)
class AggregatedGeneratorLossComponents:
    total: torch.Tensor
    adversarial: torch.Tensor
    change: torch.Tensor
    boundary: torch.Tensor
    total_variation: torch.Tensor


def aggregate_generator_losses(
    *,
    adversarial: torch.Tensor,
    change: torch.Tensor,
    boundary: torch.Tensor,
    total_variation: torch.Tensor,
    weights: GeneratorLossWeights | None = None,
) -> AggregatedGeneratorLossComponents:
    selected = weights or GeneratorLossWeights()
    selected.validate()
    components = (adversarial, change, boundary, total_variation)
    if any(component.ndim != 0 for component in components):
        raise ValueError("Generator loss components must be scalar tensors")
    if any(not bool(torch.isfinite(component).all()) for component in components):
        raise ValueError("Generator loss components must be finite")
    total = (
        selected.adversarial * adversarial
        + selected.change * change
        + selected.boundary * boundary
        + selected.total_variation * total_variation
    )
    return AggregatedGeneratorLossComponents(
        total=total,
        adversarial=adversarial,
        change=change,
        boundary=boundary,
        total_variation=total_variation,
    )


def localized_r1_gradient_penalty(
    discriminator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    real_image: torch.Tensor,
    real_mask: torch.Tensor,
    *,
    localization_radius: int = 35,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Return unscaled localized R1; a later trainer may schedule it lazily."""
    image = real_image if real_image.requires_grad else real_image.detach().requires_grad_(True)
    canonical_mask = canonicalize_discriminator_mask(
        real_mask, threshold=mask_threshold
    ).to(device=image.device, dtype=image.dtype)
    logits = discriminator(image, canonical_mask)
    weights = patch_logit_localization_weights(
        canonical_mask,
        logits,
        localization_radius=localization_radius,
        mask_threshold=mask_threshold,
    )
    flattened_logits = logits.flatten(1)
    flattened_weights = weights.flatten(1)
    localized_scores = (flattened_logits * flattened_weights).sum(dim=1) / (
        flattened_weights.sum(dim=1)
    )
    gradients = torch.autograd.grad(
        outputs=localized_scores,
        inputs=image,
        grad_outputs=torch.ones_like(localized_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    penalty = gradients.square().flatten(1).sum(dim=1).mean()
    if not bool(torch.isfinite(penalty)):
        raise RuntimeError("Localized R1 gradient penalty is non-finite")
    return penalty
