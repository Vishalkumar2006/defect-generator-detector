"""Native-valid alignment for paired real/fake discriminator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from defectgen.training.gan_losses import canonicalize_discriminator_mask

if TYPE_CHECKING:
    from .training_pairs import GANTrainingSample


VALIDITY_ALIGNMENT_VERSION = "g1_3b_discriminator_view_validity_v1"
DISCRIMINATOR_INVALID_FILL_VALUE = 0.0


@dataclass(frozen=True)
class AlignedDiscriminatorViews:
    """Paired discriminator inputs sharing one native-valid support."""

    real_discriminator_view: torch.Tensor
    fake_discriminator_view: torch.Tensor
    joint_valid_mask: torch.Tensor
    discriminator_mask: torch.Tensor


def _validate_image_pair(real_image: torch.Tensor, fake_image: torch.Tensor) -> None:
    if real_image.ndim != 4 or real_image.shape[1] != 3:
        raise ValueError("real_image must have shape [B,3,H,W]")
    if fake_image.shape != real_image.shape:
        raise ValueError("real_image and fake_image must have identical shapes")
    if not real_image.is_floating_point() or not fake_image.is_floating_point():
        raise ValueError("real_image and fake_image must be floating point")
    if real_image.dtype != fake_image.dtype:
        raise ValueError("real_image and fake_image must have identical dtypes")
    if real_image.device != fake_image.device:
        raise ValueError("real_image and fake_image must be on the same device")
    if not bool(torch.isfinite(real_image).all()) or not bool(
        torch.isfinite(fake_image).all()
    ):
        raise ValueError("real_image and fake_image must contain only finite values")
    if bool((real_image < -1).any()) or bool((real_image > 1).any()):
        raise ValueError("real_image values must be in [-1,1]")
    if bool((fake_image < -1).any()) or bool((fake_image > 1).any()):
        raise ValueError("fake_image values must be in [-1,1]")


def _validate_spatial_mask(
    mask: torch.Tensor,
    image: torch.Tensor,
    *,
    name: str,
) -> None:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W]")
    if mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"{name} must match the image batch and spatial dimensions")
    if mask.device != image.device:
        raise ValueError(f"{name} must be on the same device as the images")
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError(f"{name} must be bool or floating point")
    if mask.dtype != torch.bool:
        if not bool(torch.isfinite(mask).all()):
            raise ValueError(f"{name} must contain only finite values")
        if bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ValueError(f"{name} values must be in [0,1]")


def prepare_aligned_discriminator_views(
    real_image: torch.Tensor,
    fake_image: torch.Tensor,
    real_valid_mask: torch.Tensor,
    fake_valid_mask: torch.Tensor,
    discriminator_mask: torch.Tensor,
    *,
    generator_support_mask: torch.Tensor | None = None,
    valid_mask_threshold: float = 0.5,
    discriminator_mask_threshold: float = 0.5,
) -> AlignedDiscriminatorViews:
    """Return real/fake views with identical fixed content outside joint validity.

    The source tensors are never mutated. Canonical defect containment is strict.
    ``generator_support_mask`` is validated when supplied but may extend outside
    joint validity because feather and refinement halos are non-native context.
    """

    _validate_image_pair(real_image, fake_image)
    for name, mask in (
        ("real_valid_mask", real_valid_mask),
        ("fake_valid_mask", fake_valid_mask),
        ("discriminator_mask", discriminator_mask),
    ):
        _validate_spatial_mask(mask, real_image, name=name)
    if generator_support_mask is not None:
        _validate_spatial_mask(
            generator_support_mask, real_image, name="generator_support_mask"
        )
    if not 0 < float(valid_mask_threshold) <= 1:
        raise ValueError("valid_mask_threshold must be in (0,1]")

    real_valid = canonicalize_discriminator_mask(
        real_valid_mask, threshold=valid_mask_threshold
    ).bool()
    fake_valid = canonicalize_discriminator_mask(
        fake_valid_mask, threshold=valid_mask_threshold
    ).bool()
    joint_valid = real_valid & fake_valid
    canonical_discriminator_mask = canonicalize_discriminator_mask(
        discriminator_mask, threshold=discriminator_mask_threshold
    )
    canonical_support = canonical_discriminator_mask.bool()
    outside_counts = (canonical_support & ~joint_valid).flatten(1).sum(dim=1)
    if bool((outside_counts > 0).any()):
        affected = [
            f"{index}:{int(count)}"
            for index, count in enumerate(outside_counts.tolist())
            if count
        ]
        raise ValueError(
            "canonical discriminator mask lies outside joint validity "
            f"(batch_index:pixel_count={','.join(affected)})"
        )

    expanded_joint = joint_valid.expand_as(real_image)
    fill = real_image.new_tensor(DISCRIMINATOR_INVALID_FILL_VALUE)
    real_view = torch.where(expanded_joint, real_image, fill)
    fake_view = torch.where(expanded_joint, fake_image, fill)
    invalid = ~expanded_joint
    if not torch.equal(real_view[invalid], fake_view[invalid]):
        raise RuntimeError("Aligned discriminator padding is not bit-exact equal")
    return AlignedDiscriminatorViews(
        real_discriminator_view=real_view,
        fake_discriminator_view=fake_view,
        joint_valid_mask=joint_valid.to(dtype=torch.float32),
        discriminator_mask=canonical_discriminator_mask,
    )


def prepare_training_sample_discriminator_views(
    sample: GANTrainingSample,
    *,
    valid_mask_threshold: float = 0.5,
    discriminator_mask_threshold: float = 0.5,
) -> AlignedDiscriminatorViews:
    """Prepare one unbatched G1.3 sample for paired discriminator branches."""

    if not torch.equal(
        sample.real_discriminator_mask, sample.fake_discriminator_mask
    ):
        raise ValueError("GAN training sample real/fake discriminator masks differ")
    return prepare_aligned_discriminator_views(
        sample.real_image.unsqueeze(0),
        sample.composite_image.unsqueeze(0),
        sample.real_valid_mask.unsqueeze(0),
        sample.fake_valid_mask.unsqueeze(0),
        sample.fake_discriminator_mask.unsqueeze(0),
        generator_support_mask=sample.generator_mask.unsqueeze(0),
        valid_mask_threshold=valid_mask_threshold,
        discriminator_mask_threshold=discriminator_mask_threshold,
    )
