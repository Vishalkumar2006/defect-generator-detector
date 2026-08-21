"""Detector model architectures."""

from .gan import (
    ARCHITECTURE_VERSION,
    GANArchitectureConfig,
    MaskConditionedPatchDiscriminator,
    MaskedResidualGenerator,
    MaskedResidualGeneratorOutput,
    RESIDUAL_SEMANTICS_VERSION,
    build_gan_models,
    load_gan_architecture_config,
    range_aware_residual,
)
from .unet import UNet, count_parameters

__all__ = [
    "ARCHITECTURE_VERSION",
    "GANArchitectureConfig",
    "MaskConditionedPatchDiscriminator",
    "MaskedResidualGenerator",
    "MaskedResidualGeneratorOutput",
    "RESIDUAL_SEMANTICS_VERSION",
    "UNet",
    "build_gan_models",
    "count_parameters",
    "load_gan_architecture_config",
    "range_aware_residual",
]
