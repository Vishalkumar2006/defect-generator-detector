"""Detector model architectures."""

from .gan import (
    GANArchitectureConfig,
    MaskConditionedPatchDiscriminator,
    MaskedResidualGenerator,
    MaskedResidualGeneratorOutput,
    build_gan_models,
    load_gan_architecture_config,
)
from .unet import UNet, count_parameters

__all__ = [
    "GANArchitectureConfig",
    "MaskConditionedPatchDiscriminator",
    "MaskedResidualGenerator",
    "MaskedResidualGeneratorOutput",
    "UNet",
    "build_gan_models",
    "count_parameters",
    "load_gan_architecture_config",
]
