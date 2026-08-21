"""Training-only online GAN-input construction; no GAN model lives here."""

from .compatibility import GANPlacementCompatibilityIndex
from .dataset import GANSamplingFailure, OnlineGANInputDataset
from .discriminator_views import (
    AlignedDiscriminatorViews,
    prepare_aligned_discriminator_views,
    prepare_training_sample_discriminator_views,
)
from .geometry import ContactSides, ComponentWindow, connected_components, deterministic_component_windows
from .manifest import build_gan_input_metadata
from .normalization import gan_rgb_to_uint8, rgb_to_gan
from .pipeline import construct_coarse_gan_input
from .training_pairs import (
    GANInternalSplit,
    GANTrainingPairConfig,
    GANTrainingPairDataset,
    GANTrainingSample,
    create_internal_gan_split,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)

__all__ = [
    "ComponentWindow",
    "ContactSides",
    "AlignedDiscriminatorViews",
    "GANPlacementCompatibilityIndex",
    "GANInternalSplit",
    "GANSamplingFailure",
    "GANTrainingPairConfig",
    "GANTrainingPairDataset",
    "GANTrainingSample",
    "OnlineGANInputDataset",
    "build_gan_input_metadata",
    "connected_components",
    "construct_coarse_gan_input",
    "create_internal_gan_split",
    "deterministic_component_windows",
    "gan_rgb_to_uint8",
    "load_gan_training_pair_config",
    "load_training_pair_manifest",
    "prepare_aligned_discriminator_views",
    "prepare_training_sample_discriminator_views",
    "rgb_to_gan",
]
