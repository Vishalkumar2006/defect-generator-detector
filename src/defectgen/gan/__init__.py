"""Training-only online GAN-input construction; no GAN model lives here."""

from .dataset import OnlineGANInputDataset
from .geometry import ComponentWindow, connected_components, deterministic_component_windows
from .manifest import build_gan_input_metadata
from .normalization import gan_rgb_to_uint8, rgb_to_gan
from .pipeline import construct_coarse_gan_input

__all__ = [
    "ComponentWindow",
    "OnlineGANInputDataset",
    "build_gan_input_metadata",
    "connected_components",
    "construct_coarse_gan_input",
    "deterministic_component_windows",
    "gan_rgb_to_uint8",
    "rgb_to_gan",
]
