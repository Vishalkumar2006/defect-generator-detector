"""GAN-specific RGB conversion; detector standardization is intentionally absent."""

from __future__ import annotations

import numpy as np
import torch


def rgb_to_gan(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert HxWx3 uint8/[0,1] RGB to CxHxW float32 in [-1,1]."""
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a three-dimensional RGB image, got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] != 3:
        raise ValueError("RGB image must have exactly three channels")
    tensor = tensor.float()
    is_uint8 = image.dtype == (np.uint8 if isinstance(image, np.ndarray) else torch.uint8)
    if is_uint8:
        tensor = tensor / 255.0
    elif bool((tensor < 0).any()) or bool((tensor > 1).any()):
        raise ValueError("Floating RGB inputs must be in [0,1]")
    return tensor.mul(2.0).sub(1.0)


def gan_rgb_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Round-trip a CxHxW GAN tensor to HxWx3 uint8 RGB."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a 3xHxW GAN RGB tensor")
    if bool((image < -1).any()) or bool((image > 1).any()):
        raise ValueError("GAN RGB tensor must be in [-1,1]")
    return image.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)


def binary_mask_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("Expected HxW or 1xHxW mask")
    if not bool(torch.all((tensor == 0) | (tensor == 1))):
        raise ValueError("Masks must remain exactly binary")
    return tensor.float()
