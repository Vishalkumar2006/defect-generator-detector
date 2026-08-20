"""Deterministic synchronized spatial augmentation for segmentation samples."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SynchronizedRandomFlips:
    """Flip image, mask, and valid-region tensors with stateless seeded choices.

    Decisions are derived from seed, epoch, and sample ID, so they are stable
    across worker scheduling and exactly reconstructible after a resume.
    """

    horizontal_probability: float = 0.5
    vertical_probability: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        for name, probability in (
            ("horizontal_probability", self.horizontal_probability),
            ("vertical_probability", self.vertical_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def _uniforms(self, sample_id: str, epoch: int) -> tuple[float, float]:
        digest = hashlib.sha256(f"{self.seed}:{epoch}:{sample_id}".encode("utf-8")).digest()
        denominator = float(1 << 64)
        return (
            int.from_bytes(digest[:8], "big") / denominator,
            int.from_bytes(digest[8:16], "big") / denominator,
        )

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_region: torch.Tensor,
        *,
        sample_id: str,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or mask.ndim != 3 or valid_region.ndim != 3:
            raise ValueError("Expected CxHxW image, mask, and valid-region tensors")
        if image.shape[-2:] != mask.shape[-2:] or mask.shape != valid_region.shape:
            raise ValueError("Spatial augmentation inputs must be aligned")
        horizontal, vertical = self._uniforms(sample_id, epoch)
        dimensions: list[int] = []
        if horizontal < self.horizontal_probability:
            dimensions.append(-1)
        if vertical < self.vertical_probability:
            dimensions.append(-2)
        if dimensions:
            image = torch.flip(image, dimensions)
            mask = torch.flip(mask, dimensions)
            valid_region = torch.flip(valid_region, dimensions)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Synchronized flips must preserve binary masks")
        if not torch.all((valid_region == 0) | (valid_region == 1)):
            raise ValueError("Synchronized flips must preserve the binary valid region")
        return image, mask, valid_region
