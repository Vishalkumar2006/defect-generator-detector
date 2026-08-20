"""Manifest-only, native-geometry KSDD2 full-image PyTorch dataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .ksdd2 import interpret_binary_mask
from .patches import Padding, reflection_pad
from .splits import load_development_manifest


MODEL_WIDTH = 256
MODEL_HEIGHT = 672


@dataclass(frozen=True)
class NativeGeometry:
    original_height: int
    original_width: int
    padding: Padding


def symmetric_padding(
    original_height: int,
    original_width: int,
    target_height: int = MODEL_HEIGHT,
    target_width: int = MODEL_WIDTH,
) -> Padding:
    if original_height > target_height or original_width > target_width:
        raise ValueError(
            f"Native image {original_width}x{original_height} exceeds canvas {target_width}x{target_height}"
        )
    horizontal = target_width - original_width
    vertical = target_height - original_height
    return Padding(
        left=horizontal // 2,
        top=vertical // 2,
        right=horizontal - horizontal // 2,
        bottom=vertical - vertical // 2,
    )


def restore_to_native(prediction: torch.Tensor | np.ndarray, geometry: NativeGeometry):
    """Crop the last two dimensions back to the exact captured image area."""
    top = geometry.padding.top
    left = geometry.padding.left
    return prediction[..., top : top + geometry.original_height, left : left + geometry.original_width]


def pad_full_sample(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    target_size: tuple[int, int] = (MODEL_WIDTH, MODEL_HEIGHT),
    image_padding_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Padding]:
    """Pad an image by reflection/edge while always zero-padding its labels."""
    if image.ndim != 3 or image.shape[2] != 3 or mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError(f"Expected synchronized H×W×3 image and H×W mask, got {image.shape}, {mask.shape}")
    if image_padding_mode not in {"reflect", "edge"}:
        raise ValueError("image_padding_mode must be 'reflect' or 'edge'")
    target_width, target_height = target_size
    height, width = mask.shape
    padding = symmetric_padding(height, width, target_height, target_width)
    if image_padding_mode == "reflect":
        padded_image = reflection_pad(image, padding)
    else:
        padded_image = np.pad(
            image,
            ((padding.top, padding.bottom), (padding.left, padding.right), (0, 0)),
            mode="edge",
        )
    padded_mask = np.pad(
        mask.astype(np.float32),
        ((padding.top, padding.bottom), (padding.left, padding.right)),
        mode="constant",
        constant_values=0,
    )
    valid_region = np.zeros((target_height, target_width), dtype=np.float32)
    valid_region[padding.top : padding.top + height, padding.left : padding.left + width] = 1.0
    return padded_image, padded_mask, valid_region, padding


class KSDD2FullImageDataset(Dataset):
    """Load complete images exclusively from the tracked development manifest."""

    def __init__(
        self,
        repo_root: Path,
        development_split: str,
        manifest_path: Path | None = None,
        *,
        target_size: tuple[int, int] = (MODEL_WIDTH, MODEL_HEIGHT),
        image_padding_mode: str = "reflect",
        mean: Sequence[float] | None = None,
        standard_deviation: Sequence[float] | None = None,
        sample_ids: set[str] | None = None,
        augmentation: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        spatial_transform: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        if development_split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid development split: {development_split}")
        if image_padding_mode not in {"reflect", "edge"}:
            raise ValueError("image_padding_mode must be 'reflect' or 'edge'")
        self.repo_root = repo_root.resolve()
        all_rows = load_development_manifest(self.repo_root, manifest_path)
        self.rows = [row for row in all_rows if row["development_split"] == development_split]
        if sample_ids is not None:
            self.rows = [row for row in self.rows if row["sample_id"] in sample_ids]
            missing = sample_ids - {row["sample_id"] for row in self.rows}
            if missing:
                raise ValueError(f"Requested sample IDs are not in {development_split}: {sorted(missing)}")
        if not self.rows:
            raise ValueError(f"No samples selected for development split {development_split}")
        self.development_split = development_split
        self.target_width, self.target_height = target_size
        self.image_padding_mode = image_padding_mode
        self.augmentation = augmentation
        self.spatial_transform = spatial_transform
        self.epoch = 0
        if (mean is None) != (standard_deviation is None):
            raise ValueError("mean and standard_deviation must be supplied together")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1) if mean is not None else None
        self.standard_deviation = (
            torch.tensor(standard_deviation, dtype=torch.float32).view(3, 1, 1)
            if standard_deviation is not None
            else None
        )
        if self.standard_deviation is not None and torch.any(self.standard_deviation <= 0):
            raise ValueError("Channel standard deviations must be positive")

    @property
    def labels(self) -> list[bool]:
        return [bool(row["has_defect"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch before creating its loader iterator."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(self.repo_root / row["image_path"]) as source:
            image = np.asarray(source.convert("RGB"))
        height, width = image.shape[:2]
        if row["mask_path"]:
            with Image.open(self.repo_root / row["mask_path"]) as source:
                mask = interpret_binary_mask(np.asarray(source))
        else:
            mask = np.zeros((height, width), dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"Image/mask dimensions disagree for {row['sample_id']}")
        if bool(np.any(mask)) != bool(row["has_defect"]):
            raise ValueError(f"Mask content disagrees with manifest label for {row['sample_id']}")
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
            if image.shape[:2] != mask.shape:
                raise ValueError("Synchronized augmentation returned mismatched image/mask dimensions")
        padded_image, padded_mask, valid_region, padding = pad_full_sample(
            image,
            mask,
            target_size=(self.target_width, self.target_height),
            image_padding_mode=self.image_padding_mode,
        )
        image_tensor = torch.from_numpy(np.ascontiguousarray(padded_image.transpose(2, 0, 1))).float() / 255.0
        if self.mean is not None and self.standard_deviation is not None:
            image_tensor = (image_tensor - self.mean) / self.standard_deviation
        mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid_region).unsqueeze(0)
        if self.spatial_transform is not None:
            image_tensor, mask_tensor, valid_tensor = self.spatial_transform(
                image_tensor,
                mask_tensor,
                valid_tensor,
                sample_id=row["sample_id"],
                epoch=self.epoch,
            )
            if image_tensor.shape[-2:] != mask_tensor.shape[-2:] or mask_tensor.shape != valid_tensor.shape:
                raise ValueError("Synchronized spatial transform returned mismatched tensors")
            valid_coordinates = torch.nonzero(valid_tensor[0], as_tuple=False)
            if len(valid_coordinates) == 0:
                raise ValueError("Synchronized spatial transform removed the valid region")
            top = int(valid_coordinates[:, 0].min().item())
            bottom = int(valid_coordinates[:, 0].max().item())
            left = int(valid_coordinates[:, 1].min().item())
            right = int(valid_coordinates[:, 1].max().item())
            padding = Padding(
                left=left,
                top=top,
                right=self.target_width - right - 1,
                bottom=self.target_height - bottom - 1,
            )
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise ValueError(f"Non-binary mask after padding for {row['sample_id']}")
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "valid_region": valid_tensor,
            "has_defect": torch.tensor(bool(row["has_defect"]), dtype=torch.bool),
            "sample_id": row["sample_id"],
            "original_height": height,
            "original_width": width,
            "padding": torch.tensor(
                [padding.left, padding.top, padding.right, padding.bottom], dtype=torch.int64
            ),
        }


def geometry_from_batch(batch: dict[str, Any], index: int) -> NativeGeometry:
    offsets = batch["padding"][index].tolist()
    return NativeGeometry(
        original_height=int(batch["original_height"][index]),
        original_width=int(batch["original_width"][index]),
        padding=Padding(left=offsets[0], top=offsets[1], right=offsets[2], bottom=offsets[3]),
    )
