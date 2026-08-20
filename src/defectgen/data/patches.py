"""Deterministic, synchronized, mask-aware patch extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BoundingBox, bbox_fits_patch, bounding_box


@dataclass(frozen=True)
class Padding:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class CropCoordinates:
    """Crop origin in padded and original-image coordinate systems."""

    padded_left: int
    padded_top: int
    source_left: int
    source_top: int
    width: int
    height: int


@dataclass
class PatchResult:
    image: np.ndarray
    mask: np.ndarray
    coordinates: CropCoordinates
    padding: Padding
    defect_fully_contained: bool
    context_fully_contained: bool
    source_bbox: BoundingBox | None


def _validate_patch_size(patch_size: tuple[int, int]) -> tuple[int, int]:
    patch_width, patch_height = patch_size
    if patch_width <= 0 or patch_height <= 0:
        raise ValueError(f"Patch dimensions must be positive, got {patch_size}")
    return patch_width, patch_height


def _validate_pair(image: np.ndarray, mask: np.ndarray) -> None:
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be H×W or H×W×C, got {image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be H×W, got {mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask dimensions disagree: {image.shape[:2]} versus {mask.shape}")
    values = np.unique(mask)
    if len(values) > 2 or (len(values) == 2 and values[0] != 0):
        raise ValueError(f"Mask is not binary: {values[:8].tolist()}")


def reflection_pad(array: np.ndarray, padding: Padding) -> np.ndarray:
    """Reflection-pad spatial axes, supporting padding wider than the source."""
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D array, got {array.shape}")
    if min(padding.left, padding.top, padding.right, padding.bottom) < 0:
        raise ValueError(f"Padding cannot be negative: {padding}")
    result = array
    vertical = ((padding.top, padding.bottom), (0, 0)) + (((0, 0),) if array.ndim == 3 else ())
    horizontal = ((0, 0), (padding.left, padding.right)) + (((0, 0),) if array.ndim == 3 else ())
    if padding.top or padding.bottom:
        result = np.pad(result, vertical, mode="reflect" if array.shape[0] > 1 else "edge")
    if padding.left or padding.right:
        result = np.pad(result, horizontal, mode="reflect" if array.shape[1] > 1 else "edge")
    return result


def _zero_pad_mask(mask: np.ndarray, padding: Padding) -> np.ndarray:
    return np.pad(mask, ((padding.top, padding.bottom), (padding.left, padding.right)), mode="constant")


def _padding_for_patch(shape: tuple[int, int], patch_size: tuple[int, int], base_margin: int = 0) -> Padding:
    if base_margin < 0:
        raise ValueError("Context margin cannot be negative")
    height, width = shape
    patch_width, patch_height = _validate_patch_size(patch_size)
    left = right = top = bottom = base_margin
    missing_width = max(0, patch_width - (width + left + right))
    missing_height = max(0, patch_height - (height + top + bottom))
    left += missing_width // 2
    right += missing_width - missing_width // 2
    top += missing_height // 2
    bottom += missing_height - missing_height // 2
    return Padding(left=left, top=top, right=right, bottom=bottom)


def _random_inclusive(rng: np.random.Generator, lower: int, upper: int) -> int:
    if lower > upper:
        raise ValueError(f"Empty coordinate range: {lower}..{upper}")
    return int(rng.integers(lower, upper + 1)) if lower < upper else lower


def mask_centered_crop_coordinates(
    mask: np.ndarray,
    patch_size: tuple[int, int],
    rng: np.random.Generator,
    context_margin: int = 0,
) -> tuple[int, int, bool, bool]:
    """Choose ``(top, left)`` on a padded mask and report containment.

    The caller must ensure the mask canvas is at least as large as the patch.
    """
    patch_width, patch_height = _validate_patch_size(patch_size)
    if context_margin < 0:
        raise ValueError("Context margin cannot be negative")
    height, width = mask.shape
    if width < patch_width or height < patch_height:
        raise ValueError("Mask canvas must be padded to at least the patch dimensions")
    box = bounding_box(mask)
    if box is None:
        raise ValueError("Positive crop requested for an empty mask")

    defect_can_fit = bbox_fits_patch(box, patch_size, 0)
    context_can_fit = bbox_fits_patch(box, patch_size, context_margin)
    if defect_can_fit:
        margin = context_margin if context_can_fit else 0
        left_low = max(0, box.x_max + margin - patch_width + 1)
        left_high = min(box.x_min - margin, width - patch_width)
        top_low = max(0, box.y_max + margin - patch_height + 1)
        top_high = min(box.y_min - margin, height - patch_height)
        left = _random_inclusive(rng, left_low, left_high)
        top = _random_inclusive(rng, top_low, top_high)
    else:
        positive_pixels = np.argwhere(mask)
        selected_y, selected_x = positive_pixels[int(rng.integers(0, len(positive_pixels)))]
        left = _random_inclusive(rng, max(0, int(selected_x) - patch_width + 1), min(int(selected_x), width - patch_width))
        top = _random_inclusive(rng, max(0, int(selected_y) - patch_height + 1), min(int(selected_y), height - patch_height))

    crop = mask[top : top + patch_height, left : left + patch_width]
    fully_contained = int(np.count_nonzero(crop)) == int(np.count_nonzero(mask))
    context_contained = (
        fully_contained
        and left <= box.x_min - context_margin
        and top <= box.y_min - context_margin
        and left + patch_width - 1 >= box.x_max + context_margin
        and top + patch_height - 1 >= box.y_max + context_margin
    )
    return top, left, fully_contained, context_contained


def extract_positive_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int],
    rng: np.random.Generator,
    context_margin: int = 0,
) -> PatchResult:
    """Extract a mask-aware positive patch without resizing the defect."""
    _validate_pair(image, mask)
    binary_mask = mask != 0
    source_box = bounding_box(binary_mask)
    if source_box is None:
        raise ValueError("Positive patch requested for a zero mask")
    padding = _padding_for_patch(mask.shape, patch_size, base_margin=context_margin)
    padded_image = reflection_pad(image, padding)
    # Reflected image context is valid background, while labels outside the
    # captured image are unknown and therefore remain zero rather than mirrored.
    padded_mask = _zero_pad_mask(binary_mask, padding)
    top, left, contained, context_contained = mask_centered_crop_coordinates(
        padded_mask, patch_size, rng, context_margin
    )
    patch_width, patch_height = _validate_patch_size(patch_size)
    image_patch = padded_image[top : top + patch_height, left : left + patch_width]
    mask_patch = padded_mask[top : top + patch_height, left : left + patch_width]
    if not np.any(mask_patch):
        raise RuntimeError("Internal error: positive patch contains zero mask pixels")
    return PatchResult(
        image=image_patch,
        mask=mask_patch,
        coordinates=CropCoordinates(
            padded_left=left,
            padded_top=top,
            source_left=left - padding.left,
            source_top=top - padding.top,
            width=patch_width,
            height=patch_height,
        ),
        padding=padding,
        defect_fully_contained=contained,
        context_fully_contained=context_contained,
        source_bbox=source_box,
    )


def extract_normal_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int],
    rng: np.random.Generator,
) -> PatchResult:
    """Extract a deterministic random patch from a verified zero-mask image."""
    _validate_pair(image, mask)
    if np.any(mask):
        raise ValueError("Normal patch sampling requires a completely zero mask")
    padding = _padding_for_patch(mask.shape, patch_size)
    padded_image = reflection_pad(image, padding)
    padded_mask = _zero_pad_mask(mask != 0, padding)
    patch_width, patch_height = _validate_patch_size(patch_size)
    max_left = padded_image.shape[1] - patch_width
    max_top = padded_image.shape[0] - patch_height
    left = _random_inclusive(rng, 0, max_left)
    top = _random_inclusive(rng, 0, max_top)
    image_patch = padded_image[top : top + patch_height, left : left + patch_width]
    mask_patch = padded_mask[top : top + patch_height, left : left + patch_width]
    if np.any(mask_patch):
        raise RuntimeError("Internal error: normal crop acquired positive mask pixels")
    return PatchResult(
        image=image_patch,
        mask=mask_patch,
        coordinates=CropCoordinates(
            padded_left=left,
            padded_top=top,
            source_left=left - padding.left,
            source_top=top - padding.top,
            width=patch_width,
            height=patch_height,
        ),
        padding=padding,
        defect_fully_contained=True,
        context_fully_contained=True,
        source_bbox=None,
    )

