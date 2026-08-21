"""Deterministic conservative template transformation and coarse compositing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from defectgen.data.geometry import bounding_box

from .normalization import binary_mask_tensor, rgb_to_gan


REQUIRED_PROVENANCE_FIELDS = {
    "generated_sample_seed",
    "normal_background_sample_id",
    "source_defect_sample_id",
    "connected_component_id",
    "source_mask_bounding_box",
    "source_window_coordinates",
    "target_window_coordinates",
    "horizontal_flip",
    "vertical_flip",
    "scale",
    "translation",
    "partial_component",
    "coverage_fraction",
    "touches_native_border",
    "colour_matching",
    "patch_size",
    "manifest_sha256",
    "split_sha256",
    "pipeline_version",
}


def validate_provenance(provenance: dict[str, Any]) -> None:
    missing = REQUIRED_PROVENANCE_FIELDS - set(provenance)
    if missing:
        raise ValueError(f"GAN input provenance is missing fields: {sorted(missing)}")


def _float_rgb(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
    if image.dtype == np.uint8:
        tensor /= 255.0
    elif bool((tensor < 0).any()) or bool((tensor > 1).any()):
        raise ValueError("Floating RGB image must be in [0,1]")
    return tensor


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    pooled = F.max_pool2d(mask.float()[None, None], 2 * radius + 1, stride=1, padding=radius)
    return pooled[0, 0].bool()


def _feather(mask: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    support = _dilate(mask, radius)
    if radius <= 0:
        return mask.float(), support
    blurred = F.avg_pool2d(
        mask.float()[None, None], 2 * radius + 1, stride=1, padding=radius
    )[0, 0]
    alpha = torch.where(mask.bool(), torch.ones_like(blurred), blurred).clamp(0, 1)
    return alpha * support.float(), support


def _colour_match(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    support: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    enabled = bool(settings["enabled"])
    ring = support & ~mask.bool()
    if not enabled or int(ring.sum()) < 3:
        return source, {"enabled": enabled, "applied": False, "gain": [1.0] * 3, "offset": [0.0] * 3}
    source_values = source[ring]
    target_values = target[ring]
    source_median = source_values.median(dim=0).values
    target_median = target_values.median(dim=0).values
    source_mad = (source_values - source_median).abs().median(dim=0).values
    target_mad = (target_values - target_median).abs().median(dim=0).values
    gain = (target_mad / source_mad.clamp_min(1.0 / 255.0)).clamp(
        float(settings["minimum_gain"]), float(settings["maximum_gain"])
    )
    maximum_offset = float(settings["maximum_absolute_offset"])
    offset = (target_median - gain * source_median).clamp(-maximum_offset, maximum_offset)
    adjusted = (source * gain + offset).clamp(0, 1)
    return adjusted, {
        "enabled": True,
        "applied": True,
        "gain": [float(value) for value in gain.tolist()],
        "offset": [float(value) for value in offset.tolist()],
        "source_boundary_median": [float(value) for value in source_median.tolist()],
        "target_boundary_median": [float(value) for value in target_median.tolist()],
    }


def construct_coarse_gan_input(
    source_rgb: np.ndarray,
    source_mask: np.ndarray,
    normal_background_rgb: np.ndarray,
    valid_region: np.ndarray,
    *,
    seed: int,
    transform_settings: dict[str, Any],
    colour_settings: dict[str, Any],
    provenance_base: dict[str, Any],
) -> dict[str, Any]:
    if source_rgb.shape != normal_background_rgb.shape or source_rgb.shape[:2] != source_mask.shape:
        raise ValueError("Source, mask, and normal background must share patch geometry")
    if valid_region.shape != source_mask.shape:
        raise ValueError("Valid-region mask must match the patch")
    source_mask = source_mask.astype(bool)
    valid_region = valid_region.astype(bool)
    if not source_mask.any():
        raise ValueError("A source defect template cannot have an empty mask")
    rng = np.random.default_rng(seed)
    horizontal_flip = bool(rng.random() < float(transform_settings["horizontal_flip_probability"]))
    vertical_flip = bool(rng.random() < float(transform_settings["vertical_flip_probability"]))
    scale = float(
        rng.uniform(float(transform_settings["minimum_scale"]), float(transform_settings["maximum_scale"]))
    )
    feather_radius = int(transform_settings["feather_radius"])
    boundary_radius = int(colour_settings["boundary_radius"])
    context = max(feather_radius, boundary_radius)
    box = bounding_box(source_mask)
    assert box is not None
    height, width = source_mask.shape
    crop_left = max(0, box.x_min - context)
    crop_right = min(width, box.x_max + context + 1)
    crop_top = max(0, box.y_min - context)
    crop_bottom = min(height, box.y_max + context + 1)
    source_crop = _float_rgb(source_rgb[crop_top:crop_bottom, crop_left:crop_right])
    mask_crop = torch.from_numpy(source_mask[crop_top:crop_bottom, crop_left:crop_right])
    scaled_height = max(1, int(round(len(mask_crop) * scale)))
    scaled_width = max(1, int(round(mask_crop.shape[1] * scale)))
    resized_rgb = F.interpolate(
        source_crop.permute(2, 0, 1)[None],
        size=(scaled_height, scaled_width),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    resized_mask = F.interpolate(
        mask_crop.float()[None, None], size=(scaled_height, scaled_width), mode="nearest"
    )[0, 0].bool()
    dimensions: list[int] = []
    if horizontal_flip:
        dimensions.append(1)
    if vertical_flip:
        dimensions.append(0)
    if dimensions:
        resized_rgb = torch.flip(resized_rgb, dimensions)
        resized_mask = torch.flip(resized_mask, dimensions)
    expected_area = max(1.0, float(source_mask.sum()) * scale * scale)
    retained_fraction = float(resized_mask.sum()) / expected_area
    minimum_pixels = int(provenance_base["minimum_positive_pixels"])
    if int(resized_mask.sum()) < minimum_pixels:
        raise ValueError("transformation_below_minimum_positive_pixels")
    if retained_fraction < float(transform_settings["minimum_retained_area_fraction"]):
        raise ValueError("transformation_removed_too_much_defect_area")
    if scaled_height > height or scaled_width > width:
        raise ValueError("transformed_template_exceeds_patch")

    positions: list[tuple[int, int]] = []
    attempts = int(transform_settings["placement_attempts"])
    for _ in range(attempts):
        top = int(rng.integers(0, height - scaled_height + 1))
        left = int(rng.integers(0, width - scaled_width + 1))
        selected_valid = torch.from_numpy(valid_region[top : top + scaled_height, left : left + scaled_width])
        if bool(selected_valid[resized_mask].all()):
            positions.append((top, left))
            break
    if not positions:
        raise ValueError("no_valid_template_placement")
    target_top, target_left = positions[0]
    source_layer = torch.zeros((height, width, 3), dtype=torch.float32)
    source_content_valid = torch.zeros((height, width), dtype=torch.bool)
    transformed_mask = torch.zeros((height, width), dtype=torch.bool)
    source_layer[target_top : target_top + scaled_height, target_left : target_left + scaled_width] = resized_rgb
    source_content_valid[
        target_top : target_top + scaled_height, target_left : target_left + scaled_width
    ] = True
    transformed_mask[target_top : target_top + scaled_height, target_left : target_left + scaled_width] = resized_mask
    alpha, support = _feather(transformed_mask, feather_radius)
    support &= source_content_valid
    support &= torch.from_numpy(valid_region)
    alpha *= support.float()
    background = _float_rgb(normal_background_rgb)
    adjusted_source, colour_parameters = _colour_match(
        source_layer, background, transformed_mask, support, colour_settings
    )
    composite = background.clone()
    blended = background * (1.0 - alpha[..., None]) + adjusted_source * alpha[..., None]
    composite[support] = blended[support]
    if not torch.equal(composite[~support], background[~support]):
        raise RuntimeError("Pixels outside coarse-composite support changed")
    provenance = {
        **provenance_base,
        "generated_sample_seed": int(seed),
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "scale": scale,
        "translation": {"x": target_left, "y": target_top},
        "colour_matching": colour_parameters,
        "patch_size": {"width": width, "height": height},
        "transformed_positive_pixels": int(transformed_mask.sum()),
        "retained_area_fraction": retained_fraction,
    }
    validate_provenance(provenance)
    return {
        "normal_background": rgb_to_gan(background.numpy()),
        "source_template": rgb_to_gan(source_layer.numpy()),
        "conditioning_mask": binary_mask_tensor(transformed_mask),
        "feathered_support": alpha.unsqueeze(0),
        "support_mask": binary_mask_tensor(support),
        "valid_region": binary_mask_tensor(valid_region),
        "coarse_composite": rgb_to_gan(composite.numpy()),
        "difference_from_background": composite.sub(background).abs().permute(2, 0, 1),
        "provenance": provenance,
    }
