"""Deterministic conservative template transformation and coarse compositing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from defectgen.data.geometry import bounding_box

from .geometry import ContactSides
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
    "source_contact_sides",
    "transformed_source_contact_sides",
    "target_window_native_contact_sides",
    "target_contact_sides",
    "colour_matching",
    "patch_size",
    "source_manifest_sha256",
    "gan_manifest_content_sha256",
    "split_sha256",
    "pipeline_version",
}


def validate_provenance(provenance: dict[str, Any]) -> None:
    missing = REQUIRED_PROVENANCE_FIELDS - set(provenance)
    if missing:
        raise ValueError(f"GAN input provenance is missing fields: {sorted(missing)}")


def sample_transform_parameters(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "horizontal_flip": bool(
            rng.random() < float(settings["horizontal_flip_probability"])
        ),
        "vertical_flip": bool(rng.random() < float(settings["vertical_flip_probability"])),
        "scale": float(
            rng.uniform(float(settings["minimum_scale"]), float(settings["maximum_scale"]))
        ),
    }


def _axis_placement(
    *,
    mask_minimum: int,
    mask_maximum: int,
    valid_minimum: int,
    valid_maximum: int,
    patch_extent: int,
    content_extent: int,
    contact_minimum: bool,
    contact_maximum: bool,
    margin: int,
    rng: np.random.Generator,
    border_template: bool,
) -> int:
    lower = max(0, valid_minimum - mask_minimum)
    upper = min(patch_extent - content_extent, valid_maximum - mask_maximum)
    if contact_minimum:
        lower = max(lower, valid_minimum - mask_minimum)
        upper = min(upper, valid_minimum - mask_minimum)
    else:
        lower = max(lower, valid_minimum + margin - mask_minimum)
    if contact_maximum:
        lower = max(lower, valid_maximum - mask_maximum)
        upper = min(upper, valid_maximum - mask_maximum)
    else:
        upper = min(upper, valid_maximum - margin - mask_maximum)
    if lower > upper:
        reason = (
            "incompatible_border_placement"
            if border_template
            else "no_non_border_placement_with_required_margin"
        )
        raise ValueError(reason)
    if lower == upper:
        return lower
    return int(rng.integers(lower, upper + 1))


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
    transform_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_rgb.shape != normal_background_rgb.shape or source_rgb.shape[:2] != source_mask.shape:
        raise ValueError("Source, mask, and normal background must share patch geometry")
    if valid_region.shape != source_mask.shape:
        raise ValueError("Valid-region mask must match the patch")
    source_mask = source_mask.astype(bool)
    valid_region = valid_region.astype(bool)
    if not source_mask.any():
        raise ValueError("A source defect template cannot have an empty mask")
    parameters = (
        sample_transform_parameters(seed, transform_settings)
        if transform_parameters is None
        else transform_parameters
    )
    horizontal_flip = parameters["horizontal_flip"]
    vertical_flip = parameters["vertical_flip"]
    scale = parameters["scale"]
    source_contacts = ContactSides.from_dict(provenance_base["source_contact_sides"])
    transformed_contacts = source_contacts.transformed(
        horizontal_flip=horizontal_flip, vertical_flip=vertical_flip
    )
    target_window_contacts = ContactSides.from_dict(
        provenance_base["target_window_native_contact_sides"]
    )
    for side in ("top", "bottom", "left", "right"):
        if getattr(transformed_contacts, side) and not getattr(target_window_contacts, side):
            raise ValueError(f"target_window_missing_required_native_{side}_edge")
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

    resized_box = bounding_box(resized_mask.numpy())
    valid_box = bounding_box(valid_region)
    assert resized_box is not None and valid_box is not None
    placement_rng = np.random.default_rng(seed ^ 0xD1CEB00C)
    margin = int(transform_settings["non_border_native_margin"])
    if margin < 0:
        raise ValueError("non_border_native_margin must be non-negative")
    target_left = _axis_placement(
        mask_minimum=resized_box.x_min,
        mask_maximum=resized_box.x_max,
        valid_minimum=valid_box.x_min,
        valid_maximum=valid_box.x_max,
        patch_extent=width,
        content_extent=scaled_width,
        contact_minimum=transformed_contacts.left,
        contact_maximum=transformed_contacts.right,
        margin=margin,
        rng=placement_rng,
        border_template=transformed_contacts.any,
    )
    target_top = _axis_placement(
        mask_minimum=resized_box.y_min,
        mask_maximum=resized_box.y_max,
        valid_minimum=valid_box.y_min,
        valid_maximum=valid_box.y_max,
        patch_extent=height,
        content_extent=scaled_height,
        contact_minimum=transformed_contacts.top,
        contact_maximum=transformed_contacts.bottom,
        margin=margin,
        rng=placement_rng,
        border_template=transformed_contacts.any,
    )
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
    target_valid = torch.from_numpy(valid_region)
    support &= target_valid
    alpha *= support.float()
    if bool((transformed_mask & ~target_valid).any()):
        raise RuntimeError("Transformed defect entered target padding")
    if bool((support & ~target_valid).any()) or bool((alpha > 0)[~target_valid].any()):
        raise RuntimeError("Feathered defect support entered target padding")
    transformed_box = bounding_box(transformed_mask.numpy())
    assert transformed_box is not None
    target_contacts = ContactSides(
        top=target_window_contacts.top and transformed_box.y_min == valid_box.y_min,
        bottom=target_window_contacts.bottom and transformed_box.y_max == valid_box.y_max,
        left=target_window_contacts.left and transformed_box.x_min == valid_box.x_min,
        right=target_window_contacts.right and transformed_box.x_max == valid_box.x_max,
    )
    accidental_contacts = [
        side
        for side in ("top", "bottom", "left", "right")
        if getattr(target_contacts, side) != getattr(transformed_contacts, side)
    ]
    if accidental_contacts:
        raise RuntimeError(f"target_contact_violation:{','.join(accidental_contacts)}")
    background = _float_rgb(normal_background_rgb)
    adjusted_source, colour_parameters = _colour_match(
        source_layer, background, transformed_mask, support, colour_settings
    )
    composite = background.clone()
    blended = background * (1.0 - alpha[..., None]) + adjusted_source * alpha[..., None]
    composite[support] = blended[support]
    if not torch.equal(composite[~support], background[~support]):
        raise RuntimeError("Pixels outside coarse-composite support changed")
    maximum_difference_outside_support = float(
        composite.sub(background).abs()[~support].max().item() if bool((~support).any()) else 0.0
    )
    support_pixels_outside_valid = int((support & ~target_valid).sum())
    provenance = {
        **provenance_base,
        "generated_sample_seed": int(seed),
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "scale": scale,
        "translation": {"x": target_left, "y": target_top},
        "source_contact_sides": source_contacts.to_dict(),
        "transformed_source_contact_sides": transformed_contacts.to_dict(),
        "target_contact_sides": target_contacts.to_dict(),
        "colour_matching": colour_parameters,
        "patch_size": {"width": width, "height": height},
        "transformed_positive_pixels": int(transformed_mask.sum()),
        "retained_area_fraction": retained_fraction,
        "non_border_native_margin": margin,
        "accidental_contact_violations": 0,
        "support_pixels_outside_valid_region": support_pixels_outside_valid,
        "maximum_difference_outside_support": maximum_difference_outside_support,
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
        "placement_diagnostics": {
            "successful_target_contact_sides": target_contacts.to_dict(),
            "non_border_placement": not transformed_contacts.any,
            "accidental_contact_violations": 0,
            "support_pixels_outside_valid_region": support_pixels_outside_valid,
            "maximum_difference_outside_support": maximum_difference_outside_support,
        },
        "provenance": provenance,
    }
