"""Deterministic, online GAN input construction from training-only metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from defectgen.data.ksdd2 import interpret_binary_mask

from .compatibility import (
    GANPlacementCompatibilityIndex,
    measure_transformed_template_geometry,
)
from .geometry import ContactSides, ComponentWindow, connected_components, extract_native_window
from .manifest import assert_gan_training_rows, gan_manifest_content_hash
from .pipeline import construct_coarse_gan_input


SampleLoader = Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]


class GANSamplingFailure(ValueError):
    def __init__(self, reason: str, accounting: dict[str, Any]) -> None:
        super().__init__(reason)
        self.accounting = accounting


def _validate_metadata(metadata: dict[str, Any]) -> None:
    templates = metadata.get("templates", [])
    normals = metadata.get("normal_backgrounds", [])
    if not templates or not normals:
        raise ValueError("Online GAN inputs require templates and normal backgrounds")
    assert_gan_training_rows(templates)
    assert_gan_training_rows(normals)
    if any(not bool(row.get("has_defect")) for row in templates):
        raise ValueError("GAN templates must be defective development-training samples")
    if any(bool(row.get("has_defect")) for row in normals):
        raise ValueError("GAN backgrounds must be normal development-training samples")
    boundary = metadata.get("data_boundary", {})
    forbidden_counts = (
        "validation_rows_loaded",
        "official_test_rows_loaded",
        "validation_predictions_loaded",
    )
    if any(int(boundary.get(field, 0)) != 0 for field in forbidden_counts):
        raise ValueError("GAN metadata reports forbidden validation/test access")
    if int(metadata.get("materialized_image_files", 0)) != 0:
        raise ValueError("GAN metadata cannot describe bulk materialized images")
    if "manifest_sha256" in metadata or "source_manifest_sha256" not in metadata:
        raise ValueError("GAN metadata must use the explicit source_manifest_sha256 name")
    expected_content_hash = gan_manifest_content_hash(metadata)
    if metadata.get("gan_manifest_content_sha256") != expected_content_hash:
        raise ValueError("GAN manifest content hash mismatch")


def select_target_window(
    image_shape: tuple[int, int],
    patch_size: tuple[int, int],
    required_contacts: ContactSides,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], ContactSides]:
    """Select a window containing every native edge required by a transformed template."""
    native_height, native_width = image_shape
    patch_width, patch_height = patch_size
    maximum_top = max(0, native_height - patch_height)
    maximum_left = max(0, native_width - patch_width)
    if required_contacts.top and required_contacts.bottom and native_height > patch_height:
        raise ValueError("target_window_cannot_contain_top_and_bottom_native_edges")
    if required_contacts.left and required_contacts.right and native_width > patch_width:
        raise ValueError("target_window_cannot_contain_left_and_right_native_edges")
    if required_contacts.top:
        top = 0
    elif required_contacts.bottom:
        top = maximum_top
    else:
        top = int(rng.integers(maximum_top + 1))
    if required_contacts.left:
        left = 0
    elif required_contacts.right:
        left = maximum_left
    else:
        left = int(rng.integers(maximum_left + 1))
    window_contacts = ContactSides(
        top=top == 0,
        bottom=top + patch_height >= native_height,
        left=left == 0,
        right=left + patch_width >= native_width,
    )
    for side in ("top", "bottom", "left", "right"):
        if getattr(required_contacts, side) and not getattr(window_contacts, side):
            raise ValueError(f"target_window_missing_required_native_{side}_edge")
    return (top, left, patch_width, patch_height), window_contacts


class OnlineGANInputDataset(Dataset):
    """Generate coarse GAN inputs in memory; no generated image files are written."""

    def __init__(
        self,
        metadata: dict[str, Any],
        repo_root: Path,
        *,
        base_seed: int | None = None,
        length: int | None = None,
        sample_loader: SampleLoader | None = None,
        template_indices: list[int] | tuple[int, ...] | None = None,
        normal_indices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        _validate_metadata(metadata)
        self.metadata = metadata
        self.repo_root = Path(repo_root)
        self.base_seed = int(metadata["seed"] if base_seed is None else base_seed)
        self.templates = (
            metadata["templates"]
            if template_indices is None
            else [metadata["templates"][index] for index in template_indices]
        )
        self.normal_backgrounds = (
            metadata["normal_backgrounds"]
            if normal_indices is None
            else [metadata["normal_backgrounds"][index] for index in normal_indices]
        )
        if not self.templates or not self.normal_backgrounds:
            raise ValueError("Category selection produced no GAN templates or backgrounds")
        self.border_templates = [
            template for template in self.templates if any(template["source_contact_sides"].values())
        ]
        self.non_border_templates = [
            template for template in self.templates if not any(template["source_contact_sides"].values())
        ]
        self.sampling = metadata.get(
            "sampling",
            {
                "border_fraction_mode": "empirical",
                "border_fraction": None,
                "maximum_transform_attempts": 4,
            },
        )
        mode = self.sampling["border_fraction_mode"]
        if mode not in {"empirical", "fixed"}:
            raise ValueError("border_fraction_mode must be empirical or fixed")
        if mode == "fixed" and not 0 <= float(self.sampling["border_fraction"]) <= 1:
            raise ValueError("A fixed border fraction must be in [0,1]")
        patch_size = (int(metadata["patch"]["width"]), int(metadata["patch"]["height"]))
        self.compatibility_index = GANPlacementCompatibilityIndex(
            self.normal_backgrounds,
            patch_size=patch_size,
            non_border_margin=int(metadata["transform"]["non_border_native_margin"]),
            feather_radius=int(metadata["transform"]["feather_radius"]),
        )
        natural_length = max(len(self.templates), len(self.normal_backgrounds))
        self.length = natural_length if length is None else int(length)
        if self.length <= 0:
            raise ValueError("Online GAN dataset length must be positive")
        self._sample_loader = sample_loader or self._load_training_pair

    def __len__(self) -> int:
        return self.length

    def _load_training_pair(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        assert_gan_training_rows([row])
        with Image.open(self.repo_root / row["image_path"]) as source:
            image = np.asarray(source.convert("RGB"))
        if row.get("mask_path"):
            with Image.open(self.repo_root / row["mask_path"]) as source:
                mask = interpret_binary_mask(np.asarray(source))
        else:
            mask = np.zeros(image.shape[:2], dtype=bool)
        if mask.shape != image.shape[:2] or bool(mask.any()) != bool(row["has_defect"]):
            raise ValueError(f"Image/mask label mismatch for {row['sample_id']}")
        return image, mask

    def _sample_seed(self, index: int) -> int:
        material = (
            f"{self.base_seed}:{index}:{self.metadata['source_manifest_sha256']}:"
            f"{self.metadata['gan_manifest_content_sha256']}:"
            f"{self.metadata['split_sha256']}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    @staticmethod
    def _attempt_seed(sample_seed: int, attempt: int) -> int:
        if attempt == 0:
            return sample_seed
        material = f"{sample_seed}:transform-attempt:{attempt}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _select_template(
        self, rng: np.random.Generator
    ) -> tuple[dict[str, Any], str, float]:
        empirical = len(self.border_templates) / len(self.templates)
        target_fraction = (
            empirical
            if self.sampling["border_fraction_mode"] == "empirical"
            else float(self.sampling["border_fraction"])
        )
        if not self.border_templates:
            selected_class = "non-border"
        elif not self.non_border_templates:
            selected_class = "border"
        else:
            selected_class = "border" if rng.random() < target_fraction else "non-border"
        candidates = (
            self.border_templates if selected_class == "border" else self.non_border_templates
        )
        return candidates[int(rng.integers(len(candidates)))], selected_class, target_fraction

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        sample_seed = self._sample_seed(index)
        rng = np.random.default_rng(sample_seed)
        template, selected_class, target_border_fraction = self._select_template(rng)
        source_image, source_mask = self._sample_loader(template)

        components = connected_components(source_mask)
        component_id = int(template["component_id"])
        if not 0 <= component_id < len(components):
            raise ValueError("Template component ID is not present in its source mask")
        component = components[component_id]
        coordinates = template["source_window_coordinates"]
        source_window = ComponentWindow(
            top=int(coordinates["top"]),
            left=int(coordinates["left"]),
            width=int(coordinates["width"]),
            height=int(coordinates["height"]),
            partial_component=bool(coordinates["partial_component"]),
            coverage_fraction=float(coordinates["coverage_fraction"]),
            positive_pixels=int(coordinates["positive_pixels"]),
            source_contact_sides=ContactSides.from_dict(coordinates["source_contact_sides"]),
        )
        source_patch, component_patch, _ = extract_native_window(
            source_image, component.mask, source_window
        )
        if int(component_patch.sum()) != source_window.positive_pixels:
            raise ValueError("Template metadata no longer aligns with its source component")

        patch_width = int(self.metadata["patch"]["width"])
        patch_height = int(self.metadata["patch"]["height"])
        source_contacts = ContactSides.from_dict(template["source_contact_sides"])
        if source_contacts != component.contact_sides:
            raise ValueError("Template source-contact metadata no longer matches its component")
        maximum_attempts = int(self.sampling["maximum_transform_attempts"])
        if maximum_attempts <= 0:
            raise ValueError("maximum_transform_attempts must be positive")
        indexing_exclusions = 0
        indexing_candidates_examined = 0
        exclusion_reasons: dict[str, int] = {}
        empty_pools = 0
        actual_placement_retries = 0
        failure_reasons: list[str] = []
        empty_pool_side_combinations: dict[str, int] = {}
        failure_side_combinations: dict[str, int] = {}
        for attempt in range(maximum_attempts):
            attempt_seed = self._attempt_seed(sample_seed, attempt)
            try:
                geometry = measure_transformed_template_geometry(
                    component_patch,
                    source_contacts,
                    seed=attempt_seed,
                    transform_settings=self.metadata["transform"],
                    colour_settings=self.metadata["colour_matching"],
                    minimum_positive_pixels=int(
                        self.metadata["patch"]["minimum_positive_pixels"]
                    ),
                )
            except ValueError as error:
                failure_reasons.append(str(error))
                failure_side_combinations["transform_rejected_before_geometry"] = (
                    failure_side_combinations.get("transform_rejected_before_geometry", 0) + 1
                )
                continue
            pool = self.compatibility_index.query(geometry)
            indexing_candidates_examined += pool.candidates_examined
            indexing_exclusions += pool.candidates_excluded
            for reason, count in pool.exclusion_reasons.items():
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + count
            if not pool.background_indices:
                empty_pools += 1
                empty_pool_side_combinations[geometry.side_combination] = (
                    empty_pool_side_combinations.get(geometry.side_combination, 0) + 1
                )
                failure_side_combinations[geometry.side_combination] = (
                    failure_side_combinations.get(geometry.side_combination, 0) + 1
                )
                failure_reasons.append(f"empty_compatibility_pool:{geometry.side_combination}")
                continue
            attempt_rng = np.random.default_rng(attempt_seed ^ 0xC04A71B1E)
            normal_index = pool.background_indices[
                int(attempt_rng.integers(len(pool.background_indices)))
            ]
            normal = self.normal_backgrounds[normal_index]
            normal_image, normal_mask = self._sample_loader(normal)
            if normal_mask.any():
                raise ValueError("A GAN background loader returned defect pixels")
            if (
                normal_image.shape[0] != int(normal["native_height"])
                or normal_image.shape[1] != int(normal["native_width"])
            ):
                raise ValueError("Normal-background native geometry differs from indexed metadata")
            try:
                window, window_contacts = select_target_window(
                    normal_image.shape[:2],
                    (patch_width, patch_height),
                    geometry.transformed_contact_sides,
                    attempt_rng,
                )
                target_top, target_left, _, _ = window
                background, _, valid = extract_native_window(
                    normal_image,
                    np.zeros(normal_image.shape[:2], dtype=bool),
                    window,
                )
                valid_fraction = float(valid.mean())
                if valid_fraction < float(
                    self.metadata["patch"]["minimum_normal_valid_fraction"]
                ):
                    raise ValueError("selected_normal_below_minimum_valid_fraction")
                provenance_base = {
                    "normal_background_sample_id": normal["sample_id"],
                    "source_defect_sample_id": template["sample_id"],
                    "connected_component_id": component_id,
                    "source_mask_bounding_box": template["source_mask_bounding_box"],
                    "source_window_coordinates": coordinates,
                    "target_window_coordinates": {
                        "top": target_top,
                        "left": target_left,
                        "width": patch_width,
                        "height": patch_height,
                    },
                    "target_window_native_contact_sides": window_contacts.to_dict(),
                    "source_contact_sides": source_contacts.to_dict(),
                    "partial_component": bool(template["partial_component"]),
                    "coverage_fraction": float(template["coverage_fraction"]),
                    "touches_native_border": source_contacts.any,
                    "minimum_positive_pixels": int(
                        self.metadata["patch"]["minimum_positive_pixels"]
                    ),
                    "source_manifest_sha256": self.metadata["source_manifest_sha256"],
                    "gan_manifest_content_sha256": self.metadata[
                        "gan_manifest_content_sha256"
                    ],
                    "split_sha256": self.metadata["split_sha256"],
                    "pipeline_version": self.metadata["pipeline_version"],
                }
                sample = construct_coarse_gan_input(
                    source_patch,
                    component_patch,
                    background,
                    valid,
                    seed=attempt_seed,
                    transform_settings=self.metadata["transform"],
                    colour_settings=self.metadata["colour_matching"],
                    provenance_base=provenance_base,
                )
                accounting = {
                    "compatibility_candidates_examined": indexing_candidates_examined,
                    "compatibility_candidates_excluded": indexing_exclusions,
                    "compatibility_pool_size": len(pool.background_indices),
                    "compatibility_exclusion_reasons": dict(sorted(exclusion_reasons.items())),
                    "empty_compatibility_pools": empty_pools,
                    "actual_transform_placement_retries": attempt,
                    "actual_placement_retries": actual_placement_retries,
                    "attempts_per_successful_sample": attempt + 1,
                    "selected_template_class": selected_class,
                    "target_border_fraction": target_border_fraction,
                    "border_fraction_mode": self.sampling["border_fraction_mode"],
                    "successful_side_combination": geometry.side_combination,
                    "template_identity": f"{template['sample_id']}:{component_id}:{template['window_index']}",
                    "background_identity": normal["sample_id"],
                }
                sample["provenance"].update(
                    {"base_sample_seed": sample_seed, "sampling_accounting": accounting}
                )
                sample["placement_diagnostics"].update(accounting)
                return sample
            except ValueError as error:
                actual_placement_retries += 1
                failure_reasons.append(str(error))
                failure_side_combinations[geometry.side_combination] = (
                    failure_side_combinations.get(geometry.side_combination, 0) + 1
                )
        reasons = sorted(set(failure_reasons))
        raise GANSamplingFailure(
            "gan_sampling_attempts_exhausted:" + ",".join(reasons),
            {
                "compatibility_candidates_examined": indexing_candidates_examined,
                "compatibility_candidates_excluded": indexing_exclusions,
                "compatibility_exclusion_reasons": dict(sorted(exclusion_reasons.items())),
                "empty_compatibility_pools": empty_pools,
                "empty_pool_side_combinations": dict(sorted(empty_pool_side_combinations.items())),
                "failure_side_combinations": dict(sorted(failure_side_combinations.items())),
                "actual_transform_placement_retries": max(0, maximum_attempts - 1),
                "actual_placement_retries": actual_placement_retries,
                "attempts": maximum_attempts,
                "selected_template_class": selected_class,
                "target_border_fraction": target_border_fraction,
                "template_identity": f"{template['sample_id']}:{component_id}:{template['window_index']}",
            },
        )
