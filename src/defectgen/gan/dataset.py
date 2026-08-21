"""Deterministic, online GAN input construction from training-only metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from defectgen.data.ksdd2 import interpret_binary_mask

from .geometry import ComponentWindow, connected_components, extract_native_window
from .manifest import assert_gan_training_rows
from .pipeline import construct_coarse_gan_input


SampleLoader = Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]


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
    ) -> None:
        _validate_metadata(metadata)
        self.metadata = metadata
        self.repo_root = Path(repo_root)
        self.base_seed = int(metadata["seed"] if base_seed is None else base_seed)
        natural_length = max(len(metadata["templates"]), len(metadata["normal_backgrounds"]))
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
            f"{self.base_seed}:{index}:{self.metadata['manifest_sha256']}:"
            f"{self.metadata['split_sha256']}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        sample_seed = self._sample_seed(index)
        rng = np.random.default_rng(sample_seed)
        template = self.metadata["templates"][int(rng.integers(len(self.metadata["templates"])))]
        normal = self.metadata["normal_backgrounds"][
            int(rng.integers(len(self.metadata["normal_backgrounds"])))
        ]
        source_image, source_mask = self._sample_loader(template)
        normal_image, normal_mask = self._sample_loader(normal)
        if normal_mask.any():
            raise ValueError("A GAN background loader returned defect pixels")

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
            touches_native_border=bool(coordinates["touches_native_border"]),
        )
        source_patch, component_patch, _ = extract_native_window(
            source_image, component.mask, source_window
        )
        if int(component_patch.sum()) != source_window.positive_pixels:
            raise ValueError("Template metadata no longer aligns with its source component")

        patch_width = int(self.metadata["patch"]["width"])
        patch_height = int(self.metadata["patch"]["height"])
        maximum_top = max(0, normal_image.shape[0] - patch_height)
        maximum_left = max(0, normal_image.shape[1] - patch_width)
        target_top = int(rng.integers(maximum_top + 1))
        target_left = int(rng.integers(maximum_left + 1))
        background, _, valid = extract_native_window(
            normal_image,
            np.zeros(normal_image.shape[:2], dtype=bool),
            (target_top, target_left, patch_width, patch_height),
        )
        valid_fraction = float(valid.mean())
        if valid_fraction < float(self.metadata["patch"]["minimum_normal_valid_fraction"]):
            raise ValueError("Selected normal window is below the minimum valid fraction")

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
            "partial_component": bool(template["partial_component"]),
            "coverage_fraction": float(template["coverage_fraction"]),
            "touches_native_border": bool(template["touches_native_border"]),
            "minimum_positive_pixels": int(self.metadata["patch"]["minimum_positive_pixels"]),
            "manifest_sha256": self.metadata["manifest_sha256"],
            "split_sha256": self.metadata["split_sha256"],
            "pipeline_version": self.metadata["pipeline_version"],
        }
        return construct_coarse_gan_input(
            source_patch,
            component_patch,
            background,
            valid,
            seed=sample_seed,
            transform_settings=self.metadata["transform"],
            colour_settings=self.metadata["colour_matching"],
            provenance_base=provenance_base,
        )
