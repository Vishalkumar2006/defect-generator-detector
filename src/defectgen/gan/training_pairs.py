"""Deterministic bridge from F1.4 samples to typed GAN training pairs."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from defectgen.training.gan_losses import (
    canonicalize_discriminator_mask,
    load_gan_loss_config,
)

from .dataset import OnlineGANInputDataset, SampleLoader, _validate_metadata


DATA_BRIDGE_VERSION = "g1_3_gan_training_pairs_v1"


@dataclass(frozen=True)
class GANTrainingPairConfig:
    data_bridge_version: str
    manifest_path: str
    loss_config_path: str
    base_seed: int
    monitor_fraction: float
    image_height: int
    image_width: int
    normalization_range: tuple[float, float]
    discriminator_mask_threshold: float
    deterministic_monitoring: bool
    audit_sample_count: int

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GANTrainingPairConfig":
        required = tuple(field.name for field in fields(cls))
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"GAN training-pair config is missing: {', '.join(missing)}")
        normalized = dict(values)
        normalized["normalization_range"] = tuple(values["normalization_range"])
        config = cls(**{name: normalized[name] for name in required})
        config.validate()
        return config

    def validate(self) -> None:
        if self.data_bridge_version != DATA_BRIDGE_VERSION:
            raise ValueError(f"data_bridge_version must be {DATA_BRIDGE_VERSION!r}")
        if not 0 < float(self.monitor_fraction) < 0.5:
            raise ValueError("monitor_fraction must be in (0, 0.5)")
        if self.image_height != 512 or self.image_width != 256:
            raise ValueError("G1.3 requires image_height=512 and image_width=256")
        if self.normalization_range != (-1, 1):
            raise ValueError("normalization_range must be [-1, 1]")
        if not 0 < float(self.discriminator_mask_threshold) <= 1:
            raise ValueError("discriminator_mask_threshold must be in (0, 1]")
        if not self.deterministic_monitoring:
            raise ValueError("G1.3 requires deterministic monitoring")
        if self.audit_sample_count <= 0:
            raise ValueError("audit_sample_count must be positive")


def load_gan_training_pair_config(path: Path | str) -> GANTrainingPairConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("GAN training-pair config must contain a JSON object")
    return GANTrainingPairConfig.from_dict(values)


def _stable_rank(seed: int, category: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{identity}".encode("utf-8")).hexdigest()


def _contact_combination(template: dict[str, Any]) -> str:
    contacts = template["source_contact_sides"]
    active = [side for side in ("top", "bottom", "left", "right") if contacts[side]]
    return "+".join(active) if active else "none"


@dataclass(frozen=True)
class GANInternalSplit:
    train_template_indices: tuple[int, ...]
    monitor_template_indices: tuple[int, ...]
    train_normal_indices: tuple[int, ...]
    monitor_normal_indices: tuple[int, ...]
    train_defect_source_ids: frozenset[str]
    monitor_defect_source_ids: frozenset[str]
    train_background_ids: frozenset[str]
    monitor_background_ids: frozenset[str]
    representation_warnings: tuple[str, ...]

    def assert_disjoint(self) -> None:
        if self.train_defect_source_ids & self.monitor_defect_source_ids:
            raise RuntimeError("Defect source IDs crossed the GAN train/monitor split")
        if self.train_background_ids & self.monitor_background_ids:
            raise RuntimeError("Background IDs crossed the GAN train/monitor split")


def create_internal_gan_split(
    metadata: dict[str, Any], *, monitor_fraction: float = 0.10, seed: int = 42
) -> GANInternalSplit:
    """Create a deterministic grouped train/monitor split from GAN-training rows only."""
    _validate_metadata(metadata)
    if not 0 < monitor_fraction < 0.5:
        raise ValueError("monitor_fraction must be in (0, 0.5)")
    templates = metadata["templates"]
    normals = metadata["normal_backgrounds"]
    defect_groups: dict[str, list[int]] = defaultdict(list)
    group_categories: dict[str, set[str]] = defaultdict(set)
    for index, template in enumerate(templates):
        source_id = str(template["sample_id"])
        defect_groups[source_id].append(index)
        combination = _contact_combination(template)
        group_categories[source_id].update(
            {combination, "border" if combination != "none" else "non-border"}
        )
    category_counts = Counter(
        category for categories in group_categories.values() for category in categories
    )
    target_monitor_groups = max(1, int(round(len(defect_groups) * monitor_fraction)))
    desired = {
        category: min(count - 1, max(1, int(round(count * monitor_fraction))))
        for category, count in category_counts.items()
        if count >= 2
    }
    selected: set[str] = set()
    selected_categories: Counter[str] = Counter()
    ranked_ids = sorted(
        defect_groups, key=lambda identity: _stable_rank(seed, "defect", identity)
    )
    while len(selected) < target_monitor_groups:
        candidates: list[tuple[float, str, str]] = []
        for identity in ranked_ids:
            if identity in selected:
                continue
            categories = group_categories[identity]
            if any(
                category_counts[category] > 1
                and selected_categories[category] >= category_counts[category] - 1
                for category in categories
            ):
                continue
            score = sum(
                max(0, desired.get(category, 0) - selected_categories[category])
                / category_counts[category]
                for category in categories
            )
            candidates.append(
                (-score, _stable_rank(seed, "defect-monitor", identity), identity)
            )
        if not candidates:
            raise RuntimeError("Unable to create a disjoint grouped defect monitor split")
        _, _, chosen = min(candidates)
        selected.add(chosen)
        selected_categories.update(group_categories[chosen])

    monitor_defect_ids = frozenset(selected)
    train_defect_ids = frozenset(set(defect_groups) - selected)
    monitor_template_indices = tuple(
        index for identity in sorted(selected) for index in defect_groups[identity]
    )
    train_template_indices = tuple(
        index for identity in sorted(train_defect_ids) for index in defect_groups[identity]
    )

    normal_groups: dict[str, list[int]] = defaultdict(list)
    for index, normal in enumerate(normals):
        normal_groups[str(normal["sample_id"])].append(index)
    normal_ids = sorted(
        normal_groups, key=lambda identity: _stable_rank(seed, "background", identity)
    )
    monitor_normal_group_count = max(1, int(round(len(normal_ids) * monitor_fraction)))
    monitor_background_ids = frozenset(normal_ids[:monitor_normal_group_count])
    train_background_ids = frozenset(set(normal_ids) - monitor_background_ids)
    monitor_normal_indices = tuple(
        index
        for identity in sorted(monitor_background_ids)
        for index in normal_groups[identity]
    )
    train_normal_indices = tuple(
        index
        for identity in sorted(train_background_ids)
        for index in normal_groups[identity]
    )

    warnings: list[str] = []
    for category, count in sorted(category_counts.items()):
        train_has = any(category in group_categories[identity] for identity in train_defect_ids)
        monitor_has = any(
            category in group_categories[identity] for identity in monitor_defect_ids
        )
        if count < 2:
            warnings.append(
                f"rare_contact_category:{category}:source_groups={count}:cannot_split"
            )
        elif not train_has or not monitor_has:
            warnings.append(
                f"contact_category_not_represented_in_both:{category}:source_groups={count}"
            )
    split = GANInternalSplit(
        train_template_indices=train_template_indices,
        monitor_template_indices=monitor_template_indices,
        train_normal_indices=train_normal_indices,
        monitor_normal_indices=monitor_normal_indices,
        train_defect_source_ids=train_defect_ids,
        monitor_defect_source_ids=monitor_defect_ids,
        train_background_ids=train_background_ids,
        monitor_background_ids=monitor_background_ids,
        representation_warnings=tuple(warnings),
    )
    split.assert_disjoint()
    return split


@dataclass(frozen=True)
class GANTrainingSample:
    composite_image: torch.Tensor
    generator_mask: torch.Tensor
    fake_discriminator_mask: torch.Tensor
    real_image: torch.Tensor
    real_discriminator_mask: torch.Tensor
    fake_valid_mask: torch.Tensor
    real_valid_mask: torch.Tensor
    metadata: dict[str, Any]


def _padding_summary(valid_mask: torch.Tensor) -> dict[str, Any]:
    valid = valid_mask[0].bool()
    coordinates = torch.nonzero(valid, as_tuple=False)
    if not len(coordinates):
        raise ValueError("A training-pair valid mask cannot be empty")
    y_min, x_min = coordinates.min(dim=0).values.tolist()
    y_max, x_max = coordinates.max(dim=0).values.tolist()
    height, width = valid.shape
    return {
        "left": int(x_min),
        "top": int(y_min),
        "right": int(width - 1 - x_max),
        "bottom": int(height - 1 - y_max),
        "invalid_pixels": int((~valid).sum()),
        "valid_fraction": float(valid.float().mean()),
    }


def _validate_training_sample(sample: GANTrainingSample, height: int, width: int) -> None:
    images = (sample.composite_image, sample.real_image)
    masks = (
        sample.generator_mask,
        sample.fake_discriminator_mask,
        sample.real_discriminator_mask,
        sample.fake_valid_mask,
        sample.real_valid_mask,
    )
    if any(image.shape != (3, height, width) for image in images):
        raise RuntimeError("GAN training images violated the [3,H,W] contract")
    if any(image.dtype != torch.float32 for image in images):
        raise RuntimeError("GAN training images must be float32")
    if any(not bool(torch.isfinite(image).all()) for image in images):
        raise RuntimeError("GAN training images must be finite")
    if any(bool((image < -1).any()) or bool((image > 1).any()) for image in images):
        raise RuntimeError("GAN training images must remain in [-1,1]")
    if any(mask.shape != (1, height, width) for mask in masks):
        raise RuntimeError("GAN training masks violated the [1,H,W] contract")
    if any(mask.dtype != torch.float32 for mask in masks):
        raise RuntimeError("GAN training masks must be float32")
    if any(not bool(torch.isfinite(mask).all()) for mask in masks):
        raise RuntimeError("GAN training masks must be finite")
    if any(bool((mask < 0).any()) or bool((mask > 1).any()) for mask in masks):
        raise RuntimeError("GAN training masks must remain in [0,1]")
    if not torch.equal(sample.fake_discriminator_mask, sample.real_discriminator_mask):
        raise RuntimeError("Real/fake canonical discriminator masks diverged")
    if not bool(sample.fake_discriminator_mask.any()):
        raise RuntimeError("A transformed training mask became empty after canonicalization")


class GANTrainingPairDataset(Dataset):
    """Typed deterministic train/monitor pairs backed by the F1.4 online sampler."""

    def __init__(
        self,
        metadata: dict[str, Any],
        repo_root: Path,
        config: GANTrainingPairConfig,
        *,
        split: str,
        internal_split: GANInternalSplit | None = None,
        sample_loader: SampleLoader | None = None,
        length: int | None = None,
    ) -> None:
        _validate_metadata(metadata)
        config.validate()
        if split not in {"train", "monitor"}:
            raise ValueError("GAN training-pair split must be train or monitor")
        if int(metadata["patch"]["height"]) != config.image_height or int(
            metadata["patch"]["width"]
        ) != config.image_width:
            raise ValueError("GAN manifest patch dimensions disagree with bridge config")
        self.metadata = metadata
        self.repo_root = Path(repo_root)
        self.config = config
        self.split = split
        loss_config = load_gan_loss_config(self.repo_root / config.loss_config_path)
        if float(loss_config.canonical_mask_threshold) != float(
            config.discriminator_mask_threshold
        ):
            raise ValueError(
                "Training-pair discriminator mask threshold disagrees with G1.2"
            )
        self.internal_split = internal_split or create_internal_gan_split(
            metadata, monitor_fraction=config.monitor_fraction, seed=config.base_seed
        )
        self.internal_split.assert_disjoint()
        template_indices = (
            self.internal_split.train_template_indices
            if split == "train"
            else self.internal_split.monitor_template_indices
        )
        normal_indices = (
            self.internal_split.train_normal_indices
            if split == "train"
            else self.internal_split.monitor_normal_indices
        )
        self.epoch = 0
        self._online = OnlineGANInputDataset(
            metadata,
            repo_root,
            base_seed=self._epoch_seed(0),
            length=length,
            sample_loader=sample_loader,
            template_indices=template_indices,
            normal_indices=normal_indices,
            include_training_details=True,
        )

    def _epoch_seed(self, epoch: int) -> int:
        effective_epoch = epoch if self.split == "train" else 0
        material = (
            f"{self.config.base_seed}:{self.split}:{effective_epoch}:"
            f"{self.metadata['gan_manifest_content_sha256']}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch) if self.split == "train" else 0
        self._online.base_seed = self._epoch_seed(self.epoch)

    def __len__(self) -> int:
        return len(self._online)

    def __getitem__(self, index: int) -> GANTrainingSample:
        f1_sample = self._online[index]
        details = f1_sample["training_details"]
        generator_mask = f1_sample["feathered_support"].float().contiguous()
        real_fractional_mask = details["transformed_real_mask"].float().contiguous()
        if not torch.equal(generator_mask, real_fractional_mask):
            raise RuntimeError("Real/fake fractional masks did not share one transform")
        fake_discriminator_mask = canonicalize_discriminator_mask(
            generator_mask.unsqueeze(0),
            threshold=self.config.discriminator_mask_threshold,
        )[0]
        real_discriminator_mask = canonicalize_discriminator_mask(
            real_fractional_mask.unsqueeze(0),
            threshold=self.config.discriminator_mask_threshold,
        )[0]
        if not torch.equal(fake_discriminator_mask, real_discriminator_mask):
            raise RuntimeError("Real/fake canonical discriminator masks are not bit-exact")
        provenance = f1_sample["provenance"]
        transform = copy.deepcopy(details["shared_spatial_transform"])
        fake_valid = f1_sample["valid_region"].float().contiguous()
        real_valid = details["transformed_real_valid_region"].float().contiguous()
        sample_metadata = {
            "data_bridge_version": self.config.data_bridge_version,
            "split": self.split,
            "epoch": self.epoch,
            "sample_index": int(index),
            "template_id": provenance["sampling_accounting"]["template_identity"],
            "template_source_sample_id": provenance["source_defect_sample_id"],
            "normal_background_sample_id": provenance["normal_background_sample_id"],
            "development_split": "train",
            "official_split": "train",
            "real_transform": copy.deepcopy(transform),
            "fake_transform": copy.deepcopy(transform),
            "transform": transform,
            "placement": copy.deepcopy(provenance["translation"]),
            "source_contact_sides": copy.deepcopy(provenance["source_contact_sides"]),
            "transformed_contact_sides": copy.deepcopy(
                provenance["transformed_source_contact_sides"]
            ),
            "target_contact_sides": copy.deepcopy(provenance["target_contact_sides"]),
            "source_native_dimensions": copy.deepcopy(
                details["source_native_dimensions"]
            ),
            "normal_native_dimensions": copy.deepcopy(
                details["normal_native_dimensions"]
            ),
            "source_padding_before_transform": _padding_summary(
                details["source_patch_valid_region"]
            ),
            "real_padding_after_transform": _padding_summary(real_valid),
            "fake_padding": _padding_summary(fake_valid),
            "deterministic_sample_seed": int(provenance["generated_sample_seed"]),
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "gan_manifest_content_sha256": provenance["gan_manifest_content_sha256"],
            "split_sha256": provenance["split_sha256"],
        }
        sample = GANTrainingSample(
            composite_image=f1_sample["coarse_composite"].float().contiguous(),
            generator_mask=generator_mask,
            fake_discriminator_mask=fake_discriminator_mask.contiguous(),
            real_image=details["transformed_real_image"].float().contiguous(),
            real_discriminator_mask=real_discriminator_mask.contiguous(),
            fake_valid_mask=fake_valid,
            real_valid_mask=real_valid,
            metadata=sample_metadata,
        )
        _validate_training_sample(sample, self.config.image_height, self.config.image_width)
        return sample


def load_training_pair_manifest(
    repo_root: Path, config: GANTrainingPairConfig
) -> dict[str, Any]:
    metadata = json.loads((Path(repo_root) / config.manifest_path).read_text(encoding="utf-8"))
    _validate_metadata(metadata)
    return metadata
