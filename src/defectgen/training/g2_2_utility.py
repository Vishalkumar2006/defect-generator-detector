"""Deterministic, leakage-guarded utilities for the G2.2 detector experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from defectgen.data.augmentation import SynchronizedRandomFlips


G2_2_VERSION = "g2_2_equal_budget_detector_utility_v1"
ALLOWED_SOURCE_SPLIT = "train"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def assert_train_only_provenance(rows: Iterable[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        for field in ("official_split", "development_split"):
            if row.get(field) != ALLOWED_SOURCE_SPLIT:
                raise RuntimeError(
                    f"Synthetic row {index} has forbidden {field}={row.get(field)!r}"
                )
        source = row.get("source_provenance", {})
        for identity in ("template", "background"):
            record = source.get(identity, {})
            if record.get("official_split") != ALLOWED_SOURCE_SPLIT or record.get(
                "development_split"
            ) != ALLOWED_SOURCE_SPLIT:
                raise RuntimeError(
                    f"Synthetic row {index} has a non-training {identity} source"
                )


def paired_source_signature(row: dict[str, Any]) -> str:
    """Hash every paired field while intentionally excluding rendered checkpoint output."""
    excluded = {
        "checkpoint_step",
        "checkpoint_path",
        "checkpoint_sha256",
        "image_path",
        "image_sha256",
    }
    return canonical_sha256({key: value for key, value in row.items() if key not in excluded})


def assert_paired_manifests(
    first_rows: Sequence[dict[str, Any]], second_rows: Sequence[dict[str, Any]]
) -> None:
    if len(first_rows) != len(second_rows):
        raise RuntimeError("Paired synthetic manifests have different row counts")
    assert_train_only_provenance(first_rows)
    assert_train_only_provenance(second_rows)
    for index, (first, second) in enumerate(zip(first_rows, second_rows)):
        if paired_source_signature(first) != paired_source_signature(second):
            raise RuntimeError(f"Synthetic manifests diverge before GAN rendering at row {index}")


def deterministic_balanced_indices(
    labels: Sequence[bool], count: int, *, seed: int, stream: str
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    classes = {
        False: [index for index, label in enumerate(labels) if not label],
        True: [index for index, label in enumerate(labels) if label],
    }
    if not classes[False] or not classes[True]:
        raise ValueError("Balanced sampling requires normal and defective examples")
    local_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{stream}".encode("utf-8")).digest()[:8], "big"
    )
    generator = random.Random(local_seed)
    return [
        generator.choice(classes[bool(position % 2)]) for position in range(count)
    ]


@dataclass(frozen=True)
class ScheduleEntry:
    position: int
    optimizer_step: int
    batch_position: int
    source: str
    source_index: int
    augmentation_epoch: int


def build_equal_budget_schedule(
    labels: Sequence[bool],
    *,
    optimizer_updates: int,
    batch_size: int,
    seed: int,
    synthetic_fraction: float,
    synthetic_count: int,
    variant: str,
) -> list[ScheduleEntry]:
    """Build an exact-update schedule with fixed per-batch synthetic allocation."""
    if optimizer_updates <= 0 or batch_size <= 0:
        raise ValueError("optimizer_updates and batch_size must be positive")
    synthetic_per_batch = int(round(batch_size * synthetic_fraction))
    if abs(synthetic_per_batch / batch_size - synthetic_fraction) > 1e-12:
        raise ValueError("synthetic_fraction must be exactly representable in each batch")
    if variant == "real_only":
        synthetic_per_batch = 0
    elif variant not in {"checkpoint_1000", "checkpoint_1500"}:
        raise ValueError(f"Unknown detector variant: {variant}")
    if synthetic_per_batch and synthetic_count <= 0:
        raise ValueError("A synthetic arm requires synthetic samples")

    total = optimizer_updates * batch_size
    # The first real slots are common across every arm. The control's replacement
    # slot uses its own stream so it cannot perturb those shared identities.
    common_real_per_batch = batch_size - int(round(batch_size * synthetic_fraction))
    common = deterministic_balanced_indices(
        labels, optimizer_updates * common_real_per_batch, seed=seed, stream="common-real"
    )
    replacement = deterministic_balanced_indices(
        labels,
        optimizer_updates * (batch_size - common_real_per_batch),
        seed=seed,
        stream="control-replacement-real",
    )
    synthetic_offset = int.from_bytes(
        hashlib.sha256(f"{seed}:synthetic-order".encode()).digest()[:8], "big"
    ) % max(1, synthetic_count)
    rows: list[ScheduleEntry] = []
    common_cursor = replacement_cursor = synthetic_cursor = 0
    for step in range(optimizer_updates):
        synthetic_positions = {
            (step + offset) % batch_size for offset in range(synthetic_per_batch)
        }
        for batch_position in range(batch_size):
            if batch_position in synthetic_positions:
                source = "synthetic"
                source_index = (synthetic_offset + synthetic_cursor) % synthetic_count
                synthetic_cursor += 1
            elif common_cursor < (step + 1) * common_real_per_batch:
                source = "real"
                source_index = common[common_cursor]
                common_cursor += 1
            else:
                source = "real"
                source_index = replacement[replacement_cursor]
                replacement_cursor += 1
            rows.append(
                ScheduleEntry(
                    position=len(rows),
                    optimizer_step=step + 1,
                    batch_position=batch_position,
                    source=source,
                    source_index=int(source_index),
                    augmentation_epoch=step,
                )
            )
    if len(rows) != total:
        raise RuntimeError("Equal-budget schedule length is incorrect")
    return rows


class SyntheticDetectorDataset(Dataset):
    """Read materialized train-only synthetic samples with detector normalization."""

    def __init__(
        self,
        repo_root: Path,
        manifest: dict[str, Any],
        *,
        mean: Sequence[float],
        standard_deviation: Sequence[float],
    ) -> None:
        self.repo_root = Path(repo_root)
        self.rows = list(manifest["rows"])
        assert_train_only_provenance(self.rows)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(standard_deviation, dtype=torch.float32).view(3, 1, 1)
        if bool((self.std <= 0).any()):
            raise ValueError("Detector standard deviations must be positive")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(self.repo_root / row["image_path"]) as source:
            image = np.asarray(source.convert("RGB"))
        with Image.open(self.repo_root / row["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) > 0
        with Image.open(self.repo_root / row["valid_region_path"]) as source:
            valid = np.asarray(source.convert("L")) > 0
        if image.shape[:2] != mask.shape or mask.shape != valid.shape:
            raise RuntimeError("Synthetic detector fields are not aligned")
        if bool(np.any(mask & ~valid)):
            raise RuntimeError("Synthetic defect entered an invalid region")
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255
        return {
            "image": (image_tensor - self.mean) / self.std,
            "mask": torch.from_numpy(mask.copy()).unsqueeze(0).float(),
            "valid_region": torch.from_numpy(valid.copy()).unsqueeze(0).float(),
            "has_defect": torch.tensor(bool(mask.any())),
            "sample_id": row["sample_id"],
        }


class ScheduledMixtureDataset(Dataset):
    """Apply one shared stateless augmentation policy to an explicit source schedule."""

    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_dataset: Dataset | None,
        schedule: Sequence[ScheduleEntry],
        transform: SynchronizedRandomFlips,
    ) -> None:
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.schedule = list(schedule)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.schedule[index]
        if entry.source == "synthetic":
            if self.synthetic_dataset is None:
                raise RuntimeError("Schedule requested synthetic data in a real-only arm")
            sample = dict(self.synthetic_dataset[entry.source_index])
        else:
            sample = dict(self.real_dataset[entry.source_index])
        image, mask, valid = self.transform(
            sample["image"], sample["mask"], sample["valid_region"],
            sample_id=str(sample["sample_id"]), epoch=entry.augmentation_epoch,
        )
        # Real samples carry native-geometry bookkeeping that materialized
        # synthetic samples do not need. Return one explicit shared contract so a
        # mixed batch is always collatable regardless of its first source.
        return {
            "image": image,
            "mask": mask,
            "valid_region": valid,
            "has_defect": sample["has_defect"],
            "sample_id": sample["sample_id"],
            "schedule_source": entry.source,
            "optimizer_step": entry.optimizer_step,
        }


def stratified_validation_metrics(
    per_image: Sequence[dict[str, Any]],
    sample_ids: Sequence[str],
    strata: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    if len(per_image) != len(sample_ids):
        raise ValueError("Per-image metrics and sample IDs must align")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, sample_id in zip(per_image, sample_ids):
        group = strata.get(sample_id)
        if group is not None:
            groups[group].append(row)
    output: dict[str, dict[str, float | int]] = {}
    for group, rows in sorted(groups.items()):
        tp = sum(int(row["true_positive_pixels"]) for row in rows)
        fp = sum(int(row["false_positive_pixels"]) for row in rows)
        fn = sum(int(row["false_negative_pixels"]) for row in rows)
        normal = [row for row in rows if not bool(row["has_defect"])]
        denominator = 2 * tp + fp + fn
        output[group] = {
            "images": len(rows),
            "dice": 2 * tp / denominator if denominator else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0),
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "normal_image_false_positive_rate": (
                sum(int(row["predicted_pixels"]) > 0 for row in normal) / len(normal)
                if normal else 0.0
            ),
        }
    return output


def meaningful_winner(
    candidate: dict[str, Any], control: dict[str, Any], *, rules: dict[str, float]
) -> tuple[bool, dict[str, float | bool]]:
    dice_gain = float(candidate["overall"]["global_dice"]) - float(
        control["overall"]["global_dice"]
    )
    iou_gain = float(candidate["overall"]["global_iou"]) - float(
        control["overall"]["global_iou"]
    )
    fpr_delta = float(candidate["overall"]["normal_image_false_positive_rate"]) - float(
        control["overall"]["normal_image_false_positive_rate"]
    )
    precision_delta = float(candidate["overall"]["pixel_precision"]) - float(
        control["overall"]["pixel_precision"]
    )
    recall_delta = float(candidate["overall"]["pixel_recall"]) - float(
        control["overall"]["pixel_recall"]
    )
    passes = (
        dice_gain >= rules["minimum_global_dice_gain"]
        and iou_gain >= rules["minimum_global_iou_gain"]
        and fpr_delta <= rules["maximum_normal_fpr_regression"]
        and precision_delta >= -rules["maximum_precision_regression"]
        and recall_delta >= -rules["maximum_recall_regression"]
    )
    return passes, {
        "passes": passes,
        "global_dice_gain": dice_gain,
        "global_iou_gain": iou_gain,
        "normal_fpr_delta": fpr_delta,
        "pixel_precision_delta": precision_delta,
        "pixel_recall_delta": recall_delta,
    }


def confirmation_decision(
    comparisons: Sequence[dict[str, float | bool]], *, rules: dict[str, float]
) -> tuple[bool, dict[str, Any]]:
    """Apply the pilot's multimetric thresholds to three-seed mean deltas."""
    if len(comparisons) != 3:
        raise ValueError("G2.2 confirmation requires exactly three seeds")
    names = (
        "global_dice_gain",
        "global_iou_gain",
        "normal_fpr_delta",
        "pixel_precision_delta",
        "pixel_recall_delta",
    )
    means = {
        name: float(np.mean([float(row[name]) for row in comparisons])) for name in names
    }
    positive_dice_seeds = sum(float(row["global_dice_gain"]) > 0 for row in comparisons)
    confirmed = (
        means["global_dice_gain"] >= rules["minimum_global_dice_gain"]
        and means["global_iou_gain"] >= rules["minimum_global_iou_gain"]
        and means["normal_fpr_delta"] <= rules["maximum_normal_fpr_regression"]
        and means["pixel_precision_delta"] >= -rules["maximum_precision_regression"]
        and means["pixel_recall_delta"] >= -rules["maximum_recall_regression"]
        and positive_dice_seeds >= 2
    )
    return confirmed, {
        "confirmed": confirmed,
        "mean_deltas": means,
        "positive_dice_seeds": positive_dice_seeds,
        "required_seed_count": 3,
    }
