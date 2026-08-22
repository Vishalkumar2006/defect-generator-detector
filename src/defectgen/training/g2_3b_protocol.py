"""G2.3B mature-budget, prevalence-controlled detector-utility protocol.

G2.3B asks one causal question: does frozen GAN checkpoint 1,500 provide detector
utility beyond the effect of raising defective-sample prevalence, when detectors
are trained to a mature budget under a common training protocol?

Three arms share one scheduling framework and differ only in their precommitted
sample-source composition:

    A standard_real            50.0% normal-real / 50.0% defective-real
    B prevalence_matched_real  37.5% normal-real / 62.5% defective-real
    C gan_1500                 37.5% normal-real / 37.5% defective-real
                               / 25.0% frozen checkpoint-1500 synthetic-defective

C versus B is the primary, gated comparison. A versus B is secondary evidence
about defect prevalence alone.

This module contains protocol primitives only. It never trains, never updates the
GAN, never regenerates synthetic data, and has no code path that can construct the
official KSDD2 test split.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from defectgen.data.augmentation import SynchronizedRandomFlips
from defectgen.training.g2_3_diagnostic import OfficialTestAccessError


G2_3B_VERSION = "g2_3b_mature_budget_prevalence_controlled_utility_v1"
G2_2_TERMINAL_DECISION = "stop_not_confirmed"

ARM_STANDARD_REAL = "standard_real"
ARM_PREVALENCE_MATCHED_REAL = "prevalence_matched_real"
ARM_GAN_1500 = "gan_1500"
ARMS: tuple[str, ...] = (ARM_STANDARD_REAL, ARM_PREVALENCE_MATCHED_REAL, ARM_GAN_1500)

PRIMARY_CANDIDATE = ARM_GAN_1500
PRIMARY_CONTROL = ARM_PREVALENCE_MATCHED_REAL
SECONDARY_CANDIDATE = ARM_STANDARD_REAL
SECONDARY_CONTROL = ARM_PREVALENCE_MATCHED_REAL

G2_3B_SEEDS: tuple[int, ...] = (45, 46, 47)

SOURCE_NORMAL_REAL = "normal_real"
SOURCE_DEFECTIVE_REAL = "defective_real"
SOURCE_SYNTHETIC = "synthetic"
SOURCE_CATEGORIES: tuple[str, ...] = (
    SOURCE_NORMAL_REAL,
    SOURCE_DEFECTIVE_REAL,
    SOURCE_SYNTHETIC,
)

# G2.3B trains on development train and evaluates on development validation only.
# The "test" development split is the untouched official KSDD2 test split and has
# no reachable code path anywhere in this phase.
TRAINING_SPLIT = "train"
EVALUATION_SPLIT = "validation"
ALLOWED_SPLITS = frozenset({TRAINING_SPLIT, EVALUATION_SPLIT})
FORBIDDEN_SPLIT = "".join(("t", "e", "s", "t"))

# G2.2 evaluated only G2.1 joint steps 1,000 and 1,500 and terminally rejected
# 1,000; step 2,000 was never an authorized utility candidate. G2.3B reuses the
# frozen checkpoint-1,500 materialization and nothing else.
ALLOWED_GAN_VARIANT = "checkpoint_1500"
FORBIDDEN_GAN_VARIANTS: tuple[str, ...] = ("checkpoint_1000", "checkpoint_2000")

# The batch composition patterns are precommitted. Each arm repeats one two-batch
# (eight-slot) unit, which is the smallest unit in which 50.0%, 37.5%, 62.5%, and
# 25.0% are all exactly representable at batch size four.
BATCH_PATTERNS: dict[str, tuple[tuple[str, ...], ...]] = {
    ARM_STANDARD_REAL: (
        (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL),
        (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL),
    ),
    ARM_PREVALENCE_MATCHED_REAL: (
        (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL),
        (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_DEFECTIVE_REAL),
    ),
    ARM_GAN_1500: (
        (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_NORMAL_REAL, SOURCE_SYNTHETIC),
        (SOURCE_NORMAL_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_DEFECTIVE_REAL, SOURCE_SYNTHETIC),
    ),
}

TARGET_FRACTIONS: dict[str, dict[str, float]] = {
    ARM_STANDARD_REAL: {
        SOURCE_NORMAL_REAL: 0.5,
        SOURCE_DEFECTIVE_REAL: 0.5,
        SOURCE_SYNTHETIC: 0.0,
    },
    ARM_PREVALENCE_MATCHED_REAL: {
        SOURCE_NORMAL_REAL: 0.375,
        SOURCE_DEFECTIVE_REAL: 0.625,
        SOURCE_SYNTHETIC: 0.0,
    },
    ARM_GAN_1500: {
        SOURCE_NORMAL_REAL: 0.375,
        SOURCE_DEFECTIVE_REAL: 0.375,
        SOURCE_SYNTHETIC: 0.25,
    },
}


# --------------------------------------------------------------------------- #
# Serialization helpers (mirrors the G2.2/G2.3A atomic-write convention)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #


def assert_permitted_split(split: str) -> str:
    """Permit only development train and validation; refuse the official test."""
    if split == FORBIDDEN_SPLIT:
        raise OfficialTestAccessError(
            "G2.3B may not construct, count, inspect, or evaluate the official "
            "KSDD2 test split. A future official-test action requires a separate "
            "authorization after a precommitted G2.3B confirmation PASS."
        )
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"G2.3B may only use {sorted(ALLOWED_SPLITS)}, not {split!r}")
    return split


def assert_evaluation_split(split: str) -> str:
    """Threshold selection and reporting may read development validation only."""
    assert_permitted_split(split)
    if split != EVALUATION_SPLIT:
        raise ValueError(
            f"G2.3B threshold selection and evaluation are validation-only, not {split!r}"
        )
    return split


def assert_allowed_gan_variant(variant: str) -> str:
    """Only the frozen checkpoint-1,500 materialization may be used."""
    if variant in FORBIDDEN_GAN_VARIANTS:
        raise RuntimeError(
            f"G2.3B may not use GAN {variant}; G2.2 terminally rejected checkpoint 1,000 "
            "and never authorized checkpoint 2,000 as a utility candidate."
        )
    if variant != ALLOWED_GAN_VARIANT:
        raise ValueError(f"Unknown GAN synthetic variant for G2.3B: {variant!r}")
    return variant


def assert_train_only_provenance(rows: Iterable[Mapping[str, Any]]) -> int:
    """Every synthetic row must declare development-training provenance."""
    count = 0
    for index, row in enumerate(rows):
        for field in ("official_split", "development_split"):
            if row.get(field) != TRAINING_SPLIT:
                raise RuntimeError(
                    f"Synthetic row {index} has forbidden {field}={row.get(field)!r}"
                )
        source = row.get("source_provenance", {})
        for identity in ("template", "background"):
            record = source.get(identity, {})
            if (
                record.get("official_split") != TRAINING_SPLIT
                or record.get("development_split") != TRAINING_SPLIT
            ):
                raise RuntimeError(
                    f"Synthetic row {index} has a non-training {identity} source"
                )
        count += 1
    return count


def verify_frozen_synthetic_identity(
    repo_root: Path,
    settings: Mapping[str, Any],
    *,
    verify_row_file_hashes: bool | None = None,
) -> dict[str, Any]:
    """Re-verify the already-materialized checkpoint-1,500 synthetic dataset.

    Nothing is regenerated. The frozen GAN checkpoint file, the synthetic manifest
    content hash, the pairing report content hash, per-row file hashes, and
    train-only provenance are all re-checked against the recorded G2.2 values.
    """
    repo_root = Path(repo_root)
    variant = assert_allowed_gan_variant(str(settings["variant"]))
    if bool(settings.get("regenerate", False)):
        raise RuntimeError("G2.3B must reuse the frozen G2.2 synthetic dataset, not regenerate it")

    checkpoint_path = repo_root / settings["frozen_gan_checkpoint_path"]
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != settings["frozen_gan_checkpoint_sha256"]:
        raise RuntimeError(
            f"Frozen GAN checkpoint hash changed: {checkpoint_hash} != "
            f"{settings['frozen_gan_checkpoint_sha256']}"
        )

    manifest = json.loads((repo_root / settings["manifest_path"]).read_text(encoding="utf-8"))
    recorded_hash = manifest.pop("content_sha256")
    recomputed_hash = canonical_sha256(manifest)
    manifest["content_sha256"] = recorded_hash
    if recomputed_hash != recorded_hash:
        raise RuntimeError("Synthetic manifest content hash does not match its own content")
    if recorded_hash != settings["frozen_manifest_content_sha256"]:
        raise RuntimeError("Synthetic manifest identity differs from the frozen G2.2 value")
    if manifest["variant"] != variant:
        raise RuntimeError("Synthetic manifest declares a different GAN variant")
    if manifest.get("official_test_source_count") or manifest.get(
        "detector_validation_source_count"
    ):
        raise RuntimeError("Forbidden source rows are declared in the synthetic manifest")

    pairing = json.loads(
        (repo_root / settings["pairing_report_path"]).read_text(encoding="utf-8")
    )
    pairing_recorded = pairing.pop("content_sha256")
    pairing["content_sha256"] = pairing_recorded
    if pairing_recorded != settings["frozen_pairing_report_content_sha256"]:
        raise RuntimeError("Pairing report identity differs from the frozen G2.2 value")
    if pairing["checkpoint_hashes_after"][variant] != checkpoint_hash:
        raise RuntimeError("Pairing report GAN checkpoint hash disagrees with the file on disk")

    rows = manifest["rows"]
    expected = int(settings["expected_sample_count"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} synthetic rows, found {len(rows)}")
    assert_train_only_provenance(rows)
    for row in rows:
        if int(row["checkpoint_step"]) != int(settings["frozen_gan_checkpoint_step"]):
            raise RuntimeError("A synthetic row was rendered by a different GAN step")
        if row["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("A synthetic row records a different GAN checkpoint hash")

    if verify_row_file_hashes is None:
        verify_row_file_hashes = bool(settings.get("verify_row_file_hashes", True))
    verified_files = 0
    defective_rows = 0
    if verify_row_file_hashes:
        for row in rows:
            for path_field, hash_field in (
                ("image_path", "image_sha256"),
                ("mask_path", "mask_sha256"),
                ("valid_region_path", "valid_region_sha256"),
            ):
                actual = file_sha256(repo_root / row[path_field])
                if actual != row[hash_field]:
                    raise RuntimeError(
                        f"{row['sample_id']}: {path_field} content changed since G2.2"
                    )
                verified_files += 1
    if bool(settings.get("require_every_sample_defective", True)):
        for row in rows:
            with Image.open(repo_root / row["mask_path"]) as source:
                mask = np.asarray(source.convert("L")) > 0
            with Image.open(repo_root / row["valid_region_path"]) as source:
                valid = np.asarray(source.convert("L")) > 0
            if mask.shape != valid.shape:
                raise RuntimeError(f"{row['sample_id']}: mask and valid-region shapes disagree")
            if bool(np.any(mask & ~valid)):
                raise RuntimeError(f"{row['sample_id']}: defect support left the valid region")
            if not bool(np.any(mask & valid)):
                raise RuntimeError(f"{row['sample_id']}: synthetic sample has no defect pixel")
            defective_rows += 1

    return {
        "variant": variant,
        "row_count": len(rows),
        "frozen_gan_checkpoint_sha256": checkpoint_hash,
        "manifest_content_sha256": recorded_hash,
        "pairing_report_content_sha256": pairing_recorded,
        "row_files_verified": verified_files,
        "rows_verified_defective": defective_rows,
        "train_only_provenance": True,
        "regenerated": False,
        "official_test_source_count": int(manifest.get("official_test_source_count", 0)),
        "detector_validation_source_count": int(
            manifest.get("detector_validation_source_count", 0)
        ),
    }


# --------------------------------------------------------------------------- #
# Budget and composition arithmetic
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BudgetPlan:
    optimizer_updates_per_epoch: int
    batch_size: int
    maximum_epochs: int

    @property
    def slots_per_epoch(self) -> int:
        return self.optimizer_updates_per_epoch * self.batch_size

    @property
    def total_optimizer_updates(self) -> int:
        return self.optimizer_updates_per_epoch * self.maximum_epochs

    @property
    def total_slots(self) -> int:
        return self.slots_per_epoch * self.maximum_epochs


def budget_plan(training: Mapping[str, Any]) -> BudgetPlan:
    plan = BudgetPlan(
        optimizer_updates_per_epoch=int(training["optimizer_updates_per_epoch"]),
        batch_size=int(training["batch_size"]),
        maximum_epochs=int(training["maximum_epochs"]),
    )
    if plan.optimizer_updates_per_epoch <= 0 or plan.batch_size <= 0 or plan.maximum_epochs <= 0:
        raise ValueError("Budget components must be positive")
    if plan.slots_per_epoch != int(training["sample_slots_per_epoch"]):
        raise ValueError("sample_slots_per_epoch must equal updates_per_epoch * batch_size")
    if plan.total_optimizer_updates != int(training["total_optimizer_updates"]):
        raise ValueError("total_optimizer_updates must equal updates_per_epoch * maximum_epochs")
    return plan


def batch_pattern(arm: str) -> tuple[tuple[str, ...], ...]:
    if arm not in BATCH_PATTERNS:
        raise ValueError(f"Unknown G2.3B arm: {arm!r}")
    return BATCH_PATTERNS[arm]


def validate_batch_patterns(plan: BudgetPlan) -> None:
    """Every arm's repeating unit must tile the epoch and hit its exact fractions."""
    for arm in ARMS:
        pattern = batch_pattern(arm)
        if any(len(batch) != plan.batch_size for batch in pattern):
            raise ValueError(f"{arm}: batch pattern width does not match the batch size")
        if plan.optimizer_updates_per_epoch % len(pattern):
            raise ValueError(f"{arm}: the repeating unit does not tile the epoch exactly")
        unit_slots = len(pattern) * plan.batch_size
        counts = Counter(token for batch in pattern for token in batch)
        for category in SOURCE_CATEGORIES:
            observed = counts.get(category, 0) / unit_slots
            expected = TARGET_FRACTIONS[arm][category]
            if abs(observed - expected) > 1e-12:
                raise ValueError(
                    f"{arm}: {category} fraction {observed} does not equal precommitted {expected}"
                )


def arm_slot_counts(arm: str, plan: BudgetPlan) -> dict[str, int]:
    """Exact per-epoch slot counts for one arm, derived from its batch pattern."""
    pattern = batch_pattern(arm)
    repeats = plan.optimizer_updates_per_epoch // len(pattern)
    counts = Counter(token for batch in pattern for token in batch)
    return {category: counts.get(category, 0) * repeats for category in SOURCE_CATEGORIES}


def stream_lengths(plan: BudgetPlan) -> dict[str, int]:
    """Per-epoch draw-stream length for each class, sized for the greediest arm.

    Every arm consumes a prefix of the same stream, so the k-th draw of a class in
    an epoch is the same identity in every arm that reaches it.
    """
    lengths = {category: 0 for category in SOURCE_CATEGORIES}
    for arm in ARMS:
        counts = arm_slot_counts(arm, plan)
        for category in SOURCE_CATEGORIES:
            lengths[category] = max(lengths[category], counts[category])
    return lengths


def effective_class_balance(counts: Mapping[str, int]) -> dict[str, float]:
    """Effective defective/normal exposure, counting synthetic samples as defective."""
    normal = int(counts[SOURCE_NORMAL_REAL])
    defective_real = int(counts[SOURCE_DEFECTIVE_REAL])
    synthetic = int(counts[SOURCE_SYNTHETIC])
    total = normal + defective_real + synthetic
    if total <= 0:
        raise ValueError("Composition counts must be positive")
    effective_defective = defective_real + synthetic
    return {
        "total_sample_slots": total,
        "normal_real_samples": normal,
        "defective_real_samples": defective_real,
        "synthetic_samples": synthetic,
        "total_effective_defective_samples": effective_defective,
        "total_effective_normal_samples": normal,
        "normal_real_fraction": normal / total,
        "defective_real_fraction": defective_real / total,
        "synthetic_fraction": synthetic / total,
        "effective_defective_fraction": effective_defective / total,
        "effective_normal_fraction": normal / total,
    }


# --------------------------------------------------------------------------- #
# Deterministic class draw streams and schedules
# --------------------------------------------------------------------------- #


def deterministic_class_stream(
    pool_size: int, count: int, *, seed: int, epoch: int, stream: str
) -> list[int]:
    """Uniform-with-replacement class draws keyed only by seed, epoch, and class.

    The key deliberately excludes the arm, so all three arms share one stream per
    class and per epoch. Replacement behaviour matches the accepted baseline's
    WeightedRandomSampler, whose weights are uniform inside each class.
    """
    if pool_size <= 0:
        raise ValueError("A class pool must be non-empty")
    if count < 0:
        raise ValueError("count must be non-negative")
    material = f"{G2_3B_VERSION}:{seed}:{epoch}:{stream}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = random.Random(local_seed)
    return [generator.randrange(pool_size) for _ in range(count)]


@dataclass(frozen=True)
class ScheduleEntry:
    position: int
    epoch: int
    optimizer_step: int
    batch_position: int
    source: str
    pool_position: int
    source_index: int
    augmentation_epoch: int


def build_arm_schedule(
    arm: str,
    *,
    seed: int,
    plan: BudgetPlan,
    normal_pool: Sequence[int],
    defective_pool: Sequence[int],
    synthetic_pool_size: int,
) -> list[ScheduleEntry]:
    """Build the full deterministic schedule for one arm and seed.

    Class draw streams are arm-independent; the arm only decides how many of each
    stream it consumes and where those slots sit inside a batch.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown G2.3B arm: {arm!r}")
    validate_batch_patterns(plan)
    pattern = batch_pattern(arm)
    lengths = stream_lengths(plan)
    if TARGET_FRACTIONS[arm][SOURCE_SYNTHETIC] > 0 and synthetic_pool_size <= 0:
        raise ValueError("The GAN arm requires a non-empty synthetic pool")

    pools: dict[str, Sequence[int]] = {
        SOURCE_NORMAL_REAL: list(normal_pool),
        SOURCE_DEFECTIVE_REAL: list(defective_pool),
        SOURCE_SYNTHETIC: list(range(max(synthetic_pool_size, 0))),
    }
    entries: list[ScheduleEntry] = []
    for epoch in range(1, plan.maximum_epochs + 1):
        streams = {
            category: deterministic_class_stream(
                max(len(pools[category]), 1),
                lengths[category],
                seed=seed,
                epoch=epoch,
                stream=category,
            )
            for category in SOURCE_CATEGORIES
        }
        cursors = {category: 0 for category in SOURCE_CATEGORIES}
        for batch_index in range(plan.optimizer_updates_per_epoch):
            batch = pattern[batch_index % len(pattern)]
            for batch_position, category in enumerate(batch):
                cursor = cursors[category]
                if cursor >= len(streams[category]):
                    raise RuntimeError(f"{arm}: exhausted the {category} draw stream")
                pool_position = streams[category][cursor]
                cursors[category] = cursor + 1
                entries.append(
                    ScheduleEntry(
                        position=len(entries),
                        epoch=epoch,
                        optimizer_step=(epoch - 1) * plan.optimizer_updates_per_epoch
                        + batch_index
                        + 1,
                        batch_position=batch_position,
                        source=category,
                        pool_position=pool_position,
                        source_index=int(pools[category][pool_position]),
                        augmentation_epoch=epoch,
                    )
                )
    if len(entries) != plan.total_slots:
        raise RuntimeError(f"{arm}: schedule length {len(entries)} != {plan.total_slots}")
    return entries


def schedule_payload(entries: Sequence[ScheduleEntry]) -> list[dict[str, Any]]:
    return [entry.__dict__ for entry in entries]


def schedule_composition(
    entries: Sequence[Mapping[str, Any]], real_labels: Sequence[bool]
) -> dict[str, Any]:
    """Recount an existing schedule from the manifest labels themselves."""
    counts = {category: 0 for category in SOURCE_CATEGORIES}
    identities: dict[str, Counter[int]] = {
        category: Counter() for category in SOURCE_CATEGORIES
    }
    for position, entry in enumerate(entries):
        category = entry["source"]
        if category not in counts:
            raise ValueError(f"Unknown schedule source {category!r} at row {position}")
        index = int(entry["source_index"])
        if category != SOURCE_SYNTHETIC:
            if index < 0 or index >= len(real_labels):
                raise IndexError(f"Schedule row {position} references real index {index}")
            label = bool(real_labels[index])
            if label != (category == SOURCE_DEFECTIVE_REAL):
                raise RuntimeError(
                    f"Schedule row {position} claims {category} but the manifest label is "
                    f"{'defective' if label else 'normal'}"
                )
        counts[category] += 1
        identities[category][index] += 1
    balance = effective_class_balance(counts)
    balance["distinct_identities"] = {
        category: len(identities[category]) for category in SOURCE_CATEGORIES
    }
    return balance


def per_epoch_composition(
    entries: Sequence[Mapping[str, Any]], plan: BudgetPlan
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in range(1, plan.maximum_epochs + 1):
        counts = {category: 0 for category in SOURCE_CATEGORIES}
        updates = set()
        for entry in entries:
            if int(entry["epoch"]) != epoch:
                continue
            counts[entry["source"]] += 1
            updates.add(int(entry["optimizer_step"]))
        rows.append({"epoch": epoch, "optimizer_updates": len(updates), **counts})
    return rows


def assert_equal_budgets(schedules: Mapping[str, Sequence[Any]], plan: BudgetPlan) -> None:
    """Every arm must carry exactly the same number of slots and updates."""
    sizes = {name: len(entries) for name, entries in schedules.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"Arms do not share one optimizer-update budget: {sizes}")
    only = next(iter(sizes.values()))
    if only != plan.total_slots:
        raise RuntimeError(f"Schedules hold {only} slots, expected {plan.total_slots}")


def shared_class_stream_prefixes(
    schedules: Mapping[str, Sequence[Mapping[str, Any]]], *, category: str, epoch: int
) -> bool:
    """True when every arm's draws of one class in one epoch share a common prefix."""
    sequences = []
    for entries in schedules.values():
        sequences.append(
            [
                int(entry["pool_position"])
                for entry in entries
                if entry["source"] == category and int(entry["epoch"]) == epoch
            ]
        )
    sequences = [sequence for sequence in sequences if sequence]
    if len(sequences) < 2:
        return True
    shortest = min(len(sequence) for sequence in sequences)
    reference = sequences[0][:shortest]
    return all(sequence[:shortest] == reference for sequence in sequences)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class FrozenSyntheticDataset(Dataset):
    """Read the frozen, already-materialized checkpoint-1,500 synthetic samples."""

    def __init__(
        self,
        repo_root: Path,
        manifest: Mapping[str, Any],
        *,
        mean: Sequence[float],
        standard_deviation: Sequence[float],
    ) -> None:
        self.repo_root = Path(repo_root)
        self.rows = list(manifest["rows"])
        assert_allowed_gan_variant(str(manifest["variant"]))
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
        if not bool(np.any(mask & valid)):
            raise RuntimeError("A synthetic sample carries no defect pixel")
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255
        return {
            "image": (tensor - self.mean) / self.std,
            "mask": torch.from_numpy(mask.copy()).unsqueeze(0).float(),
            "valid_region": torch.from_numpy(valid.copy()).unsqueeze(0).float(),
            "has_defect": torch.tensor(True),
            "sample_id": row["sample_id"],
        }


class ScheduledCompositionDataset(Dataset):
    """Serve one arm's precommitted schedule through one shared augmentation policy."""

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
        if entry.source == SOURCE_SYNTHETIC:
            if self.synthetic_dataset is None:
                raise RuntimeError("Schedule requested synthetic data in a real-only arm")
            sample = dict(self.synthetic_dataset[entry.source_index])
        else:
            sample = dict(self.real_dataset[entry.source_index])
            expected = entry.source == SOURCE_DEFECTIVE_REAL
            if bool(sample["has_defect"]) != expected:
                raise RuntimeError(
                    f"Schedule slot {index} expected {entry.source} but the sample disagrees"
                )
        image, mask, valid = self.transform(
            sample["image"],
            sample["mask"],
            sample["valid_region"],
            sample_id=str(sample["sample_id"]),
            epoch=entry.augmentation_epoch,
        )
        # Real samples carry native-geometry bookkeeping that materialized
        # synthetic samples do not need; one explicit shared contract keeps a
        # mixed batch collatable regardless of its first source.
        return {
            "image": image,
            "mask": mask,
            "valid_region": valid,
            "has_defect": sample["has_defect"],
            "sample_id": sample["sample_id"],
            "schedule_source": entry.source,
            "epoch": entry.epoch,
            "optimizer_step": entry.optimizer_step,
        }


# --------------------------------------------------------------------------- #
# Precommitted threshold selection
# --------------------------------------------------------------------------- #


def threshold_grid(settings: Mapping[str, Any]) -> list[float]:
    """Deterministic fixed threshold grid built in integer hundredths."""
    grid_settings = settings["grid"]
    start = int(round(float(grid_settings["minimum"]) * 100))
    stop = int(round(float(grid_settings["maximum"]) * 100))
    step = int(round(float(grid_settings["increment"]) * 100))
    if step <= 0 or start <= 0 or stop >= 100 or start > stop:
        raise ValueError("The threshold grid must lie strictly inside (0, 1) and ascend")
    grid = [value / 100 for value in range(start, stop + step, step) if value <= stop]
    expected = int(grid_settings["point_count"])
    if len(grid) != expected:
        raise ValueError(f"Threshold grid has {len(grid)} points, expected {expected}")
    return grid


def selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Precommitted, fully deterministic threshold-selection ordering.

    Identical for every arm and every seed: maximum global Dice, then maximum mean
    defective-image Dice, then maximum pixel precision, then the smallest
    threshold. Grid thresholds are unique, so the ordering is a total order and no
    tie can survive to an implementation-defined choice.
    """
    return (
        float(row["global_dice"]),
        float(row["mean_defective_image_dice"]),
        float(row["pixel_precision"]),
        -float(row["threshold"]),
    )


def select_operating_threshold(
    sweep_rows: Sequence[Mapping[str, Any]], grid: Sequence[float]
) -> dict[str, Any]:
    """Apply the precommitted rule to one arm's validation-only sweep."""
    if not sweep_rows:
        raise ValueError("The validation sweep produced no rows")
    observed = [float(row["threshold"]) for row in sweep_rows]
    if observed != [float(value) for value in grid]:
        raise RuntimeError("The sweep did not use the precommitted threshold grid")
    if len(set(observed)) != len(observed):
        raise RuntimeError("The precommitted threshold grid must have unique thresholds")
    best = max(sweep_rows, key=selection_key)
    return {
        "selected_threshold": float(best["threshold"]),
        "objective": "maximum_validation_global_dice",
        "tie_breaking": [
            "maximum global_dice",
            "then maximum mean_defective_image_dice",
            "then maximum pixel_precision",
            "then smallest threshold",
        ],
        "grid_point_count": len(grid),
        "data_source": EVALUATION_SPLIT,
        "selected_row": dict(best),
    }


# --------------------------------------------------------------------------- #
# Precommitted confirmation gate
# --------------------------------------------------------------------------- #


GATE_METRIC_FIELDS: tuple[str, ...] = (
    "global_dice",
    "global_iou",
    "pixel_precision",
    "pixel_recall",
    "normal_image_false_positive_rate",
    "pixel_pr_auc",
)


def arm_comparison(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, float]:
    """Per-seed deltas at the precommitted selected operating threshold."""
    for name in GATE_METRIC_FIELDS:
        if name not in candidate or name not in control:
            raise KeyError(f"Comparison requires metric {name!r} in both arms")
    return {
        "global_dice_gain": float(candidate["global_dice"]) - float(control["global_dice"]),
        "global_iou_gain": float(candidate["global_iou"]) - float(control["global_iou"]),
        "pixel_precision_delta": float(candidate["pixel_precision"])
        - float(control["pixel_precision"]),
        "pixel_recall_delta": float(candidate["pixel_recall"]) - float(control["pixel_recall"]),
        "normal_fpr_delta": float(candidate["normal_image_false_positive_rate"])
        - float(control["normal_image_false_positive_rate"]),
        "pixel_pr_auc_gain": float(candidate["pixel_pr_auc"]) - float(control["pixel_pr_auc"]),
    }


def confirmation_decision(
    comparisons: Sequence[Mapping[str, float]], *, rules: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Apply the frozen G2.3B gate to the three fresh-seed primary comparisons.

    Every G2.2 criterion is carried over unchanged; the pixel PR-AUC criteria are
    added on top, so the gate can only be harder to pass than G2.2's was.
    """
    required = int(rules["required_seed_count"])
    if len(comparisons) != required:
        raise ValueError(f"G2.3B confirmation requires exactly {required} seeds")
    names = (
        "global_dice_gain",
        "global_iou_gain",
        "pixel_precision_delta",
        "pixel_recall_delta",
        "normal_fpr_delta",
        "pixel_pr_auc_gain",
    )
    means = {
        name: float(np.mean([float(row[name]) for row in comparisons])) for name in names
    }
    positive_dice_seeds = int(sum(float(row["global_dice_gain"]) > 0 for row in comparisons))
    positive_pr_auc_seeds = int(sum(float(row["pixel_pr_auc_gain"]) > 0 for row in comparisons))
    criteria = {
        "mean_global_dice_gain": means["global_dice_gain"]
        >= float(rules["minimum_mean_global_dice_gain"]),
        "mean_global_iou_gain": means["global_iou_gain"]
        >= float(rules["minimum_mean_global_iou_gain"]),
        "mean_normal_fpr_regression": means["normal_fpr_delta"]
        <= float(rules["maximum_mean_normal_fpr_regression"]),
        "mean_precision_regression": means["pixel_precision_delta"]
        >= -float(rules["maximum_mean_precision_regression"]),
        "mean_recall_regression": means["pixel_recall_delta"]
        >= -float(rules["maximum_mean_recall_regression"]),
        "positive_dice_seeds": positive_dice_seeds >= int(rules["minimum_positive_dice_seeds"]),
        "mean_pixel_pr_auc_gain": means["pixel_pr_auc_gain"]
        >= float(rules["minimum_mean_pixel_pr_auc_gain"]),
        "positive_pr_auc_seeds": positive_pr_auc_seeds
        >= int(rules["minimum_positive_pr_auc_seeds"]),
    }
    confirmed = all(criteria.values())
    return confirmed, {
        "confirmed": confirmed,
        "mean_deltas": means,
        "positive_dice_seeds": positive_dice_seeds,
        "positive_pr_auc_seeds": positive_pr_auc_seeds,
        "required_seed_count": required,
        "criteria": criteria,
        "failed_criteria": sorted(name for name, passed in criteria.items() if not passed),
        "decision": "confirmed_gan_1500_utility_beyond_prevalence"
        if confirmed
        else "stop_not_confirmed_g2_3b",
        "official_test_authorized_by_this_decision": False,
    }
