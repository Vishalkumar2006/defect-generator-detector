"""G2.3A post-hoc, validation-only diagnostic primitives.

Nothing in this module trains, fine-tunes, resumes, or re-materializes anything.
It reads already-frozen G2.2 artifacts and already-trained G2.2 detector
checkpoints and produces threshold curves, matched-operating-point comparisons,
convergence/composition audits, and synthetic-mask integrity evidence.

Every quantity produced here is labelled POST_HOC_DIAGNOSTIC. The G2.2 terminal
decision stop_not_confirmed is an input constant, never an output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


G2_3_VERSION = "g2_3a_post_hoc_validation_only_diagnostic_v1"
DIAGNOSTIC_LABEL = "POST_HOC_DIAGNOSTIC"
G2_2_TERMINAL_DECISION = "stop_not_confirmed"

# G2.3A is validation-only by construction. The "test" development split is the
# untouched official KSDD2 test split; it must never be constructed, counted, or
# inspected here.
ALLOWED_DIAGNOSTIC_SPLITS = frozenset({"validation"})
FORBIDDEN_DIAGNOSTIC_SPLITS = frozenset({"test"})

DIAGNOSTIC_CHECKPOINTS: tuple[tuple[int, str], ...] = (
    (42, "real_only"),
    (42, "checkpoint_1500"),
    (43, "real_only"),
    (43, "checkpoint_1500"),
    (44, "real_only"),
    (44, "checkpoint_1500"),
)


class OfficialTestAccessError(RuntimeError):
    """Raised whenever a diagnostic code path is asked for official-test data."""


# --------------------------------------------------------------------------- #
# Serialization helpers (mirrors the G2.2 atomic-write convention)
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
# Validation-only access guards
# --------------------------------------------------------------------------- #


def assert_validation_only_split(split: str) -> str:
    """Permit only the validation split; refuse the official test split explicitly."""
    if split in FORBIDDEN_DIAGNOSTIC_SPLITS:
        raise OfficialTestAccessError(
            "G2.3A is a validation-only diagnostic; the official test split "
            f"({split!r}) may not be constructed, counted, or evaluated."
        )
    if split not in ALLOWED_DIAGNOSTIC_SPLITS:
        raise ValueError(f"G2.3A may only read the validation split, not {split!r}")
    return split


def assert_no_forbidden_provenance(
    source_ids: Iterable[str],
    *,
    training_ids: frozenset[str],
    validation_ids: frozenset[str],
) -> dict[str, Any]:
    """Prove train-only provenance positively, without enumerating test rows.

    Development splits are a partition, so membership in train already excludes
    both validation and official test. The validation intersection is reported as
    an independent redundant check.
    """
    unique = sorted(set(source_ids))
    outside_training = sorted(value for value in unique if value not in training_ids)
    validation_overlap = sorted(value for value in unique if value in validation_ids)
    if outside_training:
        raise RuntimeError(
            f"Synthetic source identities outside development train: {outside_training[:5]}"
        )
    if validation_overlap:
        raise RuntimeError(
            f"Synthetic source identities overlap detector validation: {validation_overlap[:5]}"
        )
    return {
        "unique_source_identities": len(unique),
        "outside_development_train": 0,
        "detector_validation_overlap": 0,
        # Membership in development train is a positive proof of non-test
        # provenance; no official-test row is read, counted, or listed.
        "official_test_rows_read": 0,
        "official_test_overlap_proven_by": "development_train_membership_partition_argument",
    }


# --------------------------------------------------------------------------- #
# Checkpoint identity
# --------------------------------------------------------------------------- #


def checkpoint_identity(
    path: Path | str, payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Record an immutable identity for an already-trained G2.2 checkpoint."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing G2.2 detector checkpoint: {resolved}")
    identity: dict[str, Any] = {
        "path": resolved.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }
    if payload is not None:
        identity.update(
            {
                "experiment_version": payload.get("experiment_version"),
                "variant": payload.get("variant"),
                "seed": payload.get("seed"),
                "optimizer_updates": payload.get("optimizer_updates"),
                "initialization_sha256": payload.get("initialization_sha256"),
                "schedule_sha256": payload.get("schedule_sha256"),
            }
        )
    return identity


# --------------------------------------------------------------------------- #
# Deterministic threshold grids and pixel histograms
# --------------------------------------------------------------------------- #


def build_probability_grid(
    *,
    probability_step: float = 0.0005,
    logit_limit: float = 20.0,
    logit_step: float = 0.02,
) -> np.ndarray:
    """Build a fine, deterministic, strictly increasing probability grid.

    The grid combines a uniform probability lattice (linear resolution around the
    fixed 0.5 comparison threshold) with a uniform logit lattice (tail resolution
    where the sigmoid saturates). 0.5 is always a grid point.
    """
    if probability_step <= 0 or logit_step <= 0 or logit_limit <= 0:
        raise ValueError("Grid steps and the logit limit must be positive")
    uniform = np.arange(probability_step, 1.0, probability_step, dtype=np.float64)
    logit_count = int(round(2 * logit_limit / logit_step)) + 1
    logits = np.linspace(-logit_limit, logit_limit, logit_count, dtype=np.float64)
    tails = 1.0 / (1.0 + np.exp(-logits))
    grid = np.unique(np.concatenate([uniform, tails, np.array([0.5], dtype=np.float64)]))
    grid = grid[(grid > 0.0) & (grid < 1.0)]
    if not np.any(grid == 0.5):
        raise RuntimeError("The fixed 0.5 comparison threshold must be a grid point")
    return grid


def build_logit_grid(*, limit: float = 30.0, step: float = 0.01) -> np.ndarray:
    count = int(round(2 * limit / step)) + 1
    return np.linspace(-limit, limit, count, dtype=np.float64)


@dataclass
class PixelHistogram:
    """Exact grid-point survivor counts plus exact scalar moments."""

    grid: np.ndarray
    counts: np.ndarray = field(init=False)
    total: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def __post_init__(self) -> None:
        if self.grid.ndim != 1 or self.grid.size == 0:
            raise ValueError("grid must be a non-empty one-dimensional array")
        if not np.all(np.diff(self.grid) > 0):
            raise ValueError("grid must be strictly increasing")
        self.counts = np.zeros(self.grid.size + 1, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        positions = np.searchsorted(self.grid, values, side="right")
        self.counts += np.bincount(positions, minlength=self.grid.size + 1)
        self.total += int(values.size)
        self.value_sum += float(values.sum())
        self.value_square_sum += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def survivors(self) -> np.ndarray:
        """Exact count of accumulated values >= grid[k], for every grid point k."""
        suffix = np.cumsum(self.counts[::-1])[::-1]
        return suffix[1:].astype(np.int64)

    def below(self) -> np.ndarray:
        return self.total - self.survivors()

    def quantiles(self, probabilities: Sequence[float]) -> dict[str, float]:
        if self.total == 0:
            return {f"p{value:g}": float("nan") for value in probabilities}
        below = self.below()
        result: dict[str, float] = {}
        for probability in probabilities:
            target = probability / 100.0 * self.total
            index = int(np.searchsorted(below, target, side="left"))
            index = min(index, self.grid.size - 1)
            result[f"p{probability:g}"] = float(self.grid[index])
        return result

    def summary(
        self, quantile_points: Sequence[float] = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
    ) -> dict[str, Any]:
        if self.total == 0:
            return {"count": 0}
        mean = self.value_sum / self.total
        variance = max(0.0, self.value_square_sum / self.total - mean * mean)
        return {
            "count": int(self.total),
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "quantiles": self.quantiles(quantile_points),
            "below_grid_count": int(self.counts[0]),
            "at_or_above_top_grid_point_count": int(self.counts[-1]),
        }


# --------------------------------------------------------------------------- #
# Threshold-dependent metric curves
# --------------------------------------------------------------------------- #


def threshold_curve(
    positive: PixelHistogram,
    negative: PixelHistogram,
    *,
    normal_image_maxima: Sequence[float] = (),
    defective_image_maxima: Sequence[float] = (),
) -> dict[str, np.ndarray]:
    """Return exact grid-point Dice/IoU/precision/recall/normal-FPR curves."""
    if positive.grid.shape != negative.grid.shape or not np.array_equal(
        positive.grid, negative.grid
    ):
        raise ValueError("Positive and negative histograms must share one grid")
    grid = positive.grid
    true_positive = positive.survivors().astype(np.float64)
    false_positive = negative.survivors().astype(np.float64)
    total_positive = float(positive.total)
    false_negative = total_positive - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    predicted = true_positive + false_positive
    empty_precision = 1.0 if total_positive == 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(denominator > 0, 2 * true_positive / np.maximum(denominator, 1.0), 1.0)
        iou = np.where(union > 0, true_positive / np.maximum(union, 1.0), 1.0)
        precision = np.where(
            predicted > 0, true_positive / np.maximum(predicted, 1.0), empty_precision
        )
    recall = (
        true_positive / total_positive if total_positive > 0 else np.ones_like(true_positive)
    )
    curve = {
        "threshold": grid,
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }
    normal = np.asarray(normal_image_maxima, dtype=np.float64)
    if normal.size:
        counts = np.array([(normal >= value).sum() for value in grid], dtype=np.float64)
        curve["normal_false_positive_images"] = counts
        curve["normal_image_false_positive_rate"] = counts / float(normal.size)
    else:
        curve["normal_false_positive_images"] = np.zeros_like(grid)
        curve["normal_image_false_positive_rate"] = np.zeros_like(grid)
    defective = np.asarray(defective_image_maxima, dtype=np.float64)
    if defective.size:
        curve["defective_images_zero_detected_pixels"] = np.array(
            [(defective < value).sum() for value in grid], dtype=np.float64
        )
    else:
        curve["defective_images_zero_detected_pixels"] = np.zeros_like(grid)
    return curve


def pr_auc(recall: np.ndarray, precision: np.ndarray) -> float:
    """Step-rule pixel PR-AUC (average precision) over the diagnostic grid.

    Inputs must be ordered by ascending threshold, so recall is non-increasing.
    The step rule sum((R_k - R_{k+1}) * P_k) is used rather than trapezoidal
    interpolation, which is optimistic across precision jumps. The recall above
    the highest grid point is credited at that point's precision, the ordinary
    average-precision convention; callers report the covered recall span.
    """
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if recall.shape != precision.shape or recall.ndim != 1:
        raise ValueError("recall and precision must be one-dimensional and aligned")
    if recall.size > 1 and np.any(np.diff(recall) > 1e-12):
        raise ValueError("recall must be non-increasing along ascending thresholds")
    next_recall = np.append(recall[1:], 0.0)
    return float(np.sum((recall - next_recall) * precision))


def match_threshold_index(values: np.ndarray, target: float) -> int:
    """Index of the grid point whose metric is closest to target.

    Ties resolve to the largest threshold, the conservative choice when matching a
    recall or a normal-image false-positive-rate operating point.
    """
    difference = np.abs(np.asarray(values, dtype=np.float64) - float(target))
    best = float(difference.min())
    candidates = np.flatnonzero(difference <= best + 1e-15)
    return int(candidates[-1])


def curve_point(curve: Mapping[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(values[index]) for name, values in curve.items()}


# --------------------------------------------------------------------------- #
# Question 3 -- schedule composition
# --------------------------------------------------------------------------- #


def schedule_composition(
    entries: Sequence[Mapping[str, Any]],
    real_labels: Sequence[bool],
    *,
    synthetic_defective: bool = True,
) -> dict[str, Any]:
    """Count exact real/synthetic and normal/defective sample slots in a schedule."""
    normal_real = defective_real = synthetic = 0
    real_index_usage: Counter[int] = Counter()
    for position, entry in enumerate(entries):
        source = entry["source"]
        index = int(entry["source_index"])
        if source == "synthetic":
            synthetic += 1
        elif source == "real":
            if index < 0 or index >= len(real_labels):
                raise IndexError(f"Schedule row {position} references real index {index}")
            real_index_usage[index] += 1
            if bool(real_labels[index]):
                defective_real += 1
            else:
                normal_real += 1
        else:
            raise ValueError(f"Unknown schedule source {source!r} at row {position}")
    total = normal_real + defective_real + synthetic
    if total != len(entries):
        raise RuntimeError("Schedule composition did not account for every slot")
    effective_defective = defective_real + (synthetic if synthetic_defective else 0)
    effective_normal = normal_real + (0 if synthetic_defective else synthetic)
    return {
        "total_sample_slots": total,
        "normal_real_samples": normal_real,
        "defective_real_samples": defective_real,
        "synthetic_samples": synthetic,
        "synthetic_assumed_defective": bool(synthetic_defective),
        "total_effective_defective_samples": effective_defective,
        "total_effective_normal_samples": effective_normal,
        "normal_real_fraction": normal_real / total,
        "defective_real_fraction": defective_real / total,
        "synthetic_fraction": synthetic / total,
        "effective_defective_fraction": effective_defective / total,
        "effective_normal_fraction": effective_normal / total,
        "distinct_real_identities_used": len(real_index_usage),
    }


def expected_schedule_composition(
    *,
    variant: str,
    optimizer_updates: int,
    batch_size: int,
    synthetic_fraction: float,
    synthetic_defective: bool = True,
) -> dict[str, float]:
    """The composition the G2.2 scheduling implementation is designed to produce."""
    synthetic_per_batch = (
        0 if variant == "real_only" else int(round(batch_size * synthetic_fraction))
    )
    common_real_per_batch = batch_size - int(round(batch_size * synthetic_fraction))
    replacement_per_batch = batch_size - common_real_per_batch - synthetic_per_batch
    total = optimizer_updates * batch_size
    # deterministic_balanced_indices alternates class parity along each stream, so
    # every real stream is exactly half normal and half defective when its length
    # is even.
    if (optimizer_updates * common_real_per_batch) % 2 or (
        optimizer_updates * replacement_per_batch
    ) % 2:
        raise ValueError("Balanced real streams must have even length to split exactly")
    real_slots = optimizer_updates * (common_real_per_batch + replacement_per_batch)
    normal_real = defective_real = real_slots // 2
    synthetic = optimizer_updates * synthetic_per_batch
    effective_defective = defective_real + (synthetic if synthetic_defective else 0)
    return {
        "total_sample_slots": total,
        "normal_real_samples": normal_real,
        "defective_real_samples": defective_real,
        "synthetic_samples": synthetic,
        "total_effective_defective_samples": effective_defective,
        "total_effective_normal_samples": total - effective_defective,
        "effective_defective_fraction": effective_defective / total,
    }


def class_prevalence_confound(
    control: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify the control/arm effective-defect-prevalence difference."""
    delta = float(arm["effective_defective_fraction"]) - float(
        control["effective_defective_fraction"]
    )
    confounded = abs(delta) > 1e-12
    return {
        "control_effective_defective_fraction": float(control["effective_defective_fraction"]),
        "arm_effective_defective_fraction": float(arm["effective_defective_fraction"]),
        "effective_defective_fraction_delta": delta,
        "is_class_prevalence_confound": confounded,
        "interpretation": (
            "G2.2 changed synthetic image content and effective defect prevalence "
            "inside the same comparison"
            if confounded
            else "control and arm carry the same effective defect prevalence"
        ),
    }


# --------------------------------------------------------------------------- #
# Question 3 -- synthetic mask integrity
# --------------------------------------------------------------------------- #


def synthetic_mask_record(
    *, sample_id: str, mask: np.ndarray, valid: np.ndarray, image_shape: tuple[int, int]
) -> dict[str, Any]:
    """Hard-check one materialized synthetic mask against its valid region."""
    mask = np.asarray(mask).astype(bool)
    valid = np.asarray(valid).astype(bool)
    if mask.shape != valid.shape:
        raise RuntimeError(f"{sample_id}: mask and valid-region shapes disagree")
    if tuple(image_shape) != mask.shape:
        raise RuntimeError(f"{sample_id}: image and mask shapes disagree")
    outside = int((mask & ~valid).sum())
    return {
        "sample_id": sample_id,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "positive_pixels": int(mask.sum()),
        "positive_valid_defect_pixels": int((mask & valid).sum()),
        "support_outside_valid_region": outside,
        "valid_pixels": int(valid.sum()),
        "has_positive_valid_defect_pixel": bool((mask & valid).any()),
        "support_inside_valid_region": outside == 0,
        "aligned": True,
    }


def summarize_mask_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures_empty = [
        row["sample_id"] for row in records if not row["has_positive_valid_defect_pixel"]
    ]
    failures_outside = [
        row["sample_id"] for row in records if not row["support_inside_valid_region"]
    ]
    positives = [int(row["positive_valid_defect_pixels"]) for row in records]
    return {
        "masks_checked": len(records),
        "all_have_positive_valid_defect_pixel": not failures_empty,
        "all_support_inside_valid_region": not failures_outside,
        "empty_mask_sample_ids": failures_empty,
        "support_outside_valid_sample_ids": failures_outside,
        "minimum_positive_valid_defect_pixels": min(positives) if positives else 0,
        "median_positive_valid_defect_pixels": float(np.median(positives)) if positives else 0.0,
        "maximum_positive_valid_defect_pixels": max(positives) if positives else 0,
        "every_synthetic_sample_is_defective": not failures_empty,
    }


# --------------------------------------------------------------------------- #
# Question 1 -- convergence / control-instability audit helpers
# --------------------------------------------------------------------------- #


def baseline_update_position(
    epoch_rows: Sequence[Mapping[str, Any]], *, updates: int, updates_per_epoch: int
) -> dict[str, Any]:
    """Locate an update count on the historical baseline epoch trajectory."""
    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be positive")
    if not epoch_rows:
        raise ValueError("epoch_rows must be non-empty")
    position = updates / updates_per_epoch
    lower = max(1, min(len(epoch_rows), int(math.floor(position))))
    upper = max(1, min(len(epoch_rows), int(math.ceil(position))))

    def _pick(index: int) -> dict[str, Any]:
        row = epoch_rows[index - 1]
        return {
            "epoch": int(row["epoch"]),
            "cumulative_optimizer_updates": int(row["epoch"]) * updates_per_epoch,
            "learning_rate": float(row["learning_rate"]),
            "train_total_loss": float(row["train_total_loss"]),
            "validation_total_loss": float(row["validation_total_loss"]),
            "validation_global_dice_at_0_5": float(row["validation_global_dice_at_0_5"]),
            "validation_global_iou_at_0_5": float(row["validation_global_iou_at_0_5"]),
            "validation_pixel_precision_at_0_5": float(row["validation_pixel_precision_at_0_5"]),
            "validation_pixel_recall_at_0_5": float(row["validation_pixel_recall_at_0_5"]),
        }

    return {
        "equivalent_baseline_epochs": position,
        "bracketing_epoch_below": _pick(lower),
        "bracketing_epoch_above": _pick(upper),
        "baseline_total_updates": len(epoch_rows) * updates_per_epoch,
        "fraction_of_baseline_budget": updates / (len(epoch_rows) * updates_per_epoch),
    }


def dispersion(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


def load_epoch_metrics(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


CURVE_CSV_COLUMNS = (
    "threshold",
    "dice",
    "iou",
    "precision",
    "recall",
    "normal_image_false_positive_rate",
    "true_positive_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
    "normal_false_positive_images",
    "defective_images_zero_detected_pixels",
)


def write_curve_csv(path: Path | str, curve: Mapping[str, np.ndarray]) -> str:
    """Write a full threshold curve as a deterministic CSV and return its hash."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [np.asarray(curve[name], dtype=np.float64) for name in CURVE_CSV_COLUMNS]
    lines = [",".join(CURVE_CSV_COLUMNS)]
    for index in range(columns[0].size):
        lines.append(",".join(repr(float(column[index])) for column in columns))
    payload = "\n".join(lines) + "\n"
    destination.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
