"""Audit deterministic G1.3 GAN training pairs without materializing a dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.training_pairs import (  # noqa: E402
    GANTrainingPairDataset,
    GANTrainingSample,
    create_internal_gan_split,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)
from defectgen.training.gan_losses import (  # noqa: E402
    load_gan_loss_config,
)


LOGIT_SHAPE = (62, 30)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _sample_hash(sample: GANTrainingSample) -> str:
    digest = hashlib.sha256()
    for field in (
        "composite_image",
        "generator_mask",
        "transformed_defect_alpha",
        "fake_discriminator_mask",
        "real_image",
        "real_discriminator_mask",
        "fake_valid_mask",
        "real_valid_mask",
        "real_valid_coverage",
    ):
        tensor = getattr(sample, field).detach().cpu().contiguous()
        digest.update(field.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    digest.update(
        json.dumps(sample.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _combination(contacts: dict[str, bool]) -> str:
    active = [side for side in ("top", "bottom", "left", "right") if contacts[side]]
    return "+".join(active) if active else "none"


def _dilated_projection(
    mask: torch.Tensor, *, radius: int, output_shape: tuple[int, int]
) -> torch.Tensor:
    """Project an exact square dilation using two equivalent one-dimensional passes."""
    projected = mask.bool().float().unsqueeze(0)
    if radius:
        kernel = 2 * radius + 1
        projected = F.max_pool2d(
            projected, kernel_size=(1, kernel), stride=1, padding=(0, radius)
        )
        projected = F.max_pool2d(
            projected, kernel_size=(kernel, 1), stride=1, padding=(radius, 0)
        )
    return F.adaptive_max_pool2d(projected, output_shape) > 0


def _padding_projection(
    valid_mask: torch.Tensor, *, radius: int, output_shape: tuple[int, int]
) -> torch.Tensor:
    return _dilated_projection(
        ~valid_mask.bool(), radius=radius, output_shape=output_shape
    )


def _as_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .cpu()
        .add(1)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def _overlay(image: torch.Tensor, mask: torch.Tensor, colour: tuple[int, int, int]) -> np.ndarray:
    rgb = _as_rgb(image).astype(np.float32)
    active = mask[0].detach().cpu().numpy() > 0
    rgb[active] = 0.45 * rgb[active] + 0.55 * np.asarray(colour)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _contact_category(sample: GANTrainingSample) -> str | None:
    contacts = sample.metadata["target_contact_sides"]
    active = [side for side, value in contacts.items() if value]
    if not active:
        return "non-border"
    if contacts["left"] and contacts["right"]:
        return "left+right"
    if len(active) >= 2 and any(contacts[side] for side in ("top", "bottom")) and any(
        contacts[side] for side in ("left", "right")
    ):
        return "corner"
    if len(active) == 1:
        return "single-border"
    return None


def _write_contact_sheet(path: Path, representatives: dict[str, GANTrainingSample]) -> None:
    categories = [
        category
        for category in ("non-border", "single-border", "corner", "left+right")
        if category in representatives
    ]
    if not categories:
        raise RuntimeError("No representative training pairs were available")
    figure, axes = plt.subplots(len(categories), 4, figsize=(13, 3.8 * len(categories)))
    if len(categories) == 1:
        axes = np.asarray([axes])
    for row, category in enumerate(categories):
        sample = representatives[category]
        invalid = 1 - sample.fake_valid_mask
        panels = (
            (_as_rgb(sample.real_image), "Transformed real defect"),
            (_as_rgb(sample.composite_image), "F1.4 coarse composite"),
            (
                _overlay(sample.composite_image, sample.generator_mask, (255, 40, 40)),
                "Generator-mask overlay",
            ),
            (
                _overlay(sample.composite_image, invalid, (255, 190, 0)),
                "Native-valid overlay (invalid amber)",
            ),
        )
        for column, (panel, title) in enumerate(panels):
            axes[row, column].imshow(panel)
            if column == 0:
                metadata = sample.metadata
                title = (
                    f"{title}\n{category}: "
                    f"{metadata['template_source_sample_id']} -> "
                    f"{metadata['normal_background_sample_id']}\n"
                    f"source {_combination(metadata['source_contact_sides'])}; "
                    f"target {_combination(metadata['target_contact_sides'])}"
                )
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis("off")
    figure.suptitle("G1.3 deterministic GAN training-pair audit", fontsize=13)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=140, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def build_audit(config_path: Path, contact_sheet_path: Path) -> dict[str, Any]:
    config = load_gan_training_pair_config(config_path)
    metadata = load_training_pair_manifest(REPO_ROOT, config)
    loss_config = load_gan_loss_config(REPO_ROOT / config.loss_config_path)
    if loss_config.canonical_mask_threshold != config.discriminator_mask_threshold:
        raise ValueError("G1.3 and G1.2 discriminator mask thresholds disagree")
    internal = create_internal_gan_split(
        metadata, monitor_fraction=config.monitor_fraction, seed=config.base_seed
    )
    requested = config.audit_sample_count
    train = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        config,
        split="train",
        internal_split=internal,
        length=requested,
    )
    monitor = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        config,
        split="monitor",
        internal_split=internal,
        length=max(1, min(requested, len(internal.monitor_normal_indices))),
    )
    equality_count = 0
    transform_equality_count = 0
    empty_masks = 0
    templates: Counter[str] = Counter()
    backgrounds: Counter[str] = Counter()
    source_sides: Counter[str] = Counter()
    target_combinations: Counter[str] = Counter()
    support_inside_fractions: list[float] = []
    fake_padding_logit_fractions: list[float] = []
    real_padding_logit_fractions: list[float] = []
    padding_asymmetry: list[float] = []
    representatives: dict[str, GANTrainingSample] = {}
    shape_pass = dtype_pass = range_pass = finite_pass = True
    started = perf_counter()
    for index in range(requested):
        if index % 25 == 0:
            print(f"Auditing GAN training pair {index}/{requested}", flush=True)
        sample = train[index]
        equality_count += torch.equal(
            sample.real_discriminator_mask, sample.fake_discriminator_mask
        )
        transform_equality_count += (
            sample.metadata["real_transform"] == sample.metadata["fake_transform"]
        )
        empty_masks += not bool(sample.fake_discriminator_mask.any())
        templates[sample.metadata["template_id"]] += 1
        backgrounds[sample.metadata["normal_background_sample_id"]] += 1
        for side, active in sample.metadata["source_contact_sides"].items():
            if active:
                source_sides[side] += 1
        if not any(sample.metadata["source_contact_sides"].values()):
            source_sides["none"] += 1
        target_combinations[_combination(sample.metadata["target_contact_sides"])] += 1
        category = _contact_category(sample)
        if category is not None and category not in representatives:
            representatives[category] = sample

        images = (sample.composite_image, sample.real_image)
        masks = (
            sample.generator_mask,
            sample.fake_discriminator_mask,
            sample.real_discriminator_mask,
            sample.fake_valid_mask,
            sample.real_valid_mask,
        )
        tensors = images + masks
        shape_pass &= all(image.shape == (3, 512, 256) for image in images) and all(
            mask.shape == (1, 512, 256) for mask in masks
        )
        dtype_pass &= all(tensor.dtype == torch.float32 for tensor in tensors)
        finite_pass &= all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        range_pass &= all(
            bool((tensor >= (-1 if tensor.shape[0] == 3 else 0)).all())
            and bool((tensor <= 1).all())
            for tensor in tensors
        )
        support = sample.generator_mask > 0
        support_inside_fractions.append(
            float((support & sample.fake_valid_mask.bool()).sum() / support.sum())
        )
        active_logits = _dilated_projection(
            sample.fake_discriminator_mask,
            radius=loss_config.localization_radius,
            output_shape=LOGIT_SHAPE,
        )
        fake_padding = _padding_projection(
            sample.fake_valid_mask,
            radius=loss_config.localization_radius,
            output_shape=LOGIT_SHAPE,
        )
        real_padding = _padding_projection(
            sample.real_valid_mask,
            radius=loss_config.localization_radius,
            output_shape=LOGIT_SHAPE,
        )
        denominator = active_logits.sum()
        fake_padding_logit_fractions.append(
            float((active_logits & fake_padding).sum() / denominator)
        )
        real_padding_logit_fractions.append(
            float((active_logits & real_padding).sum() / denominator)
        )
        padding_asymmetry.append(
            abs(float(sample.fake_valid_mask.mean()) - float(sample.real_valid_mask.mean()))
        )
    elapsed = perf_counter() - started
    print(f"Auditing GAN training pair {requested}/{requested}", flush=True)

    repeat_first = train[0]
    repeat_second = train[0]
    repeat_hashes = (_sample_hash(repeat_first), _sample_hash(repeat_second))
    train.set_epoch(1)
    epoch_one_hash = _sample_hash(train[0])
    train.set_epoch(0)
    epoch_zero_hash = _sample_hash(train[0])
    monitor_zero_hash = _sample_hash(monitor[0])
    monitor.set_epoch(99)
    monitor_later_hash = _sample_hash(monitor[0])
    _write_contact_sheet(contact_sheet_path, representatives)

    split_checks = {
        "defect_source_ids_disjoint": not bool(
            internal.train_defect_source_ids & internal.monitor_defect_source_ids
        ),
        "background_ids_disjoint": not bool(
            internal.train_background_ids & internal.monitor_background_ids
        ),
    }
    tensor_checks = {
        "shapes": shape_pass,
        "dtypes": dtype_pass,
        "ranges": range_pass,
        "finiteness": finite_pass,
    }
    deterministic_checks = {
        "repeat_hashes_equal": repeat_hashes[0] == repeat_hashes[1],
        "epoch_change_changes_training_pair": epoch_zero_hash != epoch_one_hash,
        "monitor_hash_stable_across_epochs": monitor_zero_hash == monitor_later_hash,
    }
    invariants = {
        **split_checks,
        **{f"tensor_{name}": value for name, value in tensor_checks.items()},
        **deterministic_checks,
        "canonical_mask_equality_rate_is_one": equality_count == requested,
        "transform_metadata_equality_rate_is_one": transform_equality_count == requested,
        "empty_mask_count_is_zero": empty_masks == 0,
        "all_generator_support_inside_fake_valid": min(support_inside_fractions) == 1.0,
    }
    return {
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "data_bridge_version": config.data_bridge_version,
        "requested_samples": requested,
        "runtime_seconds": elapsed,
        "samples_per_second": requested / elapsed,
        "internal_split": {
            "train_defect_source_count": len(internal.train_defect_source_ids),
            "monitor_defect_source_count": len(internal.monitor_defect_source_ids),
            "train_template_count": len(internal.train_template_indices),
            "monitor_template_count": len(internal.monitor_template_indices),
            "train_background_source_count": len(internal.train_background_ids),
            "monitor_background_source_count": len(internal.monitor_background_ids),
            "checks": split_checks,
            "representation_warnings": list(internal.representation_warnings),
        },
        "tensor_contract_checks": tensor_checks,
        "canonical_mask_equality_rate": equality_count / requested,
        "transform_metadata_equality_rate": transform_equality_count / requested,
        "empty_mask_count": empty_masks,
        "template_utilization": {
            "unique": len(templates),
            "available_train": len(internal.train_template_indices),
            "counts": dict(sorted(templates.items())),
        },
        "background_utilization": {
            "unique": len(backgrounds),
            "available_train": len(internal.train_background_ids),
            "counts": dict(sorted(backgrounds.items())),
        },
        "source_contact_side_counts": dict(sorted(source_sides.items())),
        "target_side_combination_counts": dict(sorted(target_combinations.items())),
        "native_valid_statistics": {
            "generator_support_inside_fake_valid_minimum": min(support_inside_fractions),
            "generator_support_inside_fake_valid_mean": float(
                np.mean(support_inside_fractions)
            ),
            "fake_localized_logits_affected_by_padding_mean": float(
                np.mean(fake_padding_logit_fractions)
            ),
            "fake_localized_logits_affected_by_padding_maximum": max(
                fake_padding_logit_fractions
            ),
            "real_localized_logits_affected_by_padding_mean": float(
                np.mean(real_padding_logit_fractions)
            ),
            "real_localized_logits_affected_by_padding_maximum": max(
                real_padding_logit_fractions
            ),
            "real_fake_valid_fraction_asymmetry_mean": float(np.mean(padding_asymmetry)),
            "real_fake_valid_fraction_asymmetry_maximum": max(padding_asymmetry),
        },
        "determinism": {
            "repeat_hashes": list(repeat_hashes),
            "epoch_zero_hash": epoch_zero_hash,
            "epoch_one_hash": epoch_one_hash,
            "monitor_epoch_zero_hash": monitor_zero_hash,
            "monitor_epoch_later_hash": monitor_later_hash,
            "checks": deterministic_checks,
        },
        "contact_sheet": str(contact_sheet_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "contact_sheet_categories": sorted(representatives),
        "invariants": invariants,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "training_steps": 0,
        "materialized_generated_training_images": 0,
        "audit_contact_sheets": 1,
    }


def _markdown(report: dict[str, Any]) -> str:
    split = report.get("internal_split", {})
    native = report.get("native_valid_statistics", {})
    warnings = [f"- `{warning}`" for warning in split.get("representation_warnings", [])]
    if not warnings:
        warnings = ["- None"]
    invariant_lines = [
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in report.get("invariants", {}).items()
    ]
    return "\n".join(
        [
            "# G1.3 deterministic GAN training-pair audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Requested samples: {report.get('requested_samples')}",
            f"- Runtime seconds: {report.get('runtime_seconds')}",
            f"- Samples/second: {report.get('samples_per_second')}",
            f"- Train/monitor defect sources: "
            f"{split.get('train_defect_source_count')} / "
            f"{split.get('monitor_defect_source_count')}",
            f"- Train/monitor backgrounds: "
            f"{split.get('train_background_source_count')} / "
            f"{split.get('monitor_background_source_count')}",
            f"- Real/fake canonical-mask equality rate: "
            f"{report.get('canonical_mask_equality_rate')}",
            f"- Transform-metadata equality rate: "
            f"{report.get('transform_metadata_equality_rate')}",
            f"- Empty masks: {report.get('empty_mask_count')}",
            f"- Minimum support inside fake validity: "
            f"{native.get('generator_support_inside_fake_valid_minimum')}",
            f"- Mean fake localized logits affected by padding: "
            f"{native.get('fake_localized_logits_affected_by_padding_mean')}",
            f"- Mean real localized logits affected by padding: "
            f"{native.get('real_localized_logits_affected_by_padding_mean')}",
            f"- Mean real/fake valid-fraction asymmetry: "
            f"{native.get('real_fake_valid_fraction_asymmetry_mean')}",
            f"- Contact sheet: `{report.get('contact_sheet')}`",
            f"- Validation rows loaded: {report.get('validation_rows_loaded', 0)}",
            f"- Official-test rows loaded: {report.get('official_test_rows_loaded', 0)}",
            f"- Training steps: {report.get('training_steps', 0)}",
            f"- Materialized generated training images: "
            f"{report.get('materialized_generated_training_images', 0)}",
            f"- Templates used/available: "
            f"{report.get('template_utilization', {}).get('unique')} / "
            f"{report.get('template_utilization', {}).get('available_train')}",
            f"- Backgrounds used/available: "
            f"{report.get('background_utilization', {}).get('unique')} / "
            f"{report.get('background_utilization', {}).get('available_train')}",
            f"- Target side combinations: "
            f"`{json.dumps(report.get('target_side_combination_counts', {}), sort_keys=True)}`",
            "",
            "## Invariants",
            "",
            *invariant_lines,
            "",
            "## Representation warnings",
            "",
            *warnings,
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_training_pairs.json",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_training_pairs" / "pair_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_training_pairs" / "pair_audit.md",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_training_pairs" / "contact_sheet.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples is not None:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        values["audit_sample_count"] = args.samples
        temporary_config = args.config.with_name(args.config.stem + ".audit.tmp.json")
        _atomic_write(temporary_config, json.dumps(values, indent=2) + "\n")
        config_path = temporary_config
    else:
        temporary_config = None
        config_path = args.config
    try:
        report = build_audit(config_path, args.contact_sheet)
    except Exception as error:
        report = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "training_steps": 0,
            "materialized_generated_training_images": 0,
        }
    finally:
        if temporary_config is not None and temporary_config.exists():
            temporary_config.unlink()
    _atomic_write(args.json_output, json.dumps(report, indent=2) + "\n")
    _atomic_write(args.markdown_output, _markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
