"""Compare the single G1.6 discriminator clipping/LR ablation with G1.5b."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.training_pairs import (  # noqa: E402
    GANTrainingPairDataset,
    create_internal_gan_split,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)
from defectgen.models import build_gan_models, load_gan_architecture_config  # noqa: E402
from defectgen.training.gan_smoke import (  # noqa: E402
    MONITOR_CATEGORIES,
    load_gan_smoke_config,
    select_stratified_monitor_samples,
)
from defectgen.training.gan_trainer import (  # noqa: E402
    collate_gan_training_samples,
    precision_autocast,
)
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402
from scripts.train_gan_smoke import _load_detector  # noqa: E402


METRIC_NAMES = (
    "mean_probability_inside_mask",
    "mean_probability_outside_mask",
    "inside_outside_probability_contrast",
    "dice_at_0_5",
    "iou_at_0_5",
    "samples_with_any_predicted_positive_fraction",
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    files = sorted(
        path
        for root in paths
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file()
    )
    return {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in files
    }


def _configuration_differences(
    baseline_path: Path, candidate_path: Path
) -> dict[str, dict[str, Any]]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate configuration keys differ")
    return {
        key: {"baseline": baseline[key], "candidate": candidate[key]}
        for key in baseline
        if baseline[key] != candidate[key]
    }


def _records(report_directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (report_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def clipping_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    updates = [record for record in records if record["kind"] in {"warmup", "joint"}]
    joint = [record for record in records if record["kind"] == "joint"]
    final60 = joint[-60:]

    def summarize(selected: list[dict[str, Any]], branch: str) -> dict[str, Any]:
        available = [record for record in selected if branch in record]
        clipped = [record[branch]["gradient_clipping_applied"] for record in available]
        norms = [
            record[branch]["gradient_clipping"]["pre_clipping_norm"]
            for record in available
        ]
        return {
            "update_count": len(available),
            "clipped_count": sum(bool(value) for value in clipped),
            "clipped_fraction": float(np.mean(clipped)) if clipped else 0.0,
            "pre_clipping_norm": _distribution(norms) if norms else None,
        }

    return {
        "overall": {
            "discriminator": summarize(updates, "discriminator"),
            "generator": summarize(updates, "generator"),
        },
        "final_60_joint_steps": {
            "discriminator": summarize(final60, "discriminator"),
            "generator": summarize(final60, "generator"),
        },
    }


def logit_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    joint = [record for record in records if record["kind"] == "joint"]
    final60 = joint[-60:]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "real_mean": _distribution(
                [record["discriminator"]["real_logits"]["mean"] for record in selected]
            ),
            "fake_mean": _distribution(
                [record["discriminator"]["fake_logits"]["mean"] for record in selected]
            ),
            "real_minus_fake_margin": _distribution(
                [
                    record["discriminator"]["real_minus_fake_logit_margin"]
                    for record in selected
                ]
            ),
        }

    monitors = [record for record in records if record["kind"] == "monitor"]
    return {
        "overall_joint": summarize(joint),
        "final_60_joint_steps": summarize(final60),
        "final_monitor": monitors[-1]["logits"] if monitors else None,
    }


def safety_invariants(
    summary: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, bool]:
    joint = [record for record in records if record["kind"] == "joint"]
    return {
        "status_pass": summary["status"] == "PASS",
        "completed_200_generator_steps": summary["joint_generator_steps"] == 200,
        "completed_200_discriminator_steps": summary["joint_discriminator_steps"] == 200,
        "no_early_stop": not summary["early_stop"],
        "zero_locality_violations": summary["locality_violations"] == 0,
        "zero_invalid_gradient_violations": summary["invalid_gradient_violations"] == 0,
        "zero_output_range_violations": all(
            record["generator"]["output_range_violation_count"] == 0
            for record in joint
        ),
        "all_canonical_gradients_active": all(
            record["generator"]["canonical_defect_gradient_active_count"]
            == record["generator"]["canonical_defect_gradient_total_count"]
            for record in joint
        ),
        "zero_validation_rows": summary["validation_rows_loaded"] == 0,
        "zero_official_test_rows": summary["official_test_rows_loaded"] == 0,
        "zero_materialized_training_images": summary["materialized_training_images"] == 0,
    }


def _load_generator(config, checkpoint_path: Path, device: torch.device):
    architecture = load_gan_architecture_config(
        REPO_ROOT / config.architecture_config_path
    )
    generator, _ = build_gan_models(architecture)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(payload["generator_state"])
    return generator.to(device).eval()


def _gan_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .cpu()
        .float()
        .add(1)
        .div(2)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def _detector_statistics(
    evaluator,
    image: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    metrics, _ = evaluator.metrics(image, mask, valid)
    return metrics


def _mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([value[name] for value in values]))
        for name in METRIC_NAMES
    }


def _detector_distance(
    refined: dict[str, float], genuine: dict[str, float]
) -> dict[str, float]:
    differences = np.asarray(
        [refined[name] - genuine[name] for name in METRIC_NAMES], dtype=np.float64
    )
    return {
        "l2": float(np.sqrt(np.square(differences).sum())),
        "mean_absolute": float(np.abs(differences).mean()),
    }


def _write_blinded_sheets(
    output_directory: Path,
    rows: list[dict[str, Any]],
    *,
    mapping: dict[str, str],
) -> list[str]:
    paths: list[str] = []
    for replicate in range(4):
        selected = [row for row in rows if row["replicate"] == replicate]
        figure, axes = plt.subplots(7, 7, figsize=(14, 15.5), squeeze=False)
        for row_index, item in enumerate(selected):
            input_image = item["images"]["input"]
            candidate_a = item["images"][mapping["A"]]
            candidate_b = item["images"][mapping["B"]]
            genuine = item["images"]["genuine_real"]
            mask = item["mask"]
            mask_overlay = input_image.copy()
            mask_overlay[mask] = 0.45 * mask_overlay[mask] + 0.55 * np.asarray((1, 0, 0))
            panels = (
                (genuine, "Genuine real"),
                (input_image, "Input composite"),
                (candidate_a, "Candidate A"),
                (candidate_b, "Candidate B"),
                (np.clip(np.abs(candidate_a - input_image) * 8, 0, 1), "|A-input| x8"),
                (np.clip(np.abs(candidate_b - input_image) * 8, 0, 1), "|B-input| x8"),
                (mask_overlay, "Condition mask"),
            )
            for column, (image, title) in enumerate(panels):
                axes[row_index, column].imshow(image)
                axes[row_index, column].axis("off")
                if row_index == 0:
                    axes[row_index, column].set_title(title, fontsize=8)
            axes[row_index, 0].set_ylabel(
                f"{item['opaque_id']}\n{item['category']}", fontsize=7
            )
        figure.suptitle(
            f"G1.6 blinded stratified monitor sheet {replicate + 1}/4",
            fontsize=11,
        )
        figure.tight_layout()
        path = output_directory / "blinded_sheets" / f"sheet_{replicate + 1:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(figure)
        paths.append(path.relative_to(REPO_ROOT).as_posix())
    return paths


def _stratified_monitor_audit(
    baseline_config,
    candidate_config,
    *,
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    output_directory: Path,
) -> dict[str, Any]:
    configure_reproducibility(42, deterministic=True, warn_only=False)
    pair_config = load_gan_training_pair_config(
        REPO_ROOT / baseline_config.training_pair_config_path
    )
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    internal = create_internal_gan_split(
        metadata,
        monitor_fraction=pair_config.monitor_fraction,
        seed=pair_config.base_seed,
    )
    dataset = GANTrainingPairDataset(
        metadata,
        REPO_ROOT,
        pair_config,
        split="monitor",
        internal_split=internal,
        length=baseline_config.monitor_panel_scan_limit,
    )
    panel = select_stratified_monitor_samples(
        (dataset[index] for index in range(len(dataset))), per_category=4
    )
    device = torch.device("cuda")
    baseline_generator = _load_generator(
        baseline_config, baseline_checkpoint, device
    )
    candidate_generator = _load_generator(
        candidate_config, candidate_checkpoint, device
    )
    evaluator = _load_detector(baseline_config, device)
    mapping = (
        {"A": "baseline", "B": "candidate"}
        if hashlib.sha256(b"g1.6:42").digest()[0] % 2 == 0
        else {"A": "candidate", "B": "baseline"}
    )
    branch_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_category: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for category in MONITOR_CATEGORIES:
            for replicate, sample in enumerate(panel[category]):
                batch = collate_gan_training_samples([sample]).to(device)
                with precision_autocast(device, baseline_config.precision):
                    baseline_output = baseline_generator(
                        batch.composite_image, batch.generator_mask
                    )
                    candidate_output = candidate_generator(
                        batch.composite_image, batch.generator_mask
                    )
                images = {
                    "baseline": baseline_output.refined_image,
                    "candidate": candidate_output.refined_image,
                    "genuine_real": batch.real_image,
                }
                metrics = {
                    "baseline": _detector_statistics(
                        evaluator,
                        images["baseline"],
                        batch.fake_discriminator_mask,
                        batch.fake_valid_mask,
                    ),
                    "candidate": _detector_statistics(
                        evaluator,
                        images["candidate"],
                        batch.fake_discriminator_mask,
                        batch.fake_valid_mask,
                    ),
                    "genuine_real": _detector_statistics(
                        evaluator,
                        images["genuine_real"],
                        batch.fake_discriminator_mask,
                        batch.real_valid_mask,
                    ),
                }
                for branch, values in metrics.items():
                    branch_metrics[branch].append(values)
                    by_category[category][branch].append(values)
                opaque_material = (
                    f"{sample.metadata['sample_index']}:"
                    f"{sample.metadata['template_id']}:"
                    f"{sample.metadata['normal_background_sample_id']}"
                )
                rows.append(
                    {
                        "opaque_id": "S"
                        + hashlib.sha256(opaque_material.encode("utf-8")).hexdigest()[:6],
                        "category": category,
                        "replicate": replicate,
                        "sample_index": sample.metadata["sample_index"],
                        "template_id": sample.metadata["template_id"],
                        "background_id": sample.metadata[
                            "normal_background_sample_id"
                        ],
                        "metrics": metrics,
                        "images": {
                            "input": _gan_rgb(batch.composite_image[0]),
                            "baseline": _gan_rgb(images["baseline"][0]),
                            "candidate": _gan_rgb(images["candidate"][0]),
                            "genuine_real": _gan_rgb(images["genuine_real"][0]),
                        },
                        "mask": batch.fake_discriminator_mask[0, 0]
                        .bool()
                        .cpu()
                        .numpy(),
                    }
                )
    aggregate = {
        branch: _mean_metrics(values) for branch, values in branch_metrics.items()
    }
    category_summary = {
        category: {
            branch: _mean_metrics(values)
            for branch, values in branches.items()
        }
        for category, branches in by_category.items()
    }
    sheet_paths = _write_blinded_sheets(output_directory, rows, mapping=mapping)
    reveal = {
        "mapping": mapping,
        "derivation": "sha256('g1.6:42') first-byte parity",
        "sheet_paths": sheet_paths,
    }
    _atomic_write(
        output_directory / "blinding_reveal.json",
        json.dumps(reveal, indent=2) + "\n",
    )
    return {
        "sample_count": len(rows),
        "samples_per_category": 4,
        "categories": list(MONITOR_CATEGORIES),
        "unique_sample_count": len({row["opaque_id"] for row in rows}),
        "sample_provenance": [
            {key: row[key] for key in ("opaque_id", "category", "sample_index", "template_id", "background_id")}
            for row in rows
        ],
        "aggregate_detector_statistics": aggregate,
        "detector_statistic_distance_from_genuine_real": {
            "baseline": _detector_distance(aggregate["baseline"], aggregate["genuine_real"]),
            "candidate": _detector_distance(aggregate["candidate"], aggregate["genuine_real"]),
        },
        "by_category_detector_statistics": category_summary,
        "blinded_sheet_paths": sheet_paths,
        "blinding_reveal_path": (
            output_directory / "blinding_reveal.json"
        ).relative_to(REPO_ROOT).as_posix(),
        "development_training_only": True,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_generated_dataset_images": 0,
        "materialized_blinded_monitor_sheets": len(sheet_paths),
    }


def recommendation(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    visual_review: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_safe = all(baseline["safety_invariants"].values())
    candidate_safe = all(candidate["safety_invariants"].values())
    baseline_clip = baseline["clipping"]
    candidate_clip = candidate["clipping"]
    overall_improvement = (
        baseline_clip["overall"]["discriminator"]["clipped_fraction"]
        - candidate_clip["overall"]["discriminator"]["clipped_fraction"]
    )
    final60_improvement = (
        baseline_clip["final_60_joint_steps"]["discriminator"]["clipped_fraction"]
        - candidate_clip["final_60_joint_steps"]["discriminator"]["clipped_fraction"]
    )
    baseline_distance = baseline["detector_distance"]["l2"]
    candidate_distance = candidate["detector_distance"]["l2"]
    baseline_margin = baseline["logits"]["final_60_joint_steps"][
        "real_minus_fake_margin"
    ]["mean"]
    candidate_margin = candidate["logits"]["final_60_joint_steps"][
        "real_minus_fake_margin"
    ]["mean"]
    visual_preference = (visual_review or {}).get("preference", "pending")
    visual_concerns = (visual_review or {}).get("safety_concerns", [])
    candidate_dominates = (
        candidate_safe
        and overall_improvement >= 0.10
        and final60_improvement >= 0.10
        and candidate_distance <= baseline_distance + 0.02
        and candidate_margin >= baseline_margin - 0.02
        and visual_preference in {"candidate", "tie"}
        and not visual_concerns
    )
    baseline_dominates = (
        baseline_safe
        and overall_improvement <= -0.10
        and final60_improvement <= -0.10
        and baseline_distance <= candidate_distance + 0.02
        and baseline_margin >= candidate_margin - 0.02
        and visual_preference in {"baseline", "tie"}
        and not visual_concerns
    )
    selected = "candidate" if candidate_dominates else "baseline"
    reason = (
        "candidate_clearly_dominates"
        if candidate_dominates
        else "baseline_clearly_dominates"
        if baseline_dominates
        else "neither_clearly_dominates_retain_baseline"
    )
    return {
        "selected_configuration": selected,
        "reason": reason,
        "candidate_clearly_dominates": candidate_dominates,
        "baseline_clearly_dominates": baseline_dominates,
        "discriminator_clipping_improvement": {
            "overall_percentage_points": 100 * overall_improvement,
            "final_60_percentage_points": 100 * final60_improvement,
        },
        "visual_review": visual_review,
        "dominance_thresholds": {
            "minimum_clipping_improvement_percentage_points": 10,
            "maximum_detector_l2_regression": 0.02,
            "maximum_final60_margin_regression": 0.02,
            "requires_no_blinded_visual_safety_concern": True,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return f"# G1.6 discriminator-update ablation\n\n- Status: **FAIL**\n- Error: `{report.get('error')}`\n"
    baseline = report["baseline"]
    candidate = report["candidate"]
    recommendation_value = report["recommendation"]
    return "\n".join(
        [
            "# G1.6 discriminator-update ablation",
            "",
            f"- Status: **{report['status']}**",
            f"- Baseline safety invariants: {all(baseline['safety_invariants'].values())}",
            f"- Candidate safety invariants: {all(candidate['safety_invariants'].values())}",
            f"- Candidate terminal status/step/reason: {candidate['summary_status']} / {candidate['completed_joint_steps']} / `{candidate['terminal_reason']}`",
            f"- Candidate terminal active/total canonical gradients: {candidate['terminal_canonical_gradient_counts']}",
            f"- Baseline D clipping overall/final-60: {baseline['clipping']['overall']['discriminator']['clipped_fraction']} / {baseline['clipping']['final_60_joint_steps']['discriminator']['clipped_fraction']}",
            f"- Candidate D clipping overall/final-60: {candidate['clipping']['overall']['discriminator']['clipped_fraction']} / {candidate['clipping']['final_60_joint_steps']['discriminator']['clipped_fraction']}",
            f"- Baseline/candidate final-60 mean margin: {baseline['logits']['final_60_joint_steps']['real_minus_fake_margin']['mean']} / {candidate['logits']['final_60_joint_steps']['real_minus_fake_margin']['mean']}",
            f"- Baseline/candidate detector L2 distance: {baseline['detector_distance']['l2']} / {candidate['detector_distance']['l2']}",
            f"- Stratified monitor samples: {report['stratified_monitor_audit']['sample_count']}",
            f"- Validation / official-test rows: {report['validation_rows_loaded']} / {report['official_test_rows_loaded']}",
            f"- Baseline artifacts unchanged: {report['baseline_integrity']['unchanged']}",
            f"- Recommendation: **{recommendation_value['selected_configuration']}** (`{recommendation_value['reason']}`)",
            "",
            "## Configuration differences",
            "",
            f"`{json.dumps(report['configuration_differences'], sort_keys=True)}`",
            "",
            "## Blinded visual sheets",
            "",
            *[
                f"- `{path}`"
                for path in report["stratified_monitor_audit"]["blinded_sheet_paths"]
            ],
            "",
        ]
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline_config = load_gan_smoke_config(args.baseline_config)
    candidate_config = load_gan_smoke_config(args.candidate_config)
    differences = _configuration_differences(
        args.baseline_config, args.candidate_config
    )
    expected_differences = {
        "discriminator_learning_rate",
        "discriminator_gradient_clip_max_norm",
        "report_directory",
        "checkpoint_directory",
    }
    if set(differences) != expected_differences:
        raise ValueError("G1.6 configuration contains changes outside the controlled ablation")
    baseline_report = REPO_ROOT / baseline_config.report_directory
    candidate_report = REPO_ROOT / candidate_config.report_directory
    baseline_checkpoint_dir = REPO_ROOT / baseline_config.checkpoint_directory
    candidate_checkpoint_dir = REPO_ROOT / candidate_config.checkpoint_directory
    baseline_summary = json.loads(
        (baseline_report / "summary.json").read_text(encoding="utf-8")
    )
    candidate_summary = json.loads(
        (candidate_report / "summary.json").read_text(encoding="utf-8")
    )
    baseline_records = _records(baseline_report)
    candidate_records = _records(candidate_report)
    existing_comparison = args.output_directory / "comparison.json"
    if args.reuse_stratified and existing_comparison.is_file():
        stratified = json.loads(existing_comparison.read_text(encoding="utf-8"))[
            "stratified_monitor_audit"
        ]
    else:
        stratified = _stratified_monitor_audit(
            baseline_config,
            candidate_config,
            baseline_checkpoint=baseline_checkpoint_dir / "joint_200.pt",
            candidate_checkpoint=(
                candidate_checkpoint_dir / "joint_200.pt"
                if (candidate_checkpoint_dir / "joint_200.pt").is_file()
                else candidate_checkpoint_dir / "last.pt"
            ),
            output_directory=args.output_directory,
        )
    before = json.loads(args.baseline_snapshot.read_text(encoding="utf-8"))
    after_hashes = hash_paths((baseline_report, baseline_checkpoint_dir))
    integrity = {
        "before_sha256": before["sha256"],
        "after_sha256": after_hashes,
        "unchanged": before["sha256"] == after_hashes,
    }
    baseline = {
        "summary_status": baseline_summary["status"],
        "completed_joint_steps": baseline_summary["joint_generator_steps"],
        "terminal_reason": baseline_summary["early_stop_reason"],
        "safety_invariants": safety_invariants(baseline_summary, baseline_records),
        "clipping": clipping_statistics(baseline_records),
        "logits": logit_statistics(baseline_records),
        "detector_distance": stratified["detector_statistic_distance_from_genuine_real"]["baseline"],
    }
    candidate = {
        "summary_status": candidate_summary["status"],
        "completed_joint_steps": candidate_summary["joint_generator_steps"],
        "terminal_reason": candidate_summary["early_stop_reason"],
        "safety_invariants": safety_invariants(candidate_summary, candidate_records),
        "clipping": clipping_statistics(candidate_records),
        "logits": logit_statistics(candidate_records),
        "detector_distance": stratified["detector_statistic_distance_from_genuine_real"]["candidate"],
        "terminal_canonical_gradient_counts": {
            "active": candidate_records[-1]["generator"][
                "canonical_defect_gradient_active_count"
            ],
            "total": candidate_records[-1]["generator"][
                "canonical_defect_gradient_total_count"
            ],
            "fraction": candidate_records[-1]["generator"][
                "canonical_defect_gradient_coverage"
            ],
        }
        if candidate_records and candidate_records[-1]["kind"] == "joint"
        else None,
    }
    visual_review = (
        json.loads(args.visual_review.read_text(encoding="utf-8"))
        if args.visual_review and args.visual_review.is_file()
        else None
    )
    recommendation_value = recommendation(baseline, candidate, visual_review)
    report = {
        "status": "PASS"
        if all(baseline["safety_invariants"].values())
        and integrity["unchanged"]
        and stratified["sample_count"] == 28
        and stratified["official_test_rows_loaded"] == 0
        else "FAIL",
        "phase": "G1.6_single_discriminator_update_ablation",
        "configuration_differences": differences,
        "baseline": baseline,
        "candidate": candidate,
        "stratified_monitor_audit": stratified,
        "baseline_integrity": integrity,
        "recommendation": recommendation_value,
        "additional_sweeps_run": 0,
        "validation_rows_loaded": 0,
        "official_test_rows_loaded": 0,
        "materialized_generated_dataset_images": 0,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_smoke.json",
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=REPO_ROOT / "configs" / "gan_smoke_dclip10.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "reports" / "gan_training" / "g1_6_dclip10_ablation",
    )
    parser.add_argument(
        "--baseline-snapshot",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "gan_training"
        / "g1_6_dclip10_ablation"
        / "baseline_integrity_before.json",
    )
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--reuse-stratified", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_config = load_gan_smoke_config(args.baseline_config)
    baseline_paths = (
        REPO_ROOT / baseline_config.report_directory,
        REPO_ROOT / baseline_config.checkpoint_directory,
    )
    if args.snapshot_only:
        snapshot = {
            "baseline_config": args.baseline_config.relative_to(REPO_ROOT).as_posix(),
            "sha256": hash_paths(baseline_paths),
        }
        _atomic_write(
            args.baseline_snapshot, json.dumps(snapshot, indent=2) + "\n"
        )
        print(f"Wrote immutable baseline snapshot: {args.baseline_snapshot}")
        return 0
    try:
        report = build_report(args)
    except Exception as error:
        report = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "validation_rows_loaded": 0,
            "official_test_rows_loaded": 0,
            "materialized_generated_dataset_images": 0,
        }
    _atomic_write(args.output_directory / "comparison.json", json.dumps(report, indent=2) + "\n")
    _atomic_write(args.output_directory / "comparison.md", _markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
