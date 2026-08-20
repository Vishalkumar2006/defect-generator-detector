"""Build validation-only D1 comparison reports and fixed visual diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.full_image import KSDD2FullImageDataset, NativeGeometry, restore_to_native  # noqa: E402
from defectgen.data.patches import Padding  # noqa: E402
from defectgen.models import UNet  # noqa: E402
from defectgen.training.diagnostics import select_fixed_validation_ids  # noqa: E402
from defectgen.training.engine import configurations_differ_only_by_pos_weight  # noqa: E402


WEIGHTS = (1, 5, 10, 20)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_candidates(root: Path) -> dict[int, dict[str, Any]]:
    candidates = {}
    for weight in WEIGHTS:
        directory = root / f"pw_{weight}"
        summary = json.loads((directory / "candidate_summary.json").read_text(encoding="utf-8"))
        threshold = json.loads((directory / "threshold_sweep.json").read_text(encoding="utf-8"))
        candidates[weight] = {
            "directory": directory,
            "summary": summary,
            "threshold": threshold,
            "epochs": _read_csv(directory / "epoch_metrics.csv"),
            "per_image": _read_csv(directory / "per_image_metrics_best_global_threshold.csv"),
        }
    hashes = {entry["summary"]["initialization_sha256"] for entry in candidates.values()}
    if len(hashes) != 1:
        raise ValueError("Candidates did not start from identical model parameters")
    configs = [entry["summary"]["configuration"] for entry in candidates.values()]
    if not all(configurations_differ_only_by_pos_weight(configs[0], config) for config in configs[1:]):
        raise ValueError("Candidate configurations differ in more than pos_weight")
    return candidates


def _stability(epochs: list[dict[str, str]]) -> tuple[str, float]:
    validation = [float(row["validation_total_loss"]) for row in epochs]
    best = min(validation)
    ratio = validation[-1] / best if best else 1.0
    if not all(np.isfinite(validation)):
        return "non-finite", ratio
    if ratio > 1.10:
        return "obvious validation divergence", ratio
    if max(validation) > 1.5 * min(validation):
        return "large validation-loss regime shift", ratio
    return "no obvious instability", ratio


def _comparison_row(weight: int, entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry["summary"]
    fixed = summary["threshold_0_5_metrics_at_best_epoch"]
    global_best = summary["best_global_dice_threshold"]
    defective_best = summary["best_mean_defective_dice_threshold"]
    stability, last_best_ratio = _stability(entry["epochs"])
    gradient_failures = int(summary["nonfinite_gradient_steps"])
    training_fallbacks = int(summary["amp_forward_fallback_batches"])
    validation_fallbacks = int(summary["amp_validation_fallback_batches"])
    failure_count = gradient_failures + training_fallbacks + validation_fallbacks
    if failure_count:
        stability += "; substantial AMP numerical fallback activity"
    return {
        "pos_weight": weight,
        "best_epoch": summary["best_epoch"],
        "best_validation_loss": summary["best_validation_loss"],
        "threshold_0_5_global_dice": fixed["global_dice"],
        "threshold_0_5_global_iou": fixed["global_iou"],
        "threshold_0_5_pixel_precision": fixed["pixel_precision"],
        "threshold_0_5_pixel_recall": fixed["pixel_recall"],
        "threshold_0_5_mean_defective_image_dice": fixed["mean_defective_image_dice"],
        "threshold_0_5_median_defective_image_dice": fixed["median_defective_image_dice"],
        "threshold_0_5_zero_detected_defective_images": fixed["defective_images_zero_detected_pixels"],
        "threshold_0_5_mean_normal_predicted_fraction": fixed["mean_predicted_defect_fraction_normal_images"],
        "threshold_0_5_normal_image_false_positive_rate": fixed["normal_image_false_positive_rate"],
        "threshold_0_5_image_precision": fixed["image_precision"],
        "threshold_0_5_image_recall": fixed["image_recall"],
        "threshold_0_5_image_f1": fixed["image_f1"],
        "best_global_dice_threshold": global_best["threshold"],
        "best_global_dice": global_best["global_dice"],
        "global_choice_mean_defective_image_dice": global_best["mean_defective_image_dice"],
        "global_choice_pixel_precision": global_best["pixel_precision"],
        "global_choice_pixel_recall": global_best["pixel_recall"],
        "global_choice_normal_image_false_positive_rate": global_best["normal_image_false_positive_rate"],
        "global_choice_image_f1": global_best["image_f1"],
        "best_mean_defective_dice_threshold": defective_best["threshold"],
        "defective_choice_global_dice": defective_best["global_dice"],
        "best_mean_defective_image_dice": defective_best["mean_defective_image_dice"],
        "defective_choice_pixel_precision": defective_best["pixel_precision"],
        "defective_choice_pixel_recall": defective_best["pixel_recall"],
        "defective_choice_normal_image_false_positive_rate": defective_best["normal_image_false_positive_rate"],
        "defective_choice_image_f1": defective_best["image_f1"],
        "threshold_objectives_differ": summary["threshold_objectives_differ"],
        "training_seconds": sum(float(epoch["epoch_seconds"]) for epoch in entry["epochs"]),
        "runtime_seconds": summary["runtime_seconds"],
        "peak_allocated_vram_bytes": summary["peak_allocated_vram_bytes"],
        "peak_reserved_vram_bytes": summary["peak_reserved_vram_bytes"],
        "nonfinite_gradient_steps": gradient_failures,
        "amp_training_fallback_batches": training_fallbacks,
        "amp_validation_fallback_batches": validation_fallbacks,
        "failure_count": failure_count,
        "stability_assessment": stability,
        "last_to_best_validation_loss_ratio": last_best_ratio,
        "official_test_samples_loaded": summary["official_test_samples_loaded"],
    }


def _plot_comparisons(candidates: dict[int, dict[str, Any]], root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for weight, entry in candidates.items():
        epochs = [int(row["epoch"]) for row in entry["epochs"]]
        axes[0].plot(epochs, [float(row["train_total_loss"]) for row in entry["epochs"]], "--", label=f"pw={weight} train")
        axes[1].plot(epochs, [float(row["validation_total_loss"]) for row in entry["epochs"]], marker="o", label=f"pw={weight} val")
    for axis, title in zip(axes, ("Training combined loss", "Validation combined loss")):
        axis.set(title=title, xlabel="epoch", ylabel="BCE + Dice loss")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(root / "comparison_loss_curves.png", dpi=170, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for weight, entry in candidates.items():
        rows = entry["threshold"]["rows"]
        axis.plot([row["pixel_recall"] for row in rows], [row["pixel_precision"] for row in rows], marker=".", label=f"pw={weight}")
        selected = entry["threshold"]["best_global_dice"]
        axis.scatter(selected["pixel_recall"], selected["pixel_precision"], s=55)
    axis.set(title="Validation pixel precision-recall threshold sweep", xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(root / "comparison_precision_recall.png", dpi=170, facecolor="white")
    plt.close(figure)


def _geometry(sample: dict[str, Any]) -> NativeGeometry:
    left, top, right, bottom = [int(value) for value in sample["padding"].tolist()]
    return NativeGeometry(
        int(sample["original_height"]),
        int(sample["original_width"]),
        Padding(left=left, top=top, right=right, bottom=bottom),
    )


def _visual_payloads(
    candidate: dict[str, Any], fixed_ids: dict[str, str], base: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    normalization = base["detector_normalization"]
    dataset = KSDD2FullImageDataset(
        REPO_ROOT,
        "validation",
        REPO_ROOT / base["paths"]["manifest"],
        target_size=(base["model"]["input_width"], base["model"]["input_height"]),
        image_padding_mode=base["model"]["image_padding_mode"],
        mean=normalization["mean"],
        standard_deviation=normalization["standard_deviation"],
        sample_ids=set(fixed_ids.values()),
    )
    model = UNet(base_channels=base["model"]["base_channels"]).cuda().eval()
    checkpoint = torch.load(
        REPO_ROOT / candidate["summary"]["checkpoint_best"], map_location="cuda", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(candidate["summary"]["best_global_dice_threshold"]["threshold"])
    mean_array = np.asarray(normalization["mean"], dtype=np.float32).reshape(3, 1, 1)
    std_array = np.asarray(normalization["standard_deviation"], dtype=np.float32).reshape(3, 1, 1)
    payloads = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            with torch.autocast("cuda", dtype=torch.float16):
                probability = torch.sigmoid(model(sample["image"].unsqueeze(0).cuda()).float())[0, 0].cpu()
            geometry = _geometry(sample)
            image = sample["image"].numpy() * std_array + mean_array
            image = np.moveaxis(restore_to_native(image, geometry), 0, -1).clip(0, 1)
            truth = restore_to_native(sample["mask"][0].numpy(), geometry).astype(bool)
            probability_native = restore_to_native(probability.numpy(), geometry)
            prediction = probability_native >= threshold
            overlay = image.copy()
            overlay[truth] = 0.55 * overlay[truth] + 0.45 * np.array([1.0, 0.0, 0.0])
            overlay[prediction] = 0.55 * overlay[prediction] + 0.45 * np.array([0.0, 1.0, 1.0])
            payloads[sample["sample_id"]] = {
                "image": image,
                "truth": truth,
                "probability": probability_native,
                "prediction": prediction,
                "overlay": overlay,
            }
    del model
    torch.cuda.empty_cache()
    return payloads


def _save_diagnostics(candidate: dict[str, Any], fixed_ids: dict[str, str], base: dict[str, Any]) -> None:
    payloads = _visual_payloads(candidate, fixed_ids, base)
    figure, axes = plt.subplots(len(fixed_ids), 5, figsize=(13, 17), constrained_layout=True)
    columns = ("input", "ground truth", "probability", "prediction", "overlay (GT red, pred cyan)")
    for row_index, (role, sample_id) in enumerate(fixed_ids.items()):
        payload = payloads[sample_id]
        views = (payload["image"], payload["truth"], payload["probability"], payload["prediction"], payload["overlay"])
        for column_index, (axis, view) in enumerate(zip(axes[row_index], views)):
            axis.imshow(view, cmap="magma" if column_index == 2 else "gray" if column_index in (1, 3) else None, vmin=0, vmax=1)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(columns[column_index], fontsize=9)
        axes[row_index, 0].set_ylabel(f"{role.replace('_', ' ')}\n{sample_id}", fontsize=8)
    threshold = candidate["summary"]["best_global_dice_threshold"]["threshold"]
    figure.suptitle(f"pos_weight={candidate['summary']['pos_weight']}; validation threshold={threshold}", fontsize=13)
    figure.savefig(candidate["directory"] / "fixed_validation_diagnostics.png", dpi=150, facecolor="white")
    plt.close(figure)


def _write_reports(
    candidates: dict[int, dict[str, Any]], rows: list[dict[str, Any]], fixed_ids: dict[str, str], root: Path,
    recommended: int | None, reason: str,
) -> None:
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "PASS - VALIDATION-ONLY IMBALANCE-LOSS PILOT",
        "audit_status": "PROVISIONAL UNTIL AMP STEP-ACCOUNTING AUDIT IS COMPLETE",
        "historical_metrics_unchanged": True,
        "official_test_evaluated": False,
        "initialization_hash_identical": True,
        "configurations_differ_only_by_pos_weight": True,
        "fixed_validation_ids": fixed_ids,
        "candidates": rows,
        "recommendation": {"pos_weight": recommended, "reason": reason},
        "warning": (
            "D1 used automatic fp32 forward retries on numerically unsafe fp16 batches. "
            "The stored metrics describe those executed hybrid-precision runs but cannot yet establish "
            "a clean pos_weight-only fp16 comparison. No candidate is final."
        ),
    }
    (root / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    headers = [
        "pw", "best ep", "val loss", "Dice@.5", "def Dice@.5", "P@.5", "R@.5", "normal FP@.5",
        "global t", "global Dice", "global normal FP", "def t", "best def Dice", "def normal FP",
        "train sec", "total sec", "VRAM GiB", "failures", "stability",
    ]
    table_rows = []
    for row in rows:
        table_rows.append([
            row["pos_weight"], row["best_epoch"], f'{row["best_validation_loss"]:.4f}',
            f'{row["threshold_0_5_global_dice"]:.4f}', f'{row["threshold_0_5_mean_defective_image_dice"]:.4f}',
            f'{row["threshold_0_5_pixel_precision"]:.4f}', f'{row["threshold_0_5_pixel_recall"]:.4f}',
            f'{row["threshold_0_5_normal_image_false_positive_rate"]:.4f}', row["best_global_dice_threshold"],
            f'{row["best_global_dice"]:.4f}', f'{row["global_choice_normal_image_false_positive_rate"]:.4f}',
            row["best_mean_defective_dice_threshold"], f'{row["best_mean_defective_image_dice"]:.4f}',
            f'{row["defective_choice_normal_image_false_positive_rate"]:.4f}', f'{row["training_seconds"]:.1f}',
            f'{row["runtime_seconds"]:.1f}', f'{row["peak_allocated_vram_bytes"] / 2**30:.2f}',
            row["failure_count"], row["stability_assessment"],
        ])
    markdown = [
        "# D1 positive-weight pilot", "",
        "> **PROVISIONAL — AMP AUDIT REQUIRED:** D1 automatically retried non-finite fp16 forwards in fp32. "
        "The historical metrics below are unchanged and describe the runs that actually executed, but the candidates "
        "followed different hybrid-precision paths. No positive weight is final until the manual numerical audit is reviewed.",
        "", "Validation-only screening on all 1,981 development-training and 350 validation images. The official test dataset was not constructed or evaluated.", "",
        "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, values)) + " |" for values in table_rows], "",
        "The CSV and JSON contain the complete threshold-0.5 metrics, both independently selected threshold operating points, runtime, memory, failures, and normal-image behavior.", "",
        "## Threshold trade-offs", "",
    ]
    for row in rows:
        relation = "different" if row["threshold_objectives_differ"] else "the same"
        markdown.append(f"- `pos_weight={row['pos_weight']}`: global-Dice threshold {row['best_global_dice_threshold']} and defective-image-Dice threshold {row['best_mean_defective_dice_threshold']} are {relation}.")
    markdown.extend(["", "## Stability", ""])
    for row in rows:
        markdown.append(f"- `pos_weight={row['pos_weight']}`: {row['stability_assessment']} (last/best validation-loss ratio {row['last_to_best_validation_loss_ratio']:.3f}).")
    recommendation = f"Provisional screening preference `pos_weight={recommended}`: {reason}" if recommended is not None else f"No candidate recommendation: {reason}"
    markdown.extend(["", "## Recommendation", "", recommendation, "", "This phase does not authorize official-test evaluation, final baseline training, augmentation, synthetic data, or GAN work."])
    (root / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommended-pos-weight", type=int, choices=WEIGHTS)
    parser.add_argument("--recommendation-reason", default="evidence review has not selected a balanced candidate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = REPO_ROOT / "reports" / "loss_pilot"
    candidates = _load_candidates(root)
    geometry_rows = _read_csv(REPO_ROOT / "reports" / "preprocessing" / "bbox_statistics.csv")
    fixed_ids = select_fixed_validation_ids(
        geometry_rows, {str(weight): entry["per_image"] for weight, entry in candidates.items()}
    )
    (root / "fixed_validation_ids.json").write_text(json.dumps(fixed_ids, indent=2) + "\n", encoding="utf-8")
    rows = [_comparison_row(weight, entry) for weight, entry in candidates.items()]
    _plot_comparisons(candidates, root)
    base = json.loads((REPO_ROOT / "configs" / "baseline.json").read_text(encoding="utf-8"))
    for entry in candidates.values():
        _save_diagnostics(entry, fixed_ids, base)
    _write_reports(candidates, rows, fixed_ids, root, args.recommended_pos_weight, args.recommendation_reason)
    print("comparison reports and fixed validation diagnostics created", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
