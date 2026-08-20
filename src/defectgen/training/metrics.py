"""Valid-region segmentation and image-level defect metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def _safe_ratio(numerator: float, denominator: float, empty_value: float) -> float:
    return numerator / denominator if denominator else empty_value


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_region: torch.Tensor,
    threshold: float = 0.5,
    has_defect: torch.Tensor | None = None,
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets.bool()
    valid = valid_region.bool()

    def compute(selection: torch.Tensor) -> dict[str, float]:
        predicted = predictions & selection
        actual = truth & selection
        tp = int((predicted & actual).sum().item())
        fp = int((predicted & ~actual & selection).sum().item())
        fn = int((~predicted & actual & selection).sum().item())
        return {
            "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn, 1.0),
            "iou": _safe_ratio(tp, tp + fp + fn, 1.0),
            "precision": _safe_ratio(tp, tp + fp, 1.0 if tp + fn == 0 else 0.0),
            "recall": _safe_ratio(tp, tp + fn, 1.0),
            "true_positive_pixels": float(tp),
            "false_positive_pixels": float(fp),
            "false_negative_pixels": float(fn),
        }

    result = {f"pixel_{key}": value for key, value in compute(valid).items()}
    if has_defect is not None and torch.any(has_defect.bool()):
        defective_selection = valid & has_defect.bool().view(-1, 1, 1, 1)
        result.update({f"defective_pixel_{key}": value for key, value in compute(defective_selection).items()})
    return result


def image_defect_probabilities(logits: torch.Tensor, valid_region: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    masked = probabilities.masked_fill(~valid_region.bool(), float("-inf"))
    return masked.flatten(start_dim=1).amax(dim=1)


def image_classification_metrics(
    probabilities: Iterable[float], labels: Iterable[bool], threshold: float
) -> dict[str, float]:
    probabilities = list(probabilities)
    labels = list(labels)
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Non-empty probability and label sequences must have equal length")
    predictions = [probability >= threshold for probability in probabilities]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels))
    fp = sum(prediction and not label for prediction, label in zip(predictions, labels))
    fn = sum(not prediction and label for prediction, label in zip(predictions, labels))
    precision = _safe_ratio(tp, tp + fp, 1.0 if not any(labels) else 0.0)
    recall = _safe_ratio(tp, tp + fn, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall, 0.0),
        "threshold": threshold,
    }


def select_validation_threshold(
    probabilities: Iterable[float], labels: Iterable[bool], candidates: Iterable[float] | None = None
) -> tuple[float, dict[str, float]]:
    probabilities = list(probabilities)
    labels = list(labels)
    candidates = list(candidates or [value / 20 for value in range(1, 20)])
    scored = [(threshold, image_classification_metrics(probabilities, labels, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"], -item[0]))


def detailed_validation_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    valid_region: torch.Tensor,
    has_defect: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Return requested aggregate and per-image metrics at one threshold."""
    if probabilities.shape != targets.shape or targets.shape != valid_region.shape:
        raise ValueError("probabilities, targets, and valid_region must have identical shapes")
    if probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("Expected N×1×H×W segmentation tensors")
    if len(has_defect) != len(probabilities):
        raise ValueError("has_defect length must match the batch")
    predicted = probabilities >= threshold
    truth = targets.bool()
    valid = valid_region.bool()
    global_tp = global_fp = global_fn = 0
    per_image: list[dict[str, float | int]] = []
    defective_dice: list[float] = []
    zero_detected_defects = 0
    normal_predicted_fractions: list[float] = []
    normal_false_positive_images = 0
    image_probabilities: list[float] = []
    labels = [bool(value) for value in has_defect.tolist()]
    for index, label in enumerate(labels):
        selected = valid[index]
        prediction = predicted[index] & selected
        actual = truth[index] & selected
        tp = int((prediction & actual).sum().item())
        fp = int((prediction & ~actual & selected).sum().item())
        fn = int((~prediction & actual & selected).sum().item())
        predicted_pixels = int(prediction.sum().item())
        valid_pixels = int(selected.sum().item())
        dice = _safe_ratio(2 * tp, 2 * tp + fp + fn, 1.0)
        image_probability = float(probabilities[index].masked_fill(~selected, float("-inf")).max().item())
        global_tp += tp
        global_fp += fp
        global_fn += fn
        image_probabilities.append(image_probability)
        if label:
            defective_dice.append(dice)
            zero_detected_defects += int(predicted_pixels == 0)
        else:
            fraction = predicted_pixels / valid_pixels
            normal_predicted_fractions.append(fraction)
            normal_false_positive_images += int(predicted_pixels > 0)
        per_image.append(
            {
                "index": index,
                "has_defect": int(label),
                "dice": dice,
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "predicted_pixels": predicted_pixels,
                "valid_pixels": valid_pixels,
                "predicted_fraction": predicted_pixels / valid_pixels,
                "image_probability": image_probability,
            }
        )
    image_metrics = image_classification_metrics(image_probabilities, labels, threshold)
    normal_count = len(normal_predicted_fractions)
    metrics: dict[str, float | int] = {
        "global_dice": _safe_ratio(2 * global_tp, 2 * global_tp + global_fp + global_fn, 1.0),
        "global_iou": _safe_ratio(global_tp, global_tp + global_fp + global_fn, 1.0),
        "pixel_precision": _safe_ratio(global_tp, global_tp + global_fp, 1.0 if global_tp + global_fn == 0 else 0.0),
        "pixel_recall": _safe_ratio(global_tp, global_tp + global_fn, 1.0),
        "mean_defective_image_dice": float(np.mean(defective_dice)) if defective_dice else 0.0,
        "median_defective_image_dice": float(np.median(defective_dice)) if defective_dice else 0.0,
        "defective_images_zero_detected_pixels": zero_detected_defects,
        "mean_predicted_defect_fraction_normal_images": (
            float(np.mean(normal_predicted_fractions)) if normal_predicted_fractions else 0.0
        ),
        "normal_image_false_positive_rate": normal_false_positive_images / normal_count if normal_count else 0.0,
        "normal_false_positive_images": normal_false_positive_images,
        "normal_image_count": normal_count,
        "image_precision": image_metrics["precision"],
        "image_recall": image_metrics["recall"],
        "image_f1": image_metrics["f1"],
        "threshold": threshold,
    }
    return metrics, per_image


def validation_threshold_sweep(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    valid_region: torch.Tensor,
    has_defect: torch.Tensor,
    thresholds: Iterable[float] | None = None,
) -> tuple[list[dict[str, float | int]], dict[str, float | int], dict[str, float | int]]:
    """Sweep validation thresholds and select two independently optimized objectives."""
    thresholds = list(thresholds or [value / 100 for value in range(5, 100, 5)])
    rows = []
    for threshold in thresholds:
        metrics, _ = detailed_validation_metrics(
            probabilities, targets, valid_region, has_defect, threshold
        )
        rows.append(metrics)
    best_global = max(
        rows,
        key=lambda row: (row["global_dice"], row["mean_defective_image_dice"], row["pixel_precision"], -row["threshold"]),
    )
    best_defective = max(
        rows,
        key=lambda row: (row["mean_defective_image_dice"], row["global_dice"], row["pixel_recall"], -row["threshold"]),
    )
    return rows, dict(best_global), dict(best_defective)
