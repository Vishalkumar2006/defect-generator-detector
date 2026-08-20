"""Valid-region segmentation and image-level defect metrics."""

from __future__ import annotations

from collections.abc import Iterable

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

