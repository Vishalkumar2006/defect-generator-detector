"""Run a trained detector checkpoint over a single image and save a mask overlay.

This is a read-only inspection tool for understanding the V1 detector without
rerunning any experiment. It trains nothing, writes no checkpoint, touches no
report, and changes no recorded result.

It needs a locally supplied checkpoint. No checkpoint is distributed with this
repository (see docs/dataset-setup.md and THIRD_PARTY_NOTICES.md); the intended
input is the accepted real-only reference produced by
``scripts/train_final_real_baseline.py``:

    checkpoints/final_real_baseline_bf16_seed42/best.pt

Preprocessing mirrors ``KSDD2FullImageDataset`` exactly: the native image is
symmetrically padded onto the frozen 256x672 canvas (reflection for pixels,
zeros for labels), normalized with the channel statistics recorded inside the
checkpoint's own configuration, and the prediction is cropped back to native
size before anything is written. Runs on CPU by default.

Examples
--------
    # Windows PowerShell
    .\\.venv\\Scripts\\python.exe .\\scripts\\demo_segment_image.py `
        --checkpoint .\\checkpoints\\final_real_baseline_bf16_seed42\\best.pt `
        --image .\\data\\extracted\\KolektorSDD2\\train\\10000.png

    # Linux / macOS
    ./.venv/bin/python ./scripts/demo_segment_image.py \\
        --checkpoint checkpoints/final_real_baseline_bf16_seed42/best.pt \\
        --image data/extracted/KolektorSDD2/train/10000.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.data.full_image import (  # noqa: E402
    MODEL_HEIGHT,
    MODEL_WIDTH,
    NativeGeometry,
    pad_full_sample,
    restore_to_native,
)
from defectgen.models.unet import UNet, count_parameters  # noqa: E402

# The one-time validation Dice-optimal threshold frozen for the accepted
# real-only BF16 reference. 0.5 is the fixed comparison threshold.
DEFAULT_THRESHOLD = 0.5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment one image with a locally supplied detector checkpoint and "
            "save a prediction overlay. Read-only: trains nothing and alters no result."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a local detector checkpoint (.pt) saved by this project.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a single input image. KSDD2 images are tall strips of about 230x630 px.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "demo" / "prediction.png",
        help="Where to write the side-by-side overlay PNG.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold used to binarize the predicted mask.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device. CPU is sufficient for a single image.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional KSDD2 *_GT.png mask, drawn in red beside the prediction for comparison.",
    )
    return parser.parse_args()


def load_detector(checkpoint_path: Path, device: torch.device) -> tuple[UNet, dict, int | None]:
    """Rebuild the U-Net described by a checkpoint, load its weights, and report its epoch."""
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            "No checkpoint ships with this repository. Train one with "
            "scripts/train_final_real_baseline.py, or point --checkpoint at a local file."
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise SystemExit(f"{checkpoint_path} does not look like a detector checkpoint (no 'model_state').")

    configuration = payload.get("configuration", {})
    model_config = configuration.get("model", {})
    model = UNet(
        input_channels=int(model_config.get("input_channels", 3)),
        output_channels=int(model_config.get("output_channels", 1)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    model.load_state_dict(payload["model_state"])
    model.eval().to(device)
    epoch = payload.get("epoch")
    return model, configuration, int(epoch) if isinstance(epoch, int) else None


def normalization_from(configuration: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the channel statistics the checkpoint was trained with, if it recorded any."""
    stats = configuration.get("data", {}).get("detector_normalization")
    if not stats:
        return None, None
    return (
        np.asarray(stats["mean"], dtype=np.float32).reshape(3, 1, 1),
        np.asarray(stats["standard_deviation"], dtype=np.float32).reshape(3, 1, 1),
    )


def predict(
    model: UNet,
    image: np.ndarray,
    *,
    device: torch.device,
    mean: np.ndarray | None,
    standard_deviation: np.ndarray | None,
) -> np.ndarray:
    """Return native-resolution defect probabilities for one H x W x 3 uint8 image."""
    height, width = image.shape[:2]
    if height > MODEL_HEIGHT or width > MODEL_WIDTH:
        raise SystemExit(
            f"Image {width}x{height} exceeds the frozen {MODEL_WIDTH}x{MODEL_HEIGHT} canvas. "
            "This detector consumes native KSDD2 geometry without resizing."
        )

    # A zero mask is passed only to reuse the exact padding routine; it is unused.
    padded_image, _, _, padding = pad_full_sample(
        image,
        np.zeros((height, width), dtype=bool),
        target_size=(MODEL_WIDTH, MODEL_HEIGHT),
        image_padding_mode="reflect",
    )

    tensor = padded_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    if mean is not None and standard_deviation is not None:
        tensor = (tensor - mean) / standard_deviation

    batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(batch))

    geometry = NativeGeometry(original_height=height, original_width=width, padding=padding)
    return restore_to_native(probabilities, geometry)[0, 0].cpu().numpy()


def build_overlay(
    image: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    ground_truth: np.ndarray | None,
) -> Image.Image:
    """Compose [input | probability heat | prediction cyan | optional GT red] as one strip."""
    height, width = image.shape[:2]
    panels = [image]

    heat = np.zeros_like(image)
    heat[..., 0] = (np.clip(probabilities, 0.0, 1.0) * 255).astype(np.uint8)
    panels.append(heat)

    predicted = probabilities >= threshold
    prediction_panel = image.copy()
    prediction_panel[predicted] = (0.35 * prediction_panel[predicted] + 0.65 * np.array([0, 255, 255])).astype(
        np.uint8
    )
    panels.append(prediction_panel)

    if ground_truth is not None:
        truth_panel = image.copy()
        truth_panel[ground_truth] = (0.35 * truth_panel[ground_truth] + 0.65 * np.array([255, 0, 0])).astype(
            np.uint8
        )
        panels.append(truth_panel)

    gap = 8
    canvas = np.full((height, width * len(panels) + gap * (len(panels) - 1), 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        left = index * (width + gap)
        canvas[:, left : left + width] = panel
    return Image.fromarray(canvas)


def main() -> None:
    arguments = parse_arguments()
    if not 0.0 < arguments.threshold < 1.0:
        raise SystemExit("--threshold must lie strictly between 0 and 1.")
    if not arguments.image.is_file():
        raise SystemExit(f"Image not found: {arguments.image}")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable.")

    device = torch.device(arguments.device)
    model, configuration, checkpoint_epoch = load_detector(arguments.checkpoint, device)
    mean, standard_deviation = normalization_from(configuration)

    with Image.open(arguments.image) as source:
        image = np.asarray(source.convert("RGB"))

    ground_truth = None
    if arguments.ground_truth is not None:
        if not arguments.ground_truth.is_file():
            raise SystemExit(f"Ground-truth mask not found: {arguments.ground_truth}")
        with Image.open(arguments.ground_truth) as source:
            ground_truth = np.asarray(source.convert("L")) > 0
        if ground_truth.shape != image.shape[:2]:
            raise SystemExit("Ground-truth mask dimensions do not match the image.")

    probabilities = predict(
        model,
        image,
        device=device,
        mean=mean,
        standard_deviation=standard_deviation,
    )
    predicted = probabilities >= arguments.threshold

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    build_overlay(image, probabilities, arguments.threshold, ground_truth).save(arguments.output)

    summary = {
        "image": str(arguments.image),
        "checkpoint": str(arguments.checkpoint),
        "experiment_identity": configuration.get("experiment_identity"),
        "checkpoint_epoch": checkpoint_epoch,
        "device": arguments.device,
        "native_size": [int(image.shape[1]), int(image.shape[0])],
        "model_parameters": count_parameters(model),
        "threshold": arguments.threshold,
        "predicted_defect_pixels": int(predicted.sum()),
        "predicted_defect_fraction": float(predicted.mean()),
        "max_probability": float(probabilities.max()),
        "overlay": str(arguments.output),
    }
    if ground_truth is not None:
        intersection = float(np.logical_and(predicted, ground_truth).sum())
        total = float(predicted.sum() + ground_truth.sum())
        summary["ground_truth_defect_pixels"] = int(ground_truth.sum())
        summary["dice"] = (2.0 * intersection / total) if total > 0 else 1.0

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
