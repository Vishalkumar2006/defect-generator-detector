"""Build paired, train-only detector samples from two frozen G2.1 generators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan.training_pairs import (  # noqa: E402
    GANTrainingPairDataset,
    load_gan_training_pair_config,
    load_training_pair_manifest,
)
from defectgen.models.gan import build_gan_models, load_gan_architecture_config  # noqa: E402
from defectgen.training.g2_2_utility import (  # noqa: E402
    G2_2_VERSION,
    assert_paired_manifests,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
)


def _atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        Image.fromarray(array).save(name, format="PNG")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _checkpoint_generator(
    path: Path, architecture_config: Any, expected_step: int, device: torch.device
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    progress = payload.get("progress", {})
    if int(progress.get("joint_generator_steps", -1)) != expected_step:
        raise RuntimeError(f"{path} is not the expected joint generator step {expected_step}")
    generator, _ = build_gan_models(architecture_config)
    generator.load_state_dict(payload["generator_state"])
    generator.eval().requires_grad_(False).to(device)
    return generator


def _detector_canvas(field: torch.Tensor, *, image: bool) -> torch.Tensor:
    if field.shape[-2:] != (512, 256):
        raise ValueError("Expected a 512x256 GAN field")
    if image:
        return torch.nn.functional.pad(field, (0, 0, 80, 80), mode="reflect")
    return torch.nn.functional.pad(field, (0, 0, 80, 80), mode="constant", value=0)


def build(config_path: Path, *, device_name: str | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["experiment_version"] != G2_2_VERSION:
        raise ValueError("Unexpected G2.2 experiment version")
    gan = config["gan"]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    pair_config = load_gan_training_pair_config(REPO_ROOT / gan["training_pair_config_path"])
    metadata = load_training_pair_manifest(REPO_ROOT, pair_config)
    dataset = GANTrainingPairDataset(
        metadata, REPO_ROOT, pair_config, split="train", length=int(gan["synthetic_sample_count"])
    )
    architecture = load_gan_architecture_config(REPO_ROOT / gan["architecture_config_path"])
    checkpoint_paths = {
        name: REPO_ROOT / relative for name, relative in gan["checkpoints"].items()
    }
    hashes_before = {name: file_sha256(path) for name, path in checkpoint_paths.items()}
    generators = {
        name: _checkpoint_generator(path, architecture, int(gan["expected_steps"][name]), device)
        for name, path in checkpoint_paths.items()
    }
    root = REPO_ROOT / gan["synthetic_root"]
    common_root = root / "common"
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in generators}
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata_row = sample.metadata
        if metadata_row["official_split"] != "train" or metadata_row["development_split"] != "train":
            raise RuntimeError("Non-training source entered G2.2 materialization")
        sample_id = f"g2-2-synthetic-{index:06d}"
        composite = sample.composite_image.unsqueeze(0).to(device)
        condition = sample.generator_mask.unsqueeze(0).to(device)
        valid = sample.fake_valid_mask.unsqueeze(0).to(device).bool()
        target = sample.fake_discriminator_mask.bool()
        mask_canvas = (_detector_canvas(target.float(), image=False)[0] * 255).byte().numpy()
        valid_canvas = (_detector_canvas(sample.fake_valid_mask, image=False)[0] * 255).byte().numpy()
        mask_path = common_root / "masks" / f"{sample_id}.png"
        valid_path = common_root / "valid" / f"{sample_id}.png"
        _atomic_png(mask_path, mask_canvas)
        _atomic_png(valid_path, valid_canvas)
        coarse_uint8 = ((_detector_canvas(sample.composite_image, image=True).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        coarse_sha = hashlib.sha256(coarse_uint8.tobytes()).hexdigest()
        shared = {
            "sample_id": sample_id,
            "sample_index": index,
            "deterministic_sample_seed": int(metadata_row["deterministic_sample_seed"]),
            "official_split": "train",
            "development_split": "train",
            "mask_path": mask_path.relative_to(REPO_ROOT).as_posix(),
            "mask_sha256": file_sha256(mask_path),
            "valid_region_path": valid_path.relative_to(REPO_ROOT).as_posix(),
            "valid_region_sha256": file_sha256(valid_path),
            "coarse_image_content_sha256": coarse_sha,
            "source_provenance": {
                "template": {
                    "sample_id": metadata_row["template_source_sample_id"],
                    "official_split": "train",
                    "development_split": "train",
                    "template_id": metadata_row["template_id"],
                },
                "background": {
                    "sample_id": metadata_row["normal_background_sample_id"],
                    "official_split": "train",
                    "development_split": "train",
                },
                "source_contact_sides": metadata_row["source_contact_sides"],
                "transformed_contact_sides": metadata_row["transformed_contact_sides"],
                "target_contact_sides": metadata_row["target_contact_sides"],
                "placement": metadata_row["placement"],
                "transform": metadata_row["transform"],
                "gan_manifest_content_sha256": metadata_row["gan_manifest_content_sha256"],
            },
        }
        with torch.inference_mode():
            for name, generator in generators.items():
                output = generator(composite, condition).refined_image
                # Detector labels are native-only. Reflection remains context, never content.
                output = torch.where(valid.expand_as(output), output, composite)
                canvas = _detector_canvas(output[0].cpu(), image=True)
                image_array = ((canvas.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
                image_path = root / name / "images" / f"{sample_id}.png"
                _atomic_png(image_path, image_array)
                checkpoint_path = checkpoint_paths[name]
                rows[name].append(
                    {
                        **shared,
                        "checkpoint_step": int(gan["expected_steps"][name]),
                        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
                        "checkpoint_sha256": hashes_before[name],
                        "image_path": image_path.relative_to(REPO_ROOT).as_posix(),
                        "image_sha256": file_sha256(image_path),
                    }
                )
    names = list(rows)
    assert_paired_manifests(rows[names[0]], rows[names[1]])
    hashes_after = {name: file_sha256(path) for name, path in checkpoint_paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("A frozen G2.1 checkpoint changed during synthetic materialization")
    manifest_dir = REPO_ROOT / gan["manifest_directory"]
    outputs = {}
    for name, manifest_rows in rows.items():
        document = {
            "experiment_version": G2_2_VERSION,
            "variant": name,
            "rows": manifest_rows,
            "row_count": len(manifest_rows),
            "official_test_source_count": 0,
            "detector_validation_source_count": 0,
            "g2_1_checkpoint_frozen": True,
        }
        document["content_sha256"] = canonical_sha256(document)
        path = manifest_dir / f"{name}.json"
        atomic_write_json(path, document)
        outputs[name] = {"path": path.relative_to(REPO_ROOT).as_posix(), "content_sha256": document["content_sha256"]}
    pairing = {
        "experiment_version": G2_2_VERSION,
        "paired_rows": len(dataset),
        "only_intended_difference": "frozen_generator_checkpoint_and_rendered_image",
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "manifests": outputs,
        "detector_validation_source_count": 0,
        "official_test_source_count": 0,
    }
    pairing["content_sha256"] = canonical_sha256(pairing)
    atomic_write_json(manifest_dir / "pairing_report.json", pairing)
    return pairing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/g2_2_detector_utility.json")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(build(args.config, device_name=args.device), indent=2))


if __name__ == "__main__":
    main()
