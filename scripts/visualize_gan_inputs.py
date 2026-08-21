"""Render a small deterministic contact sheet of online GAN inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan import OnlineGANInputDataset, gan_rgb_to_uint8  # noqa: E402


def _rgb(tensor) -> np.ndarray:
    return gan_rgb_to_uint8(tensor).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "reports" / "gan_inputs" / "manifest.json"
    )
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        raise ValueError("Visualization count must be between 1 and 12")
    metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset = OnlineGANInputDataset(metadata, REPO_ROOT, base_seed=args.seed, length=args.count)
    default_directory = REPO_ROOT / "reports" / "gan_inputs" / "visualizations"
    output = args.output or default_directory / "gan_input_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(args.count, 7, figsize=(19, 3.5 * args.count), squeeze=False)
    headings = (
        "Normal background",
        "Source template",
        "Condition mask",
        "Feather alpha",
        "Coarse composite",
        "Absolute difference",
        "Provenance",
    )
    for row_index in range(args.count):
        sample = dataset[row_index]
        panels = (
            (_rgb(sample["normal_background"]), None),
            (_rgb(sample["source_template"]), None),
            (sample["conditioning_mask"][0].cpu().numpy(), "gray"),
            (sample["feathered_support"][0].cpu().numpy(), "magma"),
            (_rgb(sample["coarse_composite"]), None),
            (sample["difference_from_background"].mean(0).cpu().numpy(), "inferno"),
        )
        for column, (panel, colour_map) in enumerate(panels):
            axes[row_index, column].imshow(
                panel,
                cmap=colour_map,
                vmin=0 if colour_map else None,
                vmax=1 if colour_map else None,
            )
            axes[row_index, column].axis("off")
        provenance = sample["provenance"]
        axes[row_index, 6].text(
            0,
            1,
            "\n".join(
                [
                    f"seed: {provenance['generated_sample_seed']}",
                    f"background: {provenance['normal_background_sample_id']}",
                    f"source: {provenance['source_defect_sample_id']}",
                    f"component: {provenance['connected_component_id']}",
                    f"scale: {provenance['scale']:.4f}",
                    f"flip H/V: {provenance['horizontal_flip']}/{provenance['vertical_flip']}",
                    f"translation: {provenance['translation']}",
                    f"partial: {provenance['partial_component']}",
                    f"coverage: {provenance['coverage_fraction']:.4f}",
                    f"border: {provenance['touches_native_border']}",
                ]
            ),
            va="top",
            family="monospace",
            fontsize=8,
        )
        axes[row_index, 6].axis("off")
        for column, heading in enumerate(headings):
            if row_index == 0:
                axes[row_index, column].set_title(heading, fontsize=10)
    figure.suptitle("Training-only deterministic online GAN inputs", fontsize=14)
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.count}-sample contact sheet: {output}")
    print("This command does not write individual generated patches.")


if __name__ == "__main__":
    main()
