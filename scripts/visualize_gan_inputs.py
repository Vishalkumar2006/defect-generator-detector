"""Render a category-aware deterministic contact sheet of online GAN inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.gan import OnlineGANInputDataset, gan_rgb_to_uint8  # noqa: E402
from defectgen.gan.visualization import (  # noqa: E402
    CATEGORIES,
    select_visualization_members,
    summarize_placements,
)


def _rgb(tensor) -> np.ndarray:
    return gan_rgb_to_uint8(tensor).cpu().numpy()


def _active_sides(value: dict[str, bool]) -> str:
    active = [side for side, enabled in value.items() if enabled]
    return "+".join(active) if active else "none"


def _valid_boundary(axis, valid: np.ndarray) -> None:
    coordinates = np.argwhere(valid)
    if not len(coordinates):
        return
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    axis.add_patch(
        Rectangle(
            (left - 0.5, top - 0.5),
            right - left + 1,
            bottom - top + 1,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
        )
    )


def _zoom(image: np.ndarray, mask: np.ndarray, margin: int = 24) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return image
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    return image[
        max(0, top - margin) : min(len(image), bottom + margin + 1),
        max(0, left - margin) : min(image.shape[1], right + margin + 1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "reports" / "gan_inputs" / "manifest.json"
    )
    parser.add_argument("--category", choices=CATEGORIES, default="all")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        raise ValueError("Visualization count must be between 1 and 12")
    metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
    members = select_visualization_members(metadata, args.category)
    maximum_candidates = max(args.count * 50, args.count)
    dataset = OnlineGANInputDataset(
        metadata,
        REPO_ROOT,
        base_seed=args.seed,
        length=maximum_candidates,
        template_indices=members["template_indices"],
        normal_indices=members["normal_indices"],
    )
    samples = []
    rejections: list[str] = []
    for index in range(maximum_candidates):
        try:
            samples.append(dataset[index])
        except ValueError as error:
            rejections.append(str(error))
        if len(samples) == args.count:
            break
    if len(samples) != args.count:
        raise RuntimeError(
            f"Only {len(samples)} compatible samples found for category {args.category}; "
            f"rejections={sorted(set(rejections))}"
        )

    default_directory = REPO_ROOT / "reports" / "gan_inputs" / "visualizations"
    output = args.output or default_directory / f"gan_inputs_{args.category}_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(args.count, 8, figsize=(22, 3.8 * args.count), squeeze=False)
    headings = (
        "Normal + valid boundary",
        "Source template",
        "Condition mask",
        "Feather support",
        "Composite + valid boundary",
        "Absolute difference",
        "Defect zoom",
        "Provenance / checks",
    )
    for row_index, sample in enumerate(samples):
        provenance = sample["provenance"]
        background = _rgb(sample["normal_background"])
        template = _rgb(sample["source_template"])
        composite = _rgb(sample["coarse_composite"])
        condition = sample["conditioning_mask"][0].cpu().numpy().astype(bool)
        valid = sample["valid_region"][0].cpu().numpy().astype(bool)
        difference = sample["difference_from_background"].mean(0).cpu().numpy()
        axes[row_index, 0].imshow(background)
        _valid_boundary(axes[row_index, 0], valid)
        axes[row_index, 1].imshow(template)
        axes[row_index, 1].set_xlabel(
            "source: " + _active_sides(provenance["source_contact_sides"]), fontsize=8
        )
        axes[row_index, 2].imshow(condition, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(
            sample["feathered_support"][0].cpu().numpy(), cmap="magma", vmin=0, vmax=1
        )
        axes[row_index, 4].imshow(composite)
        _valid_boundary(axes[row_index, 4], valid)
        axes[row_index, 5].imshow(difference, cmap="inferno", vmin=0, vmax=1)
        axes[row_index, 6].imshow(_zoom(composite, condition))
        for column in range(7):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 7].text(
            0,
            1,
            "\n".join(
                [
                    f"seed: {provenance['generated_sample_seed']}",
                    f"background: {provenance['normal_background_sample_id']}",
                    f"source: {provenance['source_defect_sample_id']}",
                    f"component: {provenance['connected_component_id']}",
                    f"source sides: {_active_sides(provenance['source_contact_sides'])}",
                    f"transformed: {_active_sides(provenance['transformed_source_contact_sides'])}",
                    f"target sides: {_active_sides(provenance['target_contact_sides'])}",
                    f"scale: {provenance['scale']:.4f}",
                    f"flip H/V: {provenance['horizontal_flip']}/{provenance['vertical_flip']}",
                    f"translation: {provenance['translation']}",
                    f"max diff outside: {provenance['maximum_difference_outside_support']:.3g}",
                    f"support outside valid: {provenance['support_pixels_outside_valid_region']}",
                    f"accidental contacts: {provenance['accidental_contact_violations']}",
                ]
            ),
            va="top",
            family="monospace",
            fontsize=7.5,
        )
        axes[row_index, 7].axis("off")
        if row_index == 0:
            for column, heading in enumerate(headings):
                axes[row_index, column].set_title(heading, fontsize=10)
    figure.suptitle(f"Training-only GAN inputs — {args.category}", fontsize=14)
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)

    accounting = summarize_placements(samples, rejections)
    accounting.update(
        {
            "category": args.category,
            "templates_in_category": len(members["template_indices"]),
            "backgrounds_in_category": len(members["normal_indices"]),
        }
    )
    accounting_path = output.with_suffix(".accounting.json")
    accounting_path.write_text(json.dumps(accounting, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.count}-sample {args.category} contact sheet: {output}")
    print(f"Wrote placement accounting: {accounting_path}")
    print(json.dumps(accounting, indent=2))


if __name__ == "__main__":
    main()
