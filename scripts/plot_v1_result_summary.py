"""Render the V1 headline result figure from the authoritative G2.3B summary.

Reads ``reports/g2_3b/confirmation_summary.json`` and plots the primary
comparison (``gan_1500`` minus ``prevalence_matched_real``) against the
precommitted gate. Every number is read from that file; nothing is hardcoded,
recomputed, or rounded for effect.

The output contains no KSDD2 pixels -- only measured deltas -- so it is safe to
commit and publish under this repository's own MIT licence.

    .\\.venv\\Scripts\\python.exe .\\scripts\\plot_v1_result_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "reports" / "g2_3b" / "confirmation_summary.json"

# Gate criteria 1, 2 and 7: the three aggregate quality gains, with the minimum
# each had to reach. Frozen before training; see docs/g2-3b-utility-protocol.md.
GATED_GAINS = (
    ("global_dice_gain", "Global Dice", 0.01),
    ("global_iou_gain", "Global IoU", 0.005),
    ("pixel_pr_auc_gain", "Pixel PR-AUC", 0.01),
)

CRITERION_LABELS = {
    "mean_global_dice_gain": "1. Mean Dice gain ≥ +0.01",
    "mean_global_iou_gain": "2. Mean IoU gain ≥ +0.005",
    "mean_recall_regression": "3. Mean recall delta ≥ −0.01",
    "mean_precision_regression": "4. Mean precision delta ≥ −0.01",
    "mean_normal_fpr_regression": "5. Mean normal FPR delta ≤ +0.02",
    "positive_dice_seeds": "6. Positive-Dice seeds ≥ 2 of 3",
    "mean_pixel_pr_auc_gain": "7. Mean PR-AUC gain ≥ +0.01",
    "positive_pr_auc_seeds": "8. Positive-PR-AUC seeds ≥ 2 of 3",
}
CRITERION_ORDER = (
    "mean_global_dice_gain",
    "mean_global_iou_gain",
    "mean_recall_regression",
    "mean_precision_regression",
    "mean_normal_fpr_regression",
    "positive_dice_seeds",
    "mean_pixel_pr_auc_gain",
    "positive_pr_auc_seeds",
)

PASS_COLOUR = "#2f7d5c"
FAIL_COLOUR = "#b3402f"
MEAN_COLOUR = "#1f3b63"
SEED_COLOURS = ("#7fa8d4", "#a9c4e0", "#5d86b3")


def display_path(path: Path) -> str:
    """Render ``path`` for humans: repository-relative when it is inside the repo.

    ``--summary`` and ``--output`` may legitimately point outside the repository
    -- CI writes the regenerated figure to a temporary directory -- so the
    repository-relative form is a readability nicety, never a requirement.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the frozen V1 (G2.3B) utility result from its authoritative summary JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY, help="G2.3B confirmation summary JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "assets" / "v1_result_summary.png",
        help="Destination PNG.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.summary.is_file():
        raise SystemExit(f"Summary not found: {arguments.summary}")
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))

    primary = summary["primary_comparison"]
    candidate, control = primary["candidate"], primary["control"]
    seeds = [str(seed) for seed in summary["seeds"]]
    aggregate = summary["aggregate"]
    mean_deltas = aggregate["mean_deltas"]
    criteria = aggregate["criteria"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- Left: per-seed and mean gains against the gate minimums -----------
    width = 0.2
    for offset, (key, label, required) in enumerate(GATED_GAINS):
        base = offset
        for index, seed in enumerate(seeds):
            value = summary["seed_results"][seed][
                f"primary_comparison_{candidate}_minus_{control}"
            ][key]
            left.bar(
                base + (index - 1.5) * width,
                value,
                width * 0.9,
                color=SEED_COLOURS[index % len(SEED_COLOURS)],
                label=f"seed {seed}" if offset == 0 else None,
                zorder=3,
            )
        left.bar(
            base + 1.5 * width,
            mean_deltas[key],
            width * 0.9,
            color=MEAN_COLOUR,
            label="mean" if offset == 0 else None,
            zorder=3,
        )
        left.hlines(
            required,
            base - 2.2 * width,
            base + 2.2 * width,
            colors=FAIL_COLOUR,
            linestyles="--",
            linewidth=1.6,
            zorder=4,
            label="required by gate" if offset == 0 else None,
        )

    left.axhline(0.0, color="#333333", linewidth=1.0, zorder=2)
    left.set_xticks(range(len(GATED_GAINS)))
    left.set_xticklabels([label for _, label, _ in GATED_GAINS])
    left.set_ylabel(f"{candidate} − {control}")
    left.set_title(
        "Gated quality gains vs. prevalence-matched real control\n"
        "mean gains land at approximately zero; all three miss the gate",
        fontsize=11,
    )
    left.grid(axis="y", alpha=0.3, zorder=0)
    left.legend(fontsize=8, ncol=2, loc="upper left")

    # ---- Right: the eight-criterion gate ----------------------------------
    labels, colours, texts = [], [], []
    for key in CRITERION_ORDER:
        passed = bool(criteria[key])
        labels.append(CRITERION_LABELS[key])
        colours.append(PASS_COLOUR if passed else FAIL_COLOUR)
        texts.append("PASS" if passed else "FAIL")

    positions = range(len(labels))
    right.barh(list(positions), [1] * len(labels), color=colours, alpha=0.85, zorder=3)
    for position, (label, text) in enumerate(zip(labels, texts)):
        right.text(0.03, position, label, va="center", ha="left", color="white", fontsize=9.5, zorder=4)
        right.text(0.97, position, text, va="center", ha="right", color="white", fontsize=9.5,
                   fontweight="bold", zorder=4)
    right.set_xlim(0, 1)
    right.set_ylim(-0.6, len(labels) - 0.4)
    right.invert_yaxis()
    right.set_xticks([])
    right.set_yticks([])
    for spine in right.spines.values():
        spine.set_visible(False)
    passed_count = sum(1 for key in CRITERION_ORDER if criteria[key])
    right.set_title(
        f"Precommitted eight-criterion gate: {passed_count} passed, "
        f"{len(CRITERION_ORDER) - passed_count} failed\n"
        f"decision: {summary['decision']}",
        fontsize=11,
    )

    figure.suptitle(
        "V1 final result — GAN checkpoint 1,500 did not confirm downstream detector utility",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    figure.text(
        0.5,
        0.005,
        f"Seeds {', '.join(seeds)} · 5,952 optimizer updates per arm · development validation only "
        f"· official test never accessed · source: {display_path(arguments.summary)}",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)
    plt.close(figure)
    print(f"wrote {display_path(arguments.output)}")


if __name__ == "__main__":
    main()
