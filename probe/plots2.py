"""CoRL stage-wise metric plots.

Creates one wide, 4-panel figure for:
Test Accuracy, Precision, Recall, and F1-Score.

Notes:
- Validation Accuracy is retained in DATA but not plotted.
- Only the first subplot keeps the y-axis labels.
- Spines/ticks are removed for a cleaner paper-style look.
- Grid density is increased with minor grid lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MultipleLocator

plt.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "font.family": "DejaVu Sans",
})

# Editable hex colors
COLORS = {
    "Pre-Trained": "#E90000",
    "SFT": "#00b4d8",
    "Attention": "#67ff01",
    "Background": "#d5d6d5",
}

STAGES = ["Message", "Aggregate", "Update", "CBF-Head"]

# Keep all 5 in the data, but only plot 4 of them
METRICS = ["Test Accuracy", "Precision", "Recall", "F1-Score"]

DATA = {
    "Pre-Trained": {
        "Validation Accuracy": [0.9982, 0.9981, 0.9162, 0.9214],
        "Test Accuracy": [0.9980, 0.9986, 0.9193, 0.9204],
        "Precision": [0.9982, 1.0000, 0.9595, 0.9541],
        "Recall": [0.9978, 0.9972, 0.8858, 0.8833],
        "F1-Score": [0.9980, 0.9986, 0.9165, 0.9174],
    },
    "SFT": {
        "Validation Accuracy": [0.9995, 0.9981, 0.9293, 0.9242],
        "Test Accuracy": [0.9992, 0.9986, 0.9223, 0.9149],
        "Precision": [0.9994, 1.0000, 0.9440, 0.9515],
        "Recall": [0.9988, 0.9972, 0.8987, 0.8746],
        "F1-Score": [0.9992, 0.9986, 0.9204, 0.9113],
    },
    "Attention": {
        "Validation Accuracy": [0.9995, 0.9981, 0.9796, 0.9796],
        "Test Accuracy": [0.9992, 0.9986, 0.9778, 0.9778],
        "Precision": [0.9994, 1.0000, 0.9987, 0.9890],
        "Recall": [0.9988, 0.9972, 0.9577, 0.9663],
        "F1-Score": [0.9992, 0.9986, 0.9773, 0.9775],
    },
}

plt.rcParams.update({
    "font.family": "Times New Roman",

    "xtick.labelsize": 20,
    "ytick.labelsize": 20,

    "axes.titlesize": 24,
    "axes.labelsize": 24,

    "legend.fontsize": 16,
})

LINE_STYLE = {
    "linewidth": 2.4,
    "marker": "o",
    "markersize": 4.0,
}


def style_axis(ax, show_y_axis: bool) -> None:
    """Apply clean paper-style formatting."""
    ax.set_facecolor(COLORS["Background"])
    ax.set_axisbelow(True)

    # Remove all plot boundaries
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Hide tick marks everywhere
    ax.tick_params(axis="both", which="both", length=0)

    # X ticks: keep labels, but no tick marks
    x = np.arange(len(STAGES))
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, rotation=0)

    # Add denser grid boxes
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(-0.5, len(STAGES), 0.5)))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.80)
    ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.45, alpha=0.45)

    # Only the first subplot keeps the y-axis
    if show_y_axis:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["left"].set_color("black")
        ax.tick_params(axis="y", labelleft=True)
        ax.set_yticks([0.88,0.92, 0.96, 1.00])
        ax.set_ylim(0.875, 1.015)
    else:
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False, labelleft=False)


def plot_four_panel(output_dir: Path) -> None:
    x = np.arange(len(STAGES))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 4),
        sharey=False,
        gridspec_kw={"wspace": 0.03},
    )

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        style_axis(ax, show_y_axis=(idx == 0))

        for variant in ["Pre-Trained", "SFT", "Attention"]:
            ax.plot(
                x,
                DATA[variant][metric],
                color=COLORS[variant],
                label=variant,
                **LINE_STYLE,
            )

        ax.set_title(metric, fontsize=24, fontweight="normal", pad=10)
        ax.set_ylim(0.875, 1.015)

        if idx == 0:
            ax.set_ylabel("Score", fontsize=24, fontweight="normal")
            ax.legend(
                fontsize=16,
                loc="lower left",
                framealpha=0.85,
                edgecolor="black",
            )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.20, wspace=0.04)

    output_path = output_dir / "corl_stage_metrics_four_panel.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./corl_stage_metric_plots_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_four_panel(output_dir)


if __name__ == "__main__":
    main()