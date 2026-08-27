"""Two-row stage-wise metric plots.

Creates one wide, 8-panel figure:
- Top row: GCBF+
- Bottom row: InforMARL

Each row contains 4 metrics:
Test Accuracy, Precision, Recall, and F1-Score.

Controls:
- BACKGROUND_ALPHA: background opacity
- WSPACE: horizontal distance between panels
- HSPACE: vertical distance between the two rows
- FIGSIZE: overall horizontal stretch of the figure

Notes:
- Validation Accuracy is kept in the data but not plotted.
- Only the first subplot in each row shows y tick numbers.
- Metric titles appear only on the top row.
- No axis lines/spines are drawn.
- X tick labels appear only on the bottom row.
- Fonts are normal weight, with Times New Roman first and serif fallbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MultipleLocator

# -----------------------------
# Global typography
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.titlesize": 50,
    "axes.labelsize": 50,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "legend.fontsize": 50,
})

# -----------------------------
# Style controls
# -----------------------------
COLORS = {
    "Pre-Trained": "#E90000",
    "SFT": "#00b4d8",
    "Attention": "#0a0a0a",
    "Background": "#d5d6d5",
}

BACKGROUND_ALPHA = 0.3  # lower = more transparent, higher = less transparent

# Increase WSPACE to push columns farther apart.
# Increase HSPACE to push the two rows farther apart.
WSPACE = 0.05
HSPACE = 0.10

# Make the figure wider if x labels feel cramped.
FIGSIZE = (36, 8)

STAGES = [r"$P_{1}$", r"$P_{2}$", r"$P_{3}$", r"$P_{4}$"]
METRICS = ["Test Accuracy", "Precision", "Recall", "F1-Score"]

# Fixed y-axis settings for the top row
TOP_Y_LIM = (0.87, 1.01)
TOP_Y_TICKS = [0.88, 0.92, 0.96, 1.00]

# Fixed y-axis settings for the bottom row.
# Edit these manually when your bottom-row data changes.
BOTTOM_Y_LIM = (0.73, 1.01)
BOTTOM_Y_TICKS = [0.73, 0.82, 0.91, 1.00]

# Top row case study: GCBF+
GCBF_PLUS_DATA = {
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

# Bottom row case study: InforMARL
# Replace these values with your actual InforMARL results later.
INFORMARL_DATA = {
    "Pre-Trained": {
        "Validation Accuracy": [0.9992, 0.9994, 1.000, 0.9994],
        "Test Accuracy": [0.9994, 0.9957, 0.8593, 0.8374],
        "Precision": [1.000, 1.0000, 0.9408, 0.9230],
        "Recall": [0.9988, 0.9914, 0.7667, 0.7408],
        "F1-Score": [0.9993, 0.9985, 0.8448, 0.8219],
    },
    "SFT": {
        "Validation Accuracy": [0.9995, 0.9981, 0.9293, 0.9242],
        "Test Accuracy": [0.9997, 0.9984, 0.8649, 0.8519],
        "Precision": [0.9994, 1.0000, 0.9449, 0.9351],
        "Recall": [0.9988, 0.9972, 0.7741, 0.7498],
        "F1-Score": [0.9990, 0.9985, 0.8661, 0.8322],
    },
    "Attention": {
        "Validation Accuracy": [0.9950, 0.9931, 0.9990, 0.9796],
        "Test Accuracy": [0.9950, 0.9931, 0.9153, 0.9163],
        "Precision": [0.9941, 0.9959, 0.9554, 0.9499],
        "Recall": [0.9937, 0.9932, 0.8817, 0.8792],
        "F1-Score": [0.9938, 0.9945, 0.9170, 0.9131],
    },
}

CASE_STUDIES = [
    ("GCBF+", GCBF_PLUS_DATA),
    ("InforMARL", INFORMARL_DATA),
]

LINE_STYLE = {
    "linewidth": 4.0,
    "marker": "o",
    "markersize": 0.0,
}


def style_axis(ax, show_x_labels: bool, show_y_numbers: bool, y_lim, y_ticks) -> None:
    """Apply clean paper-style formatting."""
    bg_rgba = mcolors.to_rgba(COLORS["Background"], alpha=BACKGROUND_ALPHA)
    ax.set_facecolor(bg_rgba)
    ax.set_axisbelow(True)

    # Remove all boundaries.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # X axis: labels only where requested, no tick marks.
    x = np.arange(len(STAGES))
    ax.set_xlim(-0.2, len(STAGES) - 0.8) 
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES if show_x_labels else [])
    ax.tick_params(
        axis="x",
        which="both",
        bottom=show_x_labels,
        top=False,
        labelbottom=show_x_labels,
        length=0,
        pad=7,
    )

    # Y axis: fixed ticks, numbers only on the first subplot of each row.
    ax.set_ylim(*y_lim)
    ax.set_yticks(y_ticks)
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=show_y_numbers,
        labelright=False,
        length=0,
        pad=6,
    )

    # Denser grid boxes.
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(-0.5, len(STAGES), 0.5)))
    y_step = (y_lim[1] - y_lim[0]) / 12 if (y_lim[1] - y_lim[0]) > 0 else 0.01
    ax.yaxis.set_minor_locator(MultipleLocator(y_step))
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.50)
    #ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.45, alpha=0.45)


def plot_row(
    ax_row,
    row_label: str,
    data: dict,
    show_legend: bool,
    show_x_labels: bool,
    y_lim,
    y_ticks,
    show_titles: bool,
) -> None:
    """Plot one case-study row across the 4 metrics."""
    x = np.arange(len(STAGES))

    for idx, metric in enumerate(METRICS):
        ax = ax_row[idx]
        style_axis(
            ax,
            show_x_labels=show_x_labels,
            show_y_numbers=(idx == 0),
            y_lim=y_lim,
            y_ticks=y_ticks,
        )

        for variant in ["Pre-Trained", "SFT", "Attention"]:
            ax.plot(
                x,
                data[variant][metric],
                color=COLORS[variant],
                label=variant,
                **LINE_STYLE,
            )

        # Metric names only once: on the top row.
        if show_titles:
            ax.set_title(metric, pad=10, fontweight="normal")
        else:
            ax.set_title("")

        if idx == 0:
            # Row label pulled closer to the first panel.
            ax.text(
                -0.2,
                0.5,
                row_label,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=50,
                fontweight="normal",
                clip_on=False,
            )

            '''if show_legend:
                ax.legend(
                    loc="lower left",
                    framealpha=0.85,
                    edgecolor="black",
                    prop={"family": "Times New Roman", "weight": "normal", "size": 40},
                )'''


def plot_eight_panel(output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharex=True,
        sharey=False,
        gridspec_kw={"wspace": WSPACE, "hspace": HSPACE},
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS["Pre-Trained"],
            lw=4.0,
            marker="o",
            markersize=0.0,
            label="Pre-Trained",
        ),
        Line2D(
            [0], [0],
            color=COLORS["SFT"],
            lw=4.0,
            marker="o",
            markersize=0.0,
            label="SFT",
        ),
        Line2D(
            [0], [0],
            color=COLORS["Attention"],
            lw=4.0,
            marker="o",
            markersize=0.0,
            label="Probe-Informed Repair",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 1.2),
        ncol=3,
        frameon=False,
        fontsize=50,
        handlelength=2.0,
        columnspacing=2.0,
    )

    # Draw once so subplot positions are known.
    fig.canvas.draw()

    left_edge = axes[0, 0].get_position().x0
    right_edge = axes[1, -1].get_position().x1
    plot_center_x = (left_edge + right_edge) / 2.0

    # Figure-level y label.
    '''fig.text(
        left_edge - 0.07,
        0.5,
        "Evaluation Score",
        rotation=90,
        va="center",
        ha="center",
        fontsize=45,
        fontweight="normal",
    )'''

    # Top row.
    plot_row(
        axes[0, :],
        row_label="GCBF+",
        data=GCBF_PLUS_DATA,
        show_legend=True,
        show_x_labels=False,
        y_lim=TOP_Y_LIM,
        y_ticks=TOP_Y_TICKS,
        show_titles=True,
    )

    # Bottom row.
    plot_row(
        axes[1, :],
        row_label="InforMARL",
        data=INFORMARL_DATA,
        show_legend=False,
        show_x_labels=True,
        y_lim=BOTTOM_Y_LIM,
        y_ticks=BOTTOM_Y_TICKS,
        show_titles=False,
    )

    # Single x label centered over the plot block, not the full frame.
    fig.text(
        plot_center_x + 0.045,
        -0.02,
        "GNN Layers",
        va="center",
        ha="center",
        fontsize=50,
        fontweight="normal",
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.995,
        top=0.93,
        bottom=0.11,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    output_path = output_dir / "ProbeResults.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./corl_stage_metric_plots_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_eight_panel(output_dir)


if __name__ == "__main__":
    main()