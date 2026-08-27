from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, MultipleLocator
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches


# -----------------------------
# Global typography
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.titlesize": 0,
    "axes.labelsize": 50,
    "xtick.labelsize": 45,
    "ytick.labelsize": 45,
    "legend.fontsize": 42,
})


# -----------------------------
# Style controls
# -----------------------------
COLORS = {
    "Pre-Trained": "#E90000",
    "Lr 1e-2": "#00b4d8",
    "Lr 5e-2": "#C2A14A",   # mustard / beige
    "Lr 1e-1": "#0a0a0a",
    "Background": "#d5d6d5",
}

BACKGROUND_ALPHA = 0.3
WSPACE = 0.22
FIGSIZE = (32, 15)

ENV_LABELS = ["8", "16", "24", "32"]
CATEGORIES = ["Pre-Trained", "Lr 1e-2", "Lr 5e-2", "Lr 1e-1"]

ENV_GROUP_SPACING = 1.2


# -----------------------------
# Fixed y-axis for Finish Rate
# -----------------------------
FINISH_Y_LIM = (-3, 106)
FINISH_Y_TICKS = [0, 25, 50, 75, 100]


# -----------------------------
# Optional fixed y-axis for |Q|
# Set to None to auto-compute
# Example:
Q_Y_LIM = (-10, 310)
Q_Y_TICKS = [0, 75, 150, 225, 300]
# -----------------------------
#Q_Y_LIM = None
#Q_Y_TICKS = None


# -----------------------------
# CSV files
# Plug in your 16 filenames here
# Each CSV is assumed to contain BOTH:
#   - labeled_unsafe_count
#   - finish_rate
# -----------------------------
CSV_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_OG_8.csv",
        "16": "episode_metrics_OG_16.csv",
        "24": "episode_metrics_OG_24.csv",
        "32": "episode_metrics_OG_32.csv",
    },
    "Lr 1e-2": {
        "8": "episode_metrics_Radial1e-2_8.csv",
        "16": "episode_metrics_Radial1e-2_16.csv",
        "24": "episode_metrics_Radial1e-2_24.csv",
        "32": "episode_metrics_Radial1e-2_32.csv",
    },
    "Lr 5e-2": {
        "8": "episode_metrics_Radial5e-2_8.csv",
        "16": "episode_metrics_Radial5e-2_16.csv",
        "24": "episode_metrics_Radial5e-2_24.csv",
        "32": "episode_metrics_Radial5e-2_32.csv",
    },
    "Lr 1e-1": {
        "8": "episode_metrics_Attention_8.csv",
        "16": "episode_metrics_Attention_16.csv",
        "24": "episode_metrics_Attention_24.csv",
        "32": "episode_metrics_Attention_32.csv",
    },
}


# -----------------------------
# Helpers
# -----------------------------
def load_csv_column(csv_path: str | Path, column_name: str) -> list[float]:
    df = pd.read_csv(csv_path)

    normalized_cols = {col.strip().lower(): col for col in df.columns}
    if column_name.lower() not in normalized_cols:
        raise ValueError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    actual_col = normalized_cols[column_name.lower()]
    values = df[actual_col].dropna().astype(float).tolist()

    if len(values) == 0:
        raise ValueError(f"No values found in column '{actual_col}' of {csv_path}")

    if len(values) != 32:
        print(f"Warning: {csv_path} has {len(values)} values in '{actual_col}', expected 32.")

    return values


def build_data_from_csv(
    csv_map: dict,
    env_labels: list[str],
    column_name: str,
    multiply_by_100: bool = False
) -> dict:
    data = {category: {} for category in CATEGORIES}

    for category in CATEGORIES:
        for env in env_labels:
            csv_file = csv_map[category][env]
            values = load_csv_column(csv_file, column_name)
            if multiply_by_100:
                values = [100.0 * x for x in values]
            data[category][env] = values

    return data


def compute_y_lim(data: dict) -> tuple[float, float]:
    all_means = []
    all_lowers = []

    for category in data.values():
        for vals in category.values():
            arr = np.array(vals, dtype=float)
            m = arr.mean()
            s = arr.std(ddof=0)
            all_means.append(m)
            all_lowers.append(m - s)

    y_max = max(all_means)
    y_min = min(all_lowers)
    return (-5, y_max + 5)


def auto_y_ticks(y_lim: tuple[float, float], n: int = 5) -> list[float]:
    y_max = y_lim[1]
    ticks = np.linspace(0, y_max, n)
    return [int(round(t)) for t in ticks]


def style_axis(ax, y_lim, y_ticks, env_labels) -> None:
    bg_rgba = mcolors.to_rgba(COLORS["Background"], alpha=BACKGROUND_ALPHA)
    ax.set_facecolor(bg_rgba)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.arange(len(env_labels))
    ax.set_xlim(-0.35, len(env_labels) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.tick_params(
        axis="x", which="both",
        bottom=False, top=False,
        labelbottom=True,
        length=0, pad=7,
    )

    ax.set_ylim(*y_lim)
    ax.set_yticks([t for t in y_ticks if t >= 0])

    ax.tick_params(
        axis="y", which="both",
        left=False, right=False,
        labelleft=True,
        labelright=False,
        length=0, pad=6,
    )

    ax.xaxis.set_minor_locator(FixedLocator(np.arange(-0.5, len(env_labels), 0.5)))
    y_span = y_lim[1] - y_lim[0]
    y_step = y_span / 12 if y_span > 0 else 0.01
    ax.yaxis.set_minor_locator(MultipleLocator(y_step))
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.50)


def plot_panel(ax, data: dict, title: str, y_lim, y_ticks, env_labels) -> None:
    style_axis(ax, y_lim, y_ticks, env_labels)

    x = np.arange(len(env_labels))
    offsets = [-0.27, -0.09, 0.09, 0.27]

    for category, offset in zip(CATEGORIES, offsets):
        means = np.array([np.mean(data[category][env]) for env in env_labels], dtype=float)
        stds = np.array([np.std(data[category][env], ddof=0) for env in env_labels], dtype=float)
        xpos = x + offset

        ax.errorbar(
            xpos, means, yerr=stds,
            fmt="o",
            color=COLORS[category],
            ecolor=COLORS[category],
            elinewidth=3.2,
            capsize=5,
            capthick=5.0,
            markersize=16,
            markerfacecolor=COLORS[category],
            markeredgecolor="white",
            markeredgewidth=1.2,
            linestyle="none",
            zorder=3,
        )

    ax.set_title(title, pad=10, fontweight="normal")


class VerticalLineHandler(HandlerBase):
    def __init__(self, color, marker_color):
        self.color = color
        self.marker_color = marker_color
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + width / 2
        y_bottom = ydescent - height * 0.8
        y_top = ydescent - height * -0.2
        y_mid = (y_bottom + y_top) / 2

        line = plt.Line2D(
            [cx, cx], [y_bottom, y_top],
            color=self.color, linewidth=5.0, transform=trans
        )
        dot = plt.Line2D(
            [cx], [y_mid],
            marker="o", markersize=16,
            color=self.marker_color,
            markerfacecolor=self.marker_color,
            markeredgecolor="white",
            linewidth=5, transform=trans
        )
        return [line, dot]


def plot_two_panel(output_dir: Path) -> None:
    q_data = build_data_from_csv(
        CSV_FILES, ENV_LABELS, "labeled_unsafe_count", multiply_by_100=False
    )

    finish_data = build_data_from_csv(
        CSV_FILES, ENV_LABELS, "finish_rate", multiply_by_100=True
    )

    q_y_lim = Q_Y_LIM if Q_Y_LIM is not None else compute_y_lim(q_data)
    q_y_ticks = Q_Y_TICKS if Q_Y_TICKS is not None else auto_y_ticks(q_y_lim)

    fig, axes = plt.subplots(
        1, 2,
        figsize=FIGSIZE,
        sharey=False,
        gridspec_kw={"wspace": WSPACE},
    )

    plot_panel(axes[0], q_data, r"|$\mathcal{Q}$|", q_y_lim, q_y_ticks, ENV_LABELS)
    plot_panel(axes[1], finish_data, "Finish Rate", FINISH_Y_LIM, FINISH_Y_TICKS, ENV_LABELS)

    axes[0].set_ylabel(r'|$\mathcal{Q}$|', fontsize=50, fontweight="normal")
    axes[1].set_ylabel("Finish Rate (%)", fontsize=50, fontweight="normal")

    axes[0].yaxis.set_label_coords(-0.12, 0.5)
    axes[1].yaxis.set_label_coords(-0.12, 0.5)

    fig.text(
        0.52, 0.05,
        "# Agents and # Obstacles",
        va="center", ha="center",
        fontsize=50, fontweight="normal",
    )

    dummy_handles = [
        mpatches.Patch(color=COLORS["Pre-Trained"], label="\nPre-Trained"),
        mpatches.Patch(color=COLORS["Lr 1e-2"], label="\nL.R: 1e-2"),
        mpatches.Patch(color=COLORS["Lr 5e-2"], label="\nL.R: 5e-2"),
        mpatches.Patch(color=COLORS["Lr 1e-1"], label="\nL.R: 1e-1"),
    ]

    handler_map = {
        dummy_handles[i]: VerticalLineHandler(COLORS[CATEGORIES[i]], COLORS[CATEGORIES[i]])
        for i in range(4)
    }

    fig.legend(
        handles=dummy_handles,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        fontsize=42,
        handlelength=1.0,
        handleheight=2.5,
        columnspacing=2.8,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        top=0.82,
        bottom=0.16,
        wspace=WSPACE,
    )

    output_path = output_dir / "AblationLR.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./corl_falsemodes_plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_two_panel(output_dir)


if __name__ == "__main__":
    main()