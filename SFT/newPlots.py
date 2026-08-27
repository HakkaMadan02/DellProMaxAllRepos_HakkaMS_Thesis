"""One-row environment-size metric plots.

Creates one wide, 4-panel figure (single row):
- Plot 1: Unsafe Labeled Interactions — GCBF+
- Plot 2: Unsafe Labeled Interactions — InforMARL
- Plot 3: Finish Rate — GCBF+
- Plot 4: Finish Rate — InforMARL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    "xtick.labelsize": 45,
    "ytick.labelsize": 45,
    "legend.fontsize": 50,
})

# -----------------------------
# Style controls
# -----------------------------
COLORS = {
    "Pre-Trained": "#E90000",
    "SFT":         "#00b4d8",
    #"Attention":   "#67ff01",
    "Attention":   "#0a0a0a",
    "Background":  "#d5d6d5",
}

BACKGROUND_ALPHA = 0.3
WSPACE  = 0.2
FIGSIZE = (36, 8)

ENV_LABELS_2 = ["8", "16", "24", "32"]

# -----------------------------
# Finish Rate y-axis (fixed)
# -----------------------------
FINISH_Y_LIM   = (-3, 103)
FINISH_Y_TICKS = [0, 25, 50, 75, 100]

FINISH_Y_LIM2 = (-15, 410)
FINISH_Y_TICKS2 = [0, 100, 200, 300, 400]
FINISH_Y_LIM3 = (-50, 1650)
FINISH_Y_TICKS3 = [0, 400, 800, 1200, 1600]

# -----------------------------
# CSV file placeholders for GCBF+
# -----------------------------
GCBF_UNSAFE_CSV_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_OG_8.csv",
        "16": "episode_metrics_OG_16.csv",
        "24": "episode_metrics_OG_24.csv",
        "32": "episode_metrics_OG_32.csv",
    },
    "SFT": {
        "8": "episode_metrics_SFT_8.csv",
        "16": "episode_metrics_SFT_16.csv",
        "24": "episode_metrics_SFT_24.csv",
        "32": "episode_metrics_SFT_32.csv",
    },
    "Attention": {
        "8": "episode_metrics_Attention_8.csv",
        "16": "episode_metrics_Attention_16.csv",
        "24": "episode_metrics_Attention_24.csv",
        "32": "episode_metrics_Attention_32.csv",
    },
}

GCBF_FINISH_CSV_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_OG_8.csv",
        "16": "episode_metrics_OG_16.csv",
        "24": "episode_metrics_OG_24.csv",
        "32": "episode_metrics_OG_32.csv",
    },
    "SFT": {
        "8": "episode_metrics_SFT_8.csv",
        "16": "episode_metrics_SFT_16.csv",
        "24": "episode_metrics_SFT_24.csv",
        "32": "episode_metrics_SFT_32.csv",
    },
    "Attention": {
        "8": "episode_metrics_Attention_8.csv",
        "16": "episode_metrics_Attention_16.csv",
        "24": "episode_metrics_Attention_24.csv",
        "32": "episode_metrics_Attention_32.csv",
    },
}

# -----------------------------
# CSV file placeholders for InforMARL / MAPPO
# -----------------------------
INFORMARL_UNSAFE_CSV_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_MAPPO_OG_8.csv",
        "16": "episode_metrics_MAPPO_OG_16.csv",
        "24": "episode_metrics_MAPPO_OG_24.csv",
        "32": "episode_metrics_MAPPO_OG_32.csv",
    },
    "SFT": {
        "8": "episode_metrics_MAPPO_SFT_8.csv",
        "16": "episode_metrics_MAPPO_SFT_16.csv",
        "24": "episode_metrics_MAPPO_SFT_24.csv",
        "32": "episode_metrics_MAPPO_SFT_32.csv",
    },
    "Attention": {
        "8": "episode_metrics_MAPPO_Attention_8.csv",
        "16": "episode_metrics_MAPPO_Attention_16.csv",
        "24": "episode_metrics_MAPPO_Attention_24.csv",
        "32": "episode_metrics_MAPPO_Attention_32.csv",
    },
}

INFORMARL_FINISH_CSV_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_MAPPO_OG_8.csv",
        "16": "episode_metrics_MAPPO_OG_16.csv",
        "24": "episode_metrics_MAPPO_OG_24.csv",
        "32": "episode_metrics_MAPPO_OG_32.csv",
    },
    "SFT": {
        "8": "episode_metrics_MAPPO_SFT_8.csv",
        "16": "episode_metrics_MAPPO_SFT_16.csv",
        "24": "episode_metrics_MAPPO_SFT_24.csv",
        "32": "episode_metrics_MAPPO_SFT_32.csv",
    },
    "Attention": {
        "8": "episode_metrics_MAPPO_Attention_8.csv",
        "16": "episode_metrics_MAPPO_Attention_16.csv",
        "24": "episode_metrics_MAPPO_Attention_24.csv",
        "32": "episode_metrics_MAPPO_Attention_32.csv",
    },
}

# -----------------------------
# Data containers
# -----------------------------
GCBF_UNSAFE_DATA = {
    "Pre-Trained": {"8": [], "16": [], "24": [], "32": []},
    "SFT":         {"8": [], "16": [], "24": [], "32": []},
    "Attention":   {"8": [], "16": [], "24": [], "32": []},
}

GCBF_FINISH_DATA = {
    "Pre-Trained": {"8": [], "16": [], "24": [], "32": []},
    "SFT":         {"8": [], "16": [], "24": [], "32": []},
    "Attention":   {"8": [], "16": [], "24": [], "32": []},
}

INFORMARL_UNSAFE_DATA = {
    "Pre-Trained": {"8": [], "16": [], "24": [], "32": []},
    "SFT":         {"8": [], "16": [], "24": [], "32": []},
    "Attention":   {"8": [], "16": [], "24": [], "32": []},
}

INFORMARL_FINISH_DATA = {
    "Pre-Trained": {"8": [], "16": [], "24": [], "32": []},
    "SFT":         {"8": [], "16": [], "24": [], "32": []},
    "Attention":   {"8": [], "16": [], "24": [], "32": []},
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

    if len(values) != 32:
        print(f"Warning: {csv_path} has {len(values)} values in '{actual_col}', expected 32.")

    return values


def build_data_from_csv(csv_map: dict, env_labels: list[str], column_name: str, multiply_by_100: bool = False) -> dict:
    data = {
        "Pre-Trained": {},
        "SFT": {},
        "Attention": {},
    }

    for variant in ["Pre-Trained", "SFT", "Attention"]:
        for env in env_labels:
            csv_file = csv_map[variant][env]
            values = load_csv_column(csv_file, column_name)
            if multiply_by_100:
                values = [100.0 * x for x in values]
            data[variant][env] = values

    return data


def compute_y_lim(data: dict) -> tuple[float, float]:
    all_means, all_lowers = [], []
    for variant in data.values():
        for vals in variant.values():
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
    ax.set_xlim(-0.3, len(env_labels) - 0.7)
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
    offsets = [-0.18, 0.0, 0.18]

    for variant, offset in zip(["Pre-Trained", "SFT", "Attention"], offsets):
        means = np.array([np.mean(data[variant][env]) for env in env_labels], dtype=float)
        stds  = np.array([np.std(data[variant][env], ddof=0) for env in env_labels], dtype=float)
        xpos  = x + offset

        ax.errorbar(
            xpos, means, yerr=stds,
            fmt="o",
            color=COLORS[variant],
            ecolor=COLORS[variant],
            elinewidth=3.2,
            capsize=5,
            capthick=5.0,
            markersize=16,
            markerfacecolor=COLORS[variant],
            markeredgecolor="white",
            markeredgewidth=1.2,
            linestyle="none",
            zorder=3,
        )

    ax.set_title(title, pad=10, fontweight="normal")


def plot_four_panel(output_dir: Path) -> None:
    gcbf_unsafe_data = build_data_from_csv(
        GCBF_UNSAFE_CSV_FILES, ENV_LABELS_2, "labeled_unsafe_count", multiply_by_100=False
    )
    gcbf_finish_data = build_data_from_csv(
        GCBF_FINISH_CSV_FILES, ENV_LABELS_2, "finish_rate", multiply_by_100=True
    )
    informarl_unsafe_data = build_data_from_csv(
        INFORMARL_UNSAFE_CSV_FILES, ENV_LABELS_2, "labeled_unsafe_count", multiply_by_100=False
    )
    informarl_finish_data = build_data_from_csv(
        INFORMARL_FINISH_CSV_FILES, ENV_LABELS_2, "finish_rate", multiply_by_100=True
    )

    fig, axes = plt.subplots(
        1, 4,
        figsize=FIGSIZE,
        sharey=False,
        gridspec_kw={"wspace": WSPACE},
    )

    plot_panel(axes[0], gcbf_unsafe_data,      "GCBF+",     FINISH_Y_LIM2, FINISH_Y_TICKS2, ENV_LABELS_2)
    plot_panel(axes[1], informarl_unsafe_data, "InforMARL", FINISH_Y_LIM3, FINISH_Y_TICKS3, ENV_LABELS_2)
    plot_panel(axes[2], gcbf_finish_data,      "GCBF+",     FINISH_Y_LIM,  FINISH_Y_TICKS,  ENV_LABELS_2)
    plot_panel(axes[3], informarl_finish_data, "InforMARL", FINISH_Y_LIM,  FINISH_Y_TICKS,  ENV_LABELS_2)

    axes[0].set_ylabel(r'|$\mathcal{Q}$|', fontsize=50, fontweight="normal")
    axes[2].set_ylabel("Finish Rate (%)", fontsize=50, fontweight="normal")
    axes[2].yaxis.set_label_coords(-0.11, 0.5)
    axes[0].yaxis.set_label_coords(-0.15, 0.5)

    fig.canvas.draw()
    left_edge  = axes[0].get_position().x0
    right_edge = axes[3].get_position().x1
    plot_center_x = (left_edge + right_edge) / 2.0
    fig.text(
        0.55, -0.02,
        "# Agents and # Obstacles",
        va="center", ha="center",
        fontsize=50, fontweight="normal",
    )

    from matplotlib.legend_handler import HandlerBase
    import matplotlib.patches as mpatches

    class VerticalLineHandler(HandlerBase):
        def __init__(self, color, marker_color):
            self.color = color
            self.marker_color = marker_color
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            cx = xdescent + width / 2
            y_bottom = ydescent - height * 0.75
            y_top    = ydescent - height * 0.05
            y_mid    = (y_bottom + y_top) / 2

            line = plt.Line2D([cx, cx], [y_bottom, y_top],
                              color=self.color, linewidth=5.0, transform=trans)
            dot  = plt.Line2D([cx], [y_mid],
                              marker="o", markersize=16,
                              color=self.marker_color,
                              markerfacecolor=self.marker_color,
                              markeredgecolor="white",
                              linewidth=5, transform=trans)
            return [line, dot]

    variants = ["Pre-Trained", "SFT", "Attention"]
    dummy_handles = [
        mpatches.Patch(color=COLORS["Pre-Trained"], label="\nPre-Trained"),
        mpatches.Patch(color=COLORS["SFT"],         label="\nSFT          "),
        mpatches.Patch(color=COLORS["Attention"],   label="\nProbe-Informed Repair   "),
    ]
    handler_map = {
        dummy_handles[i]: VerticalLineHandler(COLORS[variants[i]], COLORS[variants[i]])
        for i in range(3)
    }

    fig.legend(
        handles=dummy_handles,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(plot_center_x, 1.26),
        ncol=3,
        frameon=False,
        fontsize=50,
        handlelength=1.0,
        handleheight=2.5,
        columnspacing=5.0,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        top=0.88,
        bottom=0.13,
        wspace=WSPACE,
    )

    pos = axes[1].get_position()
    axes[1].set_position([
        pos.x0 - 0.004,
        pos.y0,
        pos.width,
        pos.height
    ])

    pos = axes[3].get_position()
    axes[3].set_position([
        pos.x0 - 0.008,
        pos.y0,
        pos.width,
        pos.height
    ])

    output_path = output_dir / "FalseModesResultsFINAL.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./corl_falsemodes_plots")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_four_panel(output_dir)


if __name__ == "__main__":
    main()