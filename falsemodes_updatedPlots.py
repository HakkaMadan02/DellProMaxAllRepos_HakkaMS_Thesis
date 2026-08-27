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
from matplotlib.lines import Line2D
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
    "Attention":   "#67ff01",
    "Background":  "#d5d6d5",
}

BACKGROUND_ALPHA = 0.3
WSPACE  = 0.2
FIGSIZE = (36, 8)

ENV_LABELS = ["15", "20", "25", "30", "35", "40"]

# -----------------------------
# Finish Rate y-axis (fixed)
# -----------------------------
FINISH_Y_LIM   = (-3, 103)
FINISH_Y_TICKS = [0, 25, 50, 75, 100]

FINISH_Y_LIM2 = (-5, 400)
FINISH_Y_TICKS2 = [0, 100, 200, 300, 400]
FINISH_Y_LIM3 = (-5, 1200)
FINISH_Y_TICKS3 = [0, 300, 600, 900, 1200]

# -----------------------------
# Data
# -----------------------------

# Unsafe Labeled Interactions: GCBF+
GCBF_UNSAFE_DATA = {
    "Pre-Trained": {
        "15": [6,   27,  26],
        "20": [36,  98,  29],
        "25": [86,  154, 60],
        "30": [116, 210, 97],
        "35": [154, 293, 123],
        "40": [145, 363, 141],
    },
    "SFT": {
        "15": [0,  18, 4],
        "20": [2,  23, 3],
        "25": [35, 48, 12],
        "30": [30, 61, 26],
        "35": [51,  75,  41],
        "40": [69, 115, 41],
    },
    "Attention": {
        "15": [0, 0,  0],
        "20": [0, 0,  0],
        "25": [0, 0,  0],
        "30": [0, 0,  0],
        "35": [0, 0,  0],
        "40": [0, 22, 0],
    },
}

# Unsafe Labeled Interactions: InforMARL (zeros for now)
INFORMARL_UNSAFE_DATA = {

    "Pre-Trained": {"15": [143, 334, 136],
                    "20": [289, 458, 255],
                    "25": [205, 482, 318],
                    "30": [532, 743, 423],
                    "35": [658, 975, 739],
                    "40": [977, 1256, 851]},
    "SFT":         {"15": [90, 158, 118],
                    "20": [240, 379, 279],
                    "25": [175, 431, 251],
                    "30": [516, 691, 397],
                    "35": [607, 917, 681],
                    "40": [958, 1128, 816]},
    "Attention":   {"15": [46, 52, 18],
                    "20": [87, 141, 75],
                    "25": [72, 215, 103],
                    "30": [165, 278, 143],
                    "35": [221, 322, 223],
                    "40": [281, 428, 266]},
                    }

# Finish Rate: GCBF+ (zeros for now)
GCBF_FINISH_DATA = {
    "Pre-Trained": {"15": [80, 80, 93],
                    "20": [75, 80, 82],
                    "25": [80, 80, 84],
                    "30": [76, 40, 80],
                    "35": [57, 52, 71],
                    "40": [50, 47, 67]},
    "SFT":         {"15": [73, 67, 87],
                    "20": [65, 35, 85],
                    "25": [60, 40, 68],
                    "30": [50, 23, 63],
                    "35": [52, 26, 60],
                    "40": [45, 25, 57]},
    "Attention":   {"15": [54, 34, 54],
                    "20": [60, 20, 55],
                    "25": [36, 16, 32],
                    "30": [34, 14, 34],
                    "35": [23, 12, 26],
                    "40": [23, 15, 15]},
}

# Finish Rate: InforMARL (zeros for now)
INFORMARL_FINISH_DATA = {
    "Pre-Trained": {"15": [100, 100, 100],
                    "20": [100, 100, 100],
                    "25": [88, 80, 84],
                    "30": [77, 67,73 ],
                    "35": [57, 65, 51],
                    "40": [45, 55, 37.5]},
    "SFT":         {"15": [93, 80, 93],
                    "20": [95, 85, 95],
                    "25": [80, 76, 80],
                    "30": [73, 67, 73],
                    "35": [54, 63, 48],
                    "40": [40, 50, 40]},
    "Attention":   {"15": [66, 73, 66],
                    "20": [65, 75, 65],
                    "25": [60, 72, 64],
                    "30": [50, 66, 56],
                    "35": [40, 54, 46],
                    "40": [30, 40, 28]},
}
# -----------------------------
# Helpers
# -----------------------------

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
    span = max(y_max - y_min, 1.0)
    return (-5, y_max + 5)


def auto_y_ticks(y_lim: tuple[float, float], n: int = 5) -> list[float]:
    y_max = y_lim[1]
    ticks = np.linspace(0, y_max, n)
    return [int(round(t)) for t in ticks]


def style_axis(ax, y_lim, y_ticks) -> None:
    """Apply probe-plot paper style. Y labels always shown."""
    bg_rgba = mcolors.to_rgba(COLORS["Background"], alpha=BACKGROUND_ALPHA)
    ax.set_facecolor(bg_rgba)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.arange(len(ENV_LABELS))
    ax.set_xlim(-0.3, len(ENV_LABELS) - 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_LABELS)
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

    ax.xaxis.set_minor_locator(FixedLocator(np.arange(-0.5, len(ENV_LABELS), 0.5)))
    y_span = y_lim[1] - y_lim[0]
    y_step = y_span / 12 if y_span > 0 else 0.01
    ax.yaxis.set_minor_locator(MultipleLocator(y_step))
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.50)


def plot_panel(ax, data: dict, title: str, y_lim, y_ticks) -> None:
    style_axis(ax, y_lim, y_ticks)

    x = np.arange(len(ENV_LABELS))
    offsets = [-0.18, 0.0, 0.18]

    for variant, offset in zip(["Pre-Trained", "SFT", "Attention"], offsets):
        means = np.array([np.mean(data[variant][env]) for env in ENV_LABELS], dtype=float)
        stds  = np.array([np.std( data[variant][env], ddof=0) for env in ENV_LABELS], dtype=float)
        xpos  = x + offset

        ax.errorbar(
            xpos, means, yerr=stds,
            fmt="o",
            color=COLORS[variant],
            ecolor=COLORS[variant],
            elinewidth=3.2,
            capsize=5,
            capthick=5.0,
            markersize=10,
            markerfacecolor=COLORS[variant],
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=3,
        )

    ax.set_title(title, pad=10, fontweight="normal")


def plot_four_panel(output_dir: Path) -> None:
    fig, axes = plt.subplots(
        1, 4,
        figsize=FIGSIZE,
        sharey=False,
        gridspec_kw={"wspace": WSPACE},
    )

    # Y limits
    unsafe_gcbf_lim       = compute_y_lim(GCBF_UNSAFE_DATA)
    unsafe_gcbf_ticks     = auto_y_ticks(unsafe_gcbf_lim)
    unsafe_infomarl_lim   = compute_y_lim(INFORMARL_UNSAFE_DATA)
    unsafe_infomarl_ticks = auto_y_ticks(unsafe_infomarl_lim)

    # Plot 1: Unsafe — GCBF+
    plot_panel(axes[0], GCBF_UNSAFE_DATA,     "GCBF+",     FINISH_Y_LIM2,     FINISH_Y_TICKS2)
    
    # Plot 2: Unsafe — InforMARL
    plot_panel(axes[1], INFORMARL_UNSAFE_DATA, "InforMARL", FINISH_Y_LIM3, FINISH_Y_TICKS3)
    
    # Plot 3: Finish Rate — GCBF+
    plot_panel(axes[2], GCBF_FINISH_DATA,     "GCBF+",     FINISH_Y_LIM,        FINISH_Y_TICKS)
    # Plot 4: Finish Rate — InforMARL
    plot_panel(axes[3], INFORMARL_FINISH_DATA, "InforMARL", FINISH_Y_LIM,        FINISH_Y_TICKS)

    # ------------------------------------------------------------------
    # Y-axis group labels via set_ylabel on left plot of each pair
    # ------------------------------------------------------------------
    axes[0].set_ylabel(r'|$\mathcal{Q}$|', fontsize=50, fontweight="normal")
    axes[2].set_ylabel("Finish Rate (%)", fontsize=50, fontweight="normal")
    axes[2].yaxis.set_label_coords(-0.11, 0.5)
    axes[0].yaxis.set_label_coords(-0.15, 0.5)
    # ------------------------------------------------------------------
    # Common x label
    # ------------------------------------------------------------------
    fig.canvas.draw()
    left_edge  = axes[0].get_position().x0
    right_edge = axes[3].get_position().x1
    plot_center_x = (left_edge + right_edge) / 2.0
    fig.text(
        0.55, -0.02,
        "Number of Agents",
        va="center", ha="center",
        fontsize=50, fontweight="normal",
    )

    # ------------------------------------------------------------------
    # Legend Block

    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    class VerticalLineHandler(HandlerBase):
        def __init__(self, color, marker_color):
            self.color = color
            self.marker_color = marker_color
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            cx = xdescent + width / 2
            # Shift the entire line+dot UP by bumping the y range
            y_bottom = ydescent - height * 0.75
            y_top    = ydescent - height * 0.05
            y_mid    = (y_bottom + y_top) / 2

            line = plt.Line2D([cx, cx], [y_bottom, y_top],
                            color=self.color, linewidth=5.0, transform=trans)
            dot  = plt.Line2D([cx], [y_mid],
                            marker="o", markersize=10,
                            color=self.marker_color,
                            markerfacecolor=self.marker_color,
                            markeredgecolor="white",
                            linewidth=5, transform=trans)
            return [line, dot]

    variants = ["Pre-Trained", "SFT", "Attention"]
    #dummy_handles = [mpatches.Patch(color=COLORS[v], label=v) for v in variants]
    dummy_handles = [
        mpatches.Patch(color=COLORS["Pre-Trained"], label="\nPre-Trained"),
        mpatches.Patch(color=COLORS["SFT"],         label="\nSFT          "),
        mpatches.Patch(color=COLORS["Attention"],   label="\nAttention   "),
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
    # Move plot 2 slightly left
    pos = axes[1].get_position()
    axes[1].set_position([
        pos.x0 - 0.004,   # move left/right
        pos.y0,
        pos.width,
        pos.height
    ])

    # Move plot 4 slightly right
    pos = axes[3].get_position()
    axes[3].set_position([
        pos.x0 - 0.008,
        pos.y0,
        pos.width,
        pos.height
    ])

    output_path = output_dir / "FalseModesResultsFINAL.png"
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