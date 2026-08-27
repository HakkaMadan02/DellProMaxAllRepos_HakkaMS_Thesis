from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Editable colors
# ============================================================
COLORS = {
    "Pre-trained": "#880808",   # red
    "SFT": "#01696F",           # teal
    "Attention": "#7FD3D4",     # light teal
    "Axis": "#0A0A0A",
    "Grid": "#D9D9D9",
}

# ============================================================
# Data: 3 runs per environment
# Leave 35x35 as zeros for now
# ============================================================
env_labels = ["15x15", "20x20", "25x25", "30x30", "35x35", "40x40"]

data = {
    "Pre-trained": {
        "15x15": [6, 27, 26],
        "20x20": [36, 98, 29],
        "25x25": [86, 154, 60],
        "30x30": [116, 210, 97],
        "35x35": [0, 0, 0],
        "40x40": [145, 363, 141],
    },
    "SFT": {
        "15x15": [0, 18, 4],
        "20x20": [2, 23, 3],
        "25x25": [35, 48, 12],
        "30x30": [30, 61, 26],
        "35x35": [0, 0, 0],
        "40x40": [69, 115, 41],
    },
    "Attention": {
        "15x15": [0, 0, 0],
        "20x20": [0, 0, 0],
        "25x25": [0, 0, 0],
        "30x30": [0, 0, 0],
        "35x35": [0, 0, 0],
        "40x40": [0, 22, 0],
    },
}

# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["Axis"],
    "axes.labelcolor": COLORS["Axis"],
    "xtick.color": COLORS["Axis"],
    "ytick.color": COLORS["Axis"],
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "font.size": 16,
})

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(env_labels))
methods = ["Pre-trained", "SFT", "Attention"]
offsets = [-0.18, 0.0, 0.18]

all_means = []
all_lowers = []
all_uppers = []

for method, offset in zip(methods, offsets):
    means = np.array([np.mean(data[method][env]) for env in env_labels], dtype=float)
    stds = np.array([np.std(data[method][env], ddof=0) for env in env_labels], dtype=float)
    xpos = x + offset

    all_means.extend(means.tolist())
    all_lowers.extend((means - stds).tolist())
    all_uppers.extend((means + stds).tolist())

    ax.errorbar(
        xpos,
        means,
        yerr=stds,
        fmt='o',
        color=COLORS[method],
        ecolor=COLORS[method],
        elinewidth=2.8,
        capsize=4,
        capthick=2.0,
        markersize=8,
        markerfacecolor=COLORS[method],
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle='none',
        label=method,
        zorder=3,
    )

# ============================================================
# Axes formatting
# ============================================================
y_min_data = min(all_lowers)
y_max_data = max(all_means)  # keep top tight around the topmost mean

# very tight margins
lower_margin = max(2.0, 0.015 * max(y_max_data, 1.0))
upper_margin = max(1.0, 0.008 * max(y_max_data, 1.0))

ax.set_ylim(-2, y_max_data + upper_margin)
ax.set_xlim(-0.5, len(env_labels) - 0.5)

ax.set_xticks(x)
ax.set_xticklabels(env_labels)

ax.set_xlabel("Environment Size", fontsize=24, fontweight="normal")
ax.set_ylabel("Labeller Identified Collisions", fontsize=24, fontweight="normal")

ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22, color=COLORS["Grid"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    fontsize=18,
    loc="upper left",
    framealpha=0.9,
    edgecolor=COLORS["Axis"],
)

plt.tight_layout()

# ============================================================
# Save figure
# ============================================================
output_path = "labeller_collisions_case_study.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)