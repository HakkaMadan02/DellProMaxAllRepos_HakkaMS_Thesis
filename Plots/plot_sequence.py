"""
plot_sequence.py

Plots a 1×10 grid of environment snapshots starting at START_TIMESTEP,
covering timesteps [START_TIMESTEP, START_TIMESTEP + 9].

Usage (drop-in alongside test_mappo.py):
    Called from within the test() function, passing the rollout and env.
    Or run standalone after importing.

Hardcoded parameters:
    START_TIMESTEP : int  — set this to your desired starting timestep
"""

import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon

from model.utils.utils import tree_index  # matches test_mappo.py import path

# ─────────────────────────────────────────────
#  HARDCODED PARAMETERS  ← edit these
# ─────────────────────────────────────────────
START_TIMESTEP = 100   # first timestep to display
NUM_FRAMES     = 12   # always 10; produces START to START+9
# ─────────────────────────────────────────────

# colour palette (matches plot.py style)
_COLORS = {
    "agent":    "#7393B3",   # blue-grey
    "goal":     "#2fdd00",   # green
    "obstacle": "#0A0A0A",   # jet-black
    "edge_ag":  "#818589",   # gun-grey  (agent circle border)
    "edge_go":  "#818589",
}


def plot_env_sequence(
    T_graph,
    env,
    start_timestep: int = START_TIMESTEP,
    num_frames: int = NUM_FRAMES,
    save_dir: str = "./env_sequence_plots",
    filename: str | None = None,
) -> pathlib.Path:
    """
    Render `num_frames` consecutive environment snapshots in a single
    1 × num_frames figure.

    Parameters
    ----------
    T_graph        : GraphsTuple with a leading time dimension
                     (the rollout.Tp1_graph from test_mappo.py)
    env            : the DoubleIntegrator (or any MultiAgentEnv) instance
    start_timestep : first timestep to plot
    num_frames     : number of consecutive frames  (default 10)
    save_dir       : directory to write the PNG
    filename       : override the auto-generated filename if desired

    Returns
    -------
    pathlib.Path to the saved PNG
    """

    timesteps   = list(range(start_timestep, start_timestep + num_frames))
    num_agents  = env.num_agents
    car_radius  = env.params["car_radius"]
    area_size   = env.area_size
    n_rays      = env.params["n_rays"]

    # ── obstacle geometry is static; read once from t=0 ──────────────────────
    graph0  = tree_index(T_graph, 0)
    obs     = graph0.env_states.obstacle
    n_obs   = len(obs.center)
    obs_polys_template = [Polygon(obs.points[ii]) for ii in range(n_obs)]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, num_frames,
        figsize=(4.5 * num_frames, 4.8),
        constrained_layout=True,
    )
    if num_frames == 1:
        axes = [axes]

    for idx, t in enumerate(timesteps):
        ax = axes[idx]

        # ── fetch graph at timestep t ─────────────────────────────────────────
        graph = tree_index(T_graph, t)

        # agent nodes:  indices  [0 .. num_agents)
        # goal nodes:   indices  [num_agents .. 2*num_agents)
        # lidar nodes:  last  num_agents * n_rays  entries
        n_hits = num_agents * n_rays
        agent_states = graph.type_states(type_idx=0, n_type=num_agents)
        goal_states  = graph.type_states(type_idx=1, n_type=num_agents)
        agent_pos    = np.array(agent_states[:, :2])
        goal_pos     = np.array(goal_states[:, :2])

        # ── axes cosmetics ───────────────────────────────────────────────────
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_title(f"t = {t}", fontsize=10, fontweight="bold", pad=4)
        ax.set_xlabel("x", fontsize=8)
        if idx == 0:
            ax.set_ylabel("y", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

        # ── obstacles (static, re-created each frame for independent zorder) ─
        obs_patches = [Polygon(obs.points[ii]) for ii in range(n_obs)]
        obs_col = PatchCollection(
            obs_patches,
            facecolor=_COLORS["obstacle"],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            zorder=5,
        )
        ax.add_collection(obs_col)

        # ── goals (green circles, same radius as agents, drawn under agents) ─
        goal_circles = [
            Circle(
                xy=(float(goal_pos[a, 0]), float(goal_pos[a, 1])),
                radius=car_radius,
                facecolor=_COLORS["goal"],
                edgecolor="#1aa300",
                linewidth=0.8,
                alpha=0.9,
                zorder=6,
            )
            for a in range(num_agents)
        ]
        goal_col = PatchCollection(goal_circles, match_original=True, zorder=6)
        ax.add_collection(goal_col)

        # ── agents (blue-grey circles) ───────────────────────────────────────
        agent_circles = [
            Circle(
                xy=(float(agent_pos[a, 0]), float(agent_pos[a, 1])),
                radius=car_radius,
                facecolor=_COLORS["agent"],
                edgecolor=_COLORS["edge_ag"],
                linewidth=0.8,
                alpha=1.0,
                zorder=8,
            )
            for a in range(num_agents)
        ]
        agent_col = PatchCollection(agent_circles, match_original=True, zorder=8)
        ax.add_collection(agent_col)

    # ── global title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Environment snapshot  |  agents: {num_agents}  |  "
        f"area: {area_size}  |  obstacles: {n_obs}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    # ── legend (attach to last axis) ─────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_COLORS["agent"],   edgecolor=_COLORS["edge_ag"], label="Agent"),
        Patch(facecolor=_COLORS["goal"],    edgecolor="#1aa300",          label="Goal"),
        Patch(facecolor=_COLORS["obstacle"],edgecolor="black",            label="Obstacle"),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.8)

    # ── save ─────────────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    out_dir = pathlib.Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{stamp}_env_t{start_timestep}-{start_timestep + num_frames - 1}_n{num_agents}.png"
    out_path = out_dir / filename

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_env_sequence] saved → {out_path}")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: call this at the end of test() in test_mappo.py
#  by adding the two lines shown in the docstring below.
# ─────────────────────────────────────────────────────────────────────────────
def plot_from_rollout(rollout, env):
    """
    Thin wrapper so you can call from test_mappo.py with:

        from plot_sequence import plot_from_rollout
        plot_from_rollout(rollout, env)

    It uses the START_TIMESTEP hardcoded at the top of this file.
    """
    return plot_env_sequence(
        T_graph        = rollout.Tp1_graph,
        env            = env,
        start_timestep = START_TIMESTEP,
        num_frames     = NUM_FRAMES,
    )
