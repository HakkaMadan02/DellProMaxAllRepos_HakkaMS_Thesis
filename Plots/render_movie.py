"""
render_movie.py

Renders all timesteps of a rollout as an .mp4 movie.
Uses FuncAnimation (same pattern as DoubleIntegrator.render_video) but
is a standalone function that doesn't depend on the env's built-in renderer.

Usage — add these two lines at the end of the episode loop in test_mappo.py:

    from render_movie import render_movie
    render_movie(rollout, env)

Or call with full control:

    render_movie(rollout, env, fps=30, dpi=150, save_dir="./movies")
"""

import datetime
import pathlib

import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe for headless servers

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Patch

from model.utils.utils import tree_index          # same import path as test_mappo.py

# ─────────────────────────────────────────────
#  HARDCODED PARAMETERS  ← edit if desired
# ─────────────────────────────────────────────
FPS        = 15        # frames per second in the output video
DPI        = 150       # render resolution
SAVE_DIR   = "./movies"
# ─────────────────────────────────────────────

_COLORS = {
    "agent":    "#7393B3",   # blue-grey  (matches plot_sequence.py)
    "goal":     "#2fdd00",   # green
    "obstacle": "#0A0A0A",   # jet-black
    "bg":       "#F5F5F5",
}


def render_movie(
    rollout,
    env,
    fps: int     = FPS,
    dpi: int     = DPI,
    save_dir: str = SAVE_DIR,
    filename: str | None = None,
) -> pathlib.Path:
    """
    Render all timesteps in `rollout.Tp1_graph` as a single .mp4.

    Parameters
    ----------
    rollout   : RolloutResult from env.rollout_fn / rollout_fn_jitstep
    env       : DoubleIntegrator (or any MultiAgentEnv) instance
    fps       : output frame rate
    dpi       : figure resolution
    save_dir  : directory to write the .mp4
    filename  : override the auto-generated filename if desired

    Returns
    -------
    pathlib.Path to the saved .mp4
    """

    T_graph    = rollout.Tp1_graph
    num_agents = env.num_agents
    car_radius = env.params["car_radius"]
    area_size  = env.area_size
    n_rays     = env.params["n_rays"]

    # total number of graph snapshots (T+1, i.e. T steps + initial state)
    total_frames = T_graph.states.shape[0]

    # ── static obstacle geometry (read once from t=0) ────────────────────────
    graph0  = tree_index(T_graph, 0)
    obs     = graph0.env_states.obstacle
    n_obs   = len(obs.center)

    # ── figure / axes setup ──────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=dpi)
    fig.patch.set_facecolor(_COLORS["bg"])
    ax.set_facecolor(_COLORS["bg"])
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.tick_params(labelsize=8)

    # ── static: obstacles ────────────────────────────────────────────────────
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

    # ── static: goal circles (goals don't move) ──────────────────────────────
    goal_states = graph0.type_states(type_idx=1, n_type=num_agents)
    goal_pos    = np.array(goal_states[:, :2])
    goal_circles = [
        Circle(
            xy=(float(goal_pos[a, 0]), float(goal_pos[a, 1])),
            radius=car_radius,
            facecolor=_COLORS["goal"],
            edgecolor="#1aa300",
            linewidth=0.8,
            alpha=0.85,
            zorder=6,
        )
        for a in range(num_agents)
    ]
    goal_col = PatchCollection(goal_circles, match_original=True, zorder=6)
    ax.add_collection(goal_col)

    # ── dynamic: agent circles (updated every frame) ─────────────────────────
    agent_states_0 = graph0.type_states(type_idx=0, n_type=num_agents)
    agent_pos_0    = np.array(agent_states_0[:, :2])
    agent_circles = []
    for a in range(num_agents):
        c = Circle(
            xy=(float(agent_pos_0[a, 0]), float(agent_pos_0[a, 1])),
            radius=car_radius,
            facecolor=_COLORS["agent"],
            edgecolor="#5a7a99",
            linewidth=0.8,
            alpha=1.0,
            zorder=8,
        )
        ax.add_patch(c)          # add directly — set_center() works on patches
        agent_circles.append(c)

    # ── timestep label ───────────────────────────────────────────────────────
    title_text = ax.set_title(
        f"t = 0  |  agents: {num_agents}  |  obstacles: {n_obs}",
        fontsize=11, fontweight="bold", pad=6,
    )

    # ── legend ───────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=_COLORS["agent"],   edgecolor="#5a7a99", label="Agent"),
        Patch(facecolor=_COLORS["goal"],    edgecolor="#1aa300", label="Goal"),
        Patch(facecolor=_COLORS["obstacle"],edgecolor="black",   label="Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.85)

    fig.tight_layout()

    # ── animation callbacks ──────────────────────────────────────────────────
    def init_fn():
        return agent_circles + [title_text]

    def update(kk: int):
        graph     = tree_index(T_graph, kk)
        agent_pos = np.array(graph.type_states(type_idx=0, n_type=num_agents)[:, :2])

        for a in range(num_agents):
            agent_circles[a].set_center((float(agent_pos[a, 0]), float(agent_pos[a, 1])))

        title_text.set_text(
            f"t = {kk}  |  agents: {num_agents}  |  obstacles: {n_obs}"
        )
        return agent_circles + [title_text]

    # ── build animation ──────────────────────────────────────────────────────
    interval_ms = 1_000 / fps
    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init_fn,
        interval=interval_ms,
        blit=True,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    stamp   = datetime.datetime.now().strftime("%m%d-%H%M")
    out_dir = pathlib.Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{stamp}_movie_T{total_frames}_n{num_agents}.mp4"
    out_path = out_dir / filename

    # Auto-detect available writer: prefer ffmpeg, fall back to pillow (.gif)
    import shutil
    if shutil.which("ffmpeg"):
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        if out_path.suffix != ".mp4":
            out_path = out_path.with_suffix(".mp4")
    else:
        print("[render_movie] ffmpeg not found — falling back to PillowWriter (.gif)")
        writer = PillowWriter(fps=fps)
        out_path = out_path.with_suffix(".gif")

    ani.save(str(out_path), writer=writer)
    print(f"[render_movie] saved → {out_path}  ({total_frames} frames @ {fps} fps)")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper (drop into test_mappo.py episode loop)
# ─────────────────────────────────────────────────────────────────────────────
def render_movie_from_rollout(rollout, env):
    """
    Minimal wrapper:
        from render_movie import render_movie_from_rollout
        render_movie_from_rollout(rollout, env)
    """
    return render_movie(rollout, env)
