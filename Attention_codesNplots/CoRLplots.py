"""KG_Plots.py

Synthetic probe plots for attention vs distance and attention vs radial velocity.
No rollout is required: the script constructs controlled synthetic graphs,
queries the trained model for attention weights, and saves publication-style
figures using the same layout as the existing attention plots.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import yaml

from gcbfplus.algo import CentralizedCBF, DecShareCBF, GCBF, GCBFPlus, make_algo
from gcbfplus.env import make_env
from gcbfplus.utils.graph import GraphsTuple

plt.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

COLORS = {
    "agent": "#0a0a0a",
    "goal": "#818589",
    "obstacle": "#880808",
    "black": "#000000",
}

PLOT_PARAMS = {
    "agent": {"color": COLORS["agent"], "marker": "o", "label": "Agent", "size": 100},
    "goal": {"color": COLORS["goal"], "marker": "o", "label": "Goal", "size": 100},
    "obstacle": {"color": COLORS["obstacle"], "marker": "o", "label": "Obstacle", "size": 100},
    "obstacle_y": {"color": COLORS["black"], "marker": "s", "label": "Obstacle-Y", "size": 100},
}


def load_config(path: Path):
    with open(path / "config.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_env_and_algo(args):
    """Create env and load the trained algo (or initialize from args)."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)

    if args.path is not None:
        config = load_config(Path(args.path))
    else:
        config = None

    num_agents = config.num_agents if args.num_agents is None and config is not None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None and config is not None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    if args.path is not None:
        model_path = Path(args.path) / "models"
        if args.step is None:
            models = [p.name for p in model_path.iterdir() if p.name.isdigit()]
            step = max(int(m) for m in models)
        else:
            step = args.step

        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            gnn_layers=config.gnn_layers,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            horizon=config.horizon,
            lr_actor=config.lr_actor,
            lr_cbf=config.lr_cbf,
            alpha=config.alpha,
            eps=0.02,
            inner_epoch=8,
            loss_action_coef=config.loss_action_coef,
            loss_unsafe_coef=config.loss_unsafe_coef,
            loss_safe_coef=config.loss_safe_coef,
            loss_h_dot_coef=config.loss_h_dot_coef,
            max_grad_norm=2.0,
            seed=config.seed,
        )
        algo.load(str(model_path), step)
    else:
        algo = make_algo(
            algo=args.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            alpha=args.alpha,
        )

    if not hasattr(algo, "get_attention_weights"):
        raise RuntimeError(
            "The selected algorithm does not expose get_attention_weights(). "
            "This script is intended for attention-enabled models."
        )

    return env, algo


def make_probe_template(env, seed: int = 0) -> GraphsTuple:
    """Create a valid graph template so env_states has the correct structure."""
    return env.reset(jr.PRNGKey(seed))


def build_synthetic_probe_graph(
    template_graph: GraphsTuple,
    n_agents: int,
    agent_id: int,
    probe_values: np.ndarray,
    probe_kind: str,
    fixed_distance: float = 0.10,
    distance_axis: str = "x",
) -> Tuple[GraphsTuple, np.ndarray]:
    """Create a graph with one receiver agent and homogeneous obstacle probe nodes."""

    noise_std_pos = 0.02
    noise_std_vel = 0.05
    rng = np.random.default_rng(0)

    values = np.asarray(probe_values, dtype=np.float32)
    n_probe = int(values.shape[0])

    node_dim = int(template_graph.nodes.shape[-1])
    state_dim = int(template_graph.states.shape[-1])
    edge_dim = int(template_graph.edges.shape[-1])

    n_nodes = n_agents + n_probe
    nodes = np.zeros((n_nodes, node_dim), dtype=np.float32)
    states = np.zeros((n_nodes, state_dim), dtype=np.float32)
    node_type = np.zeros((n_nodes,), dtype=np.int32)

    # Agent nodes first, obstacle probe nodes after.
    nodes[:n_agents, 2] = 1.0
    node_type[:n_agents] = 0

    nodes[n_agents:, 0] = 1.0
    node_type[n_agents:] = 2

    # Keep the target agent as the receiver for all probe edges.
    senders = np.arange(n_agents, n_agents + n_probe, dtype=np.int32)
    receivers = np.full((n_probe,), int(agent_id), dtype=np.int32)

    edges = np.zeros((n_probe, edge_dim), dtype=np.float32)

    if probe_kind == "distance":
        # Sweep distance either along x or along y while keeping velocity noisy as before.
        if distance_axis == "x":
            edges[:, 0] = values
            edges[:, 1] = 0.0
        elif distance_axis == "y":
            edges[:, 0] = 0.0
            edges[:, 1] = values
        else:
            raise ValueError(f"Unknown distance_axis: {distance_axis}")

        edges[:, 2] = rng.normal(0, noise_std_vel, size=n_probe)
        edges[:, 3] = rng.normal(0, noise_std_vel, size=n_probe)

    elif probe_kind == "radial_velocity":
        # Fix distance and vary radial velocity linearly.
        edges[:, 0] = fixed_distance + 0.75 * rng.normal(0, noise_std_pos, size=n_probe)
        edges[:, 1] = rng.normal(0, noise_std_pos, size=n_probe)
        edges[:, 2] = values
        edges[:, 3] = rng.normal(0, noise_std_vel, size=n_probe)

    else:
        raise ValueError(f"Unknown probe_kind: {probe_kind}")

    graph = template_graph._replace(
        nodes=jnp.array(nodes),
        edges=jnp.array(edges),
        states=jnp.array(states),
        n_node=jnp.array(n_nodes),
        n_edge=jnp.array(n_probe),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        node_type=jnp.array(node_type),
    )

    return graph, values


def get_probe_features(probe_kind: str, args, env) -> Tuple[np.ndarray, str, str, float, float, str]:
    """Return x-axis samples and labels for the synthetic probe."""
    n_points = args.probe_points

    if probe_kind == "distance":
        max_d = args.distance_max if args.distance_max is not None else float(env.params.get("comm_radius", 0.5))
        xs = np.linspace(args.distance_min, max_d, n_points, dtype=np.float32)
        xlabel = r"Distance $d_{ij}$"
        title = "Attention vs Distance"
        file_tag = "distance"
        xlim_min = -0.05
        xlim_max = 0.7
    elif probe_kind == "radial_velocity":
        max_r = args.radial_max
        xs = np.linspace(-max_r, max_r, n_points, dtype=np.float32)
        xlabel = r"Radial Velocity $r_{ij}$"
        title = "Attention vs Radial Velocity"
        file_tag = "radial_velocity"
        xlim_min = -0.75
        xlim_max = 0.75
    else:
        raise ValueError(probe_kind)

    return xs, xlabel, title, xlim_min, xlim_max, file_tag


def plot_probe_attention(
    x_values: np.ndarray,
    attn_weights_per_head: np.ndarray,
    title: str,
    xlabel: str,
    agent_id: int,
    n_agents: int,
    n_obstacles: int,
    save_path: Path,
    legend_label: str,
    xlim_min: float,
    xlim_max: float,
    sensing_radius: float | None = None,
):
    """One image per head, same style as the existing attention plots."""
    n_heads = int(attn_weights_per_head.shape[0])

    for k in range(n_heads):
        head_attn = np.asarray(attn_weights_per_head[k])
        max_attn = max(float(head_attn.max()) * 1.15, 0.1) if len(head_attn) > 0 else 1.0

        fig, ax = plt.subplots(figsize=(10, 9))

        ax.scatter(
            x_values,
            head_attn,
            alpha=0.85,
            s=PLOT_PARAMS["obstacle"]["size"],
            c=PLOT_PARAMS["obstacle"]["color"],
            marker=PLOT_PARAMS["obstacle"]["marker"],
            label=legend_label,
            edgecolors="black",
            linewidth=0.8,
        )

        if sensing_radius is not None:
            ax.axvline(
                x=sensing_radius,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Sensing Radius",
            )

        ax.set_xlabel(xlabel, fontsize=30, fontweight="normal")
        ax.set_ylabel(r"Attention Weight $w_{ij}$", fontsize=30, fontweight="normal")
        ax.set_title("")

        ax.text(
            0.5,
            1.11,
            f"Head {k} — {title}",
            transform=ax.transAxes,
            fontsize=30,
            fontweight="normal",
            ha="center",
        )

        ax.text(
            0.5,
            1.04,
            f"Env: ({n_agents} agents, {n_obstacles} obstacles) | Timestep: 0 | $Agent_{{i}}$: {agent_id}",
            transform=ax.transAxes,
            fontsize=22,
            fontweight="normal",
            ha="center",
        )

        ax.legend(fontsize=24, loc="upper right", framealpha=0.75, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(-0.05, max_attn)

        plt.tight_layout(rect=[0, 0, 1, 0.85])

        out_path = save_path.with_name(
            f"{save_path.stem}_{title.lower().replace(' ', '_')}_head{k}{save_path.suffix}"
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved to: {out_path}")
        plt.close(fig)


def plot_distance_probe_attention_xy(
    x_values: np.ndarray,
    attn_weights_x_per_head: np.ndarray,
    attn_weights_y_per_head: np.ndarray,
    title: str,
    xlabel: str,
    agent_id: int,
    n_agents: int,
    n_obstacles: int,
    save_path: Path,
    xlim_min: float,
    xlim_max: float,
    sensing_radius: float | None = None,
):
    """One image per head with x-sweep and y-sweep overlaid on the distance plot."""
    n_heads = int(attn_weights_x_per_head.shape[0])

    for k in range(n_heads):
        head_attn_x = np.asarray(attn_weights_x_per_head[k])
        head_attn_y = np.asarray(attn_weights_y_per_head[k])

        max_attn = max(
            float(np.max([head_attn_x.max(), head_attn_y.max()])) * 1.15,
            0.1,
        )

        fig, ax = plt.subplots(figsize=(10, 9))

        # Existing x-variation points (original color/style).
        ax.scatter(
            x_values,
            head_attn_x,
            alpha=0.85,
            s=PLOT_PARAMS["obstacle"]["size"],
            c=PLOT_PARAMS["obstacle"]["color"],
            marker=PLOT_PARAMS["obstacle"]["marker"],
            label=r"$Obstacle_{j}$ : $\Delta x=d,\ \Delta y=0$",
            edgecolors="black",
            linewidth=0.8,
        )

        # New y-variation points in black.
        ax.scatter(
            x_values,
            head_attn_y,
            alpha=0.85,
            s=PLOT_PARAMS["obstacle_y"]["size"],
            c=PLOT_PARAMS["obstacle_y"]["color"],
            marker=PLOT_PARAMS["obstacle_y"]["marker"],
            label=r"$Obstacle_{j}$ : $\Delta x=0,\ \Delta y=d$",
            edgecolors="black",
            linewidth=0.8,
        )

        if sensing_radius is not None:
            ax.axvline(
                x=sensing_radius,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Sensing Radius",
            )

        ax.set_xlabel(xlabel, fontsize=30, fontweight="normal")
        ax.set_ylabel(r"Attention Weight $w_{ij}$", fontsize=30, fontweight="normal")
        ax.set_title("")

        ax.text(
            0.5,
            1.11,
            f"Head {k} — {title}",
            transform=ax.transAxes,
            fontsize=30,
            fontweight="normal",
            ha="center",
        )

        ax.text(
            0.5,
            1.04,
            f"Env: ({n_agents} agents, {n_obstacles} obstacles) | Timestep: 0 | $Agent_{{i}}$: {agent_id}",
            transform=ax.transAxes,
            fontsize=22,
            fontweight="normal",
            ha="center",
        )

        ax.legend(fontsize=20, loc="upper right", framealpha=0.75, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(-0.05, max_attn)

        plt.tight_layout(rect=[0, 0, 1, 0.85])

        out_path = save_path.with_name(
            f"{save_path.stem}_{title.lower().replace(' ', '_')}_head{k}{save_path.suffix}"
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved to: {out_path}")
        plt.close(fig)


def run_synthetic_attention_plots(args):
    env, algo = load_env_and_algo(args)
    template_graph = make_probe_template(env, seed=args.seed)

    out_dir = Path(args.output_dir) if args.output_dir is not None else Path("./KG_Plots")
    out_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    base_save = out_dir / f"{timestamp}_agent{args.attn_agent}.png"

    # Distance probe: overlay x-variation and y-variation on the same figure.
    x_vals, xlabel, title, xlim_min, xlim_max, file_tag = get_probe_features("distance", args, env)

    probe_graph_x, x_vals = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_vals,
        probe_kind="distance",
        fixed_distance=args.fixed_distance,
        distance_axis="x",
    )
    attn_x = np.asarray(algo.get_attention_weights(probe_graph_x))
    if attn_x.ndim == 1:
        attn_x = attn_x[None, :]

    probe_graph_y, _ = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_vals,
        probe_kind="distance",
        fixed_distance=args.fixed_distance,
        distance_axis="y",
    )
    attn_y = np.asarray(algo.get_attention_weights(probe_graph_y))
    if attn_y.ndim == 1:
        attn_y = attn_y[None, :]

    plot_distance_probe_attention_xy(
        x_values=x_vals,
        attn_weights_x_per_head=attn_x,
        attn_weights_y_per_head=attn_y,
        title=title,
        xlabel=xlabel,
        agent_id=args.attn_agent,
        n_agents=env.num_agents,
        n_obstacles=len(x_vals),
        save_path=base_save.with_name(f"corl_distance.png"),
        xlim_min=xlim_min,
        xlim_max=xlim_max,
        sensing_radius=0.5,
    )

    # Radial velocity probe. Leave unchanged.
    x_vals, xlabel, title, xlim_min, xlim_max, file_tag = get_probe_features("radial_velocity", args, env)
    probe_graph, x_vals = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_vals,
        probe_kind="radial_velocity",
        fixed_distance=args.fixed_distance,
    )

    attn = np.asarray(algo.get_attention_weights(probe_graph))
    plot_probe_attention(
        x_values=x_vals,
        attn_weights_per_head=attn,
        title=title,
        xlabel=xlabel,
        agent_id=args.attn_agent,
        n_agents=env.num_agents,
        n_obstacles=len(x_vals),
        save_path=base_save.with_name(f"corl_radial_velocity.png"),
        legend_label=r"$Obstacle_{j}$",
        xlim_min=xlim_min,
        xlim_max=xlim_max,
        sensing_radius=None,
    )


def main():
    parser = argparse.ArgumentParser()

    # Model / env loading (same fields as the main test script when relevant).
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    # Synthetic probe controls.
    parser.add_argument("--attn-agent", type=int, default=0)
    parser.add_argument("--probe-points", type=int, default=20)
    parser.add_argument("--distance-min", type=float, default=0.05)
    parser.add_argument("--distance-max", type=float, default=None)
    parser.add_argument("--radial-max", type=float, default=0.6)
    parser.add_argument("--fixed-distance", type=float, default=0.10)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    run_synthetic_attention_plots(args)


if __name__ == "__main__":
    main()