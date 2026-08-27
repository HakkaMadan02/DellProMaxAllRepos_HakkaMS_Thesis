"""
2headAblationPlots.py

Single-path, 4-panel attention plot for GCBF+ multi-head attention.

Panels:
1. Head 1 - Distance
2. Head 1 - Radial velocity
3. Head 2 - Distance
4. Head 2 - Radial velocity

Differences from original CorlFinalPlots.py:
- takes only ONE path
- uses TWO attention heads from the same loaded model
- first 2 panels are Head 1, last 2 panels are Head 2
- distance probe now also supports clean/noisy ("dirty") mode
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import yaml

from gcbfplus.algo import GCBFPlus, make_algo
from gcbfplus.env import make_env
from gcbfplus.utils.graph import GraphsTuple


# -------------------------------------------------
# Global paper plotting configuration
# -------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
})

COLORS = {
    "jetblack": "#0A0A0A",
    "gungrey": "#818589",
    "bloodred": "#880808",
    "white": "#FFFFFF",
    "panelbg": "#f2f2f2",
}

PLOT_PARAMS = {
    "obstacle": {
        "color": COLORS["bloodred"],
        "marker": "o",
        "label": r"$Obstacle_{j}$",
        "size": 150,
    }
}


# -------------------------------------------------
# Loading
# -------------------------------------------------
def load_config(path: Path):
    with open(path / "config.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_env_and_algo_from_path(model_root: str, step_override: Optional[int], args):
    """Load env and GCBF+ model from a specific experiment path."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)

    root = Path(model_root)
    config = load_config(root)

    num_agents = config.num_agents if args.num_agents is None else args.num_agents

    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    model_path = root / "models"
    if step_override is None:
        models = [p.name for p in model_path.iterdir() if p.name.isdigit()]
        step = max(int(m) for m in models)
    else:
        step = step_override

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

    if not (hasattr(algo, "getattentionweights") or hasattr(algo, "get_attention_weights")):
        raise RuntimeError(
            "The selected algorithm does not expose attention weights. "
            "Expected getattentionweights(...) or get_attention_weights(...)."
        )

    if not isinstance(algo, GCBFPlus):
        raise RuntimeError("This script is configured for GCBF+ only.")

    return env, algo


# -------------------------------------------------
# Synthetic probe construction
# -------------------------------------------------
def make_probe_template(env, seed: int = 0) -> GraphsTuple:
    """Create a valid graph template so env states have the correct structure."""
    return env.reset(jr.PRNGKey(seed))


def normalize_mode(mode: str) -> str:
    mode = mode.lower()
    if mode in ("noisy", "dirty"):
        return "noisy"
    if mode == "clean":
        return "clean"
    raise ValueError(f"Unknown mode: {mode}")


def build_synthetic_probe_graph(
    template_graph: GraphsTuple,
    n_agents: int,
    agent_id: int,
    probe_values: np.ndarray,
    probe_kind: str,
    fixed_distance: float = 0.10,
    distance_mode: str = "clean",
    radial_mode: str = "clean",
    noise_std_pos: float = 0.02,
    noise_std_vel: float = 0.05,
    noise_seed: int = 0,
) -> Tuple[GraphsTuple, np.ndarray]:
    """Create a graph with one receiver agent and homogeneous obstacle probe nodes."""
    rng = np.random.default_rng(noise_seed)

    distance_mode = normalize_mode(distance_mode)
    radial_mode = normalize_mode(radial_mode)

    values = np.asarray(probe_values, dtype=np.float32)
    n_probe = int(values.shape[0])

    node_dim = int(template_graph.nodes.shape[-1])
    state_dim = int(template_graph.states.shape[-1])
    edge_dim = int(template_graph.edges.shape[-1])

    n_nodes = n_agents + n_probe
    nodes = np.zeros((n_nodes, node_dim), dtype=np.float32)
    states = np.zeros((n_nodes, state_dim), dtype=np.float32)
    node_type = np.zeros((n_nodes,), dtype=np.int32)

    # agents
    nodes[:n_agents, 2] = 1.0
    node_type[:n_agents] = 0

    # obstacle probe nodes
    nodes[n_agents:, 0] = 1.0
    node_type[n_agents:] = 2

    senders = np.arange(n_agents, n_agents + n_probe, dtype=np.int32)
    receivers = np.full((n_probe,), int(agent_id), dtype=np.int32)

    edges = np.zeros((n_probe, edge_dim), dtype=np.float32)

    if probe_kind == "distance":
        # sweep p_ij in edge[:,0]
        edges[:, 0] = values

        if distance_mode == "clean":
            edges[:, 1] = 0.0
            edges[:, 2] = 0.0
            edges[:, 3] = 0.0
        elif distance_mode == "noisy":
            edges[:, 1] = rng.normal(0.0, noise_std_pos, size=n_probe)
            edges[:, 2] = rng.normal(0.0, noise_std_vel, size=n_probe)
            edges[:, 3] = rng.normal(0.0, noise_std_vel, size=n_probe)

    elif probe_kind == "radial_velocity":
        # sweep r_ij in edge[:,2]
        edges[:, 2] = values

        if radial_mode == "clean":
            edges[:, 0] = fixed_distance
            edges[:, 1] = 0.0
            edges[:, 3] = 0.0
        elif radial_mode == "noisy":
            edges[:, 0] = fixed_distance + 0.75 * rng.normal(0.0, noise_std_pos, size=n_probe)
            edges[:, 1] = rng.normal(0.0, noise_std_pos, size=n_probe)
            edges[:, 3] = rng.normal(0.0, noise_std_vel, size=n_probe)
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


# -------------------------------------------------
# Probe feature grids
# -------------------------------------------------
def get_probe_features(probe_kind: str, args, env) -> Tuple[np.ndarray, str, tuple[float, float]]:
    """Return x-axis samples, x-label, and nominal x-limits for the synthetic probe."""
    n_points = args.probe_points

    if probe_kind == "distance":
        max_p = args.distance_max if args.distance_max is not None else float(env.params.get("comm_radius", 0.5))
        xs = np.linspace(args.distance_min, max_p, n_points, dtype=np.float32)
        xlabel = r"$||p_{ij}||$"
        xlim = (0.0, float(max_p))

    elif probe_kind == "radial_velocity":
        max_r = float(args.radial_max)
        #xs = np.linspace(-max_r, max_r, n_points, dtype=np.float32)
        xs = np.linspace(0, max_r, n_points, dtype=np.float32)
        xlabel = r"$r_{ij}$"
        #xlim = (-max_r, max_r)
        xlim = (0.0, max_r)  # only show positive radial velocity for clarity

    else:
        raise ValueError(probe_kind)

    return xs, xlabel, xlim


# -------------------------------------------------
# Attention extraction
# -------------------------------------------------
'''def get_all_heads_attention(algo, probe_graph: GraphsTuple) -> np.ndarray:
    if hasattr(algo, "getattentionweights"):
        attn = algo.getattentionweights(probe_graph)
    elif hasattr(algo, "get_attention_weights"):
        attn = algo.get_attention_weights(probe_graph)
    else:
        raise RuntimeError(
            "Algorithm does not expose attention weights. "
            "Expected getattentionweights(...) or get_attention_weights(...)."
        )

    attn = np.asarray(attn, dtype=np.float32)
    attn = np.squeeze(attn)

    if attn.ndim == 1:
        attn = attn[None, :]
    elif attn.ndim != 2:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    if attn.shape[1] != int(probe_graph.edges.shape[0]):
        raise ValueError(
            f"Attention edge dimension mismatch: attn.shape={attn.shape}, "
            f"num_probe_edges={int(probe_graph.edges.shape[0])}"
        )

    return attn


def get_single_head_attention(algo, probe_graph: GraphsTuple, head_idx: int) -> np.ndarray:
    attn = get_all_heads_attention(algo, probe_graph)
    if head_idx < 0 or head_idx >= attn.shape[0]:
        raise ValueError(f"head_idx={head_idx} out of range for attention shape {attn.shape}")
    return np.asarray(attn[head_idx], dtype=np.float32).reshape(-1)'''

def get_all_heads_attention(algo, probe_graph: GraphsTuple) -> np.ndarray:
    if hasattr(algo, "getattentionweights"):
        attn = algo.getattentionweights(probe_graph)
    elif hasattr(algo, "get_attention_weights"):
        attn = algo.get_attention_weights(probe_graph)
    else:
        raise RuntimeError(
            "Algorithm does not expose attention weights. "
            "Expected getattentionweights(...) or get_attention_weights(...)."
        )

    attn = np.asarray(attn, dtype=np.float32)

    num_edges = int(probe_graph.edges.shape[0])

    # Case 1: scalar or 0-d object returned by wrapper
    if attn.ndim == 0:
        attn = np.full((1, num_edges), float(attn), dtype=np.float32)

    # Case 2: standard single-head edge vector
    elif attn.ndim == 1:
        if attn.shape[0] == num_edges:
            attn = attn[None, :]
        elif attn.shape[0] == 1:
            attn = np.full((1, num_edges), float(attn[0]), dtype=np.float32)
        else:
            raise ValueError(
                f"1D attention has incompatible shape {attn.shape}; "
                f"expected ({num_edges},) or (1,)"
            )

    # Case 3: standard multi-head
    elif attn.ndim == 2:
        if attn.shape[1] != num_edges and attn.shape[0] == num_edges:
            attn = attn.T
        if attn.shape[1] != num_edges:
            raise ValueError(
                f"2D attention edge dimension mismatch: attn.shape={attn.shape}, "
                f"num_probe_edges={num_edges}"
            )

    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    return attn.astype(np.float32)


def get_single_head_attention(algo, probe_graph: GraphsTuple, head_idx: int) -> np.ndarray:
    attn = get_all_heads_attention(algo, probe_graph)

    # If only one head is available, reuse it for any requested head index.
    if attn.shape[0] == 1 and head_idx > 0:
        print(
            f"[warning] Only one attention head available from checkpoint; "
            f"requested head_idx={head_idx}, reusing head 0."
        )
        head_idx = 0

    if head_idx < 0 or head_idx >= attn.shape[0]:
        raise ValueError(
            f"head_idx={head_idx} out of range for attention shape {attn.shape}"
        )

    return np.asarray(attn[head_idx], dtype=np.float32).reshape(-1)

# -------------------------------------------------
# Styling helpers
# -------------------------------------------------
def format_tick_labels(values: np.ndarray) -> list[str]:
    return [f"{v:.2f}" for v in values]


def compute_panel_top_ylim(y: Optional[np.ndarray]) -> float:
    if y is None or len(y) == 0:
        return 0.10
    return max(float(np.max(y)) * 1.10, 0.10)


def compute_reference_y_bottom(y_ref: Optional[np.ndarray]) -> float:
    y_top_ref = compute_panel_top_ylim(y_ref)
    return -max(0.02, 0.06 * y_top_ref)


def compute_panel_xlim(base_xlim: tuple[float, float]) -> tuple[float, float]:
    x0, x1 = base_xlim
    span = x1 - x0
    margin = max(0.02, 0.05 * span)
    return x0 - margin, x1 + margin


def compute_position_xticks() -> np.ndarray:
    return np.array([0.00, 0.15, 0.30, 0.45], dtype=float)


def compute_radial_xticks(radial_max: float) -> np.ndarray:
    #return np.linspace(-radial_max, radial_max, 4)
    return np.linspace(0.0, radial_max, 4)  # only show positive radial velocity for clarity


def style_attention_axis(
    ax,
    base_xlim: tuple[float, float],
    xlabel: str,
    y: Optional[np.ndarray],
    x_mode: str,
    shared_y_bottom: float,
    radial_max: Optional[float] = None,
):
    xmin, xmax = compute_panel_xlim(base_xlim)
    ymax = compute_panel_top_ylim(y)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(shared_y_bottom, ymax)

    if x_mode == "position":
        xticks = compute_position_xticks()
    elif x_mode == "radial":
        if radial_max is None:
            raise ValueError("radial_max must be provided for radial panels.")
        xticks = compute_radial_xticks(radial_max)
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}")

    yticks = np.linspace(0.0, ymax, 4)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(format_tick_labels(xticks), fontsize=35)
    ax.set_yticklabels(format_tick_labels(yticks), fontsize=35)

    ax.set_xlabel(xlabel, fontsize=45, fontweight="normal", labelpad=10)
    ax.set_facecolor(COLORS["panelbg"])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", which="both", length=0, width=0, pad=8)
    ax.grid(False)


# -------------------------------------------------
# Plotting
# -------------------------------------------------
def plot_attention_four_panel(
    panel_1: dict,
    panel_2: dict,
    panel_3: dict,
    panel_4: dict,
    save_png: Path,
    save_pdf: Path,
    radial_max: float,
):
    panels = [panel_1, panel_2, panel_3, panel_4]

    shared_y_bottom = compute_reference_y_bottom(panel_1["y"])

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    fig.patch.set_facecolor(COLORS["white"])
    plt.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.90, wspace=0.2)

    for ax, panel in zip(axes, panels):
        if panel["y"] is not None and panel["x"] is not None:
            ax.scatter(
                panel["x"],
                panel["y"],
                alpha=0.90,
                s=PLOT_PARAMS["obstacle"]["size"],
                c=PLOT_PARAMS["obstacle"]["color"],
                marker=PLOT_PARAMS["obstacle"]["marker"],
                edgecolors=COLORS["jetblack"],
                linewidth=0.8,
                zorder=3,
            )

        style_attention_axis(
            ax=ax,
            base_xlim=panel["xlim"],
            xlabel=panel["xlabel"],
            y=panel["y"],
            x_mode=panel["x_mode"],
            shared_y_bottom=shared_y_bottom,
            radial_max=radial_max if panel["x_mode"] == "radial" else None,
        )

        ax.text(
            0.5,
            1.05,
            panel["top_text"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=45,
            fontweight="normal",
            color=COLORS["jetblack"],
        )

    fig.text(0.0125, 0.525, "GCBF+", rotation=90, va="center", ha="center", fontsize=45)
    fig.supylabel(r"$w_{ij}$", fontsize=45, x=0.023)

    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG to: {save_png}")
    print(f"Saved PDF to: {save_pdf}")


# -------------------------------------------------
# Panel building
# -------------------------------------------------

def build_panel_pair_for_head(env, algo, args, head_idx: int, top_text: str):
    template_graph = make_probe_template(env, seed=args.seed)

    x_pos, xlabel_pos, xlim_pos = get_probe_features("distance", args, env)
    graph_pos, x_pos = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_pos,
        probe_kind="distance",
        fixed_distance=args.fixed_distance,
        distance_mode=args.distance_mode,
        radial_mode=args.radial_mode,
        noise_std_pos=args.noise_std_pos,
        noise_std_vel=args.noise_std_vel,
        noise_seed=args.noise_seed,
    )
    y_pos = get_single_head_attention(algo, graph_pos, head_idx=head_idx)

    x_rv, xlabel_rv, xlim_rv = get_probe_features("radial_velocity", args, env)
    graph_rv, x_rv = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_rv,
        probe_kind="radial_velocity",
        fixed_distance=args.fixed_distance,
        distance_mode=args.distance_mode,
        radial_mode=args.radial_mode,
        noise_std_pos=args.noise_std_pos,
        noise_std_vel=args.noise_std_vel,
        noise_seed=args.noise_seed,
    )
    y_rv = get_single_head_attention(algo, graph_rv, head_idx=head_idx)

    x_pos = np.asarray(x_pos, dtype=np.float32).reshape(-1)
    y_pos = np.asarray(y_pos, dtype=np.float32).reshape(-1)
    x_rv = np.asarray(x_rv, dtype=np.float32).reshape(-1)
    y_rv = np.asarray(y_rv, dtype=np.float32).reshape(-1)

    if len(x_pos) != len(y_pos):
        raise ValueError(f"Distance probe mismatch: len(x_pos)={len(x_pos)}, len(y_pos)={len(y_pos)}")
    if len(x_rv) != len(y_rv):
        raise ValueError(f"Radial probe mismatch: len(x_rv)={len(x_rv)}, len(y_rv)={len(y_rv)}")

    print(
        f"[head {head_idx}] distance: shape={y_pos.shape}, "
        f"min={y_pos.min():.6f}, max={y_pos.max():.6f}, sum={y_pos.sum():.6f}"
    )
    print(
        f"[head {head_idx}] radial:   shape={y_rv.shape}, "
        f"min={y_rv.min():.6f}, max={y_rv.max():.6f}, sum={y_rv.sum():.6f}"
    )

    panel_pos = {
        "x": x_pos,
        "y": y_pos,
        "xlabel": xlabel_pos,
        "xlim": xlim_pos,
        "x_mode": "position",
        "top_text": top_text,
    }

    panel_rv = {
        "x": x_rv,
        "y": y_rv,
        "xlabel": xlabel_rv,
        "xlim": xlim_rv,
        "x_mode": "radial",
        "top_text": top_text,
    }

    return panel_pos, panel_rv

# -------------------------------------------------
# Main execution
# -------------------------------------------------
def run_synthetic_attention_plots(args):
    env, algo = load_env_and_algo_from_path(args.path, args.step, args)

    panel_1, panel_2 = build_panel_pair_for_head(
        env=env,
        algo=algo,
        args=args,
        head_idx=args.head1_idx,
        top_text="Head 1",
    )

    panel_3, panel_4 = build_panel_pair_for_head(
        env=env,
        algo=algo,
        args=args,
        head_idx=args.head2_idx,
        top_text="Head 2",
    )

    out_dir = Path(args.output_dir) if args.output_dir is not None else Path("./KG_Plots")
    out_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    save_png = out_dir / f"Attention_supervised_positiveRadial.png"
    save_pdf = out_dir / f"Attention_supervised_positiveRadial.pdf"

    plot_attention_four_panel(
        panel_1=panel_1,
        panel_2=panel_2,
        panel_3=panel_3,
        panel_4=panel_4,
        save_png=save_png,
        save_pdf=save_pdf,
        radial_max=float(args.radial_max),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True)
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

    parser.add_argument("--attn-agent", type=int, default=0)
    parser.add_argument("--head1-idx", type=int, default=0)
    parser.add_argument("--head2-idx", type=int, default=1)

    parser.add_argument("--probe-points", type=int, default=25)
    parser.add_argument("--distance-min", type=float, default=0.05)
    parser.add_argument("--distance-max", type=float, default=None)
    parser.add_argument("--radial-max", type=float, default=0.6)
    parser.add_argument("--fixed-distance", type=float, default=0.10)

    parser.add_argument("--distance-mode", type=str, default="clean",
                        choices=["clean", "noisy", "dirty"])
    parser.add_argument("--radial-mode", type=str, default="clean",
                        choices=["clean", "noisy", "dirty"])

    parser.add_argument("--noise-std-pos", type=float, default=0.02)
    parser.add_argument("--noise-std-vel", type=float, default=0.05)
    parser.add_argument("--noise-seed", type=int, default=0)

    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    run_synthetic_attention_plots(args)


if __name__ == "__main__":
    main()