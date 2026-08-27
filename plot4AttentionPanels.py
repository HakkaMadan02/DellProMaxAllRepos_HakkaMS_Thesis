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

from model.env import make_env as makeenv
from model.utils.graph import GraphsTuple

from model.algo import MAPPO, make_algo


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"xtick.labelsize": 40, "ytick.labelsize": 40})


COLORS = {
    "jetblack": "#0A0A0A",
    "gungrey": "#818589",
    "bloodred": "#880808",
    "white": "#FFFFFF",
    "panelbg": "#f2f2f2",
}

PLOTPARAMS = {
    "obstacle_color": COLORS["bloodred"],
    "marker": "o",
    "label": r"Obstacle$_j$",
    "size": 150,
}


def load_config(path: Path):
    with open(path / "config.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def extract_attention_weights_from_mappo(algo, probe_graph: GraphsTuple) -> np.ndarray:
    if hasattr(algo, "critic") and hasattr(algo, "critic_train_state"):
        _, attn = algo.critic.get_value_with_attn(
            algo.critic_train_state.params,
            probe_graph,
        )
        return np.asarray(attn, dtype=np.float32)

    raise RuntimeError(
    )

    raise RuntimeError(
    )


def load_env_and_algo_from_path(model_root: str, step_override: Optional[int], args):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)

    root = Path(model_root)
    config = load_config(root)

    num_agents = config.num_agents if args.num_agents is None else args.num_agents

    env = makeenv(
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
        lr_critic=config.lr_critic,
        alpha=getattr(config, "alpha", 1.0),
        eps=0.2,
        inner_epoch=8,
        seed=config.seed,
    )

    algo.load(str(model_path), step)

    # Optional sanity check.
    _ = extract_attention_weights_from_mappo(algo, make_probe_template(env, seed=args.seed))

    if not isinstance(algo, MAPPO):
        raise RuntimeError("This script is configured for MAPPO only.")

    return env, algo


def make_probe_template(env, seed: int = 0) -> GraphsTuple:

    return env.reset(jr.PRNGKey(seed))


def build_synthetic_probe_graph(
    template_graph: GraphsTuple,
    n_agents: int,
    agent_id: int,
    probe_values: np.ndarray,
    probe_kind: str,
    fixed_distance: float = 0.10,
    radial_mode: str = "clean",
    distance_mode: str = "clean",
) -> Tuple[GraphsTuple, np.ndarray]:
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

    # Keep exactly the same structure as the GCBF+ plotting script.
    nodes[:n_agents, 2] = 1.0
    node_type[:n_agents] = 0

    nodes[n_agents:, 0] = 1.0
    node_type[n_agents:] = 2

    senders = np.arange(n_agents, n_agents + n_probe, dtype=np.int32)
    receivers = np.full((n_probe,), int(agent_id), dtype=np.int32)

    edges = np.zeros((n_probe, edge_dim), dtype=np.float32)

    if probe_kind == "distance":

        if distance_mode == "clean":
            edges[:, 0] = values
            edges[:, 1] = 0.0
            edges[:, 2] = 0.0
            edges[:, 3] = 0.0
        elif distance_mode == "noisy":
            edges[:, 0] = values
            edges[:, 1] = 0.0
            edges[:, 2] = rng.normal(0, noise_std_vel, size=n_probe)
            edges[:, 3] = rng.normal(0, noise_std_vel, size=n_probe)
    elif probe_kind == "radialvelocity":
        if radial_mode == "clean":
            edges[:, 0] = fixed_distance
            edges[:, 1] = 0.0
            edges[:, 2] = values
            edges[:, 3] = 0.0
        elif radial_mode == "noisy":
            edges[:, 0] = fixed_distance + 0.75 * rng.normal(0, noise_std_pos, size=n_probe)
            edges[:, 1] = rng.normal(0, noise_std_pos, size=n_probe)
            edges[:, 2] = values
            edges[:, 3] = rng.normal(0, noise_std_vel, size=n_probe)
        else:
            raise ValueError(f"Unknown radial_mode {radial_mode}")
    else:
        raise ValueError(f"Unknown probe_kind {probe_kind}")

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


def get_probe_features(probe_kind: str, args, env) -> Tuple[np.ndarray, str, tuple[float, float]]:
    n_points = args.probe_points

    if probe_kind == "distance":
        max_p = args.distance_max if args.distance_max is not None else float(env.params.get("comm_radius", 0.5))
        xs = np.linspace(args.distance_min, max_p, n_points, dtype=np.float32)
        xlabel = r"$||p_{ij}||$"
        xlim = (0.0, float(max_p))
    elif probe_kind == "radialvelocity":
        max_r = float(args.radial_max)
        xs = np.linspace(-max_r, max_r, n_points, dtype=np.float32)
        xlabel = r"$r_{ij}$"
        xlim = (-max_r, max_r)
    else:
        raise ValueError(probe_kind)

    return xs, xlabel, xlim


def get_single_head_attention(algo, probe_graph: GraphsTuple, head_idx: int) -> np.ndarray:
    
    attn = np.asarray(extract_attention_weights_from_mappo(algo, probe_graph))

    if attn.ndim == 1:
        attn = attn[None, :]

    if head_idx < 0 or head_idx >= attn.shape[0]:
        raise ValueError(f"head_idx={head_idx} is out of range for attention shape {attn.shape}")

    return np.asarray(attn[head_idx], dtype=np.float32)


def format_tick_labels(values: np.ndarray) -> list[str]:
    return [f"{v:.2f}" for v in values]


def compute_panel_top_ylim(y: Optional[np.ndarray]) -> float:
    if y is None or len(y) == 0:
        return 0.10
    return max(float(np.max(y)) * 1.10, 0.10)


def compute_reference_ybottom(y_ref: Optional[np.ndarray]) -> float:
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
    return np.linspace(-radial_max, radial_max, 4)


def style_attention_axis(
    ax,
    base_xlim: tuple[float, float],
    xlabel: str,
    y: Optional[np.ndarray],
    xmode: str,
    shared_ybottom: float,
    radial_max: Optional[float] = None,
):
    xmin, xmax = compute_panel_xlim(base_xlim)
    ymax = compute_panel_top_ylim(y)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(shared_ybottom, ymax)

    if xmode == "position":
        xticks = compute_position_xticks()
    elif xmode == "radial":
        if radial_max is None:
            raise ValueError("radial_max must be provided for radial panels.")
        xticks = compute_radial_xticks(radial_max)
    else:
        raise ValueError(f"Unknown xmode {xmode}")

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


def plot_attention_four_panel(
    panel1: dict,
    panel2: dict,
    panel3: dict,
    panel4: dict,
    save_png: Path,
    save_pdf: Path,
    radial_max: float,
):
    panels = [panel1, panel2, panel3, panel4]
    shared_ybottom = compute_reference_ybottom(panel1["y"])

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    fig.patch.set_facecolor(COLORS["white"])
    plt.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.90, wspace=0.2)

    for ax, panel in zip(axes, panels):
        if panel["y"] is not None and panel["x"] is not None:
            ax.scatter(
                panel["x"],
                panel["y"],
                alpha=0.90,
                s=PLOTPARAMS["size"],
                c=PLOTPARAMS["obstacle_color"],
                marker=PLOTPARAMS["marker"],
                edgecolors=COLORS["jetblack"],
                linewidth=0.8,
                zorder=3,
            )

        style_attention_axis(
            ax=ax,
            base_xlim=panel["xlim"],
            xlabel=panel["xlabel"],
            y=panel["y"],
            xmode=panel["xmode"],
            shared_ybottom=shared_ybottom,
            radial_max=radial_max if panel["xmode"] == "radial" else None,
        )

        ax.text(
            0.5,
            1.05,
            panel["toptext"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=45,
            fontweight="normal",
            color=COLORS["jetblack"],
        )

    fig.text(0.0125, 0.525, "InforMARL", rotation=90, va="center", ha="center", fontsize=45)
    fig.supylabel(r"$w_{ij}$", fontsize=45, x=0.023)

    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG to {save_png}")
    print(f"Saved PDF to {save_pdf}")


def build_panel_pair(env, algo, args, distance_mode: str, radial_mode: str, toptext: str):
    
    template_graph = make_probe_template(env, seed=args.seed)

    x_pos, xlabel_pos, xlim_pos = get_probe_features("distance", args, env)
    graph_pos, x_pos = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_pos,
        probe_kind="distance",
        fixed_distance=args.fixed_distance,
        distance_mode=distance_mode,
    )
    y_pos = get_single_head_attention(algo, graph_pos, head_idx=args.head_idx)

    x_rv, xlabel_rv, xlim_rv = get_probe_features("radialvelocity", args, env)
    graph_rv, x_rv = build_synthetic_probe_graph(
        template_graph=template_graph,
        n_agents=env.num_agents,
        agent_id=args.attn_agent,
        probe_values=x_rv,
        probe_kind="radialvelocity",
        fixed_distance=args.fixed_distance,
        radial_mode=radial_mode,
    )
    y_rv = get_single_head_attention(algo, graph_rv, head_idx=args.head_idx)

    panel_pos = {
        "x": x_pos,
        "y": y_pos,
        "xlabel": xlabel_pos,
        "xlim": xlim_pos,
        "xmode": "position",
        "toptext": toptext,
    }

    panel_rv = {
        "x": x_rv,
        "y": y_rv,
        "xlabel": xlabel_rv,
        "xlim": xlim_rv,
        "xmode": "radial",
        "toptext": toptext,
    }

    return panel_pos, panel_rv


def run_synthetic_attention_plots(args):
    env1, algo1 = load_env_and_algo_from_path(args.path1, args.step1, args)
    env2, algo2 = load_env_and_algo_from_path(args.path2, args.step2, args)

    panel1, panel2 = build_panel_pair(
        env1, algo1, args, distance_mode="noisy", radial_mode="noisy", toptext="Unsupervised"
    )
    panel3, panel4 = build_panel_pair(
        env2, algo2, args, distance_mode="clean", radial_mode="clean", toptext="Supervised"
    )

    outdir = Path(args.output_dir) if args.output_dir is not None else Path("./KGPlots")
    outdir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    save_png = outdir / f"CoRL_attention_4panel.png"
    save_pdf = outdir / f"CoRL_attention_4panel.pdf"

    plot_attention_four_panel(
        panel1=panel1,
        panel2=panel2,
        panel3=panel3,
        panel4=panel4,
        save_png=save_png,
        save_pdf=save_pdf,
        radial_max=float(args.radial_max),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path1", type=str, required=True)
    parser.add_argument("--path2", type=str, required=True)
    parser.add_argument("--step1", type=int, default=None)
    parser.add_argument("--step2", type=int, default=None)

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
    parser.add_argument("--head-idx", type=int, default=0)

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