import argparse
import datetime
import functools as ft
import os
import pathlib
import cv2
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import trange

from model.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBFK, DecShareCBF, DecOptCBF
from model.env import make_env
from model.env.base import RolloutResult
from model.trainer.utils import get_bb_cbf, plot_cbf
from model.utils.graph import GraphsTuple
from model.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap
from model.nn.utils import safe_get
from model.utils.utils import MutablePatchCollection
from matplotlib.collections import LineCollection, PatchCollection


path = './logs/TRO/DoubleIntegrator/gcbf+/seed0_20231123001301'
parser = argparse.ArgumentParser()
parser.add_argument('--num-agents', type=int, default=3)
args = parser.parse_args()

# load config
with open(os.path.join(path, "config.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

# create environments
num_agents = config.num_agents if args.num_agents is None else args.num_agents
env = make_env(
    env_id=config.env,
    num_agents=num_agents,
    num_obs=0,
    area_size=2,
    max_step=256,
    max_travel=None,
)

model_path = os.path.join(path, "models")
models = os.listdir(model_path)
step = max([int(model) for model in models if model.isdigit()])
algo = make_algo(
    algo=config.algo,  # shared
    env=env,  # shared
    node_dim=env.node_dim,  # shared
    edge_dim=env.edge_dim,  # shared
    state_dim=env.state_dim,  # shared
    action_dim=env.action_dim,  # shared
    n_agents=env.num_agents,  # shared
    gnn_layers=config.gnn_layers,  # gcbf, gcbf+, mappo
    batch_size=config.batch_size,  # gcbf, gcbf+, mappo
    buffer_size=config.buffer_size,  # gcbf, gcbf+
    horizon=config.horizon,  # gcbf+
    lr_actor=config.lr_actor,  # gcbf, gcbf+, mappo
    lr_cbf=config.lr_cbf,  # gcbf, gcbf+
    lr_critic=config.lr_critic if hasattr(config, 'lr_critic') else None,  # mappo
    alpha=config.alpha,  # gcbf, gcbf+
    eps=0.02,  # gcbf, gcbf+
    inner_epoch=8,  # gcbf, gcbf+
    loss_action_coef=config.loss_action_coef,  # gcbf, gcbf+
    loss_unsafe_coef=config.loss_unsafe_coef,  # gcbf, gcbf+
    loss_safe_coef=config.loss_safe_coef,  # gcbf, gcbf+
    loss_h_dot_coef=config.loss_h_dot_coef,  # gcbf, gcbf+
    max_grad_norm=2.0,  # gcbf, gcbf+, mappo
    seed=config.seed,  # shared
    cost_weight=config.cost_weight if hasattr(config, 'cost_weight') else None,  # mappo
)
algo.load(model_path, step)

# get different graphs

ego_agent_state = jnp.array([1.0, 1.0, 0.0, 0.0])
x1s = jnp.linspace(0.55, 0.85, 4)
x2s = jnp.linspace(1.15, 1.45, 4)

# for i in range(4):
#     agent_1 = jnp.array([x1s[i], 1.0, 0.0, 0.0])
#     agent_2 = jnp.array([x2s[i], 1.0, 0.0, 0.0])
#     agent_states = jnp.stack([ego_agent_state, agent_1, agent_2], axis=0)
#     goals = jnp.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]])
#
#     n_rng_obs = 0
#     key = jr.PRNGKey(0)
#     obstacle_key, key = jr.split(key, 2)
#     obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=2)
#     length_key, key = jr.split(key, 2)
#     obs_len = jr.uniform(
#         length_key,
#         (n_rng_obs, 2),
#         minval=0.1,
#         maxval=0.2,
#     )
#     theta_key, key = jr.split(key, 2)
#     obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
#     obstacles = env.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
#
#     env_state = env.EnvState(agent_states, goals, obstacles)
#     graph = env.get_graph(env_state)
#
#     cbf = algo.get_cbf(graph)

attn = np.array([
    [3.18154230e-06, 7.92178273e-01],
    [2.4492914e-04, 2.5193138e-02],
    [2.13179402e-02, 3.43408249e-03],
    [9.18005049e-01, 9.60669131e-05]
])

fig, axs = plt.subplots(4, 1, figsize=(10, 8), dpi=300)
for i in range(4):
    ax = axs[i]
    ax.set_xlim(0.45, 1.55)
    ax.set_ylim(0.9, 1.1)
    ax.set(aspect="equal")
    plt.axis("on")
    agent_color = "#0068ff"

    agent1_pos = np.array([x1s[i], 1.0])
    agent2_pos = np.array([x2s[i], 1.0])

    agent_circs = [
        plt.Circle(ego_agent_state[:2], 0.05, color=agent_color, linewidth=0.0),
        plt.Circle([x1s[i], 1.0], 0.05, color=agent_color, linewidth=0.0),
        plt.Circle([x2s[i], 1.0], 0.05, color=agent_color, linewidth=0.0),
    ]
    agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
    ax.add_collection(agent_col)
    e_start = np.stack([ego_agent_state[:2], ego_agent_state[:2], agent1_pos, agent2_pos], axis=0)
    e_end = np.stack([agent1_pos, agent2_pos, ego_agent_state[:2], ego_agent_state[:2]], axis=0)
    e_lines = np.stack([e_start, e_end], axis=1)
    edge_col = LineCollection(e_lines, colors="0.2", linewidths=3, alpha=0.7, zorder=3)
    ax.add_collection(edge_col)

    font_opts = dict(
        size=24,
        color="k",
        family="Times New Roman",
        # weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=7,
    )

    ax.text(ego_agent_state[0], ego_agent_state[1], "0", weight='bold', **font_opts)
    ax.text(agent1_pos[0], agent1_pos[1], "1", weight='bold', **font_opts)
    ax.text(agent2_pos[0], agent2_pos[1], "2", weight='bold', **font_opts)

    ax.text((ego_agent_state[0] + agent1_pos[0]) / 2, (ego_agent_state[1] + agent1_pos[1]) / 2 + 0.07,
            f"{attn[i, 0]:.3f}", **font_opts)
    # ax.text((ego_agent_state[0] + agent1_pos[0]) / 2, (ego_agent_state[1] + agent1_pos[1]) / 2 - 0.07,
    #         f"d = {np.abs(ego_agent_state[0] - agent1_pos[0]):.2f}", **font_opts)
    ax.text((ego_agent_state[0] + agent2_pos[0]) / 2, (ego_agent_state[1] + agent2_pos[1]) / 2 + 0.07,
            f"{attn[i, 1]:.3f}", **font_opts)
    # ax.text((ego_agent_state[0] + agent2_pos[0]) / 2, (ego_agent_state[1] + agent2_pos[1]) / 2 - 0.07,
    #         f"d = {np.abs(ego_agent_state[0] - agent2_pos[0]):.2f}", **font_opts)

    if i != 3:
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    else:
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=24, labelfontfamily='Times New Roman')

plt.savefig('figs/attn.pdf', bbox_inches='tight', pad_inches=0)
