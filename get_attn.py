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
from model.nn.gnn import save_set_attn



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

save_set_attn(True)
if os.path.exists('send.csv'):
    os.remove('send.csv')
if os.path.exists('recv.csv'):
    os.remove('recv.csv')

if os.path.exists('dist.csv'):
    os.remove('dist.csv')
if os.path.exists('attn.csv'):
    os.remove('attn.csv')
from model.utils.utils import np2jax

for i in range(4):
    agent_1 = jnp.array([x1s[i], 1.0, 0.0, 0.0])
    agent_2 = jnp.array([x2s[i], 1.0, 0.0, 0.0])
    agent_states = jnp.stack([ego_agent_state, agent_1, agent_2], axis=0)
    goals = jnp.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]])

    n_rng_obs = 0
    key = jr.PRNGKey(0)
    obstacle_key, key = jr.split(key, 2)
    obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=2)
    length_key, key = jr.split(key, 2)
    obs_len = jr.uniform(
        length_key,
        (n_rng_obs, 2),
        minval=0.1,
        maxval=0.2,
    )
    theta_key, key = jr.split(key, 2)
    obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
    obstacles = env.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

    env_state = env.EnvState(agent_states, goals, obstacles)
    graph = env.get_graph(env_state)

    cbf = algo.get_cbf(graph)

