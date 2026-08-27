import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import datetime
import pathlib
import orbax
import orbax.checkpoint
import numpy as np
import socket
import matplotlib.pyplot as plt
import functools as ft
import seaborn as sns

from typing import Callable, TYPE_CHECKING
from flax.training.checkpoints import natural_sort, ORBAX_CKPT_FILENAME
from flax import io
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey
from ..utils.graph import GraphsTuple
from .data import Rollout
import optax

if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        key: PRNGKey
) -> Rollout:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]
    key: PRNGKey

    Returns
    -------
    data: Rollout
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    def body(graph, key_):
        action, log_pi = actor(graph, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        return next_graph, (graph, action, reward, cost, done, log_pi, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    final_graph, (graphs, actions, rewards, costs, dones, log_pis, next_graphs) = \
        jax.lax.scan(body, init_graph, keys, length=env.max_episode_steps)
    data = Rollout(graphs, actions, rewards, costs, dones, log_pis, next_graphs)
    return data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def get_ckpt_manager(ckpt_dir: pathlib.Path, max_to_keep: int = 5):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep, keep_time_interval=datetime.timedelta(minutes=15), create=True
    )
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)
    return ckpt_manager


def latest_checkpoint_step(ckpt_dir: pathlib.Path, prefix: str = "") -> int:
    """Retrieve the path of the latest checkpoint in a directory.
    Args:
      ckpt_dir: str: directory of checkpoints to restore from.
      prefix: str: name prefix of checkpoint files.
    Returns:
      The latest checkpoint path or None if no checkpoints were found.
    """
    checkpoint_files = list(ckpt_dir.iterdir())
    checkpoint_files = [str(f) for f in checkpoint_files if f.is_dir()]
    checkpoint_files = natural_sort(checkpoint_files)
    if not checkpoint_files:
        raise ValueError("Didn't find any checkpoint files!")

    latest = pathlib.Path(checkpoint_files[-1])
    # Extract step from it.
    return int(latest.name[len(prefix):])


def load_ckpt(model, ckpt_path: pathlib.Path):
    # Find which step we load. Do it here, so we can log it for visibility instead of it being implicit....
    assert io.isdir(ckpt_path)
    if io.exists(ckpt_path / "default" / ORBAX_CKPT_FILENAME):
        # This means the given dir is an orbax checkpoint.
        step = orbax.checkpoint.utils.step_from_checkpoint_name(ckpt_path.name)
        ckpt_dir = ckpt_path.parent
    else:
        step = latest_checkpoint_step(ckpt_path)
        # logger.info(f"Loading ckpt from step {step}!")

        ckpt_dir = ckpt_path

    ckpt_manager = get_ckpt_manager(ckpt_dir)

    def restore_fn():
        return ckpt_manager.restore(step=step, items=model)

    loaded = restore_fn()
    if model is loaded:
        raise ValueError("restore_checkpoint failed!")
    return loaded


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def empty_grad_tx() -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return None, None

    return optax.GradientTransformation(init_fn, update_fn)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def is_connected():
    try:
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close()
        return True
    except OSError:
        pass
    print('No internet connection')
    return False


# @jax.jit
# def get_new_graph(prev_graph: GraphsTuple, new_states: State) -> GraphsTuple:
#     return prev_graph._replace(
#         edges=new_states[prev_graph.receivers] - new_states[prev_graph.senders],
#         states=new_states
#     )


def plot_cbf(
        fig: plt.Figure,
        cbf: Callable,
        env: MultiAgentEnv,
        graph: GraphsTuple,
        agent_id: int,
        x_dim: int,
        y_dim: int,
) -> plt.Figure:
    ax = fig.gca()
    n_mesh = 30
    low_lim, high_lim = env.state_lim(graph.states)
    x, y = jnp.meshgrid(
        jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh),
        jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    )
    # agent_states = graph.type_states(type_idx=0, n_type=env.num_agents)
    states = graph.states

    # generate new states
    plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    plot_states = plot_states.at[:, :, agent_id, x_dim].set(x)
    plot_states = plot_states.at[:, :, agent_id, y_dim].set(y)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    new_graph = get_new_graph_vmap(plot_states)
    h = jax.vmap(jax.vmap(cbf))(new_graph)[:, :, agent_id, :].squeeze(-1)
    plt.contourf(x, y, h, cmap=sns.color_palette("rocket", as_cmap=True), levels=15, alpha=0.5)
    plt.colorbar()
    plt.contour(x, y, h, levels=[0.0], colors='blue')
    ax.set_xlim(low_lim[0], high_lim[0])
    ax.set_ylim(low_lim[1], high_lim[1])
    plt.axis('off')

    return fig


def get_bb_cbf(cbf: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int):
    n_mesh = 20
    low_lim = jnp.array([0, 0])
    high_lim = jnp.array([env.area_size, env.area_size])
    b_xs = jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh)
    b_ys = jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
    states = graph.states

    # generate new states
    bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, x_dim].set(bb_Xs)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, y_dim].set(bb_Ys)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    bb_new_graph = get_new_graph_vmap(bb_plot_states)
    bb_h = jax.vmap(jax.vmap(cbf))(bb_new_graph)[:, :, agent_id, :].squeeze(-1)
    assert bb_h.shape == (n_mesh, n_mesh)
    return b_xs, b_ys, bb_h


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)
