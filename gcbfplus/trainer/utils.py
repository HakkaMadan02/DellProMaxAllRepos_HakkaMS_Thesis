import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import functools as ft
import seaborn as sns
import optax

from typing import Callable, TYPE_CHECKING
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey
from ..utils.graph import GraphsTuple
from ..utils.utils import tree_index
from .data import Rollout



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

def get_bb_cbf_all_agents(cbf: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int, num_agents: int):
    n_mesh = 1
    low_lim = jnp.array([0, 0])
    high_lim = jnp.array([env.area_size, env.area_size])
    
    all_b_xs = []
    all_b_ys = []
    all_bb_h = []
    
    for current_agent_id in range(num_agents):
        agent_pos = graph.states[current_agent_id, :2]
        
        # Fix: Remove float() - use JAX-compatible indexing
        b_xs = agent_pos[x_dim:x_dim+1]  # This creates [x] as 1D array
        b_ys = agent_pos[y_dim:y_dim+1]  # This creates [y] as 1D array
        bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
        
        states = graph.states
        bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
        bb_plot_states = bb_plot_states.at[:, :, current_agent_id, x_dim].set(bb_Xs)
        bb_plot_states = bb_plot_states.at[:, :, current_agent_id, y_dim].set(bb_Ys)
        
        get_new_graph = env.add_edge_feats
        get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
        bb_new_graph = get_new_graph_vmap(bb_plot_states)
        
        bb_h = jax.vmap(jax.vmap(cbf))(bb_new_graph)[:, :, current_agent_id, :].squeeze(-1)
        
        all_b_xs.append(b_xs)
        all_b_ys.append(b_ys)
        all_bb_h.append(bb_h)
    
    # Return in the expected format: three separate stacked arrays
    return jnp.stack(all_b_xs), jnp.stack(all_b_ys), jnp.stack(all_bb_h)

def get_bb_cbf_in_all_agents(cbf: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int, num_agents: int):
    n_mesh = 1
    all_agent_results = []
    
    for current_agent_id in range(num_agents):
        # Fix 1: Use current_agent_id instead of agent_id
        agent_pos = graph.states[current_agent_id, :2]
        
        # Fix 2: Use x_dim and y_dim instead of hardcoded [0] and [1]
        b_xs = agent_pos[x_dim:x_dim+1]  # JAX-compatible indexing
        b_ys = agent_pos[y_dim:y_dim+1]  # JAX-compatible indexing
        bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
        
        states = graph.states
        bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
        bb_plot_states = bb_plot_states.at[:, :, current_agent_id, x_dim].set(bb_Xs)
        bb_plot_states = bb_plot_states.at[:, :, current_agent_id, y_dim].set(bb_Ys)
        
        get_new_graph = env.add_edge_feats
        get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
        bb_new_graph = get_new_graph_vmap(bb_plot_states)
        
        # Fix 3: Use current_agent_id instead of agent_id
        bb_h_in = jax.vmap(jax.vmap(cbf))(bb_new_graph)[:, :, current_agent_id, :]
        
        all_agent_results.append(bb_h_in)
    
    return jnp.stack(all_agent_results, axis=0)  # Stack for consistent format

# In utils.py, add:

def get_gnn_msg_intermediate_all_agents(gnn_msg_fn: Callable, env: MultiAgentEnv, graph: GraphsTuple, num_agents: int):
    """Extract message MLP intermediates across all agents"""
    all_results = []
    for t in range(len(graph.states)):
        intermediate = gnn_msg_fn(tree_index(graph, t))
        all_results.append(intermediate)
    return jnp.stack(all_results, axis=0)

def get_gnn_aggr_intermediate_all_agents(gnn_aggr_fn: Callable, env: MultiAgentEnv, graph: GraphsTuple, num_agents: int):
    """Extract aggregation MLP intermediates across all agents (shape: [T, n_nodes, 128])"""
    all_results = []
    for t in range(len(graph.states)):
        intermediate = gnn_aggr_fn(tree_index(graph, t))
        all_results.append(intermediate)
    return jnp.stack(all_results, axis=0)

def get_gnn_update_intermediate_all_agents(gnn_update_fn: Callable, env: MultiAgentEnv, graph: GraphsTuple, num_agents: int):
    """Extract update MLP intermediates across all agents (shape: [T, n_nodes, 256])"""
    all_results = []
    for t in range(len(graph.states)):
        intermediate = gnn_update_fn(tree_index(graph, t))
        all_results.append(intermediate)
    return jnp.stack(all_results, axis=0)


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)
