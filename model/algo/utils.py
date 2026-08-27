import functools as ft
from typing import Tuple

import einops as ei
import ipdb
import jax
import jax.numpy as jnp

from ..env.base import MultiAgentEnv
from ..env.crazyflie import CrazyFlie
from ..env.double_integrator import DoubleIntegrator
from ..env.dubins_car import DubinsCar
from ..env.linear_drone import LinearDrone
from ..env.single_integrator import SingleIntegrator
from ..env.cf_hl import CFHL
from ..utils.graph import GraphsTuple
from ..utils.typing import Array, Done, Reward


def compute_gae_fn(
        values: Array, rewards: Reward, dones: Done, next_values: Array, gamma: float, gae_lambda: float
) -> Tuple[Array, Array]:
    """
    Compute generalized advantage estimation.
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gaes = deltas

    def scan_fn(gae, inp):
        delta, done = inp
        gae_prev = delta + gamma * gae_lambda * (1 - done) * gae
        return gae_prev, gae_prev

    _, gaes_prev = jax.lax.scan(scan_fn, gaes[-1], (deltas[:-1], dones[:-1]), reverse=True)
    gaes = jnp.concatenate([gaes_prev, gaes[-1, None]], axis=0)

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


def compute_gae(
        values: Array, rewards: Reward, dones: Done, next_values: Array, gamma: float, gae_lambda: float
) -> Tuple[Array, Array]:
    return jax.vmap(ft.partial(compute_gae_fn, gamma=gamma, gae_lambda=gae_lambda))(values, rewards, dones, next_values)


def cbf_single_integrator(graph: GraphsTuple, r: float, n_agent: int, n_rays: int):
    # (n_agents, 2)
    agent_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    # (n_obs, 2)
    obs_states = jnp.concatenate([agent_states, obs_states], axis=0)
    # (n_agents, n_obs, 2)
    x_diff = agent_states[:, None, :2] - obs_states[None, :, :2]
    # (n_agents, n_obs)
    self_dist = jnp.concatenate(
        [jnp.eye(x_diff.shape[0]) * 1e2, jnp.zeros((x_diff.shape[0], x_diff.shape[1] - x_diff.shape[0]))],
        axis=-1,
    )
    # (n_agents, n_obs)
    h0 = (x_diff ** 2).sum(axis=-1) + self_dist ** 2 - 4 * r ** 2
    # (n_rays,)
    lidar_mask = jnp.ones((n_rays,))
    # (n_agents, n_obs_lidar)
    lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * n_agent)
    # (n_agents, n_obs)
    mask = jnp.concatenate([jnp.ones((n_agent, n_agent)), lidar_mask], axis=-1)
    h0 = h0 + jnp.logical_not(mask) * 1e6

    # (n_agents, )
    aargmin = jnp.argmin(h0, axis=-1)
    assert aargmin.shape == (n_agent,)

    # (n_agents,). True if agent-obs. The first n_agent elements are agent-agent.
    aisobs = aargmin >= n_agent

    # (n_agents, 1)
    h_minoverall = h0.min(axis=-1)[:, None]
    return h_minoverall, aisobs


def pwise_cbf_single_integrator_(pos: Array, agent_idx: int, o_obs_pos: Array, a_pos: Array, r: float, k: int):
    n_agent = len(a_pos)

    all_obs_pos = jnp.concatenate([a_pos, o_obs_pos], axis=0)

    # Only consider the k obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)

    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account. Add some epsilon for qp solver error.
    k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

    k_h0 = k_dist_sq
    k_isobs = k_idx >= n_agent

    return k_h0, k_isobs


def pwise_cbf_single_integrator(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 2)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 2)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_single_integrator_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def pwise_cbf_double_integrator_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    n_agent = len(a_state)

    pos = state[:2]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :2]
    del o_obs_state

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account.
    k_dist_sq = k_dist_sq - 4 * r ** 2

    k_h0 = k_dist_sq
    assert k_h0.shape == (k,)

    k_xdiff = state[:2] - all_obs_state[k_idx][:, :2]
    k_vdiff = state[2:] - all_obs_state[k_idx][:, 2:]
    assert k_xdiff.shape == k_vdiff.shape == (k, 2)

    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    assert k_h0_dot.shape == (k,)

    k_h1 = k_h0_dot + 10.0 * k_h0

    k_isobs = k_idx >= n_agent

    return k_h1, k_isobs


def pwise_cbf_double_integrator_u_constrained_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    n_agent = len(a_state)
    u_lim = 1.0  # by default

    pos = state[:2]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :2]
    del o_obs_state

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    # k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account.
    # k_dist_sq = k_dist_sq - 4 * r ** 2

    # k_h0 = k_dist_sq
    # assert k_h0.shape == (k,)

    k_xdiff = state[:2] - all_obs_state[k_idx][:, :2]
    k_vdiff = state[2:] - all_obs_state[k_idx][:, 2:]
    assert k_xdiff.shape == k_vdiff.shape == (k, 2)

    k_isobs = k_idx >= n_agent
    safe_dist = jnp.ones_like(k_isobs) * 2 * r
    safe_dist -= k_isobs * r

    k_h = (jnp.sqrt(2 * (u_lim + u_lim) * jnp.maximum(1e-4, jnp.linalg.norm(k_xdiff, axis=-1) - safe_dist)) +
           (k_xdiff * k_vdiff).sum(axis=-1) / jnp.linalg.norm(k_xdiff, axis=-1))

    return k_h, k_isobs


def pwise_cbf_double_integrator_u_constrained(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_double_integrator_u_constrained_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def pwise_cbf_double_integrator(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_double_integrator_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def pwise_cbf_dubins_car_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    n_agent = len(a_state)
    n_obs = len(o_obs_state)

    pos = state[:2]
    vel = state[3] * jnp.array([jnp.cos(state[2]), jnp.sin(state[2])])
    assert vel.shape == (2,)

    agent_vel = a_state[:, 3, None] * jnp.stack([jnp.cos(a_state[:, 2]), jnp.sin(a_state[:, 2])], axis=-1)
    assert agent_vel.shape == (n_agent, 2)

    all_obs_pos = jnp.concatenate([a_state[:, :2], o_obs_state[:, :2]], axis=0)
    all_obs_vel = jnp.concatenate([agent_vel, jnp.zeros((n_obs, 2))], axis=0)
    del o_obs_state

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account. Add some epsilon for qp solver error.
    k_dist_sq = k_dist_sq - 4 * r ** 2

    k_h0 = k_dist_sq
    assert k_h0.shape == (k,)

    k_xdiff = state[:2] - all_obs_pos[k_idx]
    k_vdiff = agent_vel[agent_idx] - all_obs_vel[k_idx]
    assert k_xdiff.shape == k_vdiff.shape == (k, 2)

    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    assert k_h0_dot.shape == (k,)

    k_h1 = k_h0_dot + 5.0 * k_h0

    k_isobs = k_idx >= n_agent

    return k_h1, k_isobs


def pwise_cbf_dubins_car(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_dubins_car_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def pwise_cbf_crazyflie_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    # state: ( 12, )
    n_agent = len(a_state)

    pos = state[:3]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :3]
    del o_obs_state

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account. Add some epsilon for qp solver error.
    k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

    k_h0 = k_dist_sq
    assert k_h0.shape == (k,)

    # all_obs_state = all_obs_state[k_idx]

    def crazyflie_f_(x: Array) -> Array:
        Ixx, Iyy, Izz = CrazyFlie.PARAMS["Ixx"], CrazyFlie.PARAMS["Iyy"], CrazyFlie.PARAMS["Izz"]
        I = jnp.array([Ixx, Iyy, Izz])
        # roll, pitch, yaw
        phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]
        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        t_th = jnp.tan(theta)

        u, v, w = x[CrazyFlie.U], x[CrazyFlie.V], x[CrazyFlie.W]
        uvw = jnp.array([u, v, w])

        p, q, r = x[CrazyFlie.P], x[CrazyFlie.Q], x[CrazyFlie.R]
        pqr = jnp.array([p, q, r])

        # Linear velocity
        R_W_cf = jnp.array(
            [
                [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
                [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
                [-s_th, c_th * s_phi, c_th * c_phi],
            ]
        )
        v_Wcf_cf = jnp.array([u, v, w])
        v_Wcf_W = R_W_cf @ v_Wcf_cf
        assert v_Wcf_W.shape == (3,)

        # Euler angle dynamics.
        mat = jnp.array(
            [
                [0, s_phi / c_th, c_phi / c_th],
                [0, c_phi, -s_phi],
                [1, s_phi * t_th, c_phi * t_th],
            ]
        )
        deuler_rpy = mat @ pqr
        deuler_ypr = deuler_rpy[::-1]

        # Body frame linear acceleration.
        acc_cf_g = -R_W_cf[2, :] * 9.81
        acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g

        # Body frame angular acceleration.
        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        rpq_dot = pqr_dot[::-1]
        assert pqr_dot.shape == (3,)

        x_dot = jnp.concatenate([v_Wcf_W, deuler_ypr, acc_cf, rpq_dot], axis=0)
        return x_dot

    # state_dot = crazyflie_f_(state)
    # all_obs_state_dot = jax.vmap(crazyflie_f_)(all_obs_state)
    #
    # # def h0_fn(self_state: Array, obs_state: Array) -> Array:
    # #     k_xdiff = state[:3] - all_obs_state[:, :3]
    # #     return jnp.square(k_xdiff).sum(axis=-1)
    # #
    # # h0 = h0_fn(state, all_obs_state)
    #
    # state_dot_dot = jnp.matmul(jax.jacfwd(crazyflie_f_)(state).T, state_dot)  # (nx,)
    # all_obs_state_dot_dot = jax.lax.batch_matmul(jax.vmap(jax.jacfwd(crazyflie_f_))(all_obs_state).transpose((0, 2, 1)),
    #                                              all_obs_state_dot[:, :, None]).squeeze(-1)  # (k, nx)
    #
    # k_xdiff = state[:3] - all_obs_state[:, :3]
    # k_vdiff = state_dot[:3] - all_obs_state_dot[:, :3]
    # k_adiff = state_dot_dot[:3] - all_obs_state_dot_dot[:, :3]
    #
    # k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    # k_h0_dot_dot = 2 * (k_vdiff ** 2 + k_xdiff * k_adiff).sum(axis=-1)
    # alpha0 = 1.0
    # alpha1 = 1.0
    #
    # k_h2 = k_h0_dot_dot + (alpha0 + alpha1) * k_h0_dot + alpha0 * alpha1 * k_h0

    def h0(x, obs_x):
        k_xdiff = x[:3] - obs_x[:, :3]
        dist = jnp.square(k_xdiff).sum(axis=-1)
        dist = dist.at[agent_idx].set(1e2)
        return dist[k_idx] - 4 * r ** 2  # (k,)

    def h1(x, obs_x):
        x_dot = crazyflie_f_(x)  # (nx,)
        obs_x_dot = jax.vmap(crazyflie_f_)(obs_x)  # (k, nx)

        h0_x = jax.jacfwd(h0, argnums=0)(x, obs_x)  # (k, nx)
        h0_obs_x = jax.jacfwd(h0, argnums=1)(x, obs_x)  # (k, k, nx)
        h0_dot = ei.einsum(h0_x, x_dot, 'k nx, nx -> k') + \
                 ei.einsum(h0_obs_x, obs_x_dot, 'k1 k2 nx, k2 nx -> k1')  # (k,)
        return h0_dot + 30.0 * h0(x, obs_x)

    def h2(x, obs_x):
        x_dot = crazyflie_f_(x)
        obs_x_dot = jax.vmap(crazyflie_f_)(obs_x)
        h1_x = jax.jacfwd(h1, argnums=0)(x, obs_x)  # (k, nx)
        h1_obs_x = jax.jacfwd(h1, argnums=1)(x, obs_x)  # (k, k, nx)
        h1_dot = ei.einsum(h1_x, x_dot, 'k nx, nx -> k') + \
                 ei.einsum(h1_obs_x, obs_x_dot, 'k1 k2 nx, k2 nx -> k1')  # (k,)
        return h1_dot + 50.0 * h1(x, obs_x)

    k_h2 = h2(state, all_obs_state)
    assert k_h2.shape == (k,)

    # k_vdiff = state_dot[0:3] - all_obs_state_dot[:, 0:3]
    #
    # k_vdiff = state[3:6] - all_obs_state[k_idx][:, 3:6]
    #
    # assert k_xdiff.shape == k_vdiff.shape == (k, 3)
    #
    # k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    # assert k_h0_dot.shape == (k,)
    #
    # k_h1 = k_h0_dot + 10.0 * k_h0

    k_isobs = k_idx >= n_agent

    # return k_h1, k_isobs
    return k_h2, k_isobs


def pwise_cbf_crazyflie(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_crazyflie_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def pwise_cbf_linear_drone_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    # state: ( 6, )
    n_agent = len(a_state)

    pos = state[:3]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :3]
    del o_obs_state

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account. Add some epsilon for qp solver error.
    k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

    k_h0 = k_dist_sq
    assert k_h0.shape == (k,)

    k_xdiff = state[:3] - all_obs_state[k_idx][:, :3]
    k_vdiff = state[3:6] - all_obs_state[k_idx][:, 3:6]
    assert k_xdiff.shape == k_vdiff.shape == (k, 3)

    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    assert k_h0_dot.shape == (k,)

    k_h1 = k_h0_dot + 3.0 * k_h0

    k_isobs = k_idx >= n_agent

    return k_h1, k_isobs


def pwise_cbf_linear_drone(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_linear_drone_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def pwise_cbf_cfhl_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    # state: ( 6, )
    n_agent = len(a_state)

    pos = state[:3]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :3]
    del o_obs_state

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account. Add some epsilon for qp solver error.
    k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

    k_h0 = k_dist_sq
    assert k_h0.shape == (k,)

    k_xdiff = state[:3] - all_obs_state[k_idx][:, :3]

    def get_v_single_(x):
        u = x[CFHL.U]
        v = x[CFHL.V]
        w = x[CFHL.W]

        R_W_cf = CFHL.get_rotation_mat(x)
        v_Wcf_cf = jnp.array([u, v, w])
        v_Wcf_W = R_W_cf @ v_Wcf_cf  # world frame velocity
        return v_Wcf_W

    k_vdiff = get_v_single_(state) - jax.vmap(get_v_single_)(all_obs_state[k_idx])


    # k_vdiff = state[3:6] - all_obs_state[k_idx][:, 3:6]
    assert k_xdiff.shape == k_vdiff.shape == (k, 3)

    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
    assert k_h0_dot.shape == (k,)

    k_h1 = k_h0_dot + 3 * k_h0

    k_isobs = k_idx >= n_agent

    return k_h1, k_isobs


def pwise_cbf_cfhl(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_cfhl_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def cbf_double_integrator(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    agent_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    obs_states = jnp.concatenate([agent_states, obs_states], axis=0)
    x_diff = agent_states[:, None, :2] - obs_states[None, :, :2]
    v_diff = agent_states[:, None, 2:] - obs_states[None, :, 2:]
    self_dist = jnp.concatenate(
        [jnp.eye(x_diff.shape[0]) * 1e2, jnp.zeros((x_diff.shape[0], x_diff.shape[1] - x_diff.shape[0]))],
        axis=-1,
    )
    h0 = (x_diff ** 2).sum(axis=-1) + self_dist ** 2 - 4 * r ** 2
    h0_dot = 2 * (x_diff * v_diff).sum(axis=-1)
    h1 = h0_dot + 10.0 * h0
    lidar_mask = jnp.ones((n_rays,))
    lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * n_agent)
    mask = jnp.concatenate([jnp.ones((n_agent, n_agent)), lidar_mask], axis=-1)
    h1 = h1 + jnp.logical_not(mask) * 1e6

    # (n_agents, )
    aargmin = jnp.argmin(h1, axis=-1)
    assert aargmin.shape == (n_agent,)

    # (n_agents,). True if agent-obs. The first n_agent elements are agent-agent.
    aisobs = aargmin >= n_agent

    # (n_agents, 1)
    h_minoverall = h1.min(axis=-1)[:, None]

    return h_minoverall, aisobs


def cbf_double_integrator_u_constrained(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    """From Safe Barrier Certificates for Collision-Free Multirobot Systems."""
    u_lim = 1.0  # by default
    agent_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    obs_states = jnp.concatenate([agent_states, obs_states], axis=0)
    x_diff = agent_states[:, None, :2] - obs_states[None, :, :2]
    v_diff = agent_states[:, None, 2:] - obs_states[None, :, 2:]
    self_dist = jnp.concatenate(
        [jnp.eye(x_diff.shape[0]) * 1e2, jnp.zeros((x_diff.shape[0], x_diff.shape[1] - x_diff.shape[0]))],
        axis=-1,
    )
    x_diff = x_diff + self_dist
    safe_dist = jnp.concatenate(
        [jnp.ones((n_agent, n_agent)) * r * 2, jnp.ones((n_agent, n_rays)) * r], axis=-1
    )
    h = jnp.sqrt(2 * (u_lim + u_lim) * (jnp.linalg.norm(x_diff, axis=-1) - safe_dist)) + x_diff.T @ v_diff / jnp.linalg.norm(x_diff, axis=-1)

    # (n_agents, )
    aargmin = jnp.argmin(h, axis=-1)
    assert aargmin.shape == (n_agent,)

    # (n_agents,). True if agent-obs. The first n_agent elements are agent-agent.
    aisobs = aargmin >= n_agent

    # (n_agents, 1)
    h_minoverall = h.min(axis=-1)[:, None]

    return h_minoverall, aisobs


def cbf_dubins_car(graph: GraphsTuple, r: float, n_agent: int, n_rays: int):
    agent_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    obs_states = jnp.concatenate([agent_states, obs_states], axis=0)
    x_diff = agent_states[:, None, :2] - obs_states[None, :, :2]
    agent_v = jnp.concatenate(
        [
            (agent_states[:, 3] * jnp.cos(agent_states[:, 2]))[:, None],
            (agent_states[:, 3] * jnp.sin(agent_states[:, 2]))[:, None],
        ],
        axis=-1,
    )
    obs_v = jnp.zeros((obs_states.shape[0], 2))
    obs_v = obs_v.at[:n_agent].set(agent_v)
    v_diff = agent_v[:, None, :] - obs_v[None, :, :]
    self_dist = jnp.concatenate(
        [jnp.eye(x_diff.shape[0]) * 1e2, jnp.zeros((x_diff.shape[0], x_diff.shape[1] - x_diff.shape[0]))],
        axis=-1,
    )
    h0 = (x_diff ** 2).sum(axis=-1) + self_dist ** 2 - 4 * r ** 2
    h0_dot = 2 * (x_diff * v_diff).sum(axis=-1)
    h1 = h0_dot + 10.0 * h0
    lidar_mask = jnp.ones((n_rays,))
    lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * n_agent)
    mask = jnp.concatenate([jnp.ones((n_agent, n_agent)), lidar_mask], axis=-1)
    h1 = h1 + jnp.logical_not(mask) * 1e6

    # (n_agents, )
    aargmin = jnp.argmin(h1, axis=-1)
    assert aargmin.shape == (n_agent,)

    # (n_agents,). True if agent-obs. The first n_agent elements are agent-agent.
    aisobs = aargmin >= n_agent

    # (n_agents, 1)
    h_minoverall = h1.min(axis=-1)[:, None]

    return h_minoverall, aisobs


def cbf_linear_drone(graph: GraphsTuple, r: float, n_agent: int, n_rays: int):
    agent_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    obs_states = jnp.concatenate([agent_states, obs_states], axis=0)
    x_diff = agent_states[:, None, :3] - obs_states[None, :, :3]
    v_diff = agent_states[:, None, 3:] - obs_states[None, :, 3:]
    self_dist = jnp.concatenate(
        [jnp.eye(x_diff.shape[0]) * 1e2, jnp.zeros((x_diff.shape[0], x_diff.shape[1] - x_diff.shape[0]))],
        axis=-1,
    )
    h0 = (x_diff ** 2).sum(axis=-1) + self_dist ** 2 - 4 * r ** 2
    h0_dot = 2 * (x_diff * v_diff).sum(axis=-1)
    h1 = h0_dot + 15.0 * h0
    lidar_mask = jnp.ones((n_rays,))
    lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * n_agent)
    mask = jnp.concatenate([jnp.ones((n_agent, n_agent)), lidar_mask], axis=-1)
    h1 = h1 + jnp.logical_not(mask) * 1e6

    # (n_agents, )
    aargmin = jnp.argmin(h1, axis=-1)
    assert aargmin.shape == (n_agent,)

    # (n_agents,). True if agent-obs. The first n_agent elements are agent-agent.
    aisobs = aargmin >= n_agent

    # (n_agents, 1)
    h_minoverall = h1.min(axis=-1)[:, None]

    return h_minoverall, aisobs


def cbf_crazyflie(graph: GraphsTuple, r: float, n_agent: int, n_rays: int):
    agent_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    obs_states = jnp.concatenate([agent_states, obs_states], axis=0)
    x_diff = agent_states[:, None, :3] - obs_states[None, :, :3]
    v_diff = agent_states[:, None, 6:9] - obs_states[None, :, 6:9]
    self_dist = jnp.concatenate(
        [jnp.eye(x_diff.shape[0]) * 1e2, jnp.zeros((x_diff.shape[0], x_diff.shape[1] - x_diff.shape[0]))],
        axis=-1,
    )
    h0 = (x_diff ** 2).sum(axis=-1) + self_dist ** 2 - 4 * r ** 2
    h0_dot = 2 * (x_diff * v_diff).sum(axis=-1)
    h1 = h0_dot + 50.0 * h0
    lidar_mask = jnp.ones((n_rays,))
    lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * n_agent)
    mask = jnp.concatenate([jnp.ones((n_agent, n_agent)), lidar_mask], axis=-1)
    h1 = h1 + jnp.logical_not(mask) * 1e6

    # (n_agents, )
    aargmin = jnp.argmin(h1, axis=-1)
    assert aargmin.shape == (n_agent,)

    # (n_agents,). True if agent-obs. The first n_agent elements are agent-agent.
    aisobs = aargmin >= n_agent

    # (n_agents, 1)
    h_minoverall = h1.min(axis=-1)[:, None]

    return h_minoverall, aisobs


def get_cbf_fn(env: MultiAgentEnv, input_constraint: bool = False):
    if input_constraint:
        if isinstance(env, DoubleIntegrator):
            return ft.partial(cbf_double_integrator_u_constrained, r=env.params["car_radius"], n_agent=env.num_agents,
                              n_rays=env.params["n_rays"])
        else:
            raise NotImplementedError
    if isinstance(env, SingleIntegrator):
        return ft.partial(
            cbf_single_integrator, r=env.params["car_radius"], n_agent=env.num_agents, n_rays=env.params["n_rays"]
        )
    elif isinstance(env, DoubleIntegrator):
        return ft.partial(
            cbf_double_integrator, r=env.params["car_radius"], n_agent=env.num_agents, n_rays=env.params["n_rays"]
        )
    elif isinstance(env, DubinsCar):
        return ft.partial(
            cbf_dubins_car, r=env.params["car_radius"], n_agent=env.num_agents, n_rays=env.params["n_rays"]
        )
    elif isinstance(env, CrazyFlie):
        return ft.partial(cbf_crazyflie, r=env.params["drone_radius"], n_agent=env.num_agents, n_rays=env.n_rays)
    elif isinstance(env, LinearDrone):
        return ft.partial(cbf_linear_drone, r=env.params["drone_radius"], n_agent=env.num_agents, n_rays=env.n_rays)
    else:
        raise NotImplementedError


def get_pwise_cbf_fn(env: MultiAgentEnv, k: int = 3, input_constraint: bool = False):
    if input_constraint:
        if isinstance(env, DoubleIntegrator):
            return ft.partial(pwise_cbf_double_integrator_u_constrained, r=env.params["car_radius"], n_agent=env.num_agents,
                              n_rays=env.params["n_rays"], k=k)
        else:
            raise NotImplementedError
    if isinstance(env, SingleIntegrator):
        n_agent = env.num_agents
        n_rays = env.params["n_rays"]
        r = env.params["car_radius"]
        return ft.partial(pwise_cbf_single_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
    elif isinstance(env, DoubleIntegrator):
        n_agent = env.num_agents
        n_rays = env.params["n_rays"]
        r = env.params["car_radius"]
        return ft.partial(pwise_cbf_double_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
    elif isinstance(env, DubinsCar):
        r = env.params["car_radius"]
        n_agent = env.num_agents
        n_rays = env.params["n_rays"]
        return ft.partial(pwise_cbf_dubins_car, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
    elif isinstance(env, CrazyFlie):
        return ft.partial(pwise_cbf_crazyflie, r=env.params["drone_radius"], n_agent=env.num_agents, n_rays=env.n_rays,
                          k=k)
    elif isinstance(env, LinearDrone):
        return ft.partial(pwise_cbf_linear_drone, r=env.params["drone_radius"], n_agent=env.num_agents,
                          n_rays=env.n_rays, k=k)
    elif isinstance(env, CFHL):
        return ft.partial(pwise_cbf_cfhl, r=env.params["drone_radius"], n_agent=env.num_agents,
                          n_rays=env.n_rays, k=k)

    raise NotImplementedError("")
