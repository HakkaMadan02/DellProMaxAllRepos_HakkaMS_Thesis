import functools as ft
import pathlib
from typing import NamedTuple, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import jacobian
from jax.lax import while_loop
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..trainer.utils import centered_norm, jax2np, np2jax
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import ABool, Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import MutablePatchCollection, merge01, save_anim, tree_index
from .base import MultiAgentEnv, RolloutResult
from .linear_drone import LinearDrone
from .obstacle import Cuboid, Obstacle, Sphere
from .plot import get_BuRd, get_cuboid_collection, plot_graph, get_sphere_collection
from .utils import get_lidar, inside_obstacles, lqr, RK4_step


class CrazyFlie(LinearDrone):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "drone_radius": 0.05,
        "comm_radius": 1.0,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 0,
        "m": 0.0299,
        "Ixx": 1.395 * 10 ** (-5),
        "Iyy": 1.395 * 10 ** (-5),
        "Izz": 2.173 * 10 ** (-5),
        "CT": 3.1582 * 10 ** (-10),
        "CD": 7.9379 * 10 ** (-12),
        "d": 0.03973,
    }

    # State indices
    X = 0
    Y = 1
    Z = 2

    PSI = 3
    THETA = 4
    PHI = 5

    U = 6
    V = 7
    W = 8

    R = 9
    Q = 10
    P = 11

    # Control indices
    F_1 = 0
    F_2 = 1
    F_3 = 2
    F_4 = 3

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None,
    ):
        super(LinearDrone, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = 1
        self.n_rays = min(self.n_rays, self.params["n_rays"] ** 2 // 2 + 2)
        nominal_goal = jnp.zeros((self.num_agents, self.state_dim))

        self.use_kunal = False
        if self.use_kunal:
            logger.info("Using CF dynamics with errors!")
        else:
            logger.info("Using corrected CF dynamics!")

        self.normalize_by_CT = True
        if self.normalize_by_CT:
            logger.info("Normalizing by CT so that controls are forces.")
        else:
            logger.info("Not normalizing by CT. controls are rotor velocities.")

        A = self._A(nominal_goal)
        B = self._B(nominal_goal)
        Q = np.eye(self.state_dim)
        # Q = np.diag([1e0, 1e0, 1e0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        R = np.eye(self.action_dim) * 1.
        K_np = lqr(jax2np(A), jax2np(B), Q, R)
        # print('K_np: {}', K_np)
        self._K = np2jax(K_np)

    @property
    def state_dim(self) -> int:
        return 12  # x, y, z, psi, theta, phi, u, v, w, r, q, p

    @property
    def node_dim(self) -> int:
        # indicator: agent: 001, goal: 010, obstacle: 100
        # psi, theta, phi
        # u_ref
        return 10

    @property
    def edge_dim(self) -> int:
        return 12  # relative position, relative velocity, relative z direction, relative P Q R (world frame)

    @property
    def action_dim(self) -> int:
        return 4  # thrusts

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    def reset_np(self, key: Array) -> GraphsTuple:
        """an easy scenario just for debugging"""
        self._t = 0
        side_length = self.area_size
        states = jnp.zeros((self.num_agents, self.state_dim))
        goals = jnp.zeros((self.num_agents, self.state_dim))

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0

        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 3), minval=0, maxval=side_length)

        r_key, key = jr.split(key, 2)
        obs_radius = jr.uniform(r_key, (n_rng_obs,),
                                minval=self._params["obs_len_range"][0] / 2,
                                maxval=self._params["obs_len_range"][1] / 2)
        obstacles = self.create_obstacles(obs_pos, obs_radius)

        # randomly generate positions of agents and goals
        max_iter = 1024  # maximum number of iterations to find a valid initial state/goal

        # generate agents with equal distance
        # agent_interval = side_length / (self.num_agents // 2 + 1)
        # for i in range(self.num_agents // 2):
        #     states = states.at[i].set(jnp.array([agent_interval * (i + 1), 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        #     states = states.at[i + self.num_agents // 2].set(jnp.array([agent_interval * (i + 1), side_length - 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        #     goals = goals.at[i].set(jnp.array([agent_interval * (i + 1), side_length - 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        #     goals = goals.at[i + self.num_agents // 2].set(jnp.array([agent_interval * (i + 1), 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        # generate agents and goals on a circle
        r = side_length / 2 - 0.1
        for i in range(self.num_agents):
            angle = 2 * np.pi * i / self.num_agents
            states = states.at[i].set(jnp.array([r * np.cos(angle) + side_length / 2, r * np.sin(angle) + side_length / 2, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            goals = goals.at[i].set(jnp.array([r * np.cos(angle + np.pi) + side_length / 2, r * np.sin(angle + np.pi) + side_length / 2, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        env_states = self.EnvState(states, goals, obstacles)
        return self.get_graph(env_states)

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        # jax.debug.print('action: {}', action - self.u_ref(graph))
        # jax.debug.print('action: {}', action)
        action = self.clip_action(action) + self.u_eq
        # action = action + self.u_eq

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_rk4(agent_states, action)
        #
        # next_agent_states = jnp.matmul(agent_states, self._A.T) + jnp.matmul(action, self._B.T)
        # next_states = jnp.concatenate([next_agent_states, goal_states], axis=0)
        # next_states = self.clip_state(next_states)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        # dist2goal = jnp.linalg.norm(goal_states[:, :3] - next_agent_states[:, :3], axis=-1)
        # reward -= (dist2goal.mean()) ** 2
        reward -= (jnp.linalg.norm(action - self.u_ref(graph) - self.u_eq, axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goal_states, obstacles)

        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            agent_pos = agent_states[:, :3]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["drone_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def _single_agent_f(self, x: Array) -> Array:
        assert x.shape == (self.state_dim,)
        Ixx, Iyy, Izz = self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
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
        # R_W_cf = jnp.array(
        #     [
        #         [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
        #         [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
        #         [-s_th, c_th * s_phi, c_th * c_phi],
        #     ]
        # )
        R_W_cf = self.get_rotation_mat(x)
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
        deuler_ypr = mat @ pqr

        # Body frame linear acceleration.
        acc_cf_g = -R_W_cf[2, :] * 9.81
        acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g

        # Body frame angular acceleration.
        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        rpq_dot = pqr_dot[::-1]
        assert pqr_dot.shape == (3,)

        x_dot = jnp.concatenate([v_Wcf_W, deuler_ypr, acc_cf, rpq_dot], axis=0)
        assert x_dot.shape == (self.state_dim,)

        return x_dot

    def _single_agent_g(self, x: Array) -> Array:
        assert x.shape == (self.state_dim,)

        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]

        """
               3   ↑P  0
                  ╲ ╱   
               R ← ╳ 
                  ╱ ╲   
               2       1
        """

        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        dw_du = CT * jnp.full(4, 1 / m)
        dp_du = CT * np.sqrt(2) * d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * jnp.array([-1.0, 1.0, -1.0, 1.0]) / Izz
        assert dw_du.shape == dp_du.shape == dq_du.shape == dr_du.shape == (self.action_dim,)

        g = jnp.zeros((self.state_dim, self.action_dim))
        g = g.at[CrazyFlie.W, :].set(dw_du)
        g = g.at[CrazyFlie.P, :].set(dp_du)
        g = g.at[CrazyFlie.Q, :].set(dq_du)
        g = g.at[CrazyFlie.R, :].set(dr_du)

        return g

    def _single_agent_f_kunal(self, x: Array) -> Array:
        assert x.shape == (self.state_dim,)

        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]

        f = jnp.zeros_like(x)
        psi = x[CrazyFlie.PSI]
        theta = x[CrazyFlie.THETA]
        phi = x[CrazyFlie.PHI]
        u = x[CrazyFlie.U]
        v = x[CrazyFlie.V]
        w = x[CrazyFlie.W]
        r = x[CrazyFlie.R]
        q = x[CrazyFlie.Q]
        p = x[CrazyFlie.P]

        s_psi = jnp.sin(psi)
        s_phi = jnp.sin(phi)
        s_the = jnp.sin(theta)
        c_psi = jnp.cos(psi)
        c_phi = jnp.cos(phi)
        c_the = jnp.cos(theta)
        t_psi = jnp.tan(psi)
        t_phi = jnp.tan(phi)
        t_the = jnp.tan(theta)

        f = f.at[CrazyFlie.X].set(
            w * (s_psi * s_phi + c_psi * c_phi * s_the)
            - v * (s_psi * c_phi - c_psi * s_phi * s_the)
            + u * c_psi * c_the
        )
        f = f.at[CrazyFlie.Y].set(
            v * (c_psi * c_phi + s_psi * s_phi * s_the)
            - w * (c_psi * s_phi - s_psi * c_phi * s_the)
            + u * s_psi * c_the
        )
        f = f.at[CrazyFlie.Z].set(w * c_psi * c_phi - u * s_the + v * s_phi * c_the)
        f = f.at[CrazyFlie.PSI].set(p + r * c_phi * t_the + q * s_phi * t_the)
        f = f.at[CrazyFlie.THETA].set(q * c_phi - r * s_phi)
        f = f.at[CrazyFlie.PHI].set(r * c_phi / (c_the + 1e-6) + q * s_phi / (c_the + 1e-6))
        f = f.at[CrazyFlie.U].set(r * v - q * w + 9.81 * s_the)
        f = f.at[CrazyFlie.V].set(p * w - r * u - 9.81 * s_phi * c_the)
        f = f.at[CrazyFlie.W].set(q * u - p * v - 9.81 * c_phi * c_the)
        f = f.at[CrazyFlie.R].set((Ixx - Iyy) / Izz * p * q)
        f = f.at[CrazyFlie.Q].set((Izz - Ixx) / Iyy * p * r)
        f = f.at[CrazyFlie.P].set((Izz - Iyy) / Ixx * p * q)

        return f

    def _single_agent_g_kunal(self, x: Array) -> Array:
        assert x.shape == (self.state_dim,)
        g = jnp.zeros((self.state_dim, self.action_dim))

        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]

        g = g.at[CrazyFlie.W, CrazyFlie.F_1].set(1 / m)
        g = g.at[CrazyFlie.W, CrazyFlie.F_2].set(1 / m)
        g = g.at[CrazyFlie.W, CrazyFlie.F_3].set(1 / m)
        g = g.at[CrazyFlie.W, CrazyFlie.F_4].set(1 / m)

        g = g.at[CrazyFlie.R, CrazyFlie.F_1].set((1 / Izz) * (-jnp.sqrt(2) * d))
        g = g.at[CrazyFlie.R, CrazyFlie.F_2].set((1 / Izz) * (-jnp.sqrt(2) * d))
        g = g.at[CrazyFlie.R, CrazyFlie.F_3].set((1 / Izz) * (jnp.sqrt(2) * d))
        g = g.at[CrazyFlie.R, CrazyFlie.F_4].set((1 / Izz) * (jnp.sqrt(2) * d))

        g = g.at[CrazyFlie.Q, CrazyFlie.F_1].set((1 / Iyy) * (-jnp.sqrt(2) * d))
        g = g.at[CrazyFlie.Q, CrazyFlie.F_2].set((1 / Iyy) * (jnp.sqrt(2) * d))
        g = g.at[CrazyFlie.Q, CrazyFlie.F_3].set((1 / Iyy) * (jnp.sqrt(2) * d))
        g = g.at[CrazyFlie.Q, CrazyFlie.F_4].set((1 / Iyy) * (-jnp.sqrt(2) * d))

        g = g.at[CrazyFlie.P, CrazyFlie.F_1].set((1 / Ixx) * (-CD / CT))
        g = g.at[CrazyFlie.P, CrazyFlie.F_2].set((1 / Ixx) * (CD / CT))
        g = g.at[CrazyFlie.P, CrazyFlie.F_3].set((1 / Ixx) * (-CD / CT))
        g = g.at[CrazyFlie.P, CrazyFlie.F_4].set((1 / Ixx) * (CD / CT))

        return g

    def _f(self, agent_state: Array) -> Array:
        if self.use_kunal:
            return jax.vmap(self._single_agent_f_kunal)(agent_state)
        else:
            return jax.vmap(self._single_agent_f)(agent_state)

    def _g(self, agent_state: Array) -> Array:
        if self.use_kunal:
            return jax.vmap(self._single_agent_g_kunal)(agent_state)
        else:
            return jax.vmap(self._single_agent_g)(agent_state)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        f = self._f(agent_states)
        g = self._g(agent_states)
        return f + jnp.einsum("ijk,ik->ij", g, action)

    def agent_step_rk4(self, agent_state: Array, action: Array) -> Array:
        assert agent_state.shape == (self.num_agents, self.state_dim)
        assert action.shape == (self.num_agents, self.action_dim)
        n_state_agent_new = RK4_step(self.agent_xdot, agent_state, action, self.dt)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        # return n_state_agent_new
        return self.clip_state(n_state_agent_new)

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = self._f(state)
        g = self._g(state)
        return f, g

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, self.node_dim))
        node_feats = node_feats.at[:self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats
        node_feats = node_feats.at[:self.num_agents, 3:6].set(state.agent[:, 3:6])  # agent Euler angles
        u_ref = self.u_ref_inner(state.agent, state.goal)
        node_feats = node_feats.at[:self.num_agents, 6:].set(u_ref)  # u_ref
        # node_feats = node_feats.at[:self.num_agents, 6:].set(state.goal[:, :3] - state.agent[:, :3])  # agent pos - goal

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(LinearDrone.GOAL)
        node_type = node_type.at[-n_hits:].set(LinearDrone.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self.params["n_rays"],
                sense_range=self._params["comm_radius"],
                max_returns=self.n_rays,
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :3]))
        lidar_data = jnp.concatenate(
            [lidar_data, jnp.zeros((lidar_data.shape[0], self.state_dim - lidar_data.shape[1]))], axis=-1
        )
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def get_rotation_mat(self, x: State) -> State:
        phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]
        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        R_W_cf = jnp.array(
            [
                [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
                [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
                [-s_th, c_th * s_phi, c_th * c_phi],
            ]
        )
        return R_W_cf

    def edge_state(self, state: State) -> State:

        def edge_state_fn_(x: State) -> State:
            pos_W = x[:3]  # world frame position
            u = x[CrazyFlie.U]
            v = x[CrazyFlie.V]
            w = x[CrazyFlie.W]

            # calculate rotation matrix
            # phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]
            # c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
            # c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            # c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
            # # t_th = jnp.tan(theta)
            # R_W_cf = jnp.array(
            #     [
            #         [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
            #         [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
            #         [-s_th, c_th * s_phi, c_th * c_phi],
            #     ]
            # )
            R_W_cf = self.get_rotation_mat(x)
            v_Wcf_cf = jnp.array([u, v, w])
            v_Wcf_W = R_W_cf @ v_Wcf_cf  # world frame velocity

            z_W = R_W_cf[:, 2]  # world frame z-axis

            r = x[CrazyFlie.R]
            q = x[CrazyFlie.Q]
            p = x[CrazyFlie.P]
            omega = jnp.array([p, q, r])
            omega_W = R_W_cf @ omega  # world frame angular velocity

            return jnp.concatenate([pos_W, v_Wcf_W, z_W, omega_W], axis=-1)

        return jax.vmap(edge_state_fn_)(state)

        # pos = state[:, :3]
        # vel = state[]
        #
        #
        # return jnp.concatenate(
        #     [
        #         state[:, :3],
        #         jnp.cos(state[:, 3])[:, None],
        #         jnp.sin(state[:, 3])[:, None],
        #         state[:, 4:],
        #     ],
        #     axis=1,
        # )

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_state = self.edge_state(state)
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def edge_blocks(self, state: EnvState, lidar_data: Pos3d) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays

        # agent - agent connection
        agent_pos = state.agent[:, :3]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_edge_state = self.edge_state(state.agent)
        state_diff = agent_edge_state[:, None, :] - agent_edge_state[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        goal_edge_state = self.edge_state(state.goal)
        agent_goal_feats = agent_edge_state[:, None, :] - goal_edge_state[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :3].set(agent_goal_feats[:, :3] * coef)
        agent_goal_edges = EdgeBlock(agent_goal_feats, agent_goal_mask, id_agent, id_goal)

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        lidar_edge_state = self.edge_state(lidar_data)
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :3]
            lidar_feats = agent_edge_state[i, :] - lidar_edge_state[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self.n_rays))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        # x, y, z, psi, theta, phi, u, v, w, r, q, p
        low_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf,
                             -jnp.inf, -jnp.pi / 4, -jnp.pi / 4,
                             -0.3, -0.3, -0.3,
                             -10, -10, -10])
        up_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf,
                            jnp.inf, jnp.pi / 4, jnp.pi / 4,
                            0.3, 0.3, 0.3,
                            10, 10, 10])
        return low_lim, up_lim

    def action_lim(self) -> Tuple[Action, Action]:
        low_lim = jnp.array([-1.0, -1.0, -1.0, -1.0])
        up_lim = jnp.array([1.0, 1.0, 1.0, 1.0])
        return low_lim, up_lim

    @property
    def u_eq(self):
        """Return the equilibrium state.
        We want total thrust equals m*g.
        """
        u_eq = jnp.zeros((self.action_dim,))
        u_eq = u_eq.at[CrazyFlie.F_1].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_2].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_3].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_4].set(self._params["m"] * 9.81 / 4)

        if not self.normalize_by_CT:
            u_eq = u_eq / self._params["CT"]

        return u_eq

    def _A(self, goals: AgentState) -> Array:
        u0 = self.u_eq  # .repeat(self.num_agents, 1)

        def dynamics(x):
            assert x.shape == (self.state_dim,)
            f = self._f(x[None, :]).squeeze()
            g = self._g(x[None, :]).squeeze()
            xdot = f + jnp.einsum("ij,j->i", g, u0)
            return xdot

        # for i in range(goals.shape[0]):
        #     A = jacobian(dynamics, goals[None, 0]).squeeze().cpu().numpy() * self._dt + np.eye(self.state_dim)
        A = jacobian(dynamics)(goals[0]) * self._dt + np.eye(self.state_dim)

        # A = jacobian(dynamics, goals[None, 0]).squeeze().cpu().numpy() * self._dt + np.eye(self.state_dim)
        return A

    def _B(self, goals: AgentState) -> Array:
        B = self._g(goals)[0]
        return B * self._dt

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        return self.u_ref_inner(agent, goal)
        # error = goal - agent
        # error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        # error = jnp.clip(error, -error_max, error_max)
        # return self.clip_action(error @ self._K.T)

    def u_ref_inner(self, state: State, goal: State):
        error = goal - state
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self.n_rays)
        action = self.clip_action(action) + self.u_eq

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_rk4(agent_states, action)


        # n_rng_obs = self._params["n_obs"]
        # assert n_rng_obs >= 0
        #
        # obs_pos = jnp.zeros((n_rng_obs, 3))
        #
        # obs_radius = jnp.zeros((n_rng_obs,))
        # obstacles = self.create_obstacles(obs_pos, obs_radius)
        #
        # env_state = self.EnvState(next_agent_states, goal_states, obstacles)
        # return self.get_graph(env_state)

        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :3]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2.5)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # agents to not have too much pitch or roll
        # def large_angle(x):
        #     phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]
        #
        #     c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        #     c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        #     c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        #     R_W_cf = jnp.array(
        #         [
        #             [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
        #             [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
        #             [-s_th, c_th * s_phi, c_th * c_phi],
        #         ]
        #     )
        #     z_dir = R_W_cf[:, 2]
        #     z_abs = z_dir[2]
        #     max_angle = 5.0 / 180.0 * jnp.pi
        #     return jnp.array(jnp.abs(z_abs) <= jnp.cos(max_angle))
        #
        # large_angle_mask = jax.vmap(large_angle)(agent_state)
        # assert large_angle_mask.shape == (self.num_agents,)

        return collision_mask

    def render_video(
            self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None,
            dpi: int = 80, **kwargs
    ) -> None:
        fig = plt.figure(figsize=(20, 10), dpi=dpi)
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]
        axs[0].view_init(elev=90, azim=0, roll=0)
        axs[1].view_init(elev=0, azim=0, roll=0)
        self.setup_ax(axs[0])
        self.setup_ax(axs[1])

        if viz_opts is None:
            viz_opts = {}

        T_graph = rollout.Tp1_graph

        graph0 = tree_index(T_graph, 0)
        n_nodes = len(graph0.states[:-1])

        agent_color = "#FF8C00"
        goal_color = "#3CB371"
        edge_goal_color = goal_color

        # plot obstacles
        obs = graph0.env_states.obstacle
        n_obs = len(obs.center)
        for i_ax in range(2):
            obs_col = get_sphere_collection(obs, alpha=0.8, facecolor="#710000")
            axs[i_ax].add_collection(obs_col)
        # obs_col = get_cuboid_collection(obs, alpha=0.8, linewidth=1.0, edgecolor='k', facecolor="#710000")
        # axs[0].add_collection(obs_col)
        # axs[1].add_collection(obs_col)

        # plot agents
        n_radius = np.array([self._params["drone_radius"]] * self.num_agents * 2)
        n_color = [agent_color] * self.num_agents + [goal_color] * self.num_agents
        n_pos = graph0.states[:self.num_agents * 2, :3]
        plot_r = axs[0].transData.transform([self._params["drone_radius"], 0])[0] - axs[0].transData.transform([0, 0])[0]
        # agent_col = ax.plot(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2], markersize=20, mfc=agent_color, linestyle="", marker='o')[0]  # todo: the size of the agent
        agent_col = axs[0].scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent
        agent_col2 = axs[1].scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)

        # plot edges
        n_hits = self.num_agents * self.n_rays
        all_pos = graph0.states[:self.num_agents * 2 + n_hits, :3]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents * 2 + n_hits, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_colors = ["0.2" for ii in range(len(e_start))]
        e_lines = np.stack([e_start, e_end], axis=1)
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        axs[0].add_collection(edge_col)
        edge_col2 = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        axs[1].add_collection(edge_col2)
        # axs[1].add_collection(edge_col)

        # edge_col = []
        cnt_col = []

        # text for cost and reward
        cost_text = axs[0].text(0.02, 0.02, 1.00, "Cost: 1.0, Reward: 1.0", transform=axs[0].transAxes, va="bottom")
        # text for z angle
        z_angle_text = axs[0].text(0.02, 0.02, 1.15, "Z angle: 0.0", transform=axs[0].transAxes, va="bottom")
        # text for action
        action_text = axs[0].text(0.02, 0.02, 1.30, "Action: 0.0", transform=axs[0].transAxes, va="bottom")
        # Text for safety.
        safe_text = []
        # if "inside_obstacles" in rollout.T_info:
        if Ta_is_unsafe is not None:
            safe_text = [axs[0].text(0.02, 0.02, 0.9, "Unsafe: {}", transform=axs[0].transAxes, va="bottom")]

        font_opts = dict(
            size=12,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=axs[0].transData,
            clip_on=True,
            zorder=7,
        )
        agent_labels = []
        for ii in range(self.num_agents):
            pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], axs[0].get_proj())[:2]
            agent_labels.append(axs[0].text2D(pos2d[0], pos2d[1], f"{ii}", **font_opts))

        font_opts2 = dict(
            size=12,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=axs[1].transData,
            clip_on=True,
            zorder=7,
        )
        agent_labels2 = []
        for ii in range(self.num_agents):
            pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], axs[1].get_proj())[:2]
            agent_labels2.append(axs[1].text2D(pos2d[0], pos2d[1], f"{ii}", **font_opts2))


        # Text for timestep.
        kk_text = axs[0].text2D(0.99, 0.99, "kk=0", transform=axs[0].transAxes, va="top", ha="right")

        def init_fn() -> list[plt.Artist]:
            return [agent_col, agent_col2, *agent_labels, *agent_labels2, cost_text, z_angle_text, action_text, *safe_text, *cnt_col, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos = graph.states[:self.num_agents * 2, :3]

            # update agent positions
            # agent_col.set_data(n_pos[:, 0], n_pos[:, 1])
            # agent_col.set_3d_properties(n_pos[:, 2])

            agent_col.set_offsets(n_pos[:, :2])
            agent_col.set_3d_properties(n_pos[:, 2], zdir='z')
            agent_col2.set_offsets(n_pos[:, :2])
            agent_col2.set_3d_properties(n_pos[:, 2], zdir='z')
            # for ii in range(self.num_agents):
            #     agent_col[ii].set_data(n_pos[ii, 0], n_pos[ii, 1])
            #     agent_col[ii].set_3d_properties(n_pos[ii, 2])

            # Update edges.
            all_pos = graph.states[:self.num_agents * 2 + n_hits, :3]
            e_edge_index = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad = np.any(e_edge_index == self.num_agents * 2 + n_hits, axis=0)
            e_edge_index = e_edge_index[:, ~is_pad]
            e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]

            # goal => num_agents <= idx < num_agents * 2
            e_is_goal = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents * 2)
            e_is_goal = e_is_goal[~is_pad]
            e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]

            # (e, n_pts, 2)
            e_lines = np.stack([e_start, e_end], axis=1)
            edge_col.set_segments(e_lines)
            edge_col.set_colors(e_colors)
            edge_col2.set_segments(e_lines)
            edge_col2.set_colors(e_colors)

            # update agent labels
            for ii in range(self.num_agents):
                text_pos = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], axs[0].get_proj())[:2]
                agent_labels[ii].set_position(text_pos)
            for ii in range(self.num_agents):
                text_pos = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], axs[1].get_proj())[:2]
                agent_labels2[ii].set_position(text_pos)
                # agent_labels[ii].set_position(n_pos[ii])

            # update cost and safe labels
            if kk < len(rollout.T_cost):
                cost_text.set_text(
                    "Cost: {:5.2f}, Reward: {:5.2f}".format(rollout.T_cost[kk], rollout.T_reward[kk]))
            else:
                cost_text.set_text("")

            # update z angle text
            agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
            R_W_cf = jax.vmap(self.get_rotation_mat)(agent_states)
            z_dir = R_W_cf[:, :, 2]  # (n_agents, 3)
            z_angle = np.arctan2(np.linalg.norm(z_dir[:, :2], axis=-1), z_dir[:, 2])
            np.set_printoptions(precision=3)
            z_angle_text.set_text(f"z angle: {np.rad2deg(z_angle)}")

            # update action text
            if kk < len(rollout.T_action):
                action_text.set_text(f"action: {rollout.T_action[kk]}")

            if len(safe_text) > 0:
                if kk < len(Ta_is_unsafe):
                    a_is_safe = Ta_is_unsafe[kk]
                    safe_text[0].set_text("Unsafe: {}".format(1 * a_is_safe))
                else:
                    cost_text.set_text("")

            kk_text.set_text("kk={:04}".format(kk))
            return [agent_col, agent_col2, *agent_labels, *agent_labels2, cost_text, action_text, *safe_text, *cnt_col, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)
