import functools as ft
import pathlib
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import scipy

from typing import NamedTuple, Tuple, Optional
from jax.lax import while_loop
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..trainer.utils import centered_norm, np2jax
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import ABool, Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import MutablePatchCollection, merge01, save_anim, tree_index
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Cuboid, Sphere
from .plot import plot_graph, get_BuRd, get_cuboid_collection, get_sphere_collection
from .utils import get_lidar, inside_obstacles, lqr


class LinearDrone(MultiAgentEnv):
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
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.15, 0.3],
        "n_obs": 4
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        super(LinearDrone, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

        self._A = np.zeros((self.state_dim, self.state_dim))
        self._A[0, 3] = 1.
        self._A[1, 4] = 1.
        self._A[2, 5] = 1.
        self._A[3, 3] = -1.1
        self._A[4, 4] = -1.1
        self._A[5, 5] = -6.
        A_discrete = scipy.linalg.expm(self._A * self._dt)

        self._B = np.zeros((self.state_dim, self.action_dim))
        self._B[3, 0] = 10.
        self._B[4, 1] = 10.
        self._B[5, 2] = 10.  # 6.0

        self._Q = np.diag([5e1, 5e1, 5e1, 1., 1., 1.])
        self._R = np.eye(self.action_dim)
        self._K = lqr(A_discrete, self._B, self._Q, self._R)
        # self.create_obstacles = jax.vmap(Cuboid.create)
        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = 16  # consider top k rays

    @property
    def state_dim(self) -> int:
        return 6  # x, y, z, vx, vy, vz

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 6  # x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel

    @property
    def action_dim(self) -> int:
        return 3  # ax, ay, az

    # def reset_point_obs(self, key: Array) -> GraphsTuple:
    #

    def reset(self, key: Array) -> GraphsTuple:
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

        def get_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
            i_iter, this_key, _, all_nodes = reset_input
            use_key, this_key = jr.split(this_key, 2)
            i_iter += 1
            return i_iter, this_key, jr.uniform(use_key, (3,), minval=0, maxval=side_length), all_nodes

        def non_valid_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
            i_iter, _, node, all_nodes = reset_input
            dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
            collide = dist_min <= self._params["drone_radius"] * 4
            inside = inside_obstacles(node, obstacles, r=self._params["drone_radius"] * 4)
            valid = ~(collide | inside) | (i_iter >= max_iter)
            return ~valid

        def get_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
            # key, goal_candidate, agent_start_pos, all_goals
            i_iter, this_key, _, agent, all_goals = reset_input
            use_key, this_key = jr.split(this_key, 2)
            i_iter += 1
            if self.max_travel is None:
                return i_iter, this_key, jr.uniform(use_key, (3,), minval=0, maxval=side_length), agent, all_goals
            else:
                return i_iter, this_key, \
                       jr.uniform(use_key, (3,), minval=-self.max_travel, maxval=self.max_travel) + agent, \
                       agent, all_goals

        def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
            # key, goal_candidate, agent_start_pos, all_goals
            i_iter, _, goal, agent, all_goals = reset_input
            dist_min = jnp.linalg.norm(all_goals - goal, axis=1).min()
            collide = dist_min <= self._params["drone_radius"] * 4
            inside = inside_obstacles(goal, obstacles, r=self._params["drone_radius"] * 4)
            outside = jnp.any(goal < 0) | jnp.any(goal > side_length)
            if self.max_travel is None:
                too_long = np.array(False, dtype=bool)
            else:
                too_long = jnp.linalg.norm(goal - agent) > self.max_travel
            valid = (~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
            out = ~valid
            assert out.shape == tuple() and out.dtype == jnp.bool_
            return out

        def reset_body(reset_input: Tuple[int, Array, Array, Array]):
            # agent_id, key, states, goals
            agent_id, this_key, all_states, all_goals = reset_input
            agent_key, goal_key, this_key = jr.split(this_key, 3)
            agent_candidate = jr.uniform(agent_key, (3,), minval=0, maxval=side_length)
            n_iter_agent, _, agent_candidate, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node,
                init_val=(0, agent_key, agent_candidate, all_states[:, :3])
            )
            all_states = all_states.at[agent_id, :3].set(agent_candidate)

            if self.max_travel is None:
                goal_candidate = jr.uniform(goal_key, (3,), minval=0, maxval=side_length)
            else:
                goal_candidate = jr.uniform(goal_key, (3,), minval=-self.max_travel, maxval=self.max_travel) + agent_candidate

            n_iter_goal, _, goal_candidate, _, _ = while_loop(
                cond_fun=non_valid_goal, body_fun=get_goal,
                init_val=(0, goal_key, goal_candidate, agent_candidate, all_goals[:, :3])
            )
            all_goals = all_goals.at[agent_id, :3].set(goal_candidate)
            agent_id += 1

            # if no solution is found, start over
            agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
            all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
            all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals

            return agent_id, this_key, all_states, all_goals

        def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
            # agent_id, key, states, goals
            agent_id, this_key, all_states, all_goals = reset_input
            return agent_id < self.num_agents

        _, _, states, goals = while_loop(
            cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))

        env_states = self.EnvState(states, goals, obstacles)

        # # randomly generate positions of agents and goals
        # agent_key, goal_key, key = jr.split(key, 3)
        # agent_candidate = jr.uniform(agent_key, (3,), minval=0, maxval=side_length)
        # goal_candidate = jr.uniform(goal_key, (3,), minval=0, maxval=side_length)
        #
        # def get_node(reset_input: Tuple[Array, Array, Array]):  # key, node, all nodes
        #     this_key, _, all_nodes = reset_input
        #     use_key, this_key = jr.split(this_key, 2)
        #     return this_key, jr.uniform(use_key, (3,), minval=0, maxval=side_length), all_nodes
        #
        # def non_valid_node(reset_input: Tuple[Array, Array, Array]):  # key, node, all nodes
        #     _, node, all_nodes = reset_input
        #     dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
        #     collide = dist_min <= self._params['drone_radius'] * 4
        #     inside = inside_obstacles(node, obstacles, r=self._params['drone_radius'] * 4)
        #     return collide | inside
        #
        # for i in range(self.num_agents):
        #     agent_key, goal_key, key = jr.split(key, 3)
        #
        #     _, agent_candidate, _ = while_loop(
        #         cond_fun=non_valid_node, body_fun=get_node, init_val=(agent_key, agent_candidate, states[:, :3]))
        #     states = states.at[i, :3].set(agent_candidate)
        #
        #     _, goal_candidate, _ = while_loop(
        #         cond_fun=non_valid_node, body_fun=get_node, init_val=(goal_key, goal_candidate, goals[:, :3]))
        #     goals = goals.at[i, :3].set(goal_candidate)
        #
        # # randomly generate initial velocities
        # env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    def agent_step_euler(self, agent_state: Array, action: Array) -> Array:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_state.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_state, action)
        n_state_agent_new = agent_state + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        return jnp.matmul(agent_states, self._A.T) + jnp.matmul(action, self._B.T)

    def step(
        self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        next_agent_states = self.agent_step_euler(agent_states, action)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        # dist2goal = jnp.linalg.norm(goal_states[:, :3] - next_agent_states[:, :3], axis=-1)
        # reward -= (dist2goal.mean()) ** 2
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
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

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :3]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        cost = nn.relu(self._params["drone_radius"] * 2 - dist).mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["drone_radius"])
        cost += collision.mean()

        return cost

    def render(self, graph: GraphsTuple) -> Array:
        pass

    def setup_ax(self, ax: Axes3D) -> None:
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_zlim(0, self.area_size)
        ax.set(aspect="equal")
        # plt.axis("off")

    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
        ax: Axes3D
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.add_subplot(projection='3d')
        self.setup_ax(ax)

        if viz_opts is None:
            viz_opts = {}

        T_graph = rollout.Tp1_graph

        graph0 = tree_index(T_graph, 0)
        n_nodes = len(graph0.states[:-1])

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        edge_goal_color = goal_color

        # plot obstacles
        obs = graph0.env_states.obstacle
        n_obs = len(obs.center)
        obs_col = get_sphere_collection(obs, alpha=1.0, facecolor="#8a0000")
        # obs_col = get_cuboid_collection(obs, alpha=0.8, linewidth=1.0, edgecolor='k', facecolor="#710000")
        ax.add_collection(obs_col)

        # plot agents
        n_radius = np.array([self._params["drone_radius"]] * self.num_agents * 2)
        n_color = [agent_color] * self.num_agents + [goal_color] * self.num_agents
        n_pos = graph0.states[:self.num_agents * 2, :3]
        n_pos = np.flip(n_pos, axis=0)
        plot_r = ax.transData.transform([self._params["drone_radius"], 0])[0] - ax.transData.transform([0, 0])[0]
        # agent_col = ax.plot(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2], markersize=20, mfc=agent_color, linestyle="", marker='o')[0]  # todo: the size of the agent
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent

        # # plot edges
        # n_hits = self.num_agents * self.n_rays
        # all_pos = graph0.states[:self.num_agents * 2 + n_hits, :3]
        # edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        # is_pad = np.any(edge_index == self.num_agents * 2 + n_hits, axis=0)
        # e_edge_index = edge_index[:, ~is_pad]
        # e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        # e_colors = ["0.2" for ii in range(len(e_start))]
        # e_lines = np.stack([e_start, e_end], axis=1)
        # edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=3, alpha=0.5, zorder=3)
        # ax.add_collection(edge_col)

        # edge_col = []
        cnt_col = []

        # # text for cost and reward
        # cost_text = ax.text(0.02, 0.02, 1.04, "Cost: 1.0, Reward: 1.0", transform=ax.transAxes, va="bottom")
        # # Text for safety.
        # safe_text = []
        # # if "inside_obstacles" in rollout.T_info:
        # if Ta_is_unsafe is not None:
        #     safe_text = [ax.text(0.02, 0.02, 1.00, "Unsafe: {}", transform=ax.transAxes, va="bottom")]

        # font_opts = dict(
        #     size=12,
        #     color="k",
        #     family="sans-serif",
        #     weight="normal",
        #     ha="center",
        #     va="center",
        #     transform=ax.transData,
        #     clip_on=True,
        #     zorder=7,
        # )
        # agent_labels = []
        # for ii in range(self.num_agents):
        #     pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], ax.get_proj())[:2]
        #     agent_labels.append(ax.text2D(pos2d[0], pos2d[1], f"{ii}", **font_opts))
        # agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], f"{ii}", **font_opts)
        #                 for ii in range(self.num_agents)]

        # # Text for timestep.
        # kk_text = ax.text2D(0.99, 0.99, "kk=0", transform=ax.transAxes, va="top", ha="right")

        def init_fn() -> list[plt.Artist]:
            # return [agent_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]
            return [agent_col, *cnt_col]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos = graph.states[:self.num_agents * 2, :3]

            # update agent positions
            # agent_col.set_data(n_pos[:, 0], n_pos[:, 1])
            # agent_col.set_3d_properties(n_pos[:, 2])

            agent_col.set_offsets(n_pos[:, :2])
            agent_col.set_3d_properties(n_pos[:, 2], zdir='z')
            # for ii in range(self.num_agents):
            #     agent_col[ii].set_data(n_pos[ii, 0], n_pos[ii, 1])
            #     agent_col[ii].set_3d_properties(n_pos[ii, 2])

            # # Update edges.
            # all_pos = graph.states[:self.num_agents * 2 + n_hits, :3]
            # e_edge_index = np.stack([graph.senders, graph.receivers], axis=0)
            # is_pad = np.any(e_edge_index == self.num_agents * 2 + n_hits, axis=0)
            # e_edge_index = e_edge_index[:, ~is_pad]
            # e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
            #
            # # goal => num_agents <= idx < num_agents * 2
            # e_is_goal = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents * 2)
            # e_is_goal = e_is_goal[~is_pad]
            # e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
            #
            # # (e, n_pts, 2)
            # e_lines = np.stack([e_start, e_end], axis=1)
            # edge_col.set_segments(e_lines)
            # edge_col.set_colors(e_colors)

            # # update agent labels
            # for ii in range(self.num_agents):
            #     text_pos = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], ax.get_proj())[:2]
            #     agent_labels[ii].set_position(text_pos)
            #     # agent_labels[ii].set_position(n_pos[ii])
            #
            # # update cost and safe labels
            # if kk < len(rollout.T_cost):
            #     cost_text.set_text(
            #         "Cost: {:5.2f}, Reward: {:5.2f}".format(rollout.T_cost[kk], rollout.T_reward[kk]))
            # else:
            #     cost_text.set_text("")
            #
            # if len(safe_text) > 0:
            #     if kk < len(Ta_is_unsafe):
            #         a_is_safe = Ta_is_unsafe[kk]
            #         safe_text[0].set_text("Unsafe: {}".format(1 * a_is_safe))
            #     else:
            #         cost_text.set_text("")
            #
            # kk_text.set_text("kk={:04}".format(kk))
            # return [agent_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]
            return [agent_col, *cnt_col]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def edge_blocks(self, state: EnvState, lidar_data: Pos3d) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays

        # agent - agent connection
        agent_pos = state.agent[:, :3]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :3].set(agent_goal_feats[:, :3] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :3]
            lidar_feats = state.agent[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self.n_rays))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.matmul(state, self._A.T)
        g = self._B
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(LinearDrone.GOAL)
        node_type = node_type.at[-n_hits:].set(LinearDrone.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self.params['n_rays'],
                sense_range=self._params["comm_radius"],
                max_returns=self.n_rays,
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :3]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros_like(lidar_data)], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        low_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -0.5, -0.5, -0.5])
        up_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.5, 0.5, 0.5])
        return low_lim, up_lim

    def action_lim(self) -> Tuple[Action, Action]:
        low_lim = jnp.array([-1., -1., -1.])
        up_lim = jnp.array([1., 1., 1.])
        return low_lim, up_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]
        agent_vel = graph.type_states(type_idx=0, n_type=self.num_agents)[:, 3:]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        # vel_diff = agent_vel[:, None, :] - agent_vel[None, :, :]
        # vel_diff_proj = jnp.sum(vel_diff * pos_diff, axis=-1) / (dist + 1e-6)
        safe_agent = jnp.greater(dist, self._params["drone_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2.5)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]
        goal_pos = graph.env_states.goal[:, :3]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["drone_radius"] * 2
        return reach
