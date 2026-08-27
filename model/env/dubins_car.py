import functools as ft
import pathlib
import flax.linen as nn
import ipdb
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

from ..trainer.utils import centered_norm
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import ABool, Action, AgentState, Array, Cost, Done, Info, Pos2d, Reward, State
from ..utils.utils import MutablePatchCollection, merge01, save_anim, tree_index
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import plot_graph, get_BuRd
from .utils import get_lidar, inside_obstacles, lqr


class DubinsCar(MultiAgentEnv):
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
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 8,
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
        super(DubinsCar, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Rectangle.create)
        self.enable_stop = True

    @property
    def state_dim(self) -> int:
        return 4  # x, y, theta, v

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0
        side_length = self.area_size
        states = jnp.zeros((self.num_agents, 4))
        goals = jnp.zeros((self.num_agents, 4))

        # randomly generate obstacles
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (self._params["n_obs"], 2), minval=0, maxval=side_length)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (self._params["n_obs"],), minval=0, maxval=2 * np.pi)

        # add 4 fixed obstacles surrounding the area
        # depth = 0.02
        # fixed_pos = np.array(
        #     [
        #         [side_length / 2, 0.0],  # bottom
        #         [0.0, side_length / 2],  # left
        #         [side_length / 2, side_length],  # top
        #         [side_length, side_length / 2],  # right
        #     ]
        # )
        # fixed_len = np.array([[side_length + depth, depth], [depth, side_length + depth]] * 2)
        # fixed_theta = np.zeros(4)
        # obs_pos = jnp.concatenate([obs_pos, fixed_pos], axis=0)
        # obs_len = jnp.concatenate([obs_len, fixed_len], axis=0)
        # obs_theta = jnp.concatenate([obs_theta, fixed_theta], axis=0)

        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate positions of agents and goals
        agent_key, goal_key, key = jr.split(key, 3)
        agent_candidate = jr.uniform(agent_key, (2,), minval=0, maxval=side_length)
        goal_candidate = jr.uniform(goal_key, (2,), minval=0, maxval=side_length)

        def get_node(reset_input: Tuple[Array, Array, Array]):  # key, node, all nodes
            this_key, _, all_nodes = reset_input
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,), minval=0, maxval=side_length), all_nodes

        def non_valid_node(reset_input: Tuple[Array, Array, Array]):  # key, node, all nodes
            _, node, all_nodes = reset_input
            dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
            collide = dist_min <= self._params["car_radius"] * 6
            inside = inside_obstacles(node, obstacles, r=self._params["car_radius"] * 4)
            return collide | inside

        for i in range(self.num_agents):
            agent_key, goal_key, key = jr.split(key, 3)

            _, agent_candidate, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node, init_val=(agent_key, agent_candidate, states[:, :2]))
            states = states.at[i, :2].set(agent_candidate)

            _, goal_candidate, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node, init_val=(goal_key, goal_candidate, goals[:, :2]))

            goals = goals.at[i, :2].set(goal_candidate)

        # add random heading
        theta_key, key = jr.split(key, 2)
        states = states.at[:, 2].set(jr.uniform(theta_key, (self.num_agents,), minval=-np.pi, maxval=np.pi))
        goals = goals.at[:, 2].set(jnp.arctan2(goals[:, 1] - states[:, 1], goals[:, 0] - states[:, 0]))

        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)

    def agent_step_euler(self, agent_states: AgentState, action: Action, stop_mask: Array) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action) * (1 - stop_mask)[:, None]
        n_state_agent_new = agent_states + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([
            (jnp.cos(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (jnp.sin(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (action[:, 0] * 20.)[:, None],
            (action[:, 1])[:, None]
        ], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

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

        stop_mask = self.stop_mask(graph)
        if not self.enable_stop:
            # If stopping is not enabled, then set stop_mask to always be 0.
            stop_mask = 0 * stop_mask
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        # next_states = jnp.concatenate([next_agent_states, goal_states], axis=0)
        # next_states = self.clip_state(next_states)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        # dist2goal = jnp.linalg.norm(goal_states[:, :2] - next_agent_states[:, :2], axis=-1)
        # reward -= (dist2goal.mean()) ** 2
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()  # * 0.01
        cost = self.get_cost(graph)

        # compute cost: collision
        # agent_pos = agent_states[:, :2]
        # dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        # dist += jnp.eye(self.num_agents) * 1e6
        # cost = nn.relu(self._params['car_radius'] * 2 - dist).mean()

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goal_states, obstacles)

        return self.get_graph(next_state), reward, cost, done, {}

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        cost = nn.relu(self._params["car_radius"] * 2 - dist).mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render(self, graph: GraphsTuple) -> plt.Figure:
        raise NotImplementedError

    def setup_ax(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set(aspect="equal")
        plt.axis("off")

    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        self.setup_ax(ax)

        if viz_opts is None:
            viz_opts = {}

        T_graph = rollout.Tp1_graph
        anim_T = len(T_graph.n_node)
        # print("anim_T: {}".format(anim_T))
        # ipdb.set_trace()

        graph0 = tree_index(T_graph, 0)
        n_nodes = len(graph0.states[:-1])

        agent_color = "#FF8C00"
        goal_color = "#3CB371"
        edge_goal_color = goal_color

        # Plot obstacles.
        obs = graph0.env_states.obstacle
        n_obs = len(obs.center)
        obs_polys = [Polygon(obs.points[ii]) for ii in range(n_obs)]
        obs_col = PatchCollection(obs_polys, color="#710000", alpha=0.8, zorder=99)
        ax.add_collection(obs_col)

        # Plot agents.
        n_hits = self.num_agents * self._params["n_rays"]
        n_radius = np.array([self._params["car_radius"]] * self.num_agents * 2 + [0] * n_hits)
        n_color = [agent_color] * self.num_agents + [goal_color] * self.num_agents + ["#710000"] * n_hits

        n_pos = graph0.states[:-1, :2]
        agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0) for ii in range(n_nodes)]
        agent_col = MutablePatchCollection(agent_circs, match_original=True, zorder=6)
        ax.add_collection(agent_col)

        # Plot edges.
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents * 2 + n_hits, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = n_pos[e_edge_index[0, :]], n_pos[e_edge_index[1, :]]
        # (e, n_pts, 2)
        e_lines = np.stack([e_start, e_end], axis=1)
        e_colors = ["0.2" for ii in range(len(e_start))]
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)

        # Text for cost and reward.
        cost_text = ax.text(0.02, 1.04, "Cost: 1.0, Reward: 1.0", transform=ax.transAxes, va="bottom")
        # Text for safety.
        safe_text = []
        # if "inside_obstacles" in rollout.T_info:
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.02, 1.00, "Unsafe: {}", transform=ax.transAxes, va="bottom")]

        # Text for timestep.
        kk_text = ax.text(0.99, 0.99, "kk=0", transform=ax.transAxes, va="top", ha="right")

        # Maybe plot cbf.
        cnt_col = []
        if "cbf" in viz_opts:
            Tb_xs, Tb_ys, Tbb_h, cbf_num = viz_opts["cbf"]
            bb_Xs, bb_Ys = np.meshgrid(Tb_xs[0], Tb_ys[0])
            # cmap = sns.color_palette("rocket", as_cmap=True)
            norm = centered_norm(Tbb_h.min(), Tbb_h.max())
            levels = np.linspace(norm.vmin, norm.vmax, 15)

            cmap = get_BuRd().reversed()
            contour_opts = dict(cmap=cmap, norm=norm, levels=levels, alpha=0.9)
            cnt = ax.contourf(bb_Xs, bb_Ys, Tbb_h[0], **contour_opts)

            contour_line_opts = dict(levels=[0.0], colors=["black"])
            cnt_line = ax.contour(bb_Xs, bb_Ys, Tbb_h[0], **contour_line_opts)

            cbar = fig.colorbar(cnt, ax=ax)
            cbar.add_lines(cnt_line)

            cnt_col = [*cnt.collections, *cnt_line.collections]

            ax.text(0.5, 1.0, "CBF for {}".format(cbf_num), transform=ax.transAxes, va="bottom")

        font_opts = dict(
            size=12,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=ax.transData,
            clip_on=True,
            zorder=7,
        )
        agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **font_opts) for ii in range(self.num_agents)]

        def init_fn() -> list[plt.Artist]:
            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos = graph.states[:-1, :2]

            # Update agent positions.
            for ii in range(n_nodes):
                agent_circs[ii].set_center(tuple(n_pos[ii]))

            # Update edges.
            e_edge_index = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad = np.any(e_edge_index == self.num_agents * 2 + n_hits, axis=0)
            e_edge_index = e_edge_index[:, ~is_pad]
            e_start, e_end = n_pos[e_edge_index[0, :]], n_pos[e_edge_index[1, :]]

            # goal => num_agents <= idx < num_agents * 2
            e_is_goal = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents * 2)
            e_is_goal = e_is_goal[~is_pad]
            e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]

            # (e, n_pts, 2)
            e_lines = np.stack([e_start, e_end], axis=1)
            edge_col.set_segments(e_lines)
            edge_col.set_colors(e_colors)

            # Update agent labels.
            for ii in range(self.num_agents):
                agent_labels[ii].set_position(n_pos[ii])

            # Update cost and safe labels.
            if kk < len(rollout.T_cost):
                cost_text.set_text("Cost: {:5.2f}, Reward: {:5.2f}".format(rollout.T_cost[kk], rollout.T_reward[kk]))
            else:
                cost_text.set_text("")

            if len(safe_text) > 0:
                if kk < len(Ta_is_unsafe):
                    # a_inside_obstacles = rollout.T_info["inside_obstacles"][kk]
                    # a_is_safe = self.safe_mask_np(graph, a_inside_obstacles)
                    a_is_safe = Ta_is_unsafe[kk]
                    safe_text[0].set_text("Unsafe: {}".format(1 * a_is_safe))
                else:
                    cost_text.set_text("")

            # Update the contourf.
            nonlocal cnt, cnt_line
            if "cbf" in viz_opts:
                for c in cnt.collections:
                    c.remove()
                for c in cnt_line.collections:
                    c.remove()

                bb_Xs, bb_Ys = np.meshgrid(Tb_xs[kk], Tb_ys[kk])
                cnt = ax.contourf(bb_Xs, bb_Ys, Tbb_h[kk], **contour_opts)
                cnt_line = ax.contour(bb_Xs, bb_Ys, Tbb_h[kk], **contour_line_opts)

                cnt_col = [*cnt.collections, *cnt_line.collections]
            else:
                cnt_col = []

            kk_text.set_text("kk={:04}".format(kk))
            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        pos_theta_diff = state.agent[:, None, :2] - state.agent[None, :, :2]
        agent_v = jnp.concatenate([(state.agent[:, 3] * jnp.cos(state.agent[:, 2]))[:, None],
                                   (state.agent[:, 3] * jnp.sin(state.agent[:, 2]))[:, None]], axis=-1)
        v_diff = agent_v[:, None, :] - agent_v[None, :, :]
        state_diff = jnp.concatenate([pos_theta_diff, v_diff], axis=-1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        agent_goal_pos_diff = state.agent[:, :2] - state.goal[:, :2]
        agent_goal_v_diff = agent_v
        agent_goal_edge_feats = jnp.concatenate([agent_goal_pos_diff, agent_goal_v_diff], axis=-1)
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_edge_feats = agent_goal_edge_feats.at[:, :2].set(agent_goal_edge_feats[:, :2] * coef)
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        for i in range(self.num_agents):
            agent_goal_edges.append(
                EdgeBlock(agent_goal_edge_feats[i][None, None, :], jnp.ones((1, 1)), id_agent[i][None], id_goal[i][None]))


        # id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        # agent_goal_mask = jnp.eye(self.num_agents)
        # agent_goal_pos_diff = state.agent[:, None, :3] - state.goal[None, :, :3]
        # goal_v = jnp.zeros((self.num_agents, 2))
        # agent_goal_v_diff = agent_v[:, None, :] - goal_v[None, :, :]
        # agent_goal_edge_feats = jnp.concatenate([agent_goal_pos_diff, agent_goal_v_diff], axis=-1)
        # feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        # comm_radius = self._params["comm_radius"]
        # safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        # coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        # agent_goal_edge_feats = agent_goal_edge_feats.at[:, :2].set(agent_goal_edge_feats[:, :2] * coef)
        #
        # agent_goal_edges = EdgeBlock(
        #     agent_goal_edge_feats, agent_goal_mask, id_agent, id_goal
        # )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = jnp.concatenate([state.agent[i, :2], agent_v[i]]) - lidar_data[id_hits, :]
            # lidar_feats = jnp.concatenate([lidar_pos, state.agent[i, 2][None, :], agent_v[i, :][None, :]], axis=-1)
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        # return [agent_agent_edges, agent_goal_edges] + agent_obs_edges
        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([
            (jnp.cos(state[:, 2]) * state[:, 3])[:, None],
            (jnp.sin(state[:, 2]) * state[:, 3])[:, None],
            jnp.zeros((state.shape[0], 2))
        ], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.array([[10., 0.], [0., 1.]])], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], 4, 2)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        v = jnp.concatenate([(state[:, 3] * jnp.cos(state[:, 2]))[:, None],
                             (state[:, 3] * jnp.sin(state[:, 2]))[:, None]], axis=-1)
        edge_state = jnp.concatenate([state[:, :2], v], axis=-1)
        assert edge_state.shape[1] == self.edge_dim
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)
        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DubinsCar.GOAL)
        node_type = node_type.at[-n_hits:].set(DubinsCar.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 2))], axis=-1)
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
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        side = self._params['area_size']
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -0.8])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.8])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        lower_lim = jnp.ones(2) * -3.0
        upper_lim = jnp.ones(2) * 3.0
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:

        # def get_agent_action(x_goal):
        #     x = x_goal[:4]
        #     goal = x_goal[4:]
        #     # desired_theta = jnp.arctan2(goal[1] - x[1], goal[0] - x[0])
        #     x = x.at[2].set((x[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        #     desired_theta = x[2]
        #     theta_id = jnp.argmin(jnp.abs(desired_theta - self._thetas))
        #     v_id = jnp.argmin(jnp.abs(x[3] - self._vs))
        #     K = self._Ks[theta_id, v_id]
        #     error = goal - x
        #     error_max = jnp.abs(error[:2] / jnp.linalg.norm(error[:2], axis=-1, keepdims=True) * self._params["comm_radius"])
        #     error = error.at[:2].set(jnp.clip(error[:2], -error_max, error_max))
        #     # error = error.at[2].set((error[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        #     return (error[None, :] @ K.T).squeeze()
        #
        # agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        # goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        # action = jax.vmap(get_agent_action)(jnp.concatenate([agent_states, goal_states], axis=-1))
        # return self.clip_action(action)

        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        pos_diff = agent_states[:, :2] - goal_states[:, :2]

        # PID parameters
        k_omega = 1.0  # 0.5
        k_v = 2.3
        k_a = 2.5

        dist = jnp.linalg.norm(pos_diff, axis=-1)
        theta_t = jnp.arctan2(-pos_diff[:, 1], -pos_diff[:, 0]) % (2 * jnp.pi)
        theta = agent_states[:, 2] % (2 * jnp.pi)
        theta_diff = theta_t - theta
        omega = jnp.zeros(agent_states.shape[0])
        agent_dir = jnp.concatenate([jnp.cos(theta)[:, None], jnp.sin(theta)[:, None]], axis=-1)
        assert agent_dir.shape == (agent_states.shape[0], 2)
        theta_between = jnp.arccos(
            jnp.clip(jnp.matmul(-pos_diff[:, None, :], agent_dir[:, :, None]).squeeze() / (dist + 0.0001),
                     a_min=-1, a_max=1))

        # when theta <= pi
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0), theta <= jnp.pi),
                          k_omega * theta_between, omega)
        # clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0)), theta <= jnp.pi),
            -k_omega * theta_between, omega
        )

        # when theta > pi
        # clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0), theta > jnp.pi),
                          -k_omega * theta_between, omega)
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0)), theta > jnp.pi),
            k_omega * theta_between, omega
        )

        omega = jnp.clip(omega, a_min=-5., a_max=5.)

        pos_diff_norm = jnp.sqrt(1e-6 + jnp.sum(pos_diff ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(pos_diff_norm, comm_radius)
        coef = jnp.where(pos_diff_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        pos_diff = coef * pos_diff
        a = -k_a * agent_states[:, 3] + k_v * jnp.linalg.norm(pos_diff, axis=-1)

        action = jnp.concatenate([omega[:, None], a[:, None]], axis=-1)
        # jax.debug.print('uref: {}', action.max(axis=0))
        return action

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        # next_graph = next_graph._replace(edges=next_states[graph.receivers] - next_states[graph.senders],
        #                                  states=next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        heading_vec = jnp.concatenate([jnp.cos(agent_state[:, 2])[:, None],
                                       jnp.sin(agent_state[:, 2])[:, None]], axis=1)[:, None, :]
        heading_vec = heading_vec.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist ** 2 - 4 * self._params['car_radius'] ** 2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist ** 2 - self._params['car_radius'] ** 2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        return jnp.logical_or(collision_mask, unsafe_dir)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

    def stop_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        stop = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 0.5
        return stop
