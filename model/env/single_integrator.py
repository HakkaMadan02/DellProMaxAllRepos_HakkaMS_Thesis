import functools as ft
import pathlib
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from jax.lax import while_loop
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from typing import List, NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Bool, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, tree_index, MutablePatchCollection, save_anim, jax_vmap
from ..trainer.utils import np2jax, centered_norm
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import plot_graph, get_BuRd
from .utils import get_lidar, inside_obstacles, lqr


class SingleIntegrator(MultiAgentEnv):
    """"""

    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
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
        super(SingleIntegrator, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self._A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32) * self._dt + np.eye(self.state_dim)
        self._B = np.array([[1.0, 0.0], [0.0, 1.0]]) * self._dt
        self._Q = np.eye(self.state_dim) * 2
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        self.create_obstacles = jax.vmap(Rectangle.create)

    @property
    def state_dim(self) -> int:
        return 2  # x, y

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 2  # x_rel, y_rel

    @property
    def action_dim(self) -> int:
        return 2  # vx, vy

    def reset_1(self, key: Array) -> GraphsTuple:
        assert self.num_agents == 1
        side_length = self.area_size
        state = jnp.zeros((self.num_agents, 2))
        goal = jnp.zeros((self.num_agents, 2))
        state = state.at[0, :2].set(jnp.array([1.0, 2]))
        goal = goal.at[0, :2].set(jnp.array([3.0, 2]))
        obs_pos = jnp.array([[2, 2]])
        width = jnp.array([0.2])
        height = jnp.array([1.0])
        theta = jnp.array([jnp.pi / 4])
        obstacles = self.create_obstacles(obs_pos, width, height, theta)
        env_states = self.EnvState(state, goal, obstacles)

        return self.get_graph(env_states)

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0
        # return self.reset_1(key)
        side_length = self.area_size
        states = jnp.zeros((self.num_agents, 2))
        goals = jnp.zeros((self.num_agents, 2))

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0

        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=side_length)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)

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
            collide = dist_min <= self._params["car_radius"] * 4
            inside = inside_obstacles(node, obstacles, r=self._params["car_radius"] * 4)
            return collide | inside

        for i in range(self.num_agents):
            agent_key, goal_key, key = jr.split(key, 3)

            _, agent_candidate, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node, init_val=(agent_key, agent_candidate, states)
            )
            states = states.at[i].set(agent_candidate)

            _, goal_candidate, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node, init_val=(goal_key, goal_candidate, goals)
            )
            goals = goals.at[i].set(goal_candidate)

        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)

    def reset_np(self, key: Array) -> GraphsTuple:
        self._t = 0
        side_length = self.area_size
        states = np.zeros((self.num_agents, 2))
        goals = np.zeros((self.num_agents, 2))

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0

        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=side_length)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)

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

        def get_node(this_key: Array):
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,), minval=0, maxval=side_length)

        def non_valid_node(node, all_nodes):  # key, node, all nodes
            dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
            collide = dist_min <= self._params["car_radius"] * 4
            inside = inside_obstacles(node, obstacles, r=self._params["car_radius"] * 4)
            return collide | inside

        for ii in range(self.num_agents):
            agent_key, goal_key, key = jr.split(key, 3)

            while non_valid_node(agent_candidate, states):
                agent_key, agent_candidate = get_node(agent_key)
            states[ii] = np.array(agent_candidate)

            while non_valid_node(goal_candidate, goals):
                goal_key, goal_candidate = get_node(goal_key)
            goals[ii] = np.array(goal_candidate)

        env_states = self.EnvState(np2jax(states), np2jax(goals), obstacles)
        return self.get_graph(env_states)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = action
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        # dist2goal = jnp.linalg.norm(goals - next_agent_states, axis=-1)
        # reward -= (dist2goal.mean()) ** 2
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()  # * 0.01
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goals, obstacles)

        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            agent_pos = agent_states
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        cost = nn.relu(self._params["car_radius"] * 2 - dist).mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render(self, graph: GraphsTuple) -> Array:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=80)

        # plot the cars, lidar hitting points and the communication network
        pos = graph.states[:-1, :2]
        cost = self.get_cost(graph)
        edge_index = np.array(jnp.stack([graph.senders, graph.receivers], axis=0))
        n_hits = self.num_agents * self._params["n_rays"]
        is_pad = np.any(edge_index == self.num_agents * 2 + n_hits, axis=0)
        edge_index_nopad = edge_index[:, ~is_pad]
        color = ["#FF8C00"] * self.num_agents + ["#3CB371"] * self.num_agents + ["#710000"] * n_hits
        radius = [self._params["car_radius"]] * self.num_agents * 2 + [0] * n_hits
        plot_graph(
            ax,
            pos,
            radius=radius,
            color=color,
            with_label=False,
            plot_edge=True,
            edge_index=edge_index_nopad,
            alpha=0.8,
        )

        # plot the obstacles
        obs = graph.env_states.obstacle
        obs_plots = []
        for i in range(obs.center.shape[0]):
            obs_plots.append(Polygon(obs.points[i], color="#710000", alpha=0.8, zorder=99))
        obs_plots = PatchCollection(obs_plots, match_original=True)
        ax.add_collection(obs_plots)

        # show cost
        ax.set_title(f"cost: {cost:.4f}")

        # set axis limits
        low_lim, up_lim = self.state_lim(graph.states)
        ax.set_xlim(low_lim[0], up_lim[0])
        ax.set_ylim(low_lim[1], up_lim[1])
        plt.axis("off")

        # convert to numpy array
        fig.canvas.draw()
        fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        plt.close()
        return fig_np

    def setup_ax(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set(aspect="equal")
        plt.axis("off")

    def render(self, rollout: RolloutResult, u_qp):
        fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=300)
        ax.set_xlim(0.5, 3.05)
        ax.set_ylim(1.4, 2.55)
        ax.set(aspect="equal")
        plt.axis("off")

        T_graph = rollout.Tp1_graph
        T_action = rollout.T_action
        graph0 = tree_index(T_graph, 0)
        total_step = T_action.shape[0]
        n_nodes = len(graph0.states[:-1])

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obs_color = "#8a0000"
        u_ref_color = "k"
        u_color = "#ff8c00"

        # plot obstacles
        obs = graph0.env_states.obstacle
        n_obs = len(obs.center)
        obs_polys = [Polygon(obs.points[i], color=obs_color, alpha=1.0, zorder=99) for i in range(n_obs)]
        ax.add_collection(PatchCollection(obs_polys, match_original=True))

        # calculate u_ref
        scale = 0.1
        u_ref = jax_vmap(self.u_ref)(T_graph)[:-1]
        if u_qp is not None:
            u_qp *= scale

        # u_nn = T_action - u_ref

        u_ref *= scale
        # u_nn *= scale
        T_action *= scale

        for t in range(0, total_step // 5, 10):
            graph = tree_index(T_graph, t)

            # plot agents and goals
            n_hits = self.num_agents * self._params["n_rays"]
            n_radius = np.array([self._params["car_radius"]] * self.num_agents * 2 + [0] * n_hits)
            n_color = [agent_color] * self.num_agents + [goal_color] * self.num_agents + [obs_color] * n_hits
            n_pos = graph.states[:-1, :2]

            # stop when reach the goal
            if np.linalg.norm(n_pos[:self.num_agents] - n_pos[self.num_agents: self.num_agents * 2]) < 2 * self._params[
                'car_radius']:
                break
            agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0, alpha=np.clip(t / 20, 0, 1)) for ii in
                           range(n_nodes)]

            agent_col = MutablePatchCollection(agent_circs, match_original=True, zorder=6)
            ax.add_collection(agent_col)

            # plot action
            for ii in range(self.num_agents):
                width = 0.015
                alpha = 0.8
                lw = 0.1

                if u_qp is not None:
                    print('plot u_qp')
                    plt.arrow(n_pos[ii, 0], n_pos[ii, 1], u_qp[t, ii, 0], u_qp[t, ii, 1], color=u_ref_color, lw=lw,
                            width=width, zorder=100, alpha=alpha, edgecolor=None)
                else:
                    plt.arrow(n_pos[ii, 0], n_pos[ii, 1], u_ref[t, ii, 0], u_ref[t, ii, 1], color=u_ref_color, lw=lw,
                          width=width, zorder=100, alpha=alpha, edgecolor=None)

                plt.arrow(n_pos[ii, 0], n_pos[ii, 1], T_action[t, ii, 0], T_action[t, ii, 1], color=u_color, lw=lw,
                          width=width, zorder=100, alpha=alpha, edgecolor=None)

        # ax.text(2.2, 2.2, r"$\eta_\mathrm{ctrl}=10^{-2}$", fontsize=24, math_fontfamily='stix')


                # plt.arrow(n_pos[ii, 0], n_pos[ii, 1], u_nn[t, ii, 0], u_nn[t, ii, 1], color='g', width=width, zorder=7, alpha=alpha)
        plt.savefig(f'./figs/gcbf_right.pdf', dpi=300, bbox_inches='tight')



    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        self.setup_ax(ax)

        if viz_opts is None:
            viz_opts = {}

        T_graph = rollout.Tp1_graph

        graph0 = tree_index(T_graph, 0)
        n_nodes = len(graph0.states[:-1])

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        edge_goal_color = goal_color

        # Plot obstacles.
        obs = graph0.env_states.obstacle
        n_obs = len(obs.center)
        obs_polys = [Polygon(obs.points[ii]) for ii in range(n_obs)]
        obs_col = PatchCollection(obs_polys, color="#8a0000", alpha=0.8, zorder=99)
        ax.add_collection(obs_col)

        # Plot agents.
        n_hits = self.num_agents * self._params["n_rays"]
        n_radius = np.array([self._params["car_radius"]] * self.num_agents * 2 + [0] * n_hits)
        n_color = [agent_color] * self.num_agents + [goal_color] * self.num_agents + ["#8a0000"] * n_hits

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
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def edge_blocks(self, state: EnvState, lidar_data: Pos2d) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.zeros_like(state)
        g = jnp.eye(state.shape[1])
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        # node type
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(SingleIntegrator.GOAL)
        node_type = node_type.at[-n_hits:].set(SingleIntegrator.OBS)

        # edge blocks
        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent))
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
        lower_lim = jnp.ones(2) * -jnp.inf
        upper_lim = jnp.ones(2) * jnp.inf
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)
        #
        # agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)
        # goal_pos = graph.type_states(type_idx=1, n_type=self.num_agents)
        # error = goal_pos - agent_pos
        # return self.clip_action(error @ self._K.T)

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

        # next_graph = graph
        # next_graph = next_graph._replace(edges=next_states[graph.receivers] - next_states[graph.senders])
        return next_graph

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 2.5)
        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        unsafe_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return unsafe_mask

    def collision_mask(self, graph: GraphsTuple) -> Array:
        return self.unsafe_mask(graph)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach
