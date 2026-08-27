import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import flax.linen as nn
import jax.debug as jd

from typing import Tuple
from jax.lax import while_loop

from .base import MultiAgentEnv
from .single_integrator import SingleIntegrator
from .plot import plot_graph
from ..utils.graph import GraphsTuple, GetGraph
from ..utils.typing import Array, Action, State, Reward, Done, Info, Cost


class SimpleCarConnect(SingleIntegrator):
    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        'area_size': 4.0,
        'car_radius': 0.05,
        'comm_radius': 1.0,
        'obs_radius': 0.2,
        'n_obs': 4,
    }

    def __init__(
            self,
            num_agents: int,
            dt: float = 0.03,
            params: dict = None
    ):
        super(SimpleCarConnect, self).__init__(num_agents, dt, params)
        self.xy_min = 0
        self.xy_max = self._params['area_size']

    @property
    def edge_dim(self) -> int:
        return 3  # x_rel, y_rel, connect

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0
        side_length = self._params['area_size']
        comm_r = self._params['comm_radius']
        init_side = comm_r / np.sqrt(2) * 0.8
        states = jnp.zeros((self.num_agents, 2))
        goals = jnp.zeros((self.num_agents, 2))

        # randomly generate a circle obstacle
        obstacle_key, key = jr.split(key, 2)
        obs_r = self._params['obs_radius']
        obs_pos = jr.uniform(obstacle_key, (self._params['n_obs'], 2), minval=obs_r, maxval=side_length - obs_r)

        # randomly generate positions of agents and goals
        agent_key, goal_key, key = jr.split(key, 3)
        agent_candidate = jr.uniform(agent_key, (2,), minval=side_length / 4, maxval=side_length / 4 * 3)
        goal_candidate = jr.uniform(goal_key, (2,), minval=side_length / 4, maxval=side_length / 4 * 3)

        def get_node(reset_input: Tuple[Array, Array, Array, int]):  # key, previous_node, all nodes
            this_key, previous_agent, all_nodes, i_node = reset_input
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,), minval=-init_side, maxval=init_side) + all_nodes[i_node - 1], all_nodes, i_node

        def non_valid_node(reset_input: Tuple[Array, Array, Array, int]):  # key, node, all nodes
            _, node, all_nodes, i_node = reset_input
            dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
            collision = dist_min <= self._params['car_radius'] * 4
            dist_obs_min = jnp.linalg.norm(obs_pos - node, axis=1).min()
            not_valid = jnp.logical_or(collision, dist_obs_min <= self._params['obs_radius'] * 2)
            # outside = jnp.logical_or(node < 0, node > side_length).any()
            return not_valid

        # def get_goal(reset_input: Tuple[Array, Array, Array]):  # key, previous_node, all nodes
        #     this_key, previous_agent, all_nodes = reset_input
        #     use_key, this_key = jr.split(this_key, 2)
        #     return this_key, jr.uniform(use_key, (2,), minval=0, maxval=side_length), all_nodes

        for i in range(self.num_agents):
            agent_key, goal_key, key = jr.split(key, 3)

            _, agent_candidate, _, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node, init_val=(agent_key, agent_candidate, states, i))
            states = states.at[i].set(agent_candidate)

            _, goal_candidate, _, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node, init_val=(goal_key, goal_candidate, goals, i))
            goals = goals.at[i].set(goal_candidate)

        all_states = jnp.concatenate([states, goals, obs_pos], axis=0)
        self.xy_min = all_states.min(axis=0)
        self.xy_max = all_states.max(axis=0)

        return self.get_graph(all_states)

    def step(self, graph: GraphsTuple, action: Action) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params['n_obs'])
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = agent_states + action * self.dt
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)
        next_states = self.clip_state(next_states)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward
        reward = jnp.zeros(()).astype(jnp.float32)
        states = next_states[:self.num_agents, :2]
        goals = next_states[self.num_agents: self.num_agents * 2, :2]
        dist2goal = jnp.linalg.norm(goals - states, axis=-1)
        reward -= (dist2goal.mean())**2
        reward -= (jnp.linalg.norm(action, axis=1)**2).mean() * 0.01

        # # compute cost: collision
        # agent_pos = agent_states[:, :2]
        # dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        # dist += jnp.eye(self.num_agents) * 1e6
        # collision_cost = nn.relu(self._params['car_radius'] * 2 - dist).mean()
        #
        # # compute cost: connectivity
        # connect_cost = nn.relu(
        #     graph.connectivity[:self.num_agents, :self.num_agents] * dist - self._params['comm_radius']
        # ).mean()
        #
        # cost = collision_cost + connect_cost

        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        return self.get_graph(next_states), reward, cost, done, {}

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params['n_obs'])

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision_cost = nn.relu(self._params['car_radius'] * 2 - dist).mean()

        # collision between agents and obstacles
        obs_pos = obs_states[:, :2]
        dist_obs = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obs_pos, 0), axis=-1)
        collision_cost += nn.relu(self._params['car_radius'] + self._params['obs_radius'] - dist_obs).mean()

        # connectivity
        connect_cost = nn.relu(
            graph.connectivity[:self.num_agents, :self.num_agents] * dist - self._params['comm_radius']
        ).mean()

        cost = collision_cost + connect_cost

        return cost

    def render(self, graph: GraphsTuple) -> Array:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=80)

        # plot the cars and the communication network
        pos = np.array(graph.states[:-1, :2])

        # agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)
        # dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        # dist += jnp.eye(self.num_agents) * 1e6
        # collision_cost = nn.relu(self._params['car_radius'] * 2 - dist).mean()
        #
        # # compute cost: connectivity
        # connect_cost = nn.relu(
        #     graph.connectivity[:self.num_agents, :self.num_agents] * dist - self._params['comm_radius']
        # ).mean()
        #
        # cost = collision_cost + connect_cost
        cost = self.get_cost(graph)

        edge_index = np.array(jnp.stack([graph.senders, graph.receivers], axis=0))
        is_pad = np.any(edge_index == self.num_agents * 2 + self._params['n_obs'], axis=0)
        edge_index_nopad = edge_index[:, ~is_pad]
        color = ['#FF8C00'] * self.num_agents + ['#3CB371'] * self.num_agents + ['#710000'] * self._params['n_obs']
        radius = [self._params['car_radius']] * self.num_agents * 2 + [self._params['obs_radius']] * self._params['n_obs']
        edge_color = np.array(['k'] * edge_index_nopad.shape[1])
        desired_connect = graph.edges[~is_pad, -1]
        edge_color[np.where(desired_connect)] = 'r'

        # violate the connectivity constraint

        edge_color = list(edge_color)
        plot_graph(ax, pos, radius=radius, color=color, edge_color=edge_color,
                   with_label=True, plot_edge=True, edge_index=edge_index_nopad, alpha=0.8)

        ax.set_title(f'cost: {cost:.4f}')

        # set axis limits
        low_lim, up_lim = self.state_lim(graph.states)
        ax.set_xlim(low_lim[0], up_lim[0])
        ax.set_ylim(low_lim[1], up_lim[1])
        plt.axis('off')

        # convert to numpy array
        fig.canvas.draw()
        fig_np = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return fig_np

    def connect_mask(self, states: State) -> Array:
        # agents are fully connected
        connect_mask = jnp.ones((self.num_agents * 2 + self._params['n_obs'],
                                 self.num_agents * 2 + self._params['n_obs']))

        # goals and obstacles are not connected
        connect_mask = connect_mask.at[self.num_agents:, :].set(0.)

        # connect agents to their own goals
        connect_mask = connect_mask.at[jnp.arange(self.num_agents), self.num_agents: 2 * self.num_agents].set(0.)
        connect_mask = connect_mask.at[jnp.arange(self.num_agents), self.num_agents + jnp.arange(self.num_agents)].set(1)

        return connect_mask

    def get_graph(self, states: State, adjacency: Array = None, connectivity: Array = None) -> GraphsTuple:
        # node features
        n_nodes = states.shape[0]
        node_feats = jnp.zeros((self.num_agents * 2 + self._params['n_obs'], 1))  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, :].set(SimpleCar.GOAL)  # goal feats
        node_feats = node_feats.at[-self._params['n_obs']:, :].set(SimpleCar.OBS)  # obs feats
        agent_type = jnp.zeros(self.num_agents, dtype=jnp.int32)
        goal_type = jnp.ones(self.num_agents, dtype=jnp.int32)
        obs_type = jnp.ones(self._params['n_obs'], dtype=jnp.int32) * SimpleCar.OBS

        # node type
        node_type = jnp.concatenate([agent_type, goal_type, obs_type])

        # connectivity
        if connectivity is None:
            connectivity = self.desired_connect_mask(states)

        # edge features
        agent_feats = states
        recv_feats = agent_feats[:, None, :]  # (n_nodes, 1, n_feat)
        send_feats = agent_feats[None, :, :]  # (1, n_nodes, n_feat)
        edge_feats = recv_feats - send_feats  # (n_nodes, n_nodes, n_feat)
        edge_feats = jnp.concatenate([edge_feats, connectivity[:, :, None]], axis=-1)

        # adjacency matrix
        if adjacency is None:
            adjacency = self.connect_mask(states)

        # create graph
        return DenseGraph(
            n_nodes=n_nodes,
            n_edges=adjacency.shape[0] * adjacency.shape[1],
            nodes=node_feats,
            edges=edge_feats,
            states=states,
            adjacency=adjacency,
            node_type=node_type,
            connectivity=connectivity,
        ).to_padded()

    def desired_connect_mask(self, states: State) -> Array:
        # adjacency = self.connect_mask(states)

        # # compute real adjacency based on distance
        # pos = states[:, :2]
        # pos_diff = pos[:, None, :] - pos[None, :, :]  # [i, j]: i -> j
        # dist = jnp.linalg.norm(pos_diff, axis=-1)
        # dist = dist + jnp.eye(dist.shape[1]) * (self._params['comm_radius'] + 1)  # remove self connection
        # dist = dist.at[-self.num_agents:, :].set(self._params['comm_radius'] + 1)  # goals are not connected
        #
        # # connect agents to their own goals
        # dist = dist.at[jnp.arange(self.num_agents), -self.num_agents:].set(self._params['comm_radius'] + 1)
        # dist = dist.at[jnp.arange(self.num_agents), self.num_agents + jnp.arange(self.num_agents)].set(0.0)
        #
        # adjacency = jnp.less(dist, self._params['comm_radius'])

        # desired_connect = \
        #     adjacency[:self.num_agents, :self.num_agents] * np.random.rand(self.num_agents, self.num_agents) > 0.4
        # desired_connect = desired_connect.astype(jnp.float32)
        desired_connect = \
            jnp.eye(self.num_agents, k=1) + jnp.eye(self.num_agents, k=-1)
        connectivity_pad = jnp.zeros((self.num_agents * 2 + self._params['n_obs'],
                                      self.num_agents * 2 + self._params['n_obs']))
        connectivity_pad = connectivity_pad.at[:self.num_agents, :self.num_agents].set(desired_connect)
        return connectivity_pad

    def state_lim(self, state: State) -> Tuple[State, State]:
        goal_states = state[self.num_agents:]
        low_lim = jnp.array([0.0, 0.0])
        low_lim = jnp.clip(goal_states, a_max=low_lim).min(axis=0)
        up_lim = jnp.array([self._params['area_size'], self._params['area_size']])
        up_lim = jnp.clip(goal_states, a_min=up_lim).max(axis=0)
        return low_lim, up_lim

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params['n_obs'])
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = agent_states + action * self.dt
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)
        next_states = self.clip_state(next_states)

        return self.get_graph(next_states, adjacency=graph.adjacency, connectivity=graph.connectivity)

    def safe_mask(self, graph: GraphsTuple) -> Array:
        safe_mask = super().safe_mask(graph)
        # agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)
        # assert agent_pos.shape == (self.num_agents, 2)
        # pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        # dist = jnp.linalg.norm(pos_diff, axis=-1)
        # dist = dist + jnp.eye(dist.shape[1]) * (self._params['car_radius'] * 2 + 1)  # remove self connection
        #
        # # agents are not colliding
        # safe = jnp.greater(dist, self._params['car_radius'] * 3)
        # safe_agent = jnp.min(safe, axis=1)
        #
        # # agents are not colliding with obstacles
        # obs_pos = graph.type_states(type_idx=2, n_type=self._params['n_obs'])
        # pos_diff = agent_pos[:, None, :] - obs_pos[None, :, :]  # [i, j]: i -> j
        # dist = jnp.linalg.norm(pos_diff, axis=-1)
        # safe_obs = jnp.greater(dist, self._params['car_radius'] * 2 + self._params['obs_radius'])
        # safe_obs = jnp.min(safe_obs, axis=1)
        #
        # safe_mask = jnp.logical_and(safe_agent, safe_obs)

        # connectivity
        dist_edge = jnp.linalg.norm(graph.edges[:, :2], axis=-1) * graph.edges[:, -1]
        safe_dist_connect = jnp.less(dist_edge, self._params['car_radius'] * 0.8)
        safe_connect_mask = jnp.zeros_like(safe_mask)
        safe_connect_mask = safe_connect_mask.at[graph.senders].set(safe_dist_connect)
        safe_connect_mask = safe_connect_mask.at[graph.receivers].set(safe_dist_connect)
        safe_mask = jnp.logical_or(safe_mask, safe_connect_mask)

        return safe_mask

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        unsafe_mask = super().unsafe_mask(graph)
        # agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)
        # assert agent_pos.shape == (self.num_agents, 2)
        # pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        # dist = jnp.linalg.norm(pos_diff, axis=-1)
        # dist = dist + jnp.eye(dist.shape[1]) * (self._params['car_radius'] * 2 + 1)  # remove self connection
        #
        # # agents are colliding
        # unsafe = jnp.less(dist, self._params['car_radius'] * 2)
        # unsafe_agent = jnp.max(unsafe, axis=1)
        #
        # # agents are colliding with obstacles
        # obs_pos = graph.type_states(type_idx=2, n_type=self._params['n_obs'])
        # pos_diff = agent_pos[:, None, :] - obs_pos[None, :, :]  # [i, j]: i -> j
        # dist = jnp.linalg.norm(pos_diff, axis=-1)
        # unsafe_obs = jnp.less(dist, self._params['car_radius'] + self._params['obs_radius'])
        # unsafe_obs = jnp.max(unsafe_obs, axis=1)
        #
        # unsafe_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # connectivity
        dist_edge = jnp.linalg.norm(graph.edges[:, :2], axis=-1) * graph.edges[:, -1]
        unsafe_dist_connect = jnp.greater(dist_edge, self._params['car_radius'] * 0.8)
        unsafe_connect_mask = jnp.zeros_like(unsafe_mask)
        unsafe_connect_mask = unsafe_connect_mask.at[graph.senders].set(unsafe_dist_connect)
        unsafe_connect_mask = unsafe_connect_mask.at[graph.receivers].set(unsafe_dist_connect)
        unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_connect_mask)

        return unsafe_mask
