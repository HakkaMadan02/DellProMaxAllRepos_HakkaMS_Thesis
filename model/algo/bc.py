import jax.numpy as jnp
import jax.random as jr
import optax
import pathlib
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle
import einops as ei

from typing import Optional, Tuple, NamedTuple
from flax.core import unfreeze
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP, QPSolution
from jax import lax

from model.utils.typing import Action, Params, PRNGKey, Array, State
from model.utils.graph import GraphsTuple
from model.utils.utils import merge01, jax_vmap, tree_merge, mask2index
from model.trainer.data import Rollout
from model.trainer.buffer import MaskedReplayBuffer
from model.trainer.utils import has_any_nan, get_ckpt_manager, load_ckpt, compute_norm_and_clip, jax2np, np2jax
from model.env.base import MultiAgentEnv
from model.algo.module.cbf import CBF
from model.algo.module.policy import DeterministicPolicy
from .base import MultiAgentController
from .utils import get_cbf_fn


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action


class BC(MultiAgentController):
    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int,
            batch_size: int,
            buffer_size: int,
            lr_actor: float = 3e-5,
            horizon: int = 32,
            alpha: float = 1.0,
            inner_epoch: int = 8,
            loss_action_coef: float = 0.001,
            max_grad_norm: float = 2.,
            seed: int = 0,
            **kwargs
    ):
        super(BC, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.state_dim = state_dim
        self.gnn_layers = gnn_layers
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr_actor = lr_actor
        self.horizon = horizon
        self.alpha = alpha
        self.inner_epoch = inner_epoch
        self.loss_action_coef = loss_action_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        # set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        # set up actor
        key = jr.PRNGKey(seed)
        self.actor = DeterministicPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        actor_key, key = jr.split(key)
        actor_params = self.actor.net.init(actor_key, nominal_graph, self.n_agents)
        actor_optim = optax.adam(learning_rate=lr_actor)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=self.actor_optim
        )

        # set up key
        self.key = key
        self.buffer = MaskedReplayBuffer(size=buffer_size)
        self.unsafe_buffer = MaskedReplayBuffer(size=buffer_size // 2)

        self.rng = np.random.default_rng(seed=seed + 1)
        self.centralized_cbf = get_cbf_fn(env)

    @property
    def config(self) -> dict:
        return {
            'state_dim': self.state_dim,
            'gnn_layers': self.gnn_layers,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'lr_actor': self.lr_actor,
            'horizon': self.horizon,
            'alpha': self.alpha,
            'inner_epoch': self.inner_epoch,
            'loss_action_coef': self.loss_action_coef,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed
        }

    @property
    def actor_params(self) -> Params:
        return self.actor_train_state.params

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_params
        action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph), log_pi
        # return action, log_pi

    def get_centralized_cbf(self, graph: GraphsTuple) -> Array:
        return self.centralized_cbf(graph)

    def get_qp_action(
            self, graph: GraphsTuple, relax_penalty: float = 1e3, qp_settings: JaxProxQP.Settings = None
    ) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            # return self.get_cbf(new_graph, params=cbf_params)
            return self.centralized_cbf(new_graph)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state).squeeze()
        h_x = jax.jacobian(h_aug)(agent_state).squeeze()

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        Lf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i")
        Lg_h = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")
        Lg_h = Lg_h.reshape((self.n_agents, -1))

        u_lb, u_ub = self._env.action_lim()
        u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ref = self._env.u_ref(graph).reshape(-1)

        # construct QP: min x^T H x + g^T x, s.t. Cx <= b
        H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents, dtype=jnp.float32)
        H = H.at[-self.n_agents:, -self.n_agents:].set(H[-self.n_agents:, -self.n_agents:] * 10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents)])
        C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents)], axis=1)
        b = Lf_h + self.alpha * h

        r_lb = jnp.array([0.] * self.n_agents, dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf] * self.n_agents, dtype=jnp.float32)
        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        if qp_settings is None:
            qp_settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, qp_settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents,)
        u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents:]
        u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        # (n_collect, T)
        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)

        rollout_np = jax2np(rollout)
        del rollout
        # t1 = time.time()
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask)

        # inner loop
        # t2 = time.time()
        update_info = self.update_nets(rollout_batch, safe_mask_batch, unsafe_mask_batch)
        # t3 = time.time()
        # print("mask: {:.3f}, buffer: {:.3f}, update: {:.3f}".format(t1 - t0, t2 - t1, t3 - t2))

        return update_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> jnp.ndarray:
        # safe if in the horizon, the agent is always safe
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.horizon else i - self.horizon
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(
                    ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_))
                # initial state is always safe
                safe_rollout_mask = safe_rollout_mask.at[0].set(1)
            return safe_rollout_mask

        safe = jax_vmap(jax_vmap(safe_rollout, in_axes=1, out_axes=1))(unsafe_mask)
        return safe

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_qp(self, b_graph: GraphsTuple) -> Action:
        b_u_qp, bT_relaxation = jax.vmap(ft.partial(self.get_qp_action))(b_graph)
        return b_u_qp

    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask):
        update_info = {}

        # Compute b_u_qp.
        n_chunks = 8
        batch_size = len(rollout.graph.states)
        chunk_size = batch_size // n_chunks

        # t0 = time.time()
        b_u_qp = []
        for ii in range(n_chunks):
            graph = jtu.tree_map(lambda x: x[ii * chunk_size: (ii + 1) * chunk_size], rollout.graph)
            b_u_qp.append(jax2np(self.get_b_u_qp(graph)))
        b_u_qp = tree_merge(b_u_qp)
        # t1 = time.time()
        # print("qp time: {:.3f}".format(t1 - t0))
        # b_u_qp = self.get_b_u_qp(rollout.graph, self.cbf_tgt.params)

        batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_qp)

        for i_epoch in range(self.inner_epoch):
            idx = self.rng.choice(rollout.length, size=rollout.length, replace=False)
            # (n_mb, mb_size)
            batch_idx = np.stack(np.array_split(idx, idx.shape[0] // self.batch_size), axis=0)
            batch = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            actor_train_state, update_info = self.update_inner(
                self.actor_train_state, batch
            )
            self.actor_train_state = actor_train_state

        return update_info

    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        if self.buffer.length > self.batch_size:
            # print("buffer length > batch size       {} > {}".format(self.buffer.length, self.batch_size))
            # sample from memory
            memory, safe_mask_memory, unsafe_mask_memory = self.buffer.sample(rollout.length)
            unsafe_memory, safe_mask_unsafe_memory, unsafe_mask_unsafe_memory = self.unsafe_buffer.sample(
                rollout.length * rollout.time_horizon)  # todo: maybe add more unsafe data

            # append new data to memory
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # get update data
            # (b, T)
            rollout = tree_merge([memory, rollout])
            safe_mask = tree_merge([safe_mask_memory, safe_mask])
            unsafe_mask = tree_merge([unsafe_mask_memory, unsafe_mask])

            # (b, T) -> (b * T, )
            rollout = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask = merge01(safe_mask)
            unsafe_mask = merge01(unsafe_mask)
            rollout_batch = tree_merge([unsafe_memory, rollout])
            safe_mask_batch = tree_merge([safe_mask_unsafe_memory, safe_mask])
            unsafe_mask_batch = tree_merge([unsafe_mask_unsafe_memory, unsafe_mask])
        else:
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # (b, T) -> (b * T, )
            rollout_batch = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask_batch = merge01(safe_mask)
            unsafe_mask_batch = merge01(unsafe_mask)

        return rollout_batch, safe_mask_batch, unsafe_mask_batch

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner(
            self, actor_train_state: TrainState, batch: Batch
    ) -> Tuple[TrainState, dict]:

        def update_fn(carry, minibatch: Batch):
            actor = carry

            def get_loss(actor_params: Params) -> Tuple[Array, dict]:
                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action = action_fn(minibatch.graph)

                # action loss
                assert action.shape == minibatch.u_qp.shape
                loss_action = jnp.mean(jnp.square(action - minibatch.u_qp).sum(axis=-1))

                # total loss
                total_loss = self.loss_action_coef * loss_action

                return total_loss, {'loss/action': loss_action,
                                    'loss/total': total_loss}

            (loss, loss_info), grad_actor = jax.value_and_grad(
                get_loss, has_aux=True, argnums=0)(actor.params)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            actor = actor.apply_gradients(grads=grad_actor)
            grad_info = {'grad_norm/actor': grad_actor_norm}
            return actor, grad_info | loss_info

        train_state = actor_train_state
        actor_train_state, info = lax.scan(update_fn, train_state, batch)
        # get training info of the last epoch
        info = jtu.tree_map(lambda x: x[-1], info)
        return actor_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.actor_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.actor_train_state = \
            self.actor_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
