import os
import pickle
import functools as ft
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax

from flax.training.train_state import TrainState
from jax import lax

from model.utils.typing import Action, Params, PRNGKey, Array
from model.utils.graph import GraphsTuple
from model.utils.utils import merge01
from model.trainer.data import Rollout, GraphCostBatch
from model.trainer.utils import has_any_nan, compute_norm_and_clip
from model.env.base import MultiAgentEnv
from model.algo.module.policy import PPOPolicy
from model.algo.module.value import ValueNet
from .base import MultiAgentController
from .utils import compute_gae
from .read_clipped_graphs import load_and_merge_failed_batches


def empty_grad_tx() -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        zero_updates = jtu.tree_map(jnp.zeros_like, updates)
        return zero_updates, optax.EmptyState()

    return optax.GradientTransformation(init_fn, update_fn)


class MAPPO(MultiAgentController):

    def __init__(
        self,
        env: MultiAgentEnv,
        node_dim: int,
        edge_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        cost_weight: float = 0.0,
        gnn_layers: int = 1,
        gamma: float = 0.99,
        lr_actor: float = 1e-5,
        lr_critic: float = 3e-5,
        batch_size: int = 8192,
        epoch_ppo: int = 10,
        epoch_criticnet: int = 3,
        clip_eps: float = 0.25,
        gae_lambda: float = 0.95,
        coef_ent: float = 1e-2,
        max_grad_norm: float = 2.0,
        seed: int = 0,
        rollout_length: Optional[int] = None,
        **kwargs
    ):
        super(MAPPO, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        self.cost_weight = cost_weight
        self.actor_gnn_layers = gnn_layers
        self.critic_gnn_layers = gnn_layers
        self.gamma = gamma
        self.rollout_length = rollout_length if rollout_length is not None else env.max_episode_steps
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        self.epoch_criticnet = epoch_criticnet
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.seed = seed

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

        self.policy = PPOPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            gnn_layers=self.actor_gnn_layers,
        )

        key = jr.PRNGKey(seed)
        policy_key, key = jr.split(key)
        policy_params = self.policy.dist.init(policy_key, nominal_graph, self.n_agents)
        policy_optim = optax.adam(learning_rate=lr_actor)
        self.policy_optim = optax.apply_if_finite(policy_optim, 1_000_000)
        self.policy_train_state = TrainState.create(
            apply_fn=self.policy.sample_action,
            params=policy_params,
            tx=self.policy_optim
        )

        self.critic = ValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            gnn_layers=self.critic_gnn_layers
        )

        critic_key, key = jr.split(key)
        critic_params = self.critic.net.init(critic_key, nominal_graph, self.n_agents)
        critic_optim = optax.adam(learning_rate=lr_critic)
        self.critic_optim = optax.apply_if_finite(critic_optim, 1_000_000)
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.get_value,
            params=critic_params,
            tx=self.critic_optim
        )

        self.key = key
        self.rng = np.random.default_rng(seed=seed + 1)

        self.failed_sft_buffer = self._load_failed_sft_buffer()
        self.sft_batch_size = batch_size
        self.freeze_actor_during_sft = True

    def _reconstruct_env_states(self, raw_env_states, n_samples: int):
        ref_env_states = self._env.reset(jr.PRNGKey(0)).env_states
        env_state_cls = type(ref_env_states)

        try:
            if len(raw_env_states) == n_samples:
                env_states_list = [env_state_cls(*sample) for sample in raw_env_states]
                return jtu.tree_map(lambda *xs: jnp.array(np.stack(xs)), *env_states_list)
        except Exception as e:
            print(f"SFT: failed to reconstruct env_states from NPZ, using dummy zeros. Error: {e}")

        return jtu.tree_map(
            lambda leaf: jnp.zeros((n_samples,) + leaf.shape, dtype=leaf.dtype),
            ref_env_states
        )

    def _load_failed_sft_buffer(self) -> Optional[GraphCostBatch]:
        data_dir = os.path.join(os.path.dirname(__file__), "clipped_graphs_npz")
        failed_batch = load_and_merge_failed_batches(data_dir)

        if not failed_batch:
            print("SFT: no clipped_graphs_npz samples found.")
            return None

        n_samples = len(failed_batch["graph_nodes"])
        env_states = self._reconstruct_env_states(failed_batch["graph_env_states"], n_samples)

        graph = GraphsTuple(
            nodes=jnp.asarray(failed_batch["graph_nodes"]),
            edges=jnp.asarray(failed_batch["graph_edges"]),
            states=jnp.asarray(failed_batch["graph_states"]),
            n_node=jnp.asarray(failed_batch["graph_n_node"]).squeeze(),
            n_edge=jnp.asarray(failed_batch["graph_n_edge"]).squeeze(),
            senders=jnp.asarray(failed_batch["graph_senders"]),
            receivers=jnp.asarray(failed_batch["graph_receivers"]),
            node_type=jnp.asarray(failed_batch["graph_node_type"]),
            env_states=env_states,
        )

        target = jnp.asarray(failed_batch["graph_cost"], dtype=jnp.float32).reshape(-1)

        print(f"SFT: loaded {target.shape[0]} NPZ graph-cost samples.")
        return GraphCostBatch(graph=graph, target=target)

    @property
    def config(self) -> dict:
        return {
            "cost_weight": self.cost_weight,
            "actor_gnn_layers": self.actor_gnn_layers,
            "critic_gnn_layers": self.critic_gnn_layers,
            "gamma": self.gamma,
            "rollout_length": self.rollout_length,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "batch_size": self.batch_size,
            "epoch_ppo": self.epoch_ppo,
            "epoch_criticnet": self.epoch_criticnet,
            "clip_eps": self.clip_eps,
            "gae_lambda": self.gae_lambda,
            "coef_ent": self.coef_ent,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "sft_batch_size": self.sft_batch_size,
        }

    @property
    def actor_params(self) -> Params:
        return self.policy_train_state.params

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_params
        action = 2 * self.policy.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.policy_train_state.apply_fn(params, graph, key)
        action = 2 * action + self._env.u_ref(graph)
        return action, log_pi

    def update(self, rollout: Rollout, step: int) -> dict:
        _, self.key = jr.split(self.key)
        idx = np.arange(rollout.dones.shape[0] * rollout.dones.shape[1])
        np.random.shuffle(idx)
        batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // self.batch_size))

        self.critic_train_state, self.policy_train_state, update_info = self.update_inner(
            self.critic_train_state,
            self.policy_train_state,
            rollout,
            batch_idx,
        )
        return update_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
        self,
        critic_train_state: TrainState,
        policy_train_state: TrainState,
        rollout: Rollout,
        batch_idx: Array,
    ) -> Tuple[TrainState, TrainState, dict]:
        value_fn = jax.vmap(jax.vmap(ft.partial(self.critic.get_value, critic_train_state.params)))
        values = value_fn(rollout.graph)
        next_values = value_fn(rollout.next_graph)

        targets, gaes = compute_gae(
            values=values,
            rewards=rollout.rewards - self.cost_weight * rollout.costs,
            dones=rollout.dones,
            next_values=next_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        rollout = jtu.tree_map(lambda x: merge01(x), rollout)
        targets = merge01(targets)
        gaes = merge01(gaes)

        def update_critic_fn(carry, idx):
            critic = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            critic, critic_info = self.update_critic(critic, rollout_batch, targets[idx])
            return critic, critic_info

        def update_policy_fn(carry, idx):
            policy = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            policy, policy_info = self.update_policy(policy, rollout_batch, gaes[idx])
            return policy, policy_info

        critic_batch_idx = jnp.tile(batch_idx, (self.epoch_criticnet, 1, 1)).reshape(-1, batch_idx.shape[-1])
        policy_batch_idx = jnp.tile(batch_idx, (self.epoch_ppo, 1, 1)).reshape(-1, batch_idx.shape[-1])

        critic_train_state, critic_info = lax.scan(update_critic_fn, critic_train_state, critic_batch_idx)
        policy_train_state, policy_info = lax.scan(update_policy_fn, policy_train_state, policy_batch_idx)

        info = jtu.tree_map(lambda x: x[-1], critic_info | policy_info)
        return critic_train_state, policy_train_state, info

    def update_critic(
        self,
        critic_train_state: TrainState,
        rollout: Rollout,
        targets: Array
    ) -> Tuple[TrainState, dict]:

        def get_value_loss(params):
            values = jax.vmap(ft.partial(self.critic.get_value, params))(rollout.graph)
            loss_critic = optax.l2_loss(values, targets).mean()
            return loss_critic

        loss, grad = jax.value_and_grad(get_value_loss)(critic_train_state.params)
        critic_has_nan = has_any_nan(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad)

        return critic_train_state, {
            "critic/loss": loss,
            "critic/grad_norm": grad_norm,
            "critic/has_nan": critic_has_nan,
            "critic/max_target": jnp.max(targets),
            "critic/min_target": jnp.min(targets),
        }

    def update_policy(
        self,
        policy_train_state: TrainState,
        rollout: Rollout,
        gaes: Array,
    ) -> Tuple[TrainState, dict]:
        gaes = gaes[:, None]
        gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0])

        def get_policy_loss(params):
            eval_action_vmap = jax.vmap(ft.partial(self.policy.eval_action, params))
            u_refs = jax.vmap(self._env.u_ref)(rollout.graph)
            raw_actions = (rollout.actions - u_refs) / 2

            log_pis, policy_entropy = eval_action_vmap(rollout.graph, raw_actions, action_keys)
            ratio = jnp.exp(log_pis - rollout.log_pis)

            loss_policy1 = -ratio * gaes
            loss_policy2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes

            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            total_entropy = policy_entropy.mean()
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(ratio - 1.0))

            total_loss = loss_policy - self.coef_ent * total_entropy
            return total_loss, {
                "policy/clip_frac": clip_frac,
                "policy/entropy": total_entropy,
                "policy/total_variation_dist": total_variation_dist,
            }

        (loss, info), grad = jax.value_and_grad(get_policy_loss, has_aux=True)(policy_train_state.params)
        policy_has_nan = has_any_nan(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        info = {
            "policy/loss": loss,
            "policy/grad_norm": grad_norm,
            "policy/has_nan": policy_has_nan,
            "policy/log_pi_min": rollout.log_pis.min(),
        } | info
        return policy_train_state, info

    def sft_setup(
        self,
        sft_lr_critic: float = 1e-6,
        sft_batch_size: int = 256,
        freeze_actor: bool = True,
    ):
        if self.failed_sft_buffer is None:
            raise RuntimeError("SFT: no NPZ graph-cost buffer available.")

        self.freeze_actor_during_sft = freeze_actor
        self.sft_batch_size = min(sft_batch_size, int(self.failed_sft_buffer.target.shape[0]))

        critic_optim = optax.adam(learning_rate=sft_lr_critic)
        self.critic_optim = optax.apply_if_finite(critic_optim, 1_000_000)
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.get_value,
            params=self.critic_train_state.params,
            tx=self.critic_optim,
        )

        if freeze_actor:
            self.policy_train_state = TrainState.create(
                apply_fn=self.policy.sample_action,
                params=self.policy_train_state.params,
                tx=empty_grad_tx(),
            )

        print(
            f"SFT setup: {self.failed_sft_buffer.target.shape[0]} NPZ samples, "
            f"batch_size={self.sft_batch_size}, critic_lr={sft_lr_critic}, freeze_actor={freeze_actor}"
        )

    def _batch_flat_graph_cost(self, flat_batch: GraphCostBatch, indices: np.ndarray) -> GraphCostBatch:
        return jtu.tree_map(lambda x: x[indices], flat_batch)

    def _make_even_mixed_batch(
        self,
        npz_flat: GraphCostBatch,
        onpolicy_flat: GraphCostBatch,
    ) -> Tuple[GraphCostBatch, Array, Array]:
        n_npz = int(npz_flat.target.shape[0])
        n_op = int(onpolicy_flat.target.shape[0])

        batch_size = min(self.sft_batch_size, 2 * min(n_npz, n_op))
        if batch_size % 2 == 1:
            batch_size -= 1
        if batch_size <= 0:
            raise RuntimeError("SFT: mixed batch construction failed because batch_size <= 0.")

        half_bs = batch_size // 2
        n_usable = min(n_npz, n_op)
        n_minibatches = max(1, n_usable // half_bs)
        n_use = n_minibatches * half_bs

        idx_npz = self.rng.permutation(n_npz)[:n_use].reshape(n_minibatches, half_bs)
        idx_op = self.rng.permutation(n_op)[:n_use].reshape(n_minibatches, half_bs)

        npz_batch = self._batch_flat_graph_cost(npz_flat, idx_npz)
        op_batch = self._batch_flat_graph_cost(onpolicy_flat, idx_op)

        mixed_graph = jtu.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=1),
            npz_batch.graph,
            op_batch.graph,
        )
        mixed_target = jnp.concatenate([npz_batch.target, op_batch.target], axis=1)

        return (
            GraphCostBatch(graph=mixed_graph, target=mixed_target),
            jnp.mean(npz_batch.target),
            jnp.mean(op_batch.target),
        )

    def sft_step(self, step: int) -> dict:
        if self.failed_sft_buffer is None:
            raise RuntimeError("SFT: no NPZ buffer loaded.")

        n_total = int(self.failed_sft_buffer.target.shape[0])
        bs = min(self.sft_batch_size, n_total)

        perm = self.rng.permutation(n_total)
        n_usable = (n_total // bs) * bs
        if n_usable == 0:
            n_usable = n_total
            bs = n_total

        idx = perm[:n_usable].reshape(-1, bs)
        batch = jtu.tree_map(lambda x: x[idx], self.failed_sft_buffer)

        self.critic_train_state, info = self.sft_update_critic_inner(self.critic_train_state, batch)
        info = info | {
            "sft/npz_target_mean": jnp.mean(batch.target),
            "sft/onpolicy_target_mean": jnp.array(0.0),
            "sft/mix_ratio_old": jnp.array(1.0),
            "sft/mix_ratio_new": jnp.array(0.0),
        }
        return info

    def sft_step_mixed(self, step: int, rollout_fn, n_env_train: int) -> dict:
        if self.failed_sft_buffer is None:
            raise RuntimeError("SFT: no NPZ buffer loaded.")

        key_x0, self.key = jr.split(self.key)
        key_x0 = jr.split(key_x0, n_env_train)
        rollouts: Rollout = rollout_fn(self.actor_params, key_x0)

        onpolicy_flat = GraphCostBatch(
            graph=jtu.tree_map(lambda x: merge01(x), rollouts.graph),
            target=merge01(rollouts.costs).sum(axis=-1).astype(jnp.float32),
        )

        mixed_batch, npz_mean, op_mean = self._make_even_mixed_batch(
            self.failed_sft_buffer,
            onpolicy_flat,
        )

        self.critic_train_state, info = self.sft_update_critic_inner(
            self.critic_train_state,
            mixed_batch,
        )

        info = info | {
            "sft/npz_target_mean": npz_mean,
            "sft/onpolicy_target_mean": op_mean,
            "sft/mix_ratio_old": jnp.array(0.5),
            "sft/mix_ratio_new": jnp.array(0.5),
            "sft/target_mean": jnp.mean(mixed_batch.target),
        }
        return info

    @ft.partial(jax.jit, static_argnums=(0,))
    def sft_update_critic_inner(
        self,
        critic_train_state: TrainState,
        batch: GraphCostBatch,
    ) -> Tuple[TrainState, dict]:

        def update_fn(carry, minibatch: GraphCostBatch):
            critic = carry
            critic, info = self.update_critic_from_targets(critic, minibatch)
            return critic, info

        critic_train_state, info = lax.scan(update_fn, critic_train_state, batch)
        info = jtu.tree_map(lambda x: x[-1], info)
        return critic_train_state, info

    def update_critic_from_targets(
        self,
        critic_train_state: TrainState,
        batch: GraphCostBatch,
    ) -> Tuple[TrainState, dict]:

        def loss_fn(params):
            values = jax.vmap(ft.partial(self.critic.get_value, params))(batch.graph)
            loss_critic = optax.l2_loss(values, batch.target).mean()
            return loss_critic

        loss, grad = jax.value_and_grad(loss_fn)(critic_train_state.params)
        critic_has_nan = has_any_nan(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad)

        return critic_train_state, {
            "critic/loss": loss,
            "critic/grad_norm": grad_norm,
            "critic/has_nan": critic_has_nan,
            "critic/max_target": jnp.max(batch.target),
            "critic/min_target": jnp.min(batch.target),
        }

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, "actor.pkl"), "wb"))
        pickle.dump(self.critic_train_state.params, open(os.path.join(model_dir, "critic.pkl"), "wb"))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))
        self.policy_train_state = self.policy_train_state.replace(
            params=pickle.load(open(os.path.join(path, "actor.pkl"), "rb"))
        )
        self.critic_train_state = self.critic_train_state.replace(
            params=pickle.load(open(os.path.join(path, "critic.pkl"), "rb"))
        )