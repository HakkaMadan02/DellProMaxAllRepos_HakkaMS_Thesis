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

from typing import Optional, Tuple
from flax.core import unfreeze
from flax.training.train_state import TrainState
from jax import lax

from model.utils.typing import Action, Params, PRNGKey, Array
from model.utils.graph import GraphsTuple
from model.utils.utils import merge01, jax_vmap, tree_merge
from model.trainer.data import Rollout
from model.trainer.buffer import ReplayBuffer
from model.trainer.utils import has_any_nan, get_ckpt_manager, load_ckpt, compute_norm_and_clip, jax2np, np2jax
from model.env.base import MultiAgentEnv
from model.algo.module.cbf import CBF
from model.algo.module.policy import DeterministicPolicy, PPOPolicy
from model.algo.module.value import ValueNet
from .base import MultiAgentController
from .utils import compute_gae
from model.algo.module.policy import TanhNormal

def compute_radial_loss(head1_attn: Array, edges: Array, senders: Array, node_types: Array) -> Array:
    rel_px = edges[:, 0]
    rel_py = edges[:, 1]
    rel_vx = edges[:, 2]
    rel_vy = edges[:, 3]
    dist = jnp.sqrt(rel_px**2 + rel_py ** 2) + 1e-6
    r_ij = (rel_px * rel_vx + rel_py * rel_vy) / dist
    goal_mask = node_types[senders] == 1
    loss = jnp.where(goal_mask, 0.0, head1_attn * r_ij)
    return jnp.mean(loss)

class MAPPO(MultiAgentController):

    def __init__(
        self,
        env: MultiAgentEnv,
        node_dim: int,
        edge_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        cost_weight: float = 0,
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
        coef_radial: float = 1e-3,
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

        # set hyperparameters
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
        self.coef_radial = coef_radial
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

        # set up PPO policy
        self.policy = PPOPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            gnn_layers=self.actor_gnn_layers
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

        # set up PPO critic
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

        # set up key
        self.key = key

    @property
    def config(self) -> dict:
        return {
            'cost_weight': self.cost_weight,
            'coef_radial': self.coef_radial,
            'actor_gnn_layers': self.actor_gnn_layers,
            'critic_gnn_layers': self.critic_gnn_layers,
            'gamma': self.gamma,
            'rollout_length': self.rollout_length,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'batch_size': self.batch_size,
            'epoch_ppo': self.epoch_ppo,
            'epoch_criticnet': self.epoch_criticnet,
            'clip_eps': self.clip_eps,
            'gae_lambda': self.gae_lambda,
            'coef_ent': self.coef_ent,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed
        }

    @property
    def actor_params(self) -> Params:
        return self.policy_train_state.params

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_params
        action = 2*self.policy.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.policy_train_state.apply_fn(params, graph, key)
        # originally action was 2*action + u_ref
        action = 2*action + self._env.u_ref(graph)
        return action, log_pi

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)
        idx = np.arange(rollout.dones.shape[0] * rollout.dones.shape[1])
        np.random.shuffle(idx)
        batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // self.batch_size))
        self.critic_train_state, self.policy_train_state, update_info = self.update_inner(
            self.critic_train_state, self.policy_train_state, rollout, batch_idx,
        )
        return update_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
        self, critic_train_state: TrainState, policy_train_state: TrainState, rollout: Rollout, batch_idx: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        value_fn = jax.vmap(jax.vmap(ft.partial(self.critic.get_value, critic_train_state.params)))
        values = value_fn(rollout.graph)
        next_values = value_fn(rollout.next_graph)

        # calculate GAE
        targets, gaes = compute_gae(
            values=values,
            rewards=rollout.rewards - self.cost_weight * rollout.costs,
            dones=rollout.dones,
            next_values=next_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        jax.debug.print(
            "BALANCE CHECK | reward_mean={r:.4f}  cost_mean={c:.4f}  weighted_cost={wc:.4f}  ratio={ratio:.1f}x",
            r=rollout.rewards.mean(),
            c=rollout.costs.mean(),
            wc=self.cost_weight * rollout.costs.mean(),
            ratio=self.cost_weight * rollout.costs.mean() / (jnp.abs(rollout.rewards.mean()) + 1e-8)
        )
        # After compute_gae, BEFORE normalization:
        '''jax.debug.print("rewards mean={m} std={s} min={mn} max={mx}",
            m=rollout.rewards.mean(), s=rollout.rewards.std(),
            mn=rollout.rewards.min(), mx=rollout.rewards.max())

        jax.debug.print("raw gaes mean={m} std={s} min={mn} max={mx}",
            m=gaes.mean(), s=gaes.std(),
            mn=gaes.min(), mx=gaes.max())

        jax.debug.print("targets mean={m} std={s}",
            m=targets.mean(), s=targets.std())'''

        # normalize advantages
        #gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        #self normalization also enabled in inforMARL

        # flatten rollout
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

        # critic runs epoch_criticnet times, policy runs epoch_ppo times
        critic_batch_idx = jnp.tile(batch_idx, (self.epoch_criticnet, 1, 1)).reshape(-1, batch_idx.shape[-1])
        policy_batch_idx = jnp.tile(batch_idx, (self.epoch_ppo, 1, 1)).reshape(-1, batch_idx.shape[-1])

        critic_train_state, critic_info = lax.scan(
            update_critic_fn, critic_train_state, critic_batch_idx
        )
        policy_train_state, policy_info = lax.scan(
            update_policy_fn, policy_train_state, policy_batch_idx
        )

        # get training info of the last epoch
        info = jtu.tree_map(lambda x: x[-1], critic_info | policy_info)

        return critic_train_state, policy_train_state, info

    '''def update_critic(
        self, critic_train_state: TrainState, rollout: Rollout, targets: Array
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
            'critic/loss': loss,
            'critic/grad_norm': grad_norm,
            'critic/has_nan': critic_has_nan,
            'critic/max_target': jnp.max(targets),
            'critic/min_target': jnp.min(targets)
        }'''

    # REPLACE the existing update_critic method with:
    def update_critic(
        self, critic_train_state: TrainState, rollout: Rollout, targets: Array
    ) -> Tuple[TrainState, dict]:

        def get_value_loss(params):
            # --- value loss (unchanged) ---
            values = jax.vmap(ft.partial(self.critic.get_value, params))(rollout.graph)
            loss_critic = optax.l2_loss(values, targets).mean()

            # --- radial loss on critic GNN attention ---
            # mirrors GCBF+: vmap get_value_with_attn over the batch
            value_fn_with_attn = jax.vmap(ft.partial(self.critic.get_value_with_attn, params))
            _, attn_weights = value_fn_with_attn(rollout.graph)
            # attn_weights: (batch, 1, num_edges)
            head1_attn = attn_weights[:, 0, :]  # (batch, num_edges)

            def radial_loss_single(h1_attn, graph):
                return compute_radial_loss(
                    h1_attn,
                    graph.edges,
                    graph.senders,
                    graph.node_type,
                )

            loss_radial = jnp.mean(jax.vmap(radial_loss_single)(head1_attn, rollout.graph))

            total_loss = loss_critic + self.coef_radial * loss_radial
            return total_loss, {'critic/loss': loss_critic, 'critic/loss_radial': loss_radial}

        (total_loss, aux), grad = jax.value_and_grad(get_value_loss, has_aux=True)(critic_train_state.params)
        critic_has_nan = has_any_nan(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad)

        return critic_train_state, {
            'critic/loss': aux['critic/loss'],
            'critic/loss_radial': aux['critic/loss_radial'],
            'critic/grad_norm': grad_norm,
            'critic/has_nan': critic_has_nan,
            'critic/max_target': jnp.max(targets),
            'critic/min_target': jnp.min(targets),
        }

    def update_policy(
        self, policy_train_state: TrainState, rollout: Rollout, gaes: Array
    ) -> Tuple[TrainState, dict]:
        # all agents share the same GAEs
        gaes = gaes[:, None]
        gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

        '''jax.debug.print("rollout.actions mean={m} std={s} min={mn} max={mx}",
            m=rollout.actions.mean(),
            s=rollout.actions.std(),
            mn=rollout.actions.min(),
            mx=rollout.actions.max())

        jax.debug.print("rollout.log_pis mean={m} std={s} min={mn} max={mx}",
            m=rollout.log_pis.mean(),
            s=rollout.log_pis.std(),
            mn=rollout.log_pis.min(),
            mx=rollout.log_pis.max())

        jax.debug.print("gaes mean={m} std={s} min={mn} max={mx}",
            m=gaes.mean(),
            s=gaes.std(),
            mn=gaes.min(),
            mx=gaes.max())'''

        def get_policy_loss(params):
            eval_action_vmap = jax.vmap(ft.partial(self.policy.eval_action, params))
            u_refs = jax.vmap(self._env.u_ref)(rollout.graph)
            raw_actions = (rollout.actions - u_refs)/2
            log_pis, policy_entropy = eval_action_vmap(rollout.graph, raw_actions, action_keys)
            ratio = jnp.exp(log_pis - rollout.log_pis)
            loss_policy1 = -ratio * gaes
            loss_policy2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes

            '''jax.debug.print("log_pis shape={s}", s=log_pis.shape)
            jax.debug.print("ratio shape={s}", s=ratio.shape)
            jax.debug.print("gaes shape={s}", s=gaes.shape)
            jax.debug.print("loss_policy1 shape={s}", s=loss_policy1.shape)
            jax.debug.print("loss_policy1 mean={m} std={s}", m=loss_policy1.mean(), s=loss_policy1.std())

            jax.debug.print("corr ratio-gaes={c}", c=(((ratio - ratio.mean()) * (gaes - gaes.mean())).mean() / (ratio.std() * gaes.std())))'''

            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            total_entropy = policy_entropy.mean()
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(ratio - 1.0))

            '''def get_attn_single(graph):
                from model.algo.module.policy import TanhNormal
                _, attn_weights = self.policy.dist.apply(
                    params, graph, self.n_agents,
                    method=TanhNormal.call_with_attn   # ← class, not instance
                )
                return attn_weights[0]'''

            #attn_batch = jax.vmap(get_attn_single)(rollout.graph)
            #loss_radial = jnp.mean(jax.vmap(compute_radial_loss)(attn_batch, rollout.graph.edges, rollout.graph.senders, rollout.graph.node_type))

            
            policy_loss = loss_policy - self.coef_ent * total_entropy #+ self.coef_radial * loss_radial

            '''jax.debug.print("log_pis (recomputed) mean={m} std={s}",
                m=log_pis.mean(), s=log_pis.std())

            jax.debug.print("ratio mean={m} std={s} min={mn} max={mx}",
                m=ratio.mean(), s=ratio.std(), mn=ratio.min(), mx=ratio.max())

            jax.debug.print("entropy mean={m}", m=policy_entropy.mean())'''

            return policy_loss, {
                'policy/clip_frac': clip_frac,
                'policy/entropy': policy_entropy.mean(),
                'policy/total_variation_dist': total_variation_dist,
                #'policy/loss_radial': loss_radial,
            }

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0]) #* rollout.actions.shape[1]).reshape(rollout.actions.shape[:2] + (2,))
        #action_keys = jnp.zeros((rollout.actions.shape[0], 2), dtype=jnp.uint32)

        (loss, info), grad = jax.value_and_grad(get_policy_loss, has_aux=True)(policy_train_state.params)
        policy_has_nan = has_any_nan(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)

        #jax.debug.print("policy grad_norm={g}", g=grad_norm)
        #jax.debug.print("policy loss={l}", l=loss)

        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        info = {
            'policy/loss': loss,
            'policy/grad_norm': grad_norm,
            'policy/has_nan': policy_has_nan,
            'policy/log_pi_min': rollout.log_pis.min()
        } | info

        return policy_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.critic_train_state.params, open(os.path.join(model_dir, 'critic.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))
        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.critic_train_state = \
            self.critic_train_state.replace(params=pickle.load(open(os.path.join(path, 'critic.pkl'), 'rb')))