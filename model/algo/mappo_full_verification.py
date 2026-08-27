import os
import sys
import pickle
import functools as ft
from typing import Optional, Tuple, Dict, List, Any

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
from model.utils.utils import merge01, jax_vmap, tree_merge, mask2index
from model.trainer.data import Rollout
from model.trainer.utils import has_any_nan, compute_norm_and_clip
from model.env.base import MultiAgentEnv
from model.algo.module.policy import PPOPolicy
from model.algo.module.value import ValueNet
from .base import MultiAgentController
from .utils import compute_gae

_GCBF_SP = os.environ.get(
    "GCBF_SITE_PACKAGES",
    "/home/kunalgarg/miniconda3/envs/gcbfplus/lib/python3.10/site-packages"
)
if _GCBF_SP not in sys.path:
    sys.path.insert(0, _GCBF_SP)

from z3 import Real, Solver, sat as z3_sat

_RELU_INACTIVE = 0
_RELU_ACTIVE = 1
_RELU_UNSTABLE = 2


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
        self.state_dim = state_dim
        self.action_dim = action_dim

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

    @property
    def config(self) -> dict:
        return {
            'cost_weight': self.cost_weight,
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
        action = self.policy.get_action(params, graph) + (0.35 * self._env.u_ref(graph))
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.policy_train_state.apply_fn(params, graph, key)
        action = action + (0.35 * self._env.u_ref(graph))
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
        self,
        critic_train_state: TrainState,
        policy_train_state: TrainState,
        rollout: Rollout,
        batch_idx: Array
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
            gae_lambda=self.gae_lambda
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

        critic_train_state, critic_info = lax.scan(
            update_critic_fn, critic_train_state, critic_batch_idx
        )
        policy_train_state, policy_info = lax.scan(
            update_policy_fn, policy_train_state, policy_batch_idx
        )

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
            'critic/loss': loss,
            'critic/grad_norm': grad_norm,
            'critic/has_nan': critic_has_nan,
            'critic/max_target': jnp.max(targets),
            'critic/min_target': jnp.min(targets)
        }

    def update_policy(
        self,
        policy_train_state: TrainState,
        rollout: Rollout,
        gaes: Array
    ) -> Tuple[TrainState, dict]:
        gaes = gaes[:, None]
        gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0])

        def get_policy_loss(params):
            eval_action_vmap = jax.vmap(ft.partial(self.policy.eval_action, params))
            u_refs = jax.vmap(self._env.u_ref)(rollout.graph)
            raw_actions = rollout.actions - (0.35 * u_refs)
            log_pis, policy_entropy = eval_action_vmap(rollout.graph, raw_actions, action_keys)
            ratio = jnp.exp(log_pis - rollout.log_pis)
            loss_policy1 = -ratio * gaes
            loss_policy2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes

            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            total_entropy = policy_entropy.mean()
            policy_loss = loss_policy - self.coef_ent * total_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(ratio - 1.0))

            return policy_loss, {
                'policy/clip_frac': clip_frac,
                'policy/entropy': policy_entropy.mean(),
                'policy/total_variation_dist': total_variation_dist
            }

        (loss, info), grad = jax.value_and_grad(get_policy_loss, has_aux=True)(policy_train_state.params)
        policy_has_nan = has_any_nan(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
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
        self.policy_train_state = self.policy_train_state.replace(
            params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb'))
        )
        self.critic_train_state = self.critic_train_state.replace(
            params=pickle.load(open(os.path.join(path, 'critic.pkl'), 'rb'))
        )

    # =========================================================================
    # POINT-WISE VERIFICATION HELPERS
    # =========================================================================

    def _mappo_v_terms_single(
        self,
        graph: GraphsTuple,
        critic_params=None,
    ):
        assert graph.is_single

        if critic_params is None:
            critic_params = self.critic_train_state.params

        agent_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_mask, self.n_agents)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        state_dim = agent_state.shape[-1]

        def v_fn(s_flat):
            new_states = graph.states.at[agent_node_id].set(
                s_flat.reshape(self.n_agents, state_dim)
            )
            new_graph = self._env.add_edge_feats(graph, new_states)
            return self.critic.get_per_agent_logits(critic_params, new_graph).reshape(-1)

        V = np.array(v_fn(agent_state.reshape(-1)))
        jac = np.array(jax.jacobian(v_fn)(agent_state.reshape(-1)))
        jac = jac.reshape(self.n_agents, self.n_agents, state_dim)
        dV_diag = jac[np.arange(self.n_agents), np.arange(self.n_agents), :]

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)

        return (
            V,
            dV_diag,
            np.array(dyn_f),
            np.array(dyn_g),
            np.array(agent_state),
        )

    def _mappo_v_terms_trajectory(
        self,
        T_graph: GraphsTuple,
        critic_params=None,
    ):
        if critic_params is None:
            critic_params = self.critic_train_state.params

        T = T_graph.states.shape[0]

        graph_0 = jtu.tree_map(lambda x: jnp.array(x[0]), T_graph)
        agent_mask = graph_0.node_type == 0
        agent_node_id = mask2index(agent_mask, self.n_agents)

        agent_state_0 = graph_0.type_states(type_idx=0, n_type=self.n_agents)
        state_dim = agent_state_0.shape[-1]

        per_agent_fn = jax.vmap(
            ft.partial(self.critic.get_per_agent_logits, critic_params)
        )
        V_all = np.array(per_agent_fn(T_graph)).reshape(T, self.n_agents)

        T_agent_states = np.array(
            jax.vmap(lambda g: g.type_states(type_idx=0, n_type=self.n_agents))(T_graph)
        )

        @jax.jit
        def jac_single(agent_states_t, graph_t):
            def v_fn(s_flat):
                new_states = graph_t.states.at[agent_node_id].set(
                    s_flat.reshape(self.n_agents, state_dim)
                )
                new_graph = self._env.add_edge_feats(graph_t, new_states)
                return self.critic.get_per_agent_logits(critic_params, new_graph).reshape(-1)

            return jax.jacobian(v_fn)(agent_states_t.reshape(-1))

        T_dV = np.zeros((T, self.n_agents, state_dim))
        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            jac = np.array(jac_single(jnp.array(T_agent_states[t]), graph_t))
            jac = jac.reshape(self.n_agents, self.n_agents, state_dim)
            T_dV[t] = jac[np.arange(self.n_agents), np.arange(self.n_agents), :]

        dyn_f_all = np.zeros((T, self.n_agents, state_dim))
        dyn_g_all = np.zeros((T, self.n_agents, state_dim, self.action_dim))
        for t in range(T):
            agent_state_t = jnp.array(T_agent_states[t])
            dyn_f_t, dyn_g_t = self._env.control_affine_dyn(agent_state_t)
            dyn_f_all[t] = np.array(dyn_f_t)
            dyn_g_all[t] = np.array(dyn_g_t)

        return V_all, T_dV, dyn_f_all, dyn_g_all, T_agent_states

    @staticmethod
    def _maximize_box_linear(coeffs, u_lb, u_ub):
        val = 0.0
        for j in range(len(coeffs)):
            c = float(coeffs[j])
            val += c * float(u_ub[j]) if c >= 0.0 else c * float(u_lb[j])
        return float(val)

    def verify_v_z3(
        self,
        graph: GraphsTuple,
        alpha: float = 1.0,
        v_threshold: Optional[float] = None,
    ) -> np.ndarray:
        assert graph.is_single

        V, dV_diag, dyn_f, dyn_g, _ = self._mappo_v_terms_single(graph)

        if v_threshold is None:
            v_threshold = float(V.mean())

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]

        alpha = float(alpha)
        sat_results = np.zeros((self.n_agents,), dtype=np.int32)

        for i in range(self.n_agents):
            V_shifted = float(V[i]) - float(v_threshold)
            Lf_Vi = float(dV_diag[i] @ dyn_f[i])
            Lg_Vi = dV_diag[i] @ dyn_g[i]

            solver = Solver()
            u_vars = [Real(f"u_{i}_{j}") for j in range(action_dim)]

            for j in range(action_dim):
                solver.add(u_vars[j] >= float(u_lb[j]))
                solver.add(u_vars[j] <= float(u_ub[j]))

            cbf_lhs = Lf_Vi + alpha * V_shifted
            for j in range(action_dim):
                cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]

            solver.add(cbf_lhs >= 0)
            sat_results[i] = 1 if solver.check() == z3_sat else 0

        return sat_results

    def verify_v_z3_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
    ) -> np.ndarray:
        T = T_graph.states.shape[0]

        V_all, T_dV, dyn_f_all, dyn_g_all, _ = self._mappo_v_terms_trajectory(T_graph)
        v_threshold = float(V_all.mean())
        print(
            f" [V-CBF/Z3] V_min={V_all.min():.4f} V_max={V_all.max():.4f} "
            f"V_mean={v_threshold:.4f}"
        )

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]

        alpha = float(alpha)
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        for t in range(T):
            for i in range(self.n_agents):
                V_shifted = float(V_all[t, i]) - v_threshold
                Lf_Vi = float(T_dV[t, i] @ dyn_f_all[t, i])
                Lg_Vi = T_dV[t, i] @ dyn_g_all[t, i]

                solver = Solver()
                u_vars = [Real(f"u{t}_{i}_{j}") for j in range(action_dim)]

                for j in range(action_dim):
                    solver.add(u_vars[j] >= float(u_lb[j]))
                    solver.add(u_vars[j] <= float(u_ub[j]))

                cbf_lhs = Lf_Vi + alpha * V_shifted
                for j in range(action_dim):
                    cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]

                solver.add(cbf_lhs >= 0)
                T_sat[t, i] = 1 if solver.check() == z3_sat else 0

        return T_sat

    def verify_v_dreal(
        self,
        graph: GraphsTuple,
        alpha: float = 1.0,
        delta: float = 1e-3,
        v_threshold: Optional[float] = None,
    ) -> np.ndarray:
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        assert graph.is_single

        V, dV_diag, dyn_f, dyn_g, _ = self._mappo_v_terms_single(graph)

        if v_threshold is None:
            v_threshold = float(V.mean())

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]

        alpha = float(alpha)
        sat_results = np.zeros((self.n_agents,), dtype=np.int32)

        for i in range(self.n_agents):
            V_shifted = float(V[i]) - float(v_threshold)
            Lf_Vi = float(dV_diag[i] @ dyn_f[i])
            Lg_Vi = dV_diag[i] @ dyn_g[i]

            u_vars = [Variable(f"u_{i}_{j}") for j in range(action_dim)]
            constraints = []

            for j in range(action_dim):
                constraints.append(u_vars[j] >= float(u_lb[j]))
                constraints.append(u_vars[j] <= float(u_ub[j]))

            cbf_lhs = Lf_Vi + alpha * V_shifted
            for j in range(action_dim):
                cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]

            constraints.append(cbf_lhs >= 0)
            result = CheckSatisfiability(DRealAnd(*constraints), delta)
            sat_results[i] = 0 if result is None else 1

        return sat_results

    def verify_v_dreal_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
        delta: float = 1e-3,
    ) -> np.ndarray:
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        T = T_graph.states.shape[0]

        V_all, T_dV, dyn_f_all, dyn_g_all, _ = self._mappo_v_terms_trajectory(T_graph)
        v_threshold = float(V_all.mean())
        print(
            f" [V-dReal] V_min={V_all.min():.4f} V_max={V_all.max():.4f} "
            f"V_mean={v_threshold:.4f}"
        )

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]

        alpha = float(alpha)
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        for t in range(T):
            for i in range(self.n_agents):
                V_shifted = float(V_all[t, i]) - v_threshold
                Lf_Vi = float(T_dV[t, i] @ dyn_f_all[t, i])
                Lg_Vi = T_dV[t, i] @ dyn_g_all[t, i]

                u_vars = [Variable(f"u{t}_{i}_{j}") for j in range(action_dim)]
                constraints = []

                for j in range(action_dim):
                    constraints.append(u_vars[j] >= float(u_lb[j]))
                    constraints.append(u_vars[j] <= float(u_ub[j]))

                cbf_lhs = Lf_Vi + alpha * V_shifted
                for j in range(action_dim):
                    cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]

                constraints.append(cbf_lhs >= 0)
                result = CheckSatisfiability(DRealAnd(*constraints), delta)
                T_sat[t, i] = 0 if result is None else 1

        return T_sat

    def verify_v_dreal_ball(
        self,
        graph: GraphsTuple,
        alpha: float = 1.0,
        delta: float = 1e-3,
        eps: float = 0.1,
        v_threshold: Optional[float] = None,
    ) -> np.ndarray:
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        assert graph.is_single

        V, dV_diag, dyn_f, dyn_g, agent_states = self._mappo_v_terms_single(graph)

        if v_threshold is None:
            v_threshold = float(V.mean())

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)

        alpha = float(alpha)
        state_dim = agent_states.shape[-1]
        sat_results = np.zeros((self.n_agents,), dtype=np.int32)

        for i in range(self.n_agents):
            grad_i = dV_diag[i]
            V_shifted_0 = float(V[i]) - float(v_threshold)
            Lf_Vi = float(grad_i @ dyn_f[i])
            Lg_Vi = grad_i @ dyn_g[i]
            max_u_term = self._maximize_box_linear(Lg_Vi, u_lb, u_ub)

            dx_vars = [Variable(f"dx_{i}_{k}") for k in range(state_dim)]
            constraints = []

            for k in range(state_dim):
                constraints.append(dx_vars[k] >= -float(eps))
                constraints.append(dx_vars[k] <= float(eps))

            bad_lhs = Lf_Vi + max_u_term + alpha * V_shifted_0
            for k in range(state_dim):
                bad_lhs = bad_lhs + alpha * float(grad_i[k]) * dx_vars[k]

            constraints.append(bad_lhs < 0)

            result = CheckSatisfiability(DRealAnd(*constraints), delta)
            sat_results[i] = 0 if result is not None else 1

        return sat_results

    def verify_v_dreal_ball_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
        delta: float = 1e-3,
        eps: float = 0.1,
    ) -> np.ndarray:
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        T = T_graph.states.shape[0]

        V_all, T_dV, dyn_f_all, dyn_g_all, T_agent_states = self._mappo_v_terms_trajectory(T_graph)
        v_threshold = float(V_all.mean())

        print(
            f" [V-dReal-Ball] V_min={V_all.min():.4f} V_max={V_all.max():.4f} "
            f"V_mean={v_threshold:.4f} eps={eps}"
        )

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)

        alpha = float(alpha)
        state_dim = T_agent_states.shape[-1]
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        for t in range(T):
            for i in range(self.n_agents):
                grad_i = T_dV[t, i]
                V_shifted_0 = float(V_all[t, i]) - v_threshold
                Lf_Vi = float(grad_i @ dyn_f_all[t, i])
                Lg_Vi = grad_i @ dyn_g_all[t, i]
                max_u_term = self._maximize_box_linear(Lg_Vi, u_lb, u_ub)

                dx_vars = [Variable(f"dx_{t}_{i}_{k}") for k in range(state_dim)]
                constraints = []

                for k in range(state_dim):
                    constraints.append(dx_vars[k] >= -float(eps))
                    constraints.append(dx_vars[k] <= float(eps))

                bad_lhs = Lf_Vi + max_u_term + alpha * V_shifted_0
                for k in range(state_dim):
                    bad_lhs = bad_lhs + alpha * float(grad_i[k]) * dx_vars[k]

                constraints.append(bad_lhs < 0)

                result = CheckSatisfiability(DRealAnd(*constraints), delta)
                T_sat[t, i] = 0 if result is not None else 1

        return T_sat

    # =========================================================================
    # FULL FORMAL VERIFICATION HELPERS
    # =========================================================================

    @staticmethod
    def _dense_sort_key(name: str) -> Tuple[int, str]:
        toks = name.split('_')
        try:
            return int(toks[-1]), name
        except Exception:
            return 10**9, name

    def _collect_dense_leaves(self, tree: Any, prefix: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
        leaves = []
        if not isinstance(tree, dict):
            return leaves

        if 'kernel' in tree and 'bias' in tree:
            leaves.append({
                'path': prefix,
                'name': "/".join(prefix),
                'kernel': np.array(tree['kernel'], dtype=np.float64),
                'bias': np.array(tree['bias'], dtype=np.float64),
            })
            return leaves

        for k, v in tree.items():
            if isinstance(v, dict):
                leaves.extend(self._collect_dense_leaves(v, prefix + (k,)))
        return leaves

    def _find_first_subtree(self, tree: dict, token_list: List[str]):
        if not isinstance(tree, dict):
            return None, None

        for k, v in tree.items():
            if any(tok in k.lower() for tok in token_list) and isinstance(v, dict):
                return k, v

        for k, v in tree.items():
            if isinstance(v, dict):
                out_k, out_v = self._find_first_subtree(v, token_list)
                if out_v is not None:
                    return out_k, out_v

        return None, None

    def _extract_value_weights(self, critic_params: dict) -> dict:
        """
        Generic extractor for MAPPO ValueNet weights.

        Assumptions preserved from your current code:
        - critic.get_per_agent_logits(graph) exists and is the surrogate scalar per agent.
        - critic has a GNN backbone with msg / update submodules similar to your other graph models.
        - critic head has two hidden Dense layers + one scalar output Dense.
        """
        p = critic_params['params'] if 'params' in critic_params else critic_params

        gnn_key = None
        for k in p.keys():
            if 'gnn' in k.lower():
                gnn_key = k
                break
        if gnn_key is None:
            raise KeyError("Could not find GNN block in critic params.")

        gnn_block = p[gnn_key]
        if isinstance(gnn_block, dict) and len(gnn_block) == 1:
            only_k = next(iter(gnn_block.keys()))
            if isinstance(gnn_block[only_k], dict) and 'kernel' not in gnn_block[only_k]:
                gnn_layer = gnn_block[only_k]
            else:
                gnn_layer = gnn_block
        else:
            gnn_layer = gnn_block

        _, msg_block = self._find_first_subtree(gnn_layer, ['msg'])
        _, upd_block = self._find_first_subtree(gnn_layer, ['update'])
        if msg_block is None or upd_block is None:
            raise KeyError("Could not find msg/update blocks inside critic GNN.")

        msg_dense = self._collect_dense_leaves(msg_block)
        upd_dense = self._collect_dense_leaves(upd_block)
        msg_dense = sorted(msg_dense, key=lambda x: self._dense_sort_key(x['path'][-1]))
        upd_dense = sorted(upd_dense, key=lambda x: self._dense_sort_key(x['path'][-1]))

        if len(msg_dense) < 2 or len(upd_dense) < 2:
            raise ValueError("Expected at least two hidden Dense layers in msg/update blocks.")

        gnn_direct_dense = []
        for k, v in gnn_layer.items():
            if isinstance(v, dict) and 'kernel' in v and 'bias' in v:
                gnn_direct_dense.append({
                    'path': (k,),
                    'name': k,
                    'kernel': np.array(v['kernel'], dtype=np.float64),
                    'bias': np.array(v['bias'], dtype=np.float64),
                })
        gnn_direct_dense = sorted(gnn_direct_dense, key=lambda x: self._dense_sort_key(x['name']))

        msg_hidden_out = msg_dense[-1]['kernel'].shape[1]
        upd_hidden_out = upd_dense[-1]['kernel'].shape[1]

        msg_out = None
        upd_out = None
        used = set()

        for idx, leaf in enumerate(gnn_direct_dense):
            kshape = leaf['kernel'].shape
            if msg_out is None and kshape[0] == msg_hidden_out and kshape[1] != 1:
                msg_out = leaf
                used.add(idx)
                break
        for idx, leaf in enumerate(gnn_direct_dense):
            kshape = leaf['kernel'].shape
            if idx in used:
                continue
            if upd_out is None and kshape[0] == upd_hidden_out and kshape[1] != 1:
                upd_out = leaf
                used.add(idx)
                break

        if msg_out is None:
            raise KeyError("Could not identify msg output Dense in critic GNN.")
        if upd_out is None:
            raise KeyError("Could not identify update output Dense in critic GNN.")

        head_candidates = {}
        for k, v in p.items():
            if k == gnn_key:
                continue
            if isinstance(v, dict):
                head_candidates[k] = v

        head_block_name = None
        head_block = None
        for k, v in head_candidates.items():
            if any(tok in k.lower() for tok in ['value', 'critic', 'head']):
                head_block_name = k
                head_block = v
                break
        if head_block is None:
            for k, v in head_candidates.items():
                denses = self._collect_dense_leaves(v, prefix=(k,))
                if len(denses) >= 2:
                    head_block_name = k
                    head_block = v
                    break

        if head_block is None:
            raise KeyError("Could not identify critic/value head block.")

        head_dense = self._collect_dense_leaves(head_block, prefix=(head_block_name,))
        head_dense = sorted(head_dense, key=lambda x: self._dense_sort_key(x['path'][-1]))

        top_level_dense = []
        for k, v in p.items():
            if isinstance(v, dict) and 'kernel' in v and 'bias' in v:
                top_level_dense.append({
                    'path': (k,),
                    'name': k,
                    'kernel': np.array(v['kernel'], dtype=np.float64),
                    'bias': np.array(v['bias'], dtype=np.float64),
                })
        top_level_dense = sorted(top_level_dense, key=lambda x: self._dense_sort_key(x['name']))

        head_hidden = [leaf for leaf in head_dense if leaf['kernel'].shape[1] != 1]
        if len(head_hidden) < 2:
            raise ValueError("Could not identify two hidden layers in critic/value head.")

        head_out = None
        for leaf in top_level_dense:
            if leaf['kernel'].shape[1] == 1:
                head_out = leaf
                break
        if head_out is None:
            for leaf in head_dense:
                if leaf['kernel'].shape[1] == 1:
                    head_out = leaf
                    break
        if head_out is None:
            raise KeyError("Could not identify scalar output Dense for critic/value head.")

        return {
            'msg_W0': msg_dense[0]['kernel'],
            'msg_b0': msg_dense[0]['bias'],
            'msg_W1': msg_dense[1]['kernel'],
            'msg_b1': msg_dense[1]['bias'],
            'msg_Wo': msg_out['kernel'],
            'msg_bo': msg_out['bias'],
            'upd_W0': upd_dense[0]['kernel'],
            'upd_b0': upd_dense[0]['bias'],
            'upd_W1': upd_dense[1]['kernel'],
            'upd_b1': upd_dense[1]['bias'],
            'upd_Wo': upd_out['kernel'],
            'upd_bo': upd_out['bias'],
            'val_W0': head_hidden[0]['kernel'],
            'val_b0': head_hidden[0]['bias'],
            'val_W1': head_hidden[1]['kernel'],
            'val_b1': head_hidden[1]['bias'],
            'val_Wo': head_out['kernel'],
            'val_bo': head_out['bias'],
        }

    @staticmethod
    def _ia_affine(W, b, lo, hi):
        Wp = np.maximum(W, 0.0)
        Wn = np.minimum(W, 0.0)
        y_lo = lo @ Wp + hi @ Wn + b
        y_hi = hi @ Wp + lo @ Wn + b
        return y_lo, y_hi

    @staticmethod
    def _ia_relu(pre_lo, pre_hi):
        post_lo = np.maximum(pre_lo, 0.0)
        post_hi = np.maximum(pre_hi, 0.0)
        status = np.where(
            pre_hi <= 0.0,
            _RELU_INACTIVE,
            np.where(pre_lo >= 0.0, _RELU_ACTIVE, _RELU_UNSTABLE)
        )
        return post_lo, post_hi, status

    @staticmethod
    def _crown_relu(pre_lo: np.ndarray, pre_hi: np.ndarray):
        active = pre_lo >= 0.0
        inactive = pre_hi <= 0.0
        unstable = ~active & ~inactive
        denom = np.where(unstable, pre_hi - pre_lo, 1.0)

        lam_u = np.where(active, 1.0, np.where(inactive, 0.0, pre_hi / denom))
        mu_u = np.where(active, 0.0, np.where(inactive, 0.0, -pre_lo * pre_hi / denom))
        lam_l = np.where(active, 1.0, 0.0)
        mu_l = np.zeros_like(pre_lo)

        return lam_l, mu_l, lam_u, mu_u

    @staticmethod
    def _crown_fwd_ia(W, b, lo, hi):
        Wp = np.maximum(W, 0.0)
        Wn = np.minimum(W, 0.0)
        return lo @ Wp + hi @ Wn + b, hi @ Wp + lo @ Wn + b

    def _crown_propagate_box(
        self,
        weights: dict,
        recv_feat: np.ndarray,
        sender_feat: np.ndarray,
        edge_lo: np.ndarray,
        edge_hi: np.ndarray,
    ) -> dict:
        """
        Single-neighbor CROWN-style propagation for the critic surrogate logit.
        This mirrors the formal GCBF+ flow, but the scalar output is V_i logit
        rather than a tanh-CBF head.
        """
        const_tail = np.concatenate([sender_feat, recv_feat])
        msg_in_lo = np.concatenate([edge_lo, const_tail])
        msg_in_hi = np.concatenate([edge_hi, const_tail])

        pm0_lo, pm0_hi = self._crown_fwd_ia(weights['msg_W0'], weights['msg_b0'], msg_in_lo, msg_in_hi)
        hm0_lo, hm0_hi = np.maximum(pm0_lo, 0), np.maximum(pm0_hi, 0)

        pm1_lo, pm1_hi = self._crown_fwd_ia(weights['msg_W1'], weights['msg_b1'], hm0_lo, hm0_hi)
        hm1_lo, hm1_hi = np.maximum(pm1_lo, 0), np.maximum(pm1_hi, 0)

        mo_lo, mo_hi = self._crown_fwd_ia(weights['msg_Wo'], weights['msg_bo'], hm1_lo, hm1_hi)

        upd_in_lo = np.concatenate([recv_feat, mo_lo])
        upd_in_hi = np.concatenate([recv_feat, mo_hi])

        pu0_lo, pu0_hi = self._crown_fwd_ia(weights['upd_W0'], weights['upd_b0'], upd_in_lo, upd_in_hi)
        hu0_lo, hu0_hi = np.maximum(pu0_lo, 0), np.maximum(pu0_hi, 0)

        pu1_lo, pu1_hi = self._crown_fwd_ia(weights['upd_W1'], weights['upd_b1'], hu0_lo, hu0_hi)
        hu1_lo, hu1_hi = np.maximum(pu1_lo, 0), np.maximum(pu1_hi, 0)

        uo_lo, uo_hi = self._crown_fwd_ia(weights['upd_Wo'], weights['upd_bo'], hu1_lo, hu1_hi)

        pv0_lo, pv0_hi = self._crown_fwd_ia(weights['val_W0'], weights['val_b0'], uo_lo, uo_hi)
        hv0_lo, hv0_hi = np.maximum(pv0_lo, 0), np.maximum(pv0_hi, 0)

        pv1_lo, pv1_hi = self._crown_fwd_ia(weights['val_W1'], weights['val_b1'], hv0_lo, hv0_hi)
        hv1_lo, hv1_hi = np.maximum(pv1_lo, 0), np.maximum(pv1_hi, 0)

        pt_lo, pt_hi = self._crown_fwd_ia(weights['val_Wo'], weights['val_bo'], hv1_lo, hv1_hi)

        def _n_unstable(lo, hi):
            return int(np.sum((lo < 0) & (hi > 0)))

        n_unstable = (
            _n_unstable(pm0_lo, pm0_hi) + _n_unstable(pm1_lo, pm1_hi) +
            _n_unstable(pu0_lo, pu0_hi) + _n_unstable(pu1_lo, pu1_hi) +
            _n_unstable(pv0_lo, pv0_hi) + _n_unstable(pv1_lo, pv1_hi)
        )

        def _bkwd_linear(W, b_vec, cW_lo, cW_hi, cb_lo, cb_hi):
            Wt = W.T
            new_cW_lo = cW_lo @ Wt
            new_cW_hi = cW_hi @ Wt
            new_b_lo = cb_lo + float((cW_lo @ b_vec).squeeze())
            new_b_hi = cb_hi + float((cW_hi @ b_vec).squeeze())
            return new_cW_lo, new_cW_hi, new_b_lo, new_b_hi

        def _bkwd_relu(pre_lo_r, pre_hi_r, cW_lo, cW_hi, cb_lo, cb_hi):
            lam_l, mu_l, lam_u, mu_u = MAPPO._crown_relu(pre_lo_r, pre_hi_r)

            s_lo = np.where(cW_lo >= 0, lam_l[None, :], lam_u[None, :])
            m_lo = np.where(cW_lo >= 0, mu_l[None, :], mu_u[None, :])
            new_cW_lo = cW_lo * s_lo
            new_b_lo = cb_lo + float((cW_lo * m_lo).sum())

            s_hi = np.where(cW_hi >= 0, lam_u[None, :], lam_l[None, :])
            m_hi = np.where(cW_hi >= 0, mu_u[None, :], mu_l[None, :])
            new_cW_hi = cW_hi * s_hi
            new_b_hi = cb_hi + float((cW_hi * m_hi).sum())

            return new_cW_lo, new_cW_hi, new_b_lo, new_b_hi

        cW_lo = weights['val_Wo'].T.copy()
        cW_hi = weights['val_Wo'].T.copy()
        cb_lo = float(np.array(weights['val_bo']).reshape(-1)[0])
        cb_hi = float(np.array(weights['val_bo']).reshape(-1)[0])

        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_relu(pv1_lo, pv1_hi, cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['val_W1'], weights['val_b1'], cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_relu(pv0_lo, pv0_hi, cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['val_W0'], weights['val_b0'], cW_lo, cW_hi, cb_lo, cb_hi)

        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['upd_Wo'], weights['upd_bo'], cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_relu(pu1_lo, pu1_hi, cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['upd_W1'], weights['upd_b1'], cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_relu(pu0_lo, pu0_hi, cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['upd_W0'], weights['upd_b0'], cW_lo, cW_hi, cb_lo, cb_hi)

        recv_dim = recv_feat.shape[0]
        cb_lo += float((cW_lo[0, :recv_dim] * recv_feat).sum())
        cb_hi += float((cW_hi[0, :recv_dim] * recv_feat).sum())
        cW_lo = cW_lo[:, recv_dim:]
        cW_hi = cW_hi[:, recv_dim:]

        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['msg_Wo'], weights['msg_bo'], cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_relu(pm1_lo, pm1_hi, cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['msg_W1'], weights['msg_b1'], cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_relu(pm0_lo, pm0_hi, cW_lo, cW_hi, cb_lo, cb_hi)
        cW_lo, cW_hi, cb_lo, cb_hi = _bkwd_linear(weights['msg_W0'], weights['msg_b0'], cW_lo, cW_hi, cb_lo, cb_hi)

        tail_dim = const_tail.shape[0]
        edge_dim = edge_lo.shape[0]
        cb_lo += float((cW_lo[0, edge_dim:edge_dim + tail_dim] * const_tail).sum())
        cb_hi += float((cW_hi[0, edge_dim:edge_dim + tail_dim] * const_tail).sum())

        A_lo = cW_lo[0, :edge_dim].copy()
        A_hi = cW_hi[0, :edge_dim].copy()

        x_worst = np.where(A_lo >= 0, edge_lo, edge_hi)
        x_best = np.where(A_hi >= 0, edge_hi, edge_lo)
        value_lo = float(A_lo @ x_worst) + cb_lo
        value_hi = float(A_hi @ x_best) + cb_hi

        value_lo = max(value_lo, float(np.array(pt_lo).reshape(-1)[0]))
        value_hi = min(value_hi, float(np.array(pt_hi).reshape(-1)[0]))

        return {
            'value_lo': value_lo,
            'value_hi': value_hi,
            'A_lo': A_lo,
            'A_hi': A_hi,
            'b_lo': cb_lo,
            'b_hi': cb_hi,
            'n_unstable': n_unstable,
        }

    def _check_v_on_box(
        self,
        crown: dict,
        edge_lo: np.ndarray,
        edge_hi: np.ndarray,
        v_threshold: float = 0.0,
        alpha: float = 1.0,
        mass: float = 0.1,
    ) -> bool:
        """
        Conservative box certificate for surrogate condition
            Lf V + sup_u Lg V @ u + alpha*(V - v_threshold) >= 0

        The box variable is the relative state edge = [dx, dy, dvx, dvy].
        This follows the same structural idea as the GCBF+ full verifier.
        """
        V_lo = crown['value_lo'] - float(v_threshold)
        A_lo = crown['A_lo']
        A_hi = crown['A_hi']

        if edge_lo.shape[0] < 4:
            raise ValueError("Full verifier assumes 4-dim relative state edge [dx, dy, dvx, dvy].")

        def ip_lo(g_lo, g_hi, v_lo, v_hi):
            return min(g_lo * v_lo, g_lo * v_hi, g_hi * v_lo, g_hi * v_hi)

        dVdx_lo, dVdx_hi = float(A_lo[0]), float(A_hi[0])
        dVdy_lo, dVdy_hi = float(A_lo[1]), float(A_hi[1])
        dVdvx_lo, dVdvx_hi = float(A_lo[2]), float(A_hi[2])
        dVdvy_lo, dVdvy_hi = float(A_lo[3]), float(A_hi[3])

        Lf_lo = (
            ip_lo(dVdx_lo, dVdx_hi, float(edge_lo[2]), float(edge_hi[2])) +
            ip_lo(dVdy_lo, dVdy_hi, float(edge_lo[3]), float(edge_hi[3]))
        )

        def abs_lower(g_lo_v, g_hi_v):
            if g_lo_v <= 0.0 <= g_hi_v:
                return 0.0
            return min(abs(g_lo_v), abs(g_hi_v))

        sup_Lg_lo = (
            abs_lower(dVdvx_lo, dVdvx_hi) +
            abs_lower(dVdvy_lo, dVdvy_hi)
        ) / float(mass)

        alpha_eff = float(alpha) * 0.1
        cert_lo = Lf_lo + sup_Lg_lo + alpha_eff * V_lo
        return cert_lo >= 0.0

    def _verify_agent_bnb(
        self,
        weights: dict,
        recv_feat: np.ndarray,
        sender_feat: np.ndarray,
        X_full_lo: np.ndarray,
        X_full_hi: np.ndarray,
        v_threshold: float = 0.0,
        alpha: float = 1.0,
        mass: float = 0.1,
        max_depth: int = 30,
    ) -> bool:
        worklist = [(X_full_lo.copy(), X_full_hi.copy(), 0)]
        n_boxes = 0
        n_verified = 0

        while worklist:
            lo, hi, depth = worklist.pop()
            n_boxes += 1

            crown = self._crown_propagate_box(weights, recv_feat, sender_feat, lo, hi)
            verified = self._check_v_on_box(
                crown,
                lo,
                hi,
                v_threshold=v_threshold,
                alpha=alpha,
                mass=mass,
            )

            if verified:
                n_verified += 1
                continue

            if depth >= max_depth:
                print(
                    f' [BnB/V-CROWN] Max depth {max_depth} reached — UNVERIFIED: '
                    f'n_unstable={crown["n_unstable"]} '
                    f'V=[{crown["value_lo"]:.3f},{crown["value_hi"]:.3f}]'
                )
                return False

            widths = hi - lo
            split_dim = int(np.argmax(widths))
            mid = (lo[split_dim] + hi[split_dim]) / 2.0

            lo_l = lo.copy()
            hi_l = hi.copy()
            hi_l[split_dim] = mid

            lo_r = lo.copy()
            hi_r = hi.copy()
            lo_r[split_dim] = mid

            worklist.append((lo_l, hi_l, depth + 1))
            worklist.append((lo_r, hi_r, depth + 1))

        print(f' [BnB/V-CROWN] VERIFIED: {n_verified}/{n_boxes} boxes')
        return True

    def verify_v_full(
        self,
        graph,
        critic_params=None,
        area_size: float = 4.0,
        v_max: float = 0.5,
        v_threshold: float = 0.0,
        alpha: float = 1.0,
        mass: float = 0.1,
        max_depth: int = 30
    ) -> np.ndarray:
        """
        Full formal verification over the relative-state box

            X_full = {
                [dx, dy, dvx, dvy] :
                dx,dy in [-area_size, area_size],
                dvx,dvy in [-v_max, v_max]
            }

        For each receiver agent and each incoming neighbor topology, verify the
        surrogate condition over X_full with branch-and-bound + CROWN.

        Unlike trajectory pointwise checks, there is no natural rollout-derived
        mean threshold here, so v_threshold is an explicit argument.
        """
        if critic_params is None:
            critic_params = self.critic_train_state.params

        weights = self._extract_value_weights(critic_params)

        graph_np = jtu.tree_map(lambda x: np.array(x), graph)
        nodes_raw = np.array(graph_np.nodes)
        if nodes_raw.ndim == 3:
            graph_t0 = jtu.tree_map(lambda x: x[0], graph_np)
        else:
            graph_t0 = graph_np

        nodes_array = np.array(graph_t0.nodes, dtype=np.float64)
        senders_arr = np.array(graph_t0.senders, dtype=np.int32).flatten()
        receivers_arr = np.array(graph_t0.receivers, dtype=np.int32).flatten()

        X_full_lo = np.array([-area_size, -area_size, -v_max, -v_max], dtype=np.float64)
        X_full_hi = np.array([area_size, area_size, v_max, v_max], dtype=np.float64)

        results = np.zeros(self.n_agents, dtype=np.int32)
        topo_cache = {}

        for agent_i in range(self.n_agents):
            nbr_edge_idxs = np.where(receivers_arr == agent_i)[0]

            if len(nbr_edge_idxs) == 0:
                print(f"[BnB/V] agent {agent_i}: isolated -> VERIFIED (trivially)")
                results[agent_i] = 1
                continue

            recv_feat = nodes_array[agent_i]
            all_verified = True

            for k in nbr_edge_idxs:
                sender_i = int(senders_arr[k])
                sender_feat = nodes_array[sender_i]

                cache_key = (
                    agent_i,
                    sender_i,
                    tuple(recv_feat.tolist()),
                    tuple(sender_feat.tolist()),
                    float(v_threshold),
                    float(alpha),
                    float(mass),
                    float(area_size),
                    float(v_max),
                )

                if cache_key in topo_cache:
                    ok = topo_cache[cache_key]
                else:
                    print(
                        f"[BnB/V] agent {agent_i} <- sender {sender_i}: "
                        f"verifying over X_full..."
                    )
                    ok = self._verify_agent_bnb(
                        weights=weights,
                        recv_feat=recv_feat,
                        sender_feat=sender_feat,
                        X_full_lo=X_full_lo,
                        X_full_hi=X_full_hi,
                        v_threshold=v_threshold,
                        alpha=alpha,
                        mass=mass,
                        max_depth=max_depth,
                    )
                    print(f"[BnB/V] -> {'VERIFIED' if ok else 'UNVERIFIED'}")
                    topo_cache[cache_key] = ok

                if not ok:
                    all_verified = False
                    break

            results[agent_i] = 1 if all_verified else 0

        return results