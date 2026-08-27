import jax.numpy as jnp
import jax.random as jr
import optax
import pathlib
import os
import sys
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
from model.utils.utils import merge01, jax_vmap, tree_merge, mask2index
from model.trainer.data import Rollout
from model.trainer.buffer import ReplayBuffer
from model.trainer.utils import has_any_nan, get_ckpt_manager, load_ckpt, compute_norm_and_clip, jax2np, np2jax
from model.env.base import MultiAgentEnv
from model.algo.module.cbf import CBF
from model.algo.module.policy import DeterministicPolicy, PPOPolicy
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
        action = self.policy.get_action(params, graph) + (0.35*self._env.u_ref(graph))
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.policy_train_state.apply_fn(params, graph, key)
        # originally action was 2*action + u_ref
        action = action + (0.35*self._env.u_ref(graph))
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

    def update_critic(
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
            raw_actions = rollout.actions - (0.35*u_refs)
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
            policy_loss = loss_policy - self.coef_ent * total_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(ratio - 1.0))

            '''jax.debug.print("log_pis (recomputed) mean={m} std={s}",
                m=log_pis.mean(), s=log_pis.std())

            jax.debug.print("ratio mean={m} std={s} min={mn} max={mx}",
                m=ratio.mean(), s=ratio.std(), mn=ratio.min(), mx=ratio.max())

            jax.debug.print("entropy mean={m}", m=policy_entropy.mean())'''

            return policy_loss, {
                'policy/clip_frac': clip_frac,
                'policy/entropy': policy_entropy.mean(),
                'policy/total_variation_dist': total_variation_dist
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

    
    # =========================================================================
    # VERIFICATION — MAPPO value-surrogate pointwise verification
    #
    # This keeps the current MAPPO principle:
    #   h_i^V(x) := V_i(x) - v_threshold
    #
    # and checks the surrogate CBF condition
    #   L_f V_i + L_g V_i @ u_i + alpha * (V_i - v_threshold) >= 0
    #
    # using ONLY the diagonal self-gradient dV_i / dx_i, exactly like the
    # GCBF+ pointwise verifiers use the diagonal CBF gradient.
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

        V = np.array(v_fn(agent_state.reshape(-1)))  # (n_agents,)
        jac = np.array(jax.jacobian(v_fn)(agent_state.reshape(-1)))
        jac = jac.reshape(self.n_agents, self.n_agents, state_dim)
        dV_diag = jac[np.arange(self.n_agents), np.arange(self.n_agents), :]  # (n_agents, state_dim)

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)

        return (
            V,
            dV_diag,
            np.array(dyn_f),   # (n_agents, state_dim)
            np.array(dyn_g),   # (n_agents, state_dim, action_dim)
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
        V_all = np.array(per_agent_fn(T_graph)).reshape(T, self.n_agents)  # (T, n_agents)

        T_agent_states = np.array(
            jax.vmap(lambda g: g.type_states(type_idx=0, n_type=self.n_agents))(T_graph)
        )  # (T, n_agents, state_dim)

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


    def _maximize_box_linear(self, coeffs, u_lb, u_ub):
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
        """
        Single-timestep Z3 feasibility check for the MAPPO surrogate value-CBF:

            Lf_V_i + Lg_V_i @ u_i + alpha * (V_i - v_threshold) >= 0

        Returns
        -------
        sat_results : np.ndarray, shape (n_agents,)
            1 = SAT (safe control exists), 0 = UNSAT.
        """
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
            Lg_Vi = dV_diag[i] @ dyn_g[i]  # (action_dim,)

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
        """
        Trajectory-level Z3 feasibility check using the SAME surrogate threshold
        for the whole trajectory:
            v_threshold = mean_t,i V_i(x_t)
        """
        T = T_graph.states.shape[0]

        V_all, T_dV, dyn_f_all, dyn_g_all, _ = self._mappo_v_terms_trajectory(T_graph)
        v_threshold = float(V_all.mean())

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
        """
        Single-timestep dReal feasibility check for the MAPPO surrogate value-CBF.

        dReal convention here:
            result is None   -> UNSAT -> return 0
            result not None  -> SAT   -> return 1
        """
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
        """
        Trajectory-level dReal feasibility check using:
            v_threshold = mean_t,i V_i(x_t)
        """
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
        """
        Robust L-inf ball check around each agent state using the MAPPO value surrogate.

        We ask whether there EXISTS a perturbation dx with ||dx||_inf <= eps such that
        the surrogate condition fails even under the analytically best admissible control:

            Lf_V + max_{u in box} (Lg_V @ u) + alpha * (V_shifted + dV@dx) < 0

        Interpretation:
            dReal SAT   -> counterexample exists -> NOT robustly safe -> 0
            dReal UNSAT -> no counterexample     -> robustly safe     -> 1
        """
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
        """
        Robust trajectory-level ball verification using the SAME trajectory-wide shift:
            v_threshold = mean_t,i V_i(x_t)

        For each (t, i), dReal searches for a perturbation dx with ||dx||_inf <= eps
        that makes the surrogate condition fail even under the best admissible control.

        Return:
            1 -> robustly safe
            0 -> counterexample found
        """
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


    def _resolve_verify_handles(self):
        n_agents = getattr(self, "nagents", None)
        if n_agents is None:
            n_agents = getattr(self, "n_agents", None)
        if n_agents is None:
            raise AttributeError("MAPPO object has neither 'nagents' nor 'n_agents'")

        critic_state = getattr(self, "critictrainstate", None)
        if critic_state is None:
            critic_state = getattr(self, "critic_train_state", None)
        if critic_state is None:
            raise AttributeError(
                "MAPPO object has neither 'critictrainstate' nor 'critic_train_state'"
            )

        env_obj = getattr(self, "_env", None)
        if env_obj is None:
            env_obj = getattr(self, "env", None)
        if env_obj is None:
            raise AttributeError("MAPPO object has neither '_env' nor 'env'")

        critic_logits_fn = getattr(self.critic, "get_per_agent_logits", None)
        if critic_logits_fn is None:
            critic_logits_fn = getattr(self.critic, "getperagentlogits", None)
        if critic_logits_fn is None:
            raise AttributeError(
                "critic has neither 'get_per_agent_logits' nor 'getperagentlogits'"
            )

        add_edge_feats_fn = getattr(env_obj, "add_edge_feats", None)
        if add_edge_feats_fn is None:
            add_edge_feats_fn = getattr(env_obj, "addedgefeats", None)
        if add_edge_feats_fn is None:
            raise AttributeError(
                "env has neither 'add_edge_feats' nor 'addedgefeats'"
            )

        control_affine_dyn_fn = getattr(env_obj, "control_affine_dyn", None)
        if control_affine_dyn_fn is None:
            control_affine_dyn_fn = getattr(env_obj, "controlaffinedyn", None)
        if control_affine_dyn_fn is None:
            raise AttributeError(
                "env has neither 'control_affine_dyn' nor 'controlaffinedyn'"
            )

        action_lim_fn = getattr(env_obj, "action_lim", None)
        if action_lim_fn is None:
            action_lim_fn = getattr(env_obj, "actionlim", None)
        if action_lim_fn is None:
            raise AttributeError("env has neither 'action_lim' nor 'actionlim'")

        return (
            n_agents,
            critic_state,
            env_obj,
            critic_logits_fn,
            add_edge_feats_fn,
            control_affine_dyn_fn,
            action_lim_fn,
        )


    def _get_agent_ids_and_states(self, graph, n_agents):
        node_type = getattr(graph, "node_type", None)
        if node_type is None:
            node_type = getattr(graph, "nodetype", None)
        if node_type is None:
            raise AttributeError("graph has neither 'node_type' nor 'nodetype'")

        agent_node_mask = node_type == 0
        agent_node_id = mask2index(agent_node_mask, n_agents)

        if hasattr(graph, "type_states"):
            try:
                agent_state = graph.type_states(type_idx=0, n_type=n_agents)
            except TypeError:
                agent_state = graph.type_states(typeidx=0, ntype=n_agents)
        elif hasattr(graph, "typestates"):
            agent_state = graph.typestates(typeidx=0, ntype=n_agents)
        else:
            raise AttributeError("graph has neither 'type_states' nor 'typestates'")

        return agent_node_id, agent_state


    def _maximize_box_linear(self, coeffs, u_lb, u_ub):
        coeffs = np.asarray(coeffs).reshape(-1)
        out = 0.0
        for j, c in enumerate(coeffs):
            out += float(c) * (float(u_ub[j]) if float(c) >= 0.0 else float(u_lb[j]))
        return float(out)


    def verify_v_seev(
        self,
        graph: GraphsTuple,
        alpha: float = 1.0,
        kgrid: int = 3,
        v_threshold: Optional[float] = None,
    ):
        is_single = getattr(graph, "is_single", None)
        if is_single is None:
            is_single = getattr(graph, "issingle", None)
        assert bool(is_single)

        alpha = float(alpha)

        (
            n_agents,
            critic_state,
            env_obj,
            critic_logits_fn,
            add_edge_feats_fn,
            control_affine_dyn_fn,
            action_lim_fn,
        ) = self._resolve_verify_handles()

        critic_params = critic_state.params
        agent_node_id, agent_state = self._get_agent_ids_and_states(graph, n_agents)
        state_dim = agent_state.shape[-1]

        def v_fn(s_flat):
            new_states = graph.states.at[agent_node_id].set(
                s_flat.reshape(n_agents, state_dim)
            )
            new_graph = add_edge_feats_fn(graph, new_states)
            return critic_logits_fn(critic_params, new_graph).reshape(-1)

        V = np.array(v_fn(agent_state.reshape(-1)))
        if v_threshold is None:
            v_threshold = float(V.mean())

        jac = np.array(jax.jacobian(v_fn)(agent_state.reshape(-1)))
        jac = jac.reshape(n_agents, n_agents, state_dim)
        dV = jac[np.arange(n_agents), np.arange(n_agents), :]

        dyn_f, dyn_g = control_affine_dyn_fn(agent_state)
        dyn_f = np.array(dyn_f)
        dyn_g = np.array(dyn_g)

        u_lb_jax, u_ub_jax = action_lim_fn()
        u_lb = np.array(u_lb_jax)
        u_ub = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]

        dt = float(env_obj.dt)
        nominal_u = np.array(self.act(graph))

        if kgrid == 1:
            per_agent_grids = [nominal_u[i][None, :] for i in range(n_agents)]
        else:
            grid_axes = [
                np.linspace(float(u_lb[d]), float(u_ub[d]), kgrid)
                for d in range(action_dim)
            ]
            shared_grid = np.stack(
                np.meshgrid(*grid_axes, indexing="ij"), axis=-1
            ).reshape(-1, action_dim)
            per_agent_grids = [shared_grid for _ in range(n_agents)]

        sat_results = np.zeros(n_agents, dtype=np.int32)
        tier1_pass_count = 0
        tier2_fail_count = 0

        for i in range(n_agents):
            grad_i = dV[i]
            V_i = float(V[i] - v_threshold)
            f_i = dyn_f[i]
            g_i = dyn_g[i]

            Lf_Vi = float(grad_i @ f_i)
            Lg_Vi = grad_i @ g_i
            max_Lg_u = self._maximize_box_linear(Lg_Vi, u_lb, u_ub)

            # Tier 1: base-point sufficient condition
            if Lf_Vi + max_Lg_u + alpha * V_i < 0.0:
                sat_results[i] = 0
                continue

            tier1_pass_count += 1
            sat_results[i] = 1

            # Tier 2: exact one-step reachable checks over a control grid
            for u_i in per_agent_grids[i]:
                next_agent_state = np.array(agent_state)
                next_agent_state[i] = np.array(agent_state[i]) + dt * (
                    f_i + g_i @ np.asarray(u_i)
                )

                next_states = graph.states.at[agent_node_id].set(jnp.array(next_agent_state))
                next_graph = add_edge_feats_fn(graph, next_states)

                V_next = np.array(
                    critic_logits_fn(critic_params, next_graph)
                ).reshape(-1)
                V_i_next = float(V_next[i] - v_threshold)

                def v_next_fn(s_flat):
                    newer_states = next_graph.states.at[agent_node_id].set(
                        s_flat.reshape(n_agents, state_dim)
                    )
                    newer_graph = add_edge_feats_fn(next_graph, newer_states)
                    return critic_logits_fn(critic_params, newer_graph).reshape(-1)

                jac_next = np.array(jax.jacobian(v_next_fn)(next_agent_state.reshape(-1)))
                jac_next = jac_next.reshape(n_agents, n_agents, state_dim)
                grad_i_next = jac_next[i, i]

                dyn_f_next, dyn_g_next = control_affine_dyn_fn(jnp.array(next_agent_state))
                f_i_next = np.array(dyn_f_next[i])
                g_i_next = np.array(dyn_g_next[i])

                Lf_Vi_next = float(grad_i_next @ f_i_next)
                Lg_Vi_next = grad_i_next @ g_i_next
                max_Lg_u_next = self._maximize_box_linear(Lg_Vi_next, u_lb, u_ub)

                if Lf_Vi_next + max_Lg_u_next + alpha * V_i_next < 0.0:
                    sat_results[i] = 0
                    tier2_fail_count += 1
                    break

        step_stats = {
            "tier1fail": n_agents - tier1_pass_count,
            "tier1pass": tier1_pass_count,
            "tier2fail": tier2_fail_count,
            "tier2pass": tier1_pass_count - tier2_fail_count,
        }
        return sat_results, step_stats


    def verify_v_seev_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
        kgrid: int = 3,
    ):
        T = T_graph.states.shape[0]

        (
            n_agents,
            critic_state,
            _env_obj,
            critic_logits_fn,
            _add_edge_feats_fn,
            _control_affine_dyn_fn,
            _action_lim_fn,
        ) = self._resolve_verify_handles()

        T_sat = np.zeros((T, n_agents), dtype=np.int32)

        critic_params = critic_state.params
        per_agent_fn = jax.vmap(ft.partial(critic_logits_fn, critic_params))
        V_all = np.array(per_agent_fn(T_graph)).reshape(T, n_agents)
        v_threshold = float(V_all.mean())

        total_agents_checked = 0
        total_tier1fail = 0
        total_tier1pass = 0
        total_tier2fail = 0
        total_tier2pass = 0

        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            T_sat[t], step_stats = self.verify_v_seev(
                graph_t,
                alpha=alpha,
                kgrid=kgrid,
                v_threshold=v_threshold,
            )
            total_agents_checked += n_agents
            total_tier1fail += step_stats["tier1fail"]
            total_tier1pass += step_stats["tier1pass"]
            total_tier2fail += step_stats["tier2fail"]
            total_tier2pass += step_stats["tier2pass"]

        tier_stats = {
            "totalchecked": total_agents_checked,
            "tier1fail": total_tier1fail,
            "tier1pass": total_tier1pass,
            "tier2fail": total_tier2fail,
            "tier2pass": total_tier2pass,
            "tier1failrate": (
                total_tier1fail / total_agents_checked if total_agents_checked > 0 else 0.0
            ),
            "tier2failrate": (
                total_tier2fail / total_tier1pass if total_tier1pass > 0 else 0.0
            ),
        }
        return T_sat, tier_stats