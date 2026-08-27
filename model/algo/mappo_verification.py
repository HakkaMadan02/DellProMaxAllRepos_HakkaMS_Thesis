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
    # VERIFICATION — Z3 with V(x) as CBF surrogate (Option 1, V_min shifted)
    #
    # For each agent i, check via Z3 whether there EXISTS a control
    # u_i in action bounds such that the value-CBF condition holds:
    #
    #   L_f V~_i  +  L_g V~_i . u_i  +  alpha * V~_i(x)  >= 0
    #
    # where V~_i(x) = V_i(x) - V_min + eps  (shifted to be strictly positive)
    # V_min is computed over the full trajectory before the per-step loop.
    #
    # Gradients ∇V~_i = ∇V_i (shift doesn't affect gradients).
    # L_f, L_g computed via control-affine dynamics from env.
    # Exactly mirrors verify_cbf_z3 in GCBF+.
    # =========================================================================

    #@staticmethod
    #def _jit_grad_v(v_fn, agent_state_flat):
    #    return jax.jit(jax.grad(v_fn))(agent_state_flat)

    '''def verify_v_z3(
        self,
        graph: GraphsTuple,
        v_min: float,
        alpha: float = 1.0,
        delta_t: Optional[float] = None,
    ) -> np.ndarray:
        """
        Single-timestep V-CBF check via Z3.

        Parameters
        ----------
        graph  : single unbatched GraphsTuple at time t
        v_min  : trajectory-level minimum of V (precomputed), used to shift V >= 0
        alpha  : CBF decay rate
        delta_t: override env dt if needed

        Returns
        -------
        sat_results : np.ndarray (n_agents,)
            1 = SAT (safe control exists), 0 = UNSAT (no safe control found).
        """
        assert graph.is_single

        alpha = float(alpha)

        # --- identify agent nodes ---
        agent_node_mask = graph.node_type == 0
        agent_node_id   = mask2index(agent_node_mask, self.n_agents)
        agent_state     = graph.type_states(type_idx=0, n_type=self.n_agents)
        # agent_state shape: (n_agents, state_dim)

        # get_value returns scalar () — same value for all agents
        V_scalar = self.critic.get_value(self.critic_train_state.params, graph)
        V_scalar_np = float(np.array(V_scalar))

        # Shift: V~ = V - v_min + 1e-3  (strictly positive)
        V_shifted_scalar = V_scalar_np - v_min + 1e-3   # single float, shared across agents

        # --- ∇V via JAX jacobian ---
        # We need d(V_i)/d(x_i) for each agent i — the diagonal block
        # of the full (n_agents, n_agents * state_dim) jacobian.
        # v_fn returns (n_agents,) — jacobian gives (n_agents, n_agents * state_dim)
        def v_fn(agent_state_flat):
            new_states = graph.states.at[agent_node_id].set(
                agent_state_flat.reshape(self.n_agents, -1)
            )
            new_graph = self._env.add_edge_feats(graph, new_states)
            return self.critic.get_value(
                self.critic_train_state.params, new_graph
            )  # must return (n_agents,), NOT a scalar

        state_dim = agent_state.shape[-1]

        # ONE grad call — get_value is scalar, grad gives (n_agents * state_dim,)
        def v_fn(agent_state_flat):
            new_states = graph.states.at[agent_node_id].set(
                agent_state_flat.reshape(self.n_agents, -1)
            )
            new_graph = self._env.add_edge_feats(graph, new_states)
            return self.critic.get_value(self.critic_train_state.params, new_graph)

        grad_all = np.array(self._jit_grad_v(v_fn, agent_state.reshape(-1)))
        # grad_all: (n_agents * state_dim,) — same gradient for all agents
        # Each agent's block: dV/dx_i
        dV = grad_all.reshape(self.n_agents, state_dim)  # (n_agents, state_dim)

        # --- control-affine dynamics ---
        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        f_np = np.array(dyn_f)   # (n_agents, state_dim)
        g_np = np.array(dyn_g)   # (n_agents, state_dim, action_dim)

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)   # (action_dim,)
        u_ub = np.array(u_ub_jax)   # (action_dim,)
        action_dim = u_lb.shape[0]

        sat_results = np.zeros(self.n_agents, dtype=np.int32)

        for i in range(self.n_agents):
            # Lie derivatives
            Lf_Vi = float(dV[i] @ f_np[i])                  # scalar
            Lg_Vi = dV[i] @ g_np[i]                          # (action_dim,)
            Vi    = V_shifted_scalar                                # scalar, shared across agents

            solver = Solver()
            u_vars = [Real(f'u_{i}_{j}') for j in range(action_dim)]

            # Box constraints: u in [u_lb, u_ub]
            for j in range(action_dim):
                solver.add(u_vars[j] >= float(u_lb[j]))
                solver.add(u_vars[j] <= float(u_ub[j]))

            # V-CBF condition: L_f V~_i + L_g V~_i . u + alpha * V~_i >= 0
            cbf_lhs = Lf_Vi + alpha * Vi
            for j in range(action_dim):
                cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]

            solver.add(cbf_lhs >= 0)

            result = solver.check()
            sat_results[i] = 1 if result == z3_sat else 0

        return sat_results   # (n_agents,)'''

    '''def verify_v_z3_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
        delta_t: Optional[float] = None,
    ) -> np.ndarray:
        T = T_graph.states.shape[0]

        # --- Step 1: get agent node ids from first timestep (fixed across traj) ---
        graph_0    = jtu.tree_map(lambda x: jnp.array(x[0]), T_graph)
        agent_mask = graph_0.node_type == 0
        agent_node_id = mask2index(agent_mask, self.n_agents)
        state_dim  = graph_0.type_states(type_idx=0, n_type=self.n_agents).shape[-1]

        # --- Step 2: precompute V for all T timesteps in one vmap call ---
        #value_fn = jax.vmap(ft.partial(self.critic.get_value, self.critic_train_state.params))
        #V_all    = np.array(value_fn(T_graph))   # (T,) scalars
        #v_min    = float(V_all.min())
        #print(f"  [V-CBF] V_min={v_min:.4f}  V_max={V_all.max():.4f}  V_mean={V_all.mean():.4f}")
        # Per-agent values: (T, n_agents)
        per_agent_fn = jax.vmap(
            ft.partial(self.critic.get_per_agent_logits, self.critic_train_state.params)
        )
        V_all = np.array(per_agent_fn(T_graph))  # (T, n_agents)
        v_threshold = float(V_all.mean())
        print(f"  [V-CBF] V_min={V_all.min():.4f}  V_max={V_all.max():.4f}  V_mean={v_threshold:.4f}")

        # --- Step 3: precompute ALL gradients in ONE batched JIT call ---
        # critic_params fixed; only graph states vary across timesteps
        critic_params = self.critic_train_state.params

        # Extract all agent states across T: shape (T, n_agents, state_dim)
        T_agent_states = np.array(
            jax.vmap(lambda g: g.type_states(type_idx=0, n_type=self.n_agents))(T_graph)
        )   # (T, n_agents, state_dim)'''

        # JIT-compiled grad function — compiled ONCE, runs for all T
        '''@jax.jit
        def grad_single(agent_states_t, graph_t):
            # agent_states_t: (n_agents, state_dim)
            def v_fn(s_flat):
                new_states = graph_t.states.at[agent_node_id].set(
                    s_flat.reshape(self.n_agents, -1)
                )
                new_graph = self._env.add_edge_feats(graph_t, new_states)
                return self.critic.get_value(critic_params, new_graph)
            return jax.grad(v_fn)(agent_states_t.reshape(-1))
            # returns (n_agents * state_dim,)'''
        '''@jax.jit
        def jac_single(agent_states_t, graph_t):
            def v_fn(s_flat):
                new_states = graph_t.states.at[agent_node_id].set(
                    s_flat.reshape(self.n_agents, -1)
                )
                new_graph = self._env.add_edge_feats(graph_t, new_states)
                return self.critic.get_per_agent_logits(critic_params, new_graph)  # (n_agents,)
            # jacobian: (n_agents,) output x (n_agents * state_dim,) input
            return jax.jacobian(v_fn)(agent_states_t.reshape(-1))

        # T_dV[t, i] = gradient of V_i w.r.t. agent i's own state
        T_dV = np.zeros((T, self.n_agents, state_dim))
        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            jac = np.array(jac_single(jnp.array(T_agent_states[t]), graph_t))
            # jac shape: (n_agents, n_agents * state_dim) — diagonal blocks = dV_i/dx_i
            jac = jac.reshape(self.n_agents, self.n_agents, state_dim)
            T_dV[t] = jac[np.arange(self.n_agents), np.arange(self.n_agents), :]
            # T_dV[t, i] = dV_i/dx_i (only agent i's own-state gradient)'''

        # Run grad for all T timesteps — first call compiles, rest are fast
        '''print(f"  [V-CBF] Precomputing gradients for {T} timesteps ...")
        T_dV = np.zeros((T, self.n_agents, state_dim))
        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            T_dV[t] = np.array(
                grad_single(jnp.array(T_agent_states[t]), graph_t)
            ).reshape(self.n_agents, state_dim)
        print(f"  [V-CBF] Gradients done. Running Z3 ...")'''

        # --- Step 4: dynamics (same for all t since env is fixed) ---
        '''dyn_f_all = np.zeros((T, self.n_agents, state_dim))
        dyn_g_all = np.zeros((T, self.n_agents, state_dim, self.action_dim))
        for t in range(T):
            agent_state_t = jnp.array(T_agent_states[t])
            dyn_f_all[t] = np.array(self._env.control_affine_dyn(agent_state_t)[0])
            dyn_g_all[t] = np.array(self._env.control_affine_dyn(agent_state_t)[1])

        u_lb = np.array(self._env.action_lim()[0])
        u_ub = np.array(self._env.action_lim()[1])
        action_dim = u_lb.shape[0]

        # --- Step 5: Z3 loop — now just linear arithmetic, no JAX ---
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)
        for t in range(T):
            #V_shifted = float(V_all[t]) - v_min + 1e-3

            for i in range(self.n_agents):
                V_shifted = float(V_all[t, i]) - v_threshold 

                Lf_Vi = float(T_dV[t, i] @ dyn_f_all[t, i])
                Lg_Vi = T_dV[t, i] @ dyn_g_all[t, i]   # (action_dim,)

                solver = Solver()
                u_vars = [Real(f'u{t}_{i}_{j}') for j in range(action_dim)]
                for j in range(action_dim):
                    solver.add(u_vars[j] >= float(u_lb[j]))
                    solver.add(u_vars[j] <= float(u_ub[j]))

                cbf_lhs = Lf_Vi + alpha * V_shifted
                for j in range(action_dim):
                    cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]
                solver.add(cbf_lhs >= 0)

                T_sat[t, i] = 1 if solver.check() == z3_sat else 0

        return T_sat'''

    '''def verify_v_dreal_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
        delta: float = 1e-3,
    ) -> np.ndarray:
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        T = T_graph.states.shape[0]

        # --- Step 1: agent node ids from first timestep ---
        graph_0 = jtu.tree_map(lambda x: jnp.array(x[0]), T_graph)
        agent_mask = graph_0.node_type == 0
        agent_node_id = mask2index(agent_mask, self.n_agents)
        state_dim = graph_0.type_states(type_idx=0, n_type=self.n_agents).shape[-1]

        # --- Step 2: per-agent gate logits for all T timesteps ---
        critic_params = self.critic_train_state.params
        per_agent_fn = jax.vmap(
            ft.partial(self.critic.get_per_agent_logits, critic_params)
        )
        V_all = np.array(per_agent_fn(T_graph))          # (T, n_agents)
        v_threshold = float(V_all.mean())
        print(f"  [V-dReal] V_min={V_all.min():.4f}  V_max={V_all.max():.4f}  V_mean={v_threshold:.4f}")

        # --- Step 3: precompute all Jacobians in one batched JIT call ---
        T_agent_states = np.array(
            jax.vmap(lambda g: g.type_states(type_idx=0, n_type=self.n_agents))(T_graph)
        )  # (T, n_agents, state_dim)'''

        '''@jax.jit
        def jac_single(agent_states_t, graph_t):
            def v_fn(s_flat):
                new_states = graph_t.states.at[agent_node_id].set(
                    s_flat.reshape(self.n_agents, -1)
                )
                new_graph = self._env.add_edge_feats(graph_t, new_states)
                return self.critic.get_per_agent_logits(critic_params, new_graph)  # (n_agents,)
            return jax.jacobian(v_fn)(agent_states_t.reshape(-1))
            # returns (n_agents, n_agents * state_dim)'''

        '''print(f"  [V-dReal] Precomputing gradients for {T} timesteps ...")
        T_dV = np.zeros((T, self.n_agents, state_dim))
        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            jac = np.array(jac_single(jnp.array(T_agent_states[t]), graph_t))
            jac = jac.reshape(self.n_agents, self.n_agents, state_dim)
            T_dV[t] = jac[np.arange(self.n_agents), np.arange(self.n_agents), :]
        print(f"  [V-dReal] Gradients done. Running dReal ...")

        # --- Step 4: dynamics for all T timesteps ---
        dyn_f_all = np.zeros((T, self.n_agents, state_dim))
        dyn_g_all = np.zeros((T, self.n_agents, state_dim, self.action_dim))
        for t in range(T):
            agent_state_t = jnp.array(T_agent_states[t])
            dyn_f_all[t] = np.array(self._env.control_affine_dyn(agent_state_t)[0])
            dyn_g_all[t] = np.array(self._env.control_affine_dyn(agent_state_t)[1])

        u_lb = np.array(self._env.action_lim()[0])   # (action_dim,)
        u_ub = np.array(self._env.action_lim()[1])   # (action_dim,)
        action_dim = u_lb.shape[0]

        # --- Step 5: dReal loop — mirrors verify_cbf_dreal in GCBF+ exactly ---
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)
        for t in range(T):
            for i in range(self.n_agents):
                V_shifted = float(V_all[t, i]) - v_threshold
                Lf_Vi    = float(T_dV[t, i] @ dyn_f_all[t, i])
                Lg_Vi    = T_dV[t, i] @ dyn_g_all[t, i]   # (action_dim,)

                u_vars = [Variable(f"u{t}_{i}_{j}") for j in range(action_dim)]
                constraints = []

                # box constraints
                for j in range(action_dim):
                    constraints.append(u_vars[j] >= float(u_lb[j]))
                    constraints.append(u_vars[j] <= float(u_ub[j]))

                # V-CBF condition: Lf_V + Lg_V @ u + alpha * V_shifted >= 0
                cbf_lhs = Lf_Vi + alpha * V_shifted
                for j in range(action_dim):
                    cbf_lhs = cbf_lhs + float(Lg_Vi[j]) * u_vars[j]
                constraints.append(cbf_lhs >= 0)

                result = CheckSatisfiability(DRealAnd(*constraints), delta)
                # dReal: None = UNSAT (0), not None = SAT (1) — opposite sign to Z3
                T_sat[t, i] = 0 if result is None else 1

        return T_sat   # (T, n_agents)'''

    '''def verify_v_dreal_ball_trajectory(
        self,
        T_graph: GraphsTuple,
        alpha: float = 1.0,
        delta: float = 1e-3,
        eps: float = 0.1,
    ) -> np.ndarray:
        """
        Robust ball V-CBF check via dReal.

        For each agent i at each timestep t, checks whether there EXISTS
        a state perturbation dx in an L-inf ball of radius eps such that
        the V-CBF condition FAILS even under the analytically best control:

            Lf_V(x+dx) + max_{u in bounds} Lg_V(x+dx)@u + alpha * V_shifted(x+dx) < 0

        u is eliminated analytically:
            max Lg_V @ u = sum_j |Lg_V_j| * u_ub[j]   (symmetric box assumed)

        dReal SAT  (result is not None) → counterexample found → NOT robustly safe → 0
        dReal UNSAT (result is None)   → no bad perturbation exists → robustly safe  → 1

        Note: SAT/UNSAT interpretation is FLIPPED vs verify_v_dreal_trajectory.
        """
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        T = T_graph.states.shape[0]

        # --- Step 1: agent node ids ---
        graph_0 = jtu.tree_map(lambda x: jnp.array(x[0]), T_graph)
        agent_mask = graph_0.node_type == 0
        agent_node_id = mask2index(agent_mask, self.n_agents)
        state_dim = graph_0.type_states(type_idx=0, n_type=self.n_agents).shape[-1]

        # --- Step 2: per-agent gate logits for all T timesteps ---
        critic_params = self.critic_train_state.params
        per_agent_fn = jax.vmap(
            ft.partial(self.critic.get_per_agent_logits, critic_params)
        )
        V_all = np.array(per_agent_fn(T_graph))          # (T, n_agents)
        v_threshold = float(V_all.mean())
        print(f"  [V-dReal-Ball] V_min={V_all.min():.4f}  V_max={V_all.max():.4f}  V_mean={v_threshold:.4f}  eps={eps}")

        # --- Step 3: precompute all Jacobians ---
        T_agent_states = np.array(
            jax.vmap(lambda g: g.type_states(type_idx=0, n_type=self.n_agents))(T_graph)
        )  # (T, n_agents, state_dim)'''

        '''@jax.jit
        def jac_single(agent_states_t, graph_t):
            def v_fn(s_flat):
                new_states = graph_t.states.at[agent_node_id].set(
                    s_flat.reshape(self.n_agents, -1)
                )
                new_graph = self._env.add_edge_feats(graph_t, new_states)
                return self.critic.get_per_agent_logits(critic_params, new_graph)
            return jax.jacobian(v_fn)(agent_states_t.reshape(-1))
            # returns (n_agents, n_agents * state_dim)

        print(f"  [V-dReal-Ball] Precomputing gradients for {T} timesteps ...")
        T_dV = np.zeros((T, self.n_agents, state_dim))
        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            jac = np.array(jac_single(jnp.array(T_agent_states[t]), graph_t))
            jac = jac.reshape(self.n_agents, self.n_agents, state_dim)
            T_dV[t] = jac[np.arange(self.n_agents), np.arange(self.n_agents), :]
        print(f"  [V-dReal-Ball] Gradients done. Running dReal ...")

        # --- Step 4: dynamics for all T timesteps ---
        dyn_f_all = np.zeros((T, self.n_agents, state_dim))
        dyn_g_all = np.zeros((T, self.n_agents, state_dim, self.action_dim))
        for t in range(T):
            agent_state_t = jnp.array(T_agent_states[t])
            dyn_f_all[t] = np.array(self._env.control_affine_dyn(agent_state_t)[0])
            dyn_g_all[t] = np.array(self._env.control_affine_dyn(agent_state_t)[1])

        u_lb = np.array(self._env.action_lim()[0])   # (action_dim,)
        u_ub = np.array(self._env.action_lim()[1])   # (action_dim,)'''

        # --- Step 5: dReal ball loop ---
        '''T_sat = np.zeros((T, self.n_agents), dtype=np.int32)
        for t in range(T):
            for i in range(self.n_agents):
                grad_Vi   = T_dV[t, i]                     # (state_dim,)  ∇V_i at x_hat
                V_shifted = float(V_all[t, i]) - v_threshold
                f_i       = dyn_f_all[t, i]                # (state_dim,)
                g_i       = dyn_g_all[t, i]                # (state_dim, action_dim)

                # dReal variables: ONLY perturbation dx, u eliminated analytically
                dx_vars = [Variable(f"dx{t}_{i}_{k}") for k in range(state_dim)]
                constraints = []

                # Ball constraint: L-inf radius eps
                for k in range(state_dim):
                    constraints.append(dx_vars[k] >= -eps)
                    constraints.append(dx_vars[k] <=  eps)

                # Linearized V: V(x+dx) ≈ V_shifted + grad_Vi @ dx
                V_lin = V_shifted
                for k in range(state_dim):
                    V_lin = V_lin + float(grad_Vi[k]) * dx_vars[k]

                # Linearized Lf_V: grad_Vi @ f_i  (constant — hessian term omitted)
                Lf_V_lin = float(grad_Vi @ f_i)

                # Analytical worst-case Lg_V @ u (u eliminated):
                # max_{u in bounds} Lg_V @ u = sum_j |Lg_V_j| * u_ub[j]
                Lg_V_nominal = grad_Vi @ g_i               # (action_dim,)
                Lg_V_worst   = float(np.sum(np.abs(Lg_V_nominal) * u_ub))

                # Negated V-CBF condition: check if safety can FAIL in the ball
                # i.e., Lf_V_lin + Lg_V_worst + alpha * V_lin < 0
                # UNSAT → no bad perturbation → robustly safe → 1
                # SAT   → counterexample found → not robustly safe → 0
                cbf_expr = Lf_V_lin + Lg_V_worst + alpha * V_lin
                constraints.append(cbf_expr < 0)

                result = CheckSatisfiability(DRealAnd(*constraints), delta)
                T_sat[t, i] = 0 if result is not None else 1  # FLIPPED vs pointwise'''

        #return T_sat   # (T, n_agents)

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