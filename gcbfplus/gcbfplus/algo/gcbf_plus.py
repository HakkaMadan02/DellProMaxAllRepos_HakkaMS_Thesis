import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei

import time
from z3 import Real, Solver, sat as z3_sat
#from dReal import Variable, CheckSatisfiability, And as DRealAnd
from typing import Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array, State
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import merge01, jax_vmap, mask2index, tree_merge
from gcbfplus.trainer.data import Rollout
from gcbfplus.trainer.buffer import MaskedReplayBuffer
from gcbfplus.trainer.utils import compute_norm_and_clip, jax2np, tree_copy, empty_grad_tx
from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.algo.module.cbf import CBF
from gcbfplus.algo.module.policy import DeterministicPolicy
from .gcbf import GCBF


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action


class GCBFPlus(GCBF):

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
            horizon: int = 32,
            lr_actor: float = 3e-5,
            lr_cbf: float = 3e-5,
            alpha: float = 1.0,
            eps: float = 0.02,
            inner_epoch: int = 8,
            loss_action_coef: float = 0.001,
            loss_unsafe_coef: float = 1.,
            loss_safe_coef: float = 1.,
            loss_h_dot_coef: float = 0.2,
            max_grad_norm: float = 2.,
            seed: int = 0,
            **kwargs
    ):
        super(GCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_cbf = lr_cbf
        self.alpha = alpha
        self.eps = eps
        self.inner_epoch = inner_epoch
        self.loss_action_coef = loss_action_coef
        self.loss_unsafe_coef = loss_unsafe_coef
        self.loss_safe_coef = loss_safe_coef
        self.loss_h_dot_coef = loss_h_dot_coef
        self.gnn_layers = gnn_layers
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.horizon = horizon

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

        # set up CBF
        self.cbf = CBF(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers
        )
        key = jr.PRNGKey(seed)
        cbf_key, key = jr.split(key)
        cbf_params = self.cbf.net.init(cbf_key, nominal_graph, self.n_agents)
        cbf_optim = optax.adamw(learning_rate=lr_cbf, weight_decay=1e-3)
        self.cbf_optim = optax.apply_if_finite(cbf_optim, 1_000_000)
        self.cbf_train_state = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=cbf_params,
            tx=self.cbf_optim
        )
        self.cbf_tgt = TrainState.create(apply_fn=self.cbf.get_cbf, params=tree_copy(cbf_params), tx=empty_grad_tx())

        # set up actor
        self.actor = DeterministicPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        actor_key, key = jr.split(key)
        actor_params = self.actor.net.init(actor_key, nominal_graph, self.n_agents)
        actor_optim = optax.adamw(learning_rate=lr_actor, weight_decay=1e-3)
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

    @property
    def config(self) -> dict:
        return {
            'batch_size': self.batch_size,
            'lr_actor': self.lr_actor,
            'lr_cbf': self.lr_cbf,
            'alpha': self.alpha,
            'eps': self.eps,
            'inner_epoch': self.inner_epoch,
            'loss_action_coef': self.loss_action_coef,
            'loss_unsafe_coef': self.loss_unsafe_coef,
            'loss_safe_coef': self.loss_safe_coef,
            'loss_h_dot_coef': self.loss_h_dot_coef,
            'gnn_layers': self.gnn_layers,
            'seed': self.seed,
            'max_grad_norm': self.max_grad_norm,
            'horizon': self.horizon
        }

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

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_train_state.params
        action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph), log_pi

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=1)
    def update_tgt(self, cbf_tgt: TrainState, cbf: TrainState, tau: float) -> TrainState:
        tgt_params = optax.incremental_update(cbf.params, cbf_tgt.params, tau)
        return cbf_tgt.replace(params=tgt_params)

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_qp(self, b_graph: GraphsTuple, params) -> Action:
        b_u_qp, bT_relaxation = jax_vmap(ft.partial(self.get_qp_action, cbf_params=params))(b_graph)
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
            b_u_qp.append(jax2np(self.get_b_u_qp(graph, self.cbf_tgt.params)))
        b_u_qp = tree_merge(b_u_qp)

        batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_qp)

        for i_epoch in range(self.inner_epoch):
            idx = self.rng.choice(rollout.length, size=rollout.length, replace=False)
            # (n_mb, mb_size)
            batch_idx = np.stack(np.array_split(idx, idx.shape[0] // self.batch_size), axis=0)
            batch = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_train_state, actor_train_state, update_info = self.update_inner(
                self.cbf_train_state, self.actor_train_state, batch
            )
            self.cbf_train_state = cbf_train_state
            self.actor_train_state = actor_train_state

        # Update target.
        self.cbf_tgt = self.update_tgt(self.cbf_tgt, self.cbf_train_state, 0.5)

        return update_info

    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        if self.buffer.length > self.batch_size:
            # sample from memory
            memory, safe_mask_memory, unsafe_mask_memory = self.buffer.sample(rollout.length)
            try:
                unsafe_memory, safe_mask_unsafe_memory, unsafe_mask_unsafe_memory = self.unsafe_buffer.sample(
                    rollout.length * rollout.time_horizon)
            except ValueError:
                unsafe_memory = jtu.tree_map(lambda x: merge01(x), memory)
                safe_mask_unsafe_memory = merge01(safe_mask_memory)
                unsafe_mask_unsafe_memory = merge01(unsafe_mask_memory)

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

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        # (n_collect, T)
        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)

        rollout_np = jax2np(rollout)
        del rollout
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask)

        # inner loop
        update_info = self.update_nets(rollout_batch, safe_mask_batch, unsafe_mask_batch)

        return update_info

    def verify_cbf_z3(
        self,
        graph: GraphsTuple,
        cbf_params=None,
    ) -> np.ndarray:
        """
        For each agent in the (single) graph, check via Z3 whether there EXISTS
        a control u_i in the action bounds such that the CBF condition holds:

            Lf_h_i + Lg_h_i @ u_i + alpha * h_i >= 0

        Uses only the diagonal of the cross-agent Jacobian (agent i's CBF
        gradient w.r.t. agent i's own state). This is the 1-hop formulation.

        Returns
        -------
        sat_results : np.ndarray of shape (n_agents,)
            1 if Z3 returns SAT (a safe control exists), 0 if UNSAT.
        """
        from z3 import Real, Solver, sat as z3_sat

        assert graph.is_single  # must be a single (unbatched) graph

        if cbf_params is None:
            cbf_params = self.cbf_train_state.params

        # ------------------------------------------------------------------ #
        # Step 1: Compute h, full cross-agent Jacobian h_x, f, g
        #         Mirrors get_qp_action() exactly — no new computation here.
        # ------------------------------------------------------------------ #
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state):
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)

        # h shape: (n_agents,)
        h = h_aug(agent_state).squeeze(-1)

        # h_x shape: (n_agents_output, 1, n_agents_input, state_dim)
        # h_x[i, 0, j, :] = d(h_i) / d(x_j)
        h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)

        # f shape: (n_agents, state_dim)
        # g shape: (n_agents, state_dim, action_dim)
        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)

        # ------------------------------------------------------------------ #
        # Step 2: Extract per-agent diagonal Lie derivatives
        #
        # For agent i we use ONLY h_x[i, i, :] — the gradient of h_i
        # w.r.t. x_i. Off-diagonal terms are cross-agent and not controllable
        # by agent i alone (1-hop formulation).
        #
        # Lf_h_i  = h_x[i,i,:] @ f_i          — scalar
        # Lg_h_i  = h_x[i,i,:] @ g_i          — shape (action_dim,)
        # ------------------------------------------------------------------ #

        # Move to numpy for Z3 (Z3 works on Python floats, not JAX arrays)
        h_np    = np.array(h)                   # (n_agents,)
        h_x_np  = np.array(h_x)                 # (n_agents, n_agents, state_dim)
        dyn_f_np = np.array(dyn_f)              # (n_agents, state_dim)
        dyn_g_np = np.array(dyn_g)              # (n_agents, state_dim, action_dim)

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb = np.array(u_lb_jax)              # (action_dim,)
        u_ub = np.array(u_ub_jax)              # (action_dim,)
        action_dim = u_lb.shape[0]

        sat_results = np.zeros(self.n_agents, dtype=np.int32)

        for i in range(self.n_agents):

            # Diagonal gradient: d(h_i)/d(x_i), shape (state_dim,)
            grad_hi_xi = h_x_np[i, i, :]       # shape (state_dim,)

            # Lf_h_i = grad_hi_xi . f_i  (dot product → scalar)
            Lf_h_i = float(grad_hi_xi @ dyn_f_np[i])

            # Lg_h_i = grad_hi_xi . g_i  (matrix product → shape (action_dim,))
            Lg_h_i = grad_hi_xi @ dyn_g_np[i]  # shape (action_dim,)

            h_i    = float(h_np[i])
            alpha  = float(self.alpha)

            # ------------------------------------------------------------ #
            # Step 3: Build and solve Z3 query
            #
            # Variables: u_0, u_1, ..., u_{action_dim-1}
            # Constraints:
            #   (a) u_lb[j] <= u_j <= u_ub[j]   for all j  (box constraints)
            #   (b) Lf_h_i + sum_j(Lg_h_i[j] * u_j) + alpha * h_i >= 0
            # ------------------------------------------------------------ #
            solver = Solver()

            u_vars = [Real(f"u_{i}_{j}") for j in range(action_dim)]

            # Box constraints
            for j in range(action_dim):
                solver.add(u_vars[j] >= float(u_lb[j]))
                solver.add(u_vars[j] <= float(u_ub[j]))

            # CBF condition: Lf_h + Lg_h @ u + alpha * h >= 0
            cbf_lhs = Lf_h_i + alpha * h_i
            for j in range(action_dim):
                cbf_lhs = cbf_lhs + float(Lg_h_i[j]) * u_vars[j]

            solver.add(cbf_lhs >= 0)

            result = solver.check()
            sat_results[i] = 1 if result == z3_sat else 0

        return sat_results  # shape (n_agents,), dtype int32

    def verify_cbf_z3_trajectory(
        self,
        T_graph: GraphsTuple,
        cbf_params=None,
    ) -> np.ndarray:
        """
        Run Z3 CBF verification over all timesteps in a trajectory.

        Parameters
        ----------
        T_graph : GraphsTuple
            Batched graph with T timesteps (as returned by rollout.Tp1_graph).

        Returns
        -------
        T_sat : np.ndarray of shape (T, n_agents)
            sat_results[t, i] = 1 if agent i is Z3-verifiably safe at timestep t.
        """
        T = T_graph.states.shape[0]
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        for t in range(T):
            #graph_t = jtu.tree_map(lambda x: x[t], T_graph)
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph) 
            T_sat[t] = self.verify_cbf_z3(graph_t, cbf_params=cbf_params)

        return T_sat  # shape (T, n_agents)

    def verify_cbf_dreal(
        self,
        graph: GraphsTuple,
        cbf_params=None,
        delta: float = 1e-3,
    ) -> np.ndarray:
        from dreal import Variable, CheckSatisfiability, And as DRealAnd
        assert graph.is_single

        if cbf_params is None:
            cbf_params = self.cbf_train_state.params

        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state):
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)

        h            = h_aug(agent_state).squeeze(-1)
        h_x          = jax.jacobian(h_aug)(agent_state).squeeze(1)
        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)

        h_np     = np.array(h)
        h_x_np   = np.array(h_x)
        dyn_f_np = np.array(dyn_f)
        dyn_g_np = np.array(dyn_g)

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb       = np.array(u_lb_jax)
        u_ub       = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]

        sat_results = np.zeros(self.n_agents, dtype=np.int32)

        for i in range(self.n_agents):
            grad_hi_xi = h_x_np[i, i, :]
            Lf_h_i     = float(grad_hi_xi @ dyn_f_np[i])
            Lg_h_i     = grad_hi_xi @ dyn_g_np[i]
            h_i        = float(h_np[i])
            alpha      = float(self.alpha)

            u_vars      = [Variable(f"u_{i}_{j}") for j in range(action_dim)]
            constraints = []

            for j in range(action_dim):
                constraints.append(u_vars[j] >= float(u_lb[j]))
                constraints.append(u_vars[j] <= float(u_ub[j]))

            cbf_lhs = Lf_h_i + alpha * h_i
            for j in range(action_dim):
                cbf_lhs = cbf_lhs + float(Lg_h_i[j]) * u_vars[j]
            constraints.append(cbf_lhs >= 0)

            result      = CheckSatisfiability(DRealAnd(*constraints), delta)
            sat_results[i] = 0 if result is None else 1

        return sat_results

    def verify_cbf_dreal_ball(
        self,
        graph: GraphsTuple,
        cbf_params=None,
        delta: float = 1e-3,
        eps: float = 0.1,
    ) -> np.ndarray:
        """
        Robust ball CBF feasibility check using dReal v4.

        For each agent i, checks whether there EXISTS any state perturbation
        delta_x within an L-inf ball of radius eps around the observed state x_i,
        AND any control u_i in bounds, such that the CBF condition FAILS:

            (Lf_h(x) + Lg_h(x) @ u) + alpha * h(x) < 0

        where h(x), Lf_h(x), Lg_h(x) are linearized around x_hat_i using
        the gradient and Hessian of h.

        If dReal returns UNSAT on this negated query -> ball is robustly safe (return 1).
        If dReal returns SAT -> a perturbation exists that breaks safety (return 0).

        Returns
        -------
        sat_results : np.ndarray of shape (n_agents,)
            1 if robustly safe over the ball, 0 if a counterexample exists.
        """
        from dreal import Variable, CheckSatisfiability, And as DRealAnd

        assert graph.is_single

        if cbf_params is None:
            cbf_params = self.cbf_train_state.params

        agent_node_mask = graph.node_type == 0
        agent_node_id   = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state):
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)

        # h: (n_agents,)
        h = h_aug(agent_state).squeeze(-1)

        # h_x: (n_agents, n_agents, state_dim)  — first-order gradient
        h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)

        # h_xx: (n_agents, n_agents, state_dim, n_agents, state_dim)
        # We want diagonal: h_xx[i, i, :, i, :] = d²h_i / dx_i²
        # shape after diagonal selection: (n_agents, state_dim, state_dim)
        #h_xx_full = jax.hessian(h_aug)(agent_state).squeeze(1) #Uncomment for 2nd order approximation
        # h_xx_full shape: (n_agents, n_agents, state_dim, n_agents, state_dim)
        # diagonal over agent axis: [i, i, :, i, :]
        #h_xx = np.array(
        #    jnp.stack([h_xx_full[i, i, :, i, :] for i in range(self.n_agents)])
        #)  # (n_agents, state_dim, state_dim)

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)

        h_np     = np.array(h)
        h_x_np   = np.array(h_x)
        dyn_f_np = np.array(dyn_f)
        dyn_g_np = np.array(dyn_g)

        u_lb_jax, u_ub_jax = self._env.action_lim()
        u_lb       = np.array(u_lb_jax)
        u_ub       = np.array(u_ub_jax)
        action_dim = u_lb.shape[0]
        state_dim  = agent_state.shape[-1]

        sat_results = np.zeros(self.n_agents, dtype=np.int32)

        '''for i in range(self.n_agents):

            # Observed quantities at x_hat_i
            grad_hi   = h_x_np[i, i, :]            # (state_dim,)  ∇h_i
            hess_hi   = h_xx[i]                     # (state_dim, state_dim)  H_i
            h_i       = float(h_np[i])
            f_i       = dyn_f_np[i]                 # (state_dim,)
            g_i       = dyn_g_np[i]                 # (state_dim, action_dim)
            alpha     = float(self.alpha)

            # dReal variables
            # delta_x: perturbation around x_hat_i  (state_dim vars)
            # u: control  (action_dim vars)
            dx_vars = [Variable(f"dx_{i}_{k}") for k in range(state_dim)]
            u_vars  = [Variable(f"u_{i}_{j}")  for j in range(action_dim)]

            constraints = []

            # --- Ball constraint: L-inf, radius eps ---
            for k in range(state_dim):
                constraints.append(dx_vars[k] >= -eps)
                constraints.append(dx_vars[k] <=  eps)

            # --- Control box constraints ---
            for j in range(action_dim):
                constraints.append(u_vars[j] >= float(u_lb[j]))
                constraints.append(u_vars[j] <= float(u_ub[j]))

            # --- Linearized h at x_hat + dx ---
            # h(x) ≈ h_i + grad_hi @ dx + 0.5 * dx @ hess_hi @ dx
            h_lin = h_i
            for k in range(state_dim):
                h_lin = h_lin + float(grad_hi[k]) * dx_vars[k]
            for k in range(state_dim):
                for l in range(state_dim):
                    h_lin = h_lin + 0.5 * float(hess_hi[k, l]) * dx_vars[k] * dx_vars[l]

            # --- Linearized Lf_h at x_hat + dx ---
            # ∇h(x) ≈ grad_hi + hess_hi @ dx
            # Lf_h(x) = ∇h(x) @ f_i  (f is constant for DoubleIntegrator)
            Lf_h_lin = 0.0
            for k in range(state_dim):
                # (grad_hi + hess_hi @ dx)[k] * f_i[k]
                coeff_k = float(grad_hi[k]) * float(f_i[k])
                Lf_h_lin = Lf_h_lin + coeff_k
                for l in range(state_dim):
                    Lf_h_lin = Lf_h_lin + float(hess_hi[k, l]) * dx_vars[l] * float(f_i[k])

            # --- Linearized Lg_h @ u at x_hat + dx ---
            # Lg_h(x) = ∇h(x) @ g_i
            # (∇h(x) @ g_i)[j] = sum_k (grad_hi[k] + sum_l hess_hi[k,l]*dx[l]) * g_i[k,j]
            Lgu_lin = 0.0
            for j in range(action_dim):
                for k in range(state_dim):
                    # constant part: grad_hi[k] * g_i[k,j] * u_j
                    Lgu_lin = Lgu_lin + float(grad_hi[k]) * float(g_i[k, j]) * u_vars[j]
                    # perturbed part: hess_hi[k,l] * dx[l] * g_i[k,j] * u_j
                    for l in range(state_dim):
                        Lgu_lin = Lgu_lin + float(hess_hi[k, l]) * dx_vars[l] * float(g_i[k, j]) * u_vars[j]

            # --- Negated CBF condition ---
            # We check if: Lf_h(x) + Lg_h(x)@u + alpha*h(x) < 0
            # i.e., the CBF condition FAILS somewhere in the ball
            # If UNSAT -> no such (x, u) exists -> robustly safe -> return 1
            # If SAT   -> a counterexample exists -> not robustly safe -> return 0
            cbf_expr = Lf_h_lin + Lgu_lin + alpha * h_lin
            constraints.append(cbf_expr < 0)

            result      = CheckSatisfiability(DRealAnd(*constraints), delta)
            sat_results[i] = 0 if result is not None else 1'''

        for i in range(self.n_agents):

            grad_hi   = h_x_np[i, i, :]          # (state_dim,)
            #hess_hi   = h_xx[i]                   # (state_dim, state_dim)
            h_i       = float(h_np[i])
            f_i       = dyn_f_np[i]              # (state_dim,)
            g_i       = dyn_g_np[i]              # (state_dim, action_dim)
            alpha     = float(self.alpha)

            # dReal variables: ONLY delta_x, no u variables
            dx_vars = [Variable(f"dx_{i}_{k}") for k in range(state_dim)]

            constraints = []

            # --- Ball constraint: L-inf, radius eps ---
            for k in range(state_dim):
                constraints.append(dx_vars[k] >= -eps)
                constraints.append(dx_vars[k] <=  eps)

            # --- Linearized h at x_hat + dx ---
            # h(x) ≈ h_i + grad_hi @ dx + 0.5 * dx @ hess_hi @ dx
            h_lin = h_i
            for k in range(state_dim):
                h_lin = h_lin + float(grad_hi[k]) * dx_vars[k]
            #for k in range(state_dim):
            #    for l in range(state_dim):
            #        h_lin = h_lin + 0.5 * float(hess_hi[k, l]) * dx_vars[k] * dx_vars[l]

            # --- Linearized Lf_h at x_hat + dx ---
            # Lf_h(x) = (grad_hi + hess_hi @ dx) @ f_i
            Lf_h_lin = 0.0
            for k in range(state_dim):
                Lf_h_lin = Lf_h_lin + float(grad_hi[k]) * float(f_i[k])
            #    for l in range(state_dim):
            #        Lf_h_lin = Lf_h_lin + float(hess_hi[k, l]) * dx_vars[l] * float(f_i[k])

            # --- Analytical worst-case Lg_h @ u (u eliminated) ---
            # max_{u in [u_lb, u_ub]} Lg_h(x+dx) @ u
            # = sum_j max(Lg_h_j(x+dx) * u_ub[j], Lg_h_j(x+dx) * u_lb[j])
            # Since box is symmetric [-1,1]: = sum_j |Lg_h_j(x+dx)|
            #
            # Lg_h_j(x+dx) = sum_k (grad_hi[k] + sum_l hess_hi[k,l]*dx[l]) * g_i[k,j]
            # We bound this conservatively:
            # |Lg_h_j(x+dx)| <= |Lg_h_j(x_hat)| + |d(Lg_h_j)/dx| * eps
            # This gives a pure float (no dReal variables in u), making the query
            # a polynomial in dx_vars only.
            Lg_h_nominal = grad_hi @ g_i                  # (action_dim,)
            #dLg_h_ddx    = hess_hi @ g_i                  # (state_dim, action_dim): d(Lg_h_j)/d(dx_k)
            # worst-case bound on |Lg_h_j(x+dx)|:
            # |Lg_h_j(x+dx)| <= |Lg_h_j(x_hat)| + ||d(Lg_h_j)/dx||_1 * eps
            Lg_h_worst = float(
                np.sum(np.abs(Lg_h_nominal) * u_ub)
                #+ np.sum(np.abs(dLg_h_ddx)) * eps * float(np.max(u_ub))
            )

            # --- Negated CBF condition ---
            # Check if: Lf_h(x+dx) i+ Lg_h_worst + alpha*h(x+dx) < 0
            # i.e., even the BEST possible u cannot satisfy CBF somewhere in the ball
            # If UNSAT -> robustly safe -> return 1
            # If SAT   -> counterexample found -> return 0
            cbf_expr = Lf_h_lin + Lg_h_worst + alpha * h_lin
            constraints.append(cbf_expr < 0)

            result = CheckSatisfiability(DRealAnd(*constraints), delta)
            sat_results[i] = 0 if result is not None else 1  # flipped vs pointwise
        return sat_results


    def verify_cbf_dreal_ball_trajectory(
        self,
        T_graph: GraphsTuple,
        cbf_params=None,
        delta: float = 1e-3,
        eps: float = 0.1,
    ) -> np.ndarray:
        T     = T_graph.states.shape[0]
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        for t in range(T):
            graph_t  = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            T_sat[t] = self.verify_cbf_dreal_ball(
                graph_t, cbf_params=cbf_params, delta=delta, eps=eps
            )

        return T_sat

    def verify_cbf_dreal_trajectory(
        self,
        T_graph: GraphsTuple,
        cbf_params=None,
        delta: float = 1e-3,
    ) -> np.ndarray:
        T     = T_graph.states.shape[0]
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        for t in range(T):
            graph_t = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            T_sat[t] = self.verify_cbf_dreal(graph_t, cbf_params=cbf_params, delta=delta)

        return T_sat

    def verify_cbf_seev(
        self,
        graph: GraphsTuple,
        cbf_params=None,
        k_grid: int = 3,
    ) -> np.ndarray:

    '''
    This is the Hamilton Jacobi Version of SEEV,
    The faster version is Heuristicd based, just make the corner points manually depending on k args
    '''
    
        assert graph.is_single

        if cbf_params is None:
            cbf_params = self.cbf_train_state.params

        agent_node_mask = graph.node_type == 0
        agent_node_id   = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state):
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)

        # --- ONE Jacobian at base — same cost as Z3, no additional passes ---
        h          = h_aug(agent_state).squeeze(-1)                # (n_agents,)
        h_x        = jax.jacobian(h_aug)(agent_state).squeeze(1)  # (n_agents, n_agents, state_dim)
        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)

        h_np    = np.array(h)            # (n_agents,)
        h_x_np  = np.array(h_x)         # (n_agents, n_agents, state_dim)
        f_np    = np.array(dyn_f)        # (n_agents, state_dim)
        g_np    = np.array(dyn_g)        # (n_agents, state_dim, action_dim)
        x_np    = np.array(agent_state)  # (n_agents, state_dim)

        u_lb_np    = np.array(self._env.action_lim()[0])   # (action_dim,)
        u_ub_np    = np.array(self._env.action_lim()[1])   # (action_dim,)
        action_dim = u_lb_np.shape[0]
        alpha      = float(self.alpha)
        dt         = float(self._env.dt)

        # Nominal next states for j != i (one policy call, then pure numpy)
        nominal_u_np   = np.array(self.act(graph))          # (n_agents, action_dim)
        x_next_nominal = x_np + dt * (                      # (n_agents, state_dim)
            f_np + np.einsum('jkl,jl->jk', g_np, nominal_u_np)
        )

        # k_grid=1: use actual actor action as the single evaluation point
        if k_grid == 1:
            grid_points = nominal_u_np   # shape (n_agents, action_dim) — per-agent
        else:
            grid_axes = [np.linspace(u_lb_np[d], u_ub_np[d], k_grid) for d in range(action_dim)]
            grid_points = np.stack(
                np.meshgrid(*grid_axes, indexing='ij'), axis=-1
            ).reshape(-1, action_dim)    # shape (k^action_dim, action_dim) — shared

        sat_results = np.zeros(self.n_agents, dtype=np.int32)

        # Add to verify_cbf_seev, just before the for loop
        #u_lb_jax, u_ub_jax = self._env.action_lim()
        #print(f"[SEEV diag] u_lb = {np.array(u_lb_jax)}, u_ub = {np.array(u_ub_jax)}")
        # print(f"[SEEV diag] symmetric = {np.allclose(np.array(u_lb_jax), -np.array(u_ub_jax))}")
        tier1_pass_count = 0
        tier2_fail_count = 0

        for i in range(self.n_agents):
            grad_h_i = h_x_np[i, i, :]   # d(h_i)/d(x_i), constant in activation region
            h_i      = h_np[i]
            f_i      = f_np[i]
            g_i      = g_np[i]

            # Lg_h_i and max_Lgu: computed ONCE per agent, reused for all grid points.
            # Valid because: grad_h_i is constant in region AND g is constant for DI.
            Lg_h_i  = grad_h_i @ g_i                              # (action_dim,)
            max_Lgu = float(np.sum(np.abs(Lg_h_i) * u_ub_np))    # scalar constant

            # --- Tier 1 at base point ---
            Lf_h_i = float(grad_h_i @ f_i)
            if (Lf_h_i + max_Lgu + alpha * float(h_i)) < 0.0:
                sat_results[i] = 0
                continue   # base already fails, no neighbourhood cert possible
            tier1_pass_count += 1
            sat_results[i] = 1  # tentatively SAT, pending Tier 2 checks

            # --- Tier 2: corners-outer, agents-inner, scalar grad ---
           
            def h_all_scalar(s):
                return h_aug(s).squeeze(-1).sum()

            for u_corn in grid_points:                             # k² iterations

                # Build next states for ALL agents under u_corn — one per agent (n_agents vectors)
                # x_next_per_agent[i] = nominal but with agent i stepped by u_corn
                # Shape: (n_agents, n_agents, state_dim)
                x_next_per_agent = np.tile(x_next_nominal, (self.n_agents, 1, 1))   # copy nominal
                for i in range(self.n_agents):
                    x_i_next_corn = x_np[i] + dt * (f_np[i] + g_np[i] @ u_corn)
                    x_next_per_agent[i, i] = x_i_next_corn       # only agent i differs

                for i in range(self.n_agents):
                    if sat_results[i] == 0:
                        continue                                   # already failed, skip

                    x_i_next  = x_next_per_agent[i, i]            # (state_dim,) agent i's next state
                    x_next_jax = jnp.array(x_next_per_agent[i])   # (n_agents, state_dim)

                    # ONE h_aug call for all agents — n_agents times cheaper than per-agent scalar
                    h_next_val, grad_all = jax.value_and_grad(h_all_scalar)(x_next_jax)
                    # grad_all shape: (n_agents, state_dim)
                    # grad_all[i] == d(h_i)/d(x_next_jax[i]) — exact, not approximate
                    h_i_next_val  = float(h_aug(x_next_jax).squeeze(-1)[i])  # get h_i specifically
                    grad_h_i_next = np.array(grad_all[i])          # (state_dim,)

                    f_i_next     = np.array([x_i_next[2], x_i_next[3], 0.0, 0.0])
                    Lf_next      = float(grad_h_i_next @ f_i_next)
                    Lg_next      = grad_h_i_next @ g_np[i]
                    max_Lgu_next = float(np.sum(np.abs(Lg_next) * u_ub_np))

                    if (Lf_next + max_Lgu_next + alpha * h_i_next_val) < 0.0:
                        sat_results[i] = 0
                        tier2_fail_count += 1

        # --- ADD: print summary for this single timestep graph ---
        '''print(
            f"[SEEV counters] n_agents={self.n_agents} | "
            f"Tier1 pass={tier1_pass_count} | "
            f"Tier2 fail (given T1 pass)={tier2_fail_count} | "
            f"pure T1 cert={tier1_pass_count - tier2_fail_count}"
        )'''

        step_stats = {
            'tier1_fail': self.n_agents - tier1_pass_count,
            'tier1_pass': tier1_pass_count,
            'tier2_fail': tier2_fail_count,
            'tier2_pass': tier1_pass_count - tier2_fail_count,
        }
        return sat_results, step_stats

    def verify_cbf_seev_trajectory(
        self,
        T_graph: GraphsTuple,
        cbf_params=None,
        k_grid: int = 3,
    ) -> np.ndarray:
        """
        Run verify_cbf_seev over every timestep in a trajectory.

        Parameters
        ----------
        T_graph    : GraphsTuple with T timesteps (e.g. rollout.Tp1_graph).
        cbf_params : CBF params (defaults to cbf_train_state.params).
        k_grid     : Grid density per action dim for Tier 2 (default 3 → 9 pts for 2D).

        Returns
        -------
        T_sat : np.ndarray (T, n_agents) — 1 = SEEV-certified, 0 = not certified.
        """
        T     = T_graph.states.shape[0]
        T_sat = np.zeros((T, self.n_agents), dtype=np.int32)

        total_agents_checked = 0
        total_tier1_fail     = 0   # Tier 1 failed at base point
        total_tier1_pass     = 0   # Tier 1 passed
        total_tier2_fail     = 0   # Tier 2 failed given Tier 1 passed
        total_tier2_pass     = 0   # Both tiers passed → certified

        for t in range(T):
            graph_t  = jtu.tree_map(lambda x: jnp.array(x[t]), T_graph)
            T_sat[t], step_stats = self.verify_cbf_seev(
                graph_t, cbf_params=cbf_params, k_grid=k_grid
            )
            total_agents_checked += self.n_agents
            total_tier1_fail     += step_stats['tier1_fail']
            total_tier1_pass     += step_stats['tier1_pass']
            total_tier2_fail     += step_stats['tier2_fail']
            total_tier2_pass     += step_stats['tier2_pass']

        tier_stats = {
            'total_checked'  : total_agents_checked,
            'tier1_fail'     : total_tier1_fail,
            'tier1_pass'     : total_tier1_pass,
            'tier2_fail'     : total_tier2_fail,
            'tier2_pass'     : total_tier2_pass,
            'tier1_fail_rate': total_tier1_fail / total_agents_checked,
            'tier2_fail_rate': total_tier2_fail / total_tier1_pass if total_tier1_pass > 0 else 0.0,
        }

        return T_sat, tier_stats
    

    def get_qp_action(
            self,
            graph: GraphsTuple,
            relax_penalty: float = 1e3,
            cbf_params=None,
            qp_settings: JaxProxQP.Settings = None,
    ) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state).squeeze(-1)
        h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)

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
        b = Lf_h + self.alpha * 0.1 * h

        r_lb = jnp.array([0.] * self.n_agents, dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf] * self.n_agents, dtype=jnp.float32)
        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        if qp_settings is None:
            qp_settings = JaxProxQP.Settings.default()
        qp_settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, qp_settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents,)
        u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents:]
        u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner(
            self, cbf_train_state: TrainState, actor_train_state: TrainState, batch: Batch
    ) -> tuple[TrainState, TrainState, dict]:
        def update_fn(carry, minibatch: Batch):
            cbf, actor = carry
            # (batch_size, n_agents) -> (minibatch_size * n_agents, )
            safe_mask_batch = merge01(minibatch.safe_mask)
            unsafe_mask_batch = merge01(minibatch.unsafe_mask)

            def get_loss(cbf_params: Params, actor_params: Params) -> Tuple[Array, dict]:
                # get CBF values
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
                cbf_fn_no_grad = jax_vmap(ft.partial(self.cbf.get_cbf, jax.lax.stop_gradient(cbf_params)))
                # (minibatch_size, n_agents)
                h = cbf_fn(minibatch.graph).squeeze(-1)
                # (minibatch_size * n_agents,)
                h = merge01(h)

                # unsafe region h(x) < 0
                unsafe_data_ratio = jnp.mean(unsafe_mask_batch)
                h_unsafe = jnp.where(unsafe_mask_batch, h, -jnp.ones_like(h) * self.eps * 2)
                max_val_unsafe = jax.nn.relu(h_unsafe + self.eps)
                loss_unsafe = jnp.sum(max_val_unsafe) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)
                acc_unsafe_mask = jnp.where(unsafe_mask_batch, h, jnp.ones_like(h))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)

                # safe region h(x) > 0
                h_safe = jnp.where(safe_mask_batch, h, jnp.ones_like(h) * self.eps * 2)
                max_val_safe = jax.nn.relu(-h_safe + self.eps)
                loss_safe = jnp.sum(max_val_safe) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)
                acc_safe_mask = jnp.where(safe_mask_batch, h, -jnp.ones_like(h))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)

                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action = action_fn(minibatch.graph)

                # get next graph
                forward_fn = jax_vmap(self._env.forward_graph)
                next_graph = forward_fn(minibatch.graph, action)
                h_next = merge01(cbf_fn(next_graph).squeeze(-1))
                h_dot = (h_next - h) / self._env.dt

                # stop gradient and get next graph
                h_no_grad = jax.lax.stop_gradient(h)
                h_next_no_grad = merge01(cbf_fn_no_grad(next_graph).squeeze(-1))
                h_dot_no_grad = (h_next_no_grad - h_no_grad) / self._env.dt

                # h_dot + alpha * h > 0 (backpropagate to action, and backpropagate to h when labeled)
                labeled_mask = jnp.logical_or(unsafe_mask_batch, safe_mask_batch)
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
                max_val_h_dot_no_grad = jax.nn.relu(-h_dot_no_grad - self.alpha * h + self.eps)
                max_val_h_dot = jnp.where(labeled_mask, max_val_h_dot, max_val_h_dot_no_grad)
                loss_h_dot = jnp.mean(max_val_h_dot)
                acc_h_dot = jnp.mean(jnp.greater(h_dot + self.alpha * h, 0))

                # action loss
                assert action.shape == minibatch.u_qp.shape
                loss_action = jnp.mean(jnp.square(action - minibatch.u_qp).sum(axis=-1))

                # total loss
                total_loss = (
                        self.loss_action_coef * loss_action
                        + self.loss_unsafe_coef * loss_unsafe
                        + self.loss_safe_coef * loss_safe
                        + self.loss_h_dot_coef * loss_h_dot
                )

                return total_loss, {'loss/action': loss_action,
                                    'loss/unsafe': loss_unsafe,
                                    'loss/safe': loss_safe,
                                    'loss/h_dot': loss_h_dot,
                                    'loss/total': total_loss,
                                    'acc/unsafe': acc_unsafe,
                                    'acc/safe': acc_safe,
                                    'acc/h_dot': acc_h_dot,
                                    'acc/unsafe_data_ratio': unsafe_data_ratio}

            (loss, loss_info), (grad_cbf, grad_actor) = jax.value_and_grad(
                get_loss, has_aux=True, argnums=(0, 1))(cbf.params, actor.params)
            grad_cbf, grad_cbf_norm = compute_norm_and_clip(grad_cbf, self.max_grad_norm)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            cbf = cbf.apply_gradients(grads=grad_cbf)
            actor = actor.apply_gradients(grads=grad_actor)
            grad_info = {'grad_norm/cbf': grad_cbf_norm, 'grad_norm/actor': grad_actor_norm}
            return (cbf, actor), grad_info | loss_info

        train_state = (cbf_train_state, actor_train_state)
        (cbf_train_state, actor_train_state), info = lax.scan(update_fn, train_state, batch)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)
        return cbf_train_state, actor_train_state, info
