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

# ── dReal ReLU status codes ────────────────────────────────────────
_RELU_INACTIVE = 0   # pre_hi <= 0  →  post = 0 always
_RELU_ACTIVE   = 1   # pre_lo >= 0  →  post = pre always
_RELU_UNSTABLE = 2   # pre_lo < 0 < pre_hi  →  case split needed

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

    # ══════════════════════════════════════════════════════════════════
    # FORMAL VERIFICATION — dReal
    # ══════════════════════════════════════════════════════════════════

    def _extract_weights(self, cbf_params: dict) -> dict:
        """
        Extract all Dense kernels and biases from the Flax param tree
        as float64 numpy arrays with explicit names.

        Confirmed tree layout from pprint:
            params
            ├── CBFHead / Dense_0 (128→256 ReLU), Dense_1 (256→256 ReLU)
            ├── Dense_0 (256→1, linear, pre-tanh)
            └── GNN_0 / GNNLayer_0
                    ├── msg    / Dense_0 (10→256 ReLU), Dense_1 (256→256 ReLU)
                    ├── Dense_0  (256→128, linear, msg output)
                    ├── attn   / Dense_0 (128→128 ReLU), Dense_1 (128→128 ReLU)
                    ├── Dense_1  (128→1,   linear, attn logit output)
                    ├── update / Dense_0 (131→256 ReLU), Dense_1 (256→256 ReLU)
                    └── Dense_2  (256→128, linear, update output)
        """
        p = cbf_params['params']
        gnn = p['GNN_0']['GNNLayer_0']
        from numpy.linalg import svd

        def np64(x):
            return np.array(x, dtype=np.float64)

        return {
            # ── MLP_msg ───────────────────────────────────────────────
            # hidden layers (ReLU): in msg sub-dict
            # output layer (linear): GNNLayer_0/Dense_0
            'msg_W0':  np64(gnn['msg']['Dense_0']['kernel']),  # (10, 256)
            'msg_b0':  np64(gnn['msg']['Dense_0']['bias']),    # (256,)
            'msg_W1':  np64(gnn['msg']['Dense_1']['kernel']),  # (256, 256)
            'msg_b1':  np64(gnn['msg']['Dense_1']['bias']),    # (256,)
            'msg_Wo':  np64(gnn['Dense_0']['kernel']),         # (256, 128)
            'msg_bo':  np64(gnn['Dense_0']['bias']),           # (128,)

            # ── MLP_attn ──────────────────────────────────────────────
            # hidden layers (ReLU): in attn sub-dict
            # output layer (linear): GNNLayer_0/Dense_1 → scalar logit
            'atn_W0':  np64(gnn['attn']['Dense_0']['kernel']),  # (128, 128)
            'atn_b0':  np64(gnn['attn']['Dense_0']['bias']),    # (128,)
            'atn_W1':  np64(gnn['attn']['Dense_1']['kernel']),  # (128, 128)
            'atn_b1':  np64(gnn['attn']['Dense_1']['bias']),    # (128,)
            'atn_Wo':  np64(gnn['Dense_1']['kernel']),          # (128, 1)
            'atn_bo':  np64(gnn['Dense_1']['bias']),            # (1,)

            # ── MLP_update ────────────────────────────────────────────
            # hidden layers (ReLU): in update sub-dict
            # output layer (linear): GNNLayer_0/Dense_2
            'upd_W0':  np64(gnn['update']['Dense_0']['kernel']),  # (131, 256)
            'upd_b0':  np64(gnn['update']['Dense_0']['bias']),    # (256,)
            'upd_W1':  np64(gnn['update']['Dense_1']['kernel']),  # (256, 256)
            'upd_b1':  np64(gnn['update']['Dense_1']['bias']),    # (256,)
            'upd_Wo':  np64(gnn['Dense_2']['kernel']),            # (256, 128)
            'upd_bo':  np64(gnn['Dense_2']['bias']),              # (128,)

            # ── CBF Head ──────────────────────────────────────────────
            # hidden layers (ReLU): in CBFHead sub-dict
            # output layer (linear): top-level Dense_0 → pre-tanh scalar
            'cbf_W0':  np64(p['CBFHead']['Dense_0']['kernel']),  # (128, 256)
            'cbf_b0':  np64(p['CBFHead']['Dense_0']['bias']),    # (256,)
            'cbf_W1':  np64(p['CBFHead']['Dense_1']['kernel']),  # (256, 256)
            'cbf_b1':  np64(p['CBFHead']['Dense_1']['bias']),    # (256,)
            'cbf_Wo':  np64(p['Dense_0']['kernel']),             # (256, 1)
            'cbf_bo':  np64(p['Dense_0']['bias']),               # (1,)
            'cbf_W0_norm': float(svd(np64(p['CBFHead']['Dense_0']['kernel']), compute_uv=False).max()),
            'cbf_W1_norm': float(svd(np64(p['CBFHead']['Dense_1']['kernel']), compute_uv=False).max()),
            'cbf_Wo_norm': float(svd(np64(p['Dense_0']['kernel']), compute_uv=False).max()),
        }


    @staticmethod
    def _ia_affine(W, b, lo, hi):
        """
        Interval arithmetic through an affine layer  y = x @ W + b.
        W: (in_dim, out_dim).  lo, hi: (in_dim,).
        Returns (y_lo, y_hi), each (out_dim,).
        """
        Wp = np.maximum(W, 0.0)
        Wn = np.minimum(W, 0.0)
        y_lo = lo @ Wp + hi @ Wn + b
        y_hi = hi @ Wp + lo @ Wn + b
        return y_lo, y_hi


    @staticmethod
    def _ia_relu(pre_lo, pre_hi):
        """
        Interval arithmetic through ReLU.
        Returns (post_lo, post_hi, status) where status ∈ {0,1,2}.
        """
        post_lo = np.maximum(pre_lo, 0.0)
        post_hi = np.maximum(pre_hi, 0.0)
        status = np.where(pre_hi <= 0.0, _RELU_INACTIVE,
                np.where(pre_lo >= 0.0, _RELU_ACTIVE, _RELU_UNSTABLE))
        return post_lo, post_hi, status


    def _interval_propagate(
        self,
        graph,
        agent_i: int,
        weights: dict,
        eps: float = 0.0,
    ) -> dict:
        """
        Phase 1 — forward interval arithmetic through the full GNN-CBF network
        for agent_i.

        eps=0 → point intervals (concrete inputs from rollout).
                All ReLUs are deterministically classified. n_unstable == 0.
        eps>0 → ball of radius eps around the observed state.
                Edge features (relative states) are perturbed by ±eps.
                Some ReLUs may become unstable.

        Returns ia: dict with keys
            'msg_relu'  : list[neighbor_idx] of [(pre_lo,pre_hi,status)×2_layers]
            'atn_relu'  : same structure as msg_relu
            'upd_relu'  : [(pre_lo,pre_hi,status)×2_layers]
            'cbf_relu'  : [(pre_lo,pre_hi,status)×2_layers]
            'pre_tanh_lo', 'pre_tanh_hi'  : scalar bounds on input to tanh
            'n_unstable': total count of unstable ReLU neurons
            'attn_weights_mid': concrete softmax weights (midpoints)
            'aggr_lo', 'aggr_hi': bounds on aggregated message (128,)
        """
        nodes     = np.array(graph.nodes,     dtype=np.float64)  # (n_nodes, 3)
        edges     = np.array(graph.edges,     dtype=np.float64)  # (n_edges, 4)
        senders   = np.array(graph.senders,   dtype=np.int32)
        receivers = np.array(graph.receivers, dtype=np.int32)

        # All edges where agent_i is the receiver
        nbr_edge_idxs = np.where(receivers == agent_i)[0]
        n_nbrs = len(nbr_edge_idxs)

        recv_feat = nodes[agent_i]  # (3,) — concrete, never perturbed (one-hot)

        ia = {
            'msg_relu': [],
            'atn_relu': [],
        }

        msg_lo_list    = []
        msg_hi_list    = []
        logit_lo_list  = []
        logit_hi_list  = []

        # ── Per-neighbor: MLP_msg then MLP_attn ───────────────────────
        for k in nbr_edge_idxs:
            edge_feat   = edges[k]                    # (4,) relative state
            sender_feat = nodes[senders[k]]           # (3,) one-hot, exact

            msg_in = np.concatenate([edge_feat, sender_feat, recv_feat])  # (10,)

            # Only edge features (relative states) are perturbed under eps-ball
            msg_in_lo = msg_in.copy()
            msg_in_hi = msg_in.copy()
            if eps > 0.0:
                msg_in_lo[:4] -= eps
                msg_in_hi[:4] += eps

            # Physical state space bounds — edge features are relative states
            # bounded by comm_radius = 0.5 in all 4 dimensions
            #msg_in_lo = msg_in.copy()
            #msg_in_hi = msg_in.copy()
            #msg_in_lo[:4] = -0.01   # Change here for tractibility issue 
            #msg_in_hi[:4] =  0.01

            # MLP_msg hidden layer 0  (10 → 256) + ReLU
            p0_lo, p0_hi = self._ia_affine(weights['msg_W0'], weights['msg_b0'], msg_in_lo, msg_in_hi)
            h0_lo, h0_hi, s0 = self._ia_relu(p0_lo, p0_hi)

            # MLP_msg hidden layer 1  (256 → 256) + ReLU
            p1_lo, p1_hi = self._ia_affine(weights['msg_W1'], weights['msg_b1'], h0_lo, h0_hi)
            h1_lo, h1_hi, s1 = self._ia_relu(p1_lo, p1_hi)

            # MLP_msg output layer  (256 → 128) linear — no activation
            mo_lo, mo_hi = self._ia_affine(weights['msg_Wo'], weights['msg_bo'], h1_lo, h1_hi)

            msg_lo_list.append(mo_lo)
            msg_hi_list.append(mo_hi)
            ia['msg_relu'].append([(p0_lo, p0_hi, s0), (p1_lo, p1_hi, s1)])

            # MLP_attn hidden layer 0  (128 → 128) + ReLU
            a0_lo, a0_hi = self._ia_affine(weights['atn_W0'], weights['atn_b0'], mo_lo, mo_hi)
            ah0_lo, ah0_hi, as0 = self._ia_relu(a0_lo, a0_hi)

            # MLP_attn hidden layer 1  (128 → 128) + ReLU
            a1_lo, a1_hi = self._ia_affine(weights['atn_W1'], weights['atn_b1'], ah0_lo, ah0_hi)
            ah1_lo, ah1_hi, as1 = self._ia_relu(a1_lo, a1_hi)

            # MLP_attn output layer  (128 → 1) linear — scalar logit
            lo_lo, lo_hi = self._ia_affine(weights['atn_Wo'], weights['atn_bo'], ah1_lo, ah1_hi)

            logit_lo_list.append(lo_lo[0])
            logit_hi_list.append(lo_hi[0])
            ia['atn_relu'].append([(a0_lo, a0_hi, as0), (a1_lo, a1_hi, as1)])

        # ── Softmax → attention weights ───────────────────────────────
        # Use midpoint logits for concrete attn weights (sound for eps=0,
        # conservative approximation for eps>0 — Wei et al. bounds come in Step 3)
        logit_mid = np.array([(lo + hi) / 2.0
                            for lo, hi in zip(logit_lo_list, logit_hi_list)])
        if n_nbrs > 0:
            logit_mid -= logit_mid.max()          # numerical stability
            exp_mid = np.exp(logit_mid)
            attn_mid = exp_mid / exp_mid.sum()    # (n_nbrs,)
        else:
            attn_mid = np.array([])

        ia['attn_weights_mid'] = attn_mid

        # ── Aggregate messages ─────────────────────────────────────────
        if n_nbrs > 0:
            msg_lo_arr = np.stack(msg_lo_list)   # (n_nbrs, 128)
            msg_hi_arr = np.stack(msg_hi_list)
            msg_mid_arr = (msg_lo_arr + msg_hi_arr) / 2.0

            # eps=0: exact aggregation (midpoints == bounds)
            aggr_lo = attn_mid @ msg_lo_arr      # (128,)
            aggr_hi = attn_mid @ msg_hi_arr

            # eps>0: conservative — weights are in (0,1) and sum to 1,
            # so aggr ∈ [min msg across nbrs, max msg across nbrs]
            if eps > 0.0:
                aggr_lo = msg_lo_arr.min(axis=0)
                aggr_hi = msg_hi_arr.max(axis=0)
        else:
            # Isolated agent — no neighbors, aggregation is zero
            aggr_lo = np.zeros(128, dtype=np.float64)
            aggr_hi = np.zeros(128, dtype=np.float64)

        ia['aggr_lo'] = aggr_lo
        ia['aggr_hi'] = aggr_hi

        # ── MLP_update ────────────────────────────────────────────────
        # input = concat(node_feat_i, aggr)  (3 + 128 = 131)
        upd_in_lo = np.concatenate([recv_feat, aggr_lo])
        upd_in_hi = np.concatenate([recv_feat, aggr_hi])

        # hidden layer 0  (131 → 256) + ReLU
        up0_lo, up0_hi = self._ia_affine(weights['upd_W0'], weights['upd_b0'], upd_in_lo, upd_in_hi)
        uh0_lo, uh0_hi, us0 = self._ia_relu(up0_lo, up0_hi)

        # hidden layer 1  (256 → 256) + ReLU
        up1_lo, up1_hi = self._ia_affine(weights['upd_W1'], weights['upd_b1'], uh0_lo, uh0_hi)
        uh1_lo, uh1_hi, us1 = self._ia_relu(up1_lo, up1_hi)

        # output layer  (256 → 128) linear
        uo_lo, uo_hi = self._ia_affine(weights['upd_Wo'], weights['upd_bo'], uh1_lo, uh1_hi)

        ia['upd_relu'] = [(up0_lo, up0_hi, us0), (up1_lo, up1_hi, us1)]

        # ── CBF Head ──────────────────────────────────────────────────
        # hidden layer 0  (128 → 256) + ReLU
        cp0_lo, cp0_hi = self._ia_affine(weights['cbf_W0'], weights['cbf_b0'], uo_lo, uo_hi)
        ch0_lo, ch0_hi, cs0 = self._ia_relu(cp0_lo, cp0_hi)

        # hidden layer 1  (256 → 256) + ReLU
        cp1_lo, cp1_hi = self._ia_affine(weights['cbf_W1'], weights['cbf_b1'], ch0_lo, ch0_hi)
        ch1_lo, ch1_hi, cs1 = self._ia_relu(cp1_lo, cp1_hi)

        # output layer  (256 → 1) linear → pre-tanh scalar
        pt_lo, pt_hi = self._ia_affine(weights['cbf_Wo'], weights['cbf_bo'], ch1_lo, ch1_hi)

        ia['cbf_relu'] = [(cp0_lo, cp0_hi, cs0), (cp1_lo, cp1_hi, cs1)]
        ia['pre_tanh_lo'] = float(pt_lo[0])
        ia['pre_tanh_hi'] = float(pt_hi[0])

        # ── Count unstable ReLU neurons ───────────────────────────────
        n_unstable = 0
        for nbr_layers in ia['msg_relu']:
            for _, _, s in nbr_layers:
                n_unstable += int(np.sum(s == _RELU_UNSTABLE))
        for nbr_layers in ia['atn_relu']:
            for _, _, s in nbr_layers:
                n_unstable += int(np.sum(s == _RELU_UNSTABLE))
        for _, _, s in ia['upd_relu']:
            n_unstable += int(np.sum(s == _RELU_UNSTABLE))
        for _, _, s in ia['cbf_relu']:
            n_unstable += int(np.sum(s == _RELU_UNSTABLE))

        ia['n_unstable'] = n_unstable
        return ia

    # ══════════════════════════════════════════════════════════════════
    # CROWN-style bound propagation — replaces IA-based box verifier
    # ══════════════════════════════════════════════════════════════════
    #
    # WHY: The old midpoint/radius hybrid in _interval_propagate_box blew up
    # because IA box widths compound multiplicatively across 7 deep layers,
    # producing pre-activation ranges of +-3600 / +-150k (all neurons
    # "unstable"), making the Jacobian-bound CBF check useless.
    #
    # CROWN instead backpropagates a LINEAR relaxation of the full network
    # from the scalar pre-tanh output back to the 4-dim edge input.
    # Unstable ReLU neurons get a linear envelope (chord upper, zero lower).
    # The final certificate is:  A_lo @ x + b_lo <= pre_tanh(x) <= A_hi @ x + b_hi
    # Minimise A_lo @ x + b_lo over the box analytically -> tight scalar bound.
    # Much tighter than IA, so BnB needs far fewer splits.
    #
    # WHAT STAYS THE SAME:
    #   _ia_affine, _ia_relu, _interval_propagate (eps-ball, rollout stats)
    #   _extract_weights, verify_cbf_full, all training/QP code
    # WHAT CHANGES:
    #   _interval_propagate_box  -> _crown_propagate_box  (new, replaces old)
    #   _check_cbf_on_box        -> updated to consume crown dict
    #   _verify_agent_bnb        -> calls the two new methods above
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _crown_relu(pre_lo: np.ndarray, pre_hi: np.ndarray):
        # Per-neuron CROWN linear envelope of ReLU(x) for x in [pre_lo, pre_hi].
        # Returns (lam_l, mu_l, lam_u, mu_u) each shape (n,) such that:
        #   lam_l*x + mu_l <= ReLU(x) <= lam_u*x + mu_u  for all x in [lo,hi]
        #
        # ACTIVE   (pre_lo >= 0):  lam_l=lam_u=1, mu_l=mu_u=0  (exact)
        # INACTIVE (pre_hi <= 0):  lam_l=lam_u=0, mu_l=mu_u=0  (exact)
        # UNSTABLE (lo<0<hi):
        #   upper chord: lam_u = hi/(hi-lo),  mu_u = -lo*hi/(hi-lo)
        #   lower zero:  lam_l = 0,            mu_l = 0
        active   = pre_lo >= 0.0
        inactive = pre_hi <= 0.0
        unstable = ~active & ~inactive
        denom    = np.where(unstable, pre_hi - pre_lo, 1.0)
        lam_u    = np.where(active, 1.0, np.where(inactive, 0.0, pre_hi / denom))
        mu_u     = np.where(active, 0.0, np.where(inactive, 0.0, -pre_lo * pre_hi / denom))
        lam_l    = np.where(active, 1.0, 0.0)
        mu_l     = np.zeros_like(pre_lo)
        return lam_l, mu_l, lam_u, mu_u

    @staticmethod
    def _crown_fwd_ia(W, b, lo, hi):
        # IA forward pass y = x @ W + b  (same as _ia_affine, used in Phase 1).
        Wp = np.maximum(W, 0.0); Wn = np.minimum(W, 0.0)
        return lo @ Wp + hi @ Wn + b, hi @ Wp + lo @ Wn + b

    def _crown_propagate_box(
        self,
        weights: dict,
        recv_feat: np.ndarray,
        sender_feat: np.ndarray,
        edge_lo: np.ndarray,
        edge_hi: np.ndarray,
    ) -> dict:
        # Phase 1: IA forward pass to get pre-activation intervals [lo,hi]
        # for every ReLU. These fix the CROWN slopes.
        # Phase 2: CROWN backward pass. Start from the scalar pre-tanh
        # output, backprop a (1 x n) linear certificate through each layer.
        # Phase 3: Minimise/maximise the linear cert over the box analytically.
        #
        # Returns dict with pre_tanh_lo/hi, A_lo/A_hi (4,), b_lo/b_hi, n_unstable.

        # ── Phase 1: IA forward ─────────────────────────────────────────
        const_tail = np.concatenate([sender_feat, recv_feat])   # (6,)
        msg_in_lo  = np.concatenate([edge_lo, const_tail])      # (10,)
        msg_in_hi  = np.concatenate([edge_hi, const_tail])

        pm0_lo, pm0_hi = self._crown_fwd_ia(weights['msg_W0'], weights['msg_b0'], msg_in_lo, msg_in_hi)
        hm0_lo, hm0_hi = np.maximum(pm0_lo, 0), np.maximum(pm0_hi, 0)
        pm1_lo, pm1_hi = self._crown_fwd_ia(weights['msg_W1'], weights['msg_b1'], hm0_lo, hm0_hi)
        hm1_lo, hm1_hi = np.maximum(pm1_lo, 0), np.maximum(pm1_hi, 0)
        mo_lo, mo_hi   = self._crown_fwd_ia(weights['msg_Wo'],  weights['msg_bo'],  hm1_lo, hm1_hi)

        upd_in_lo = np.concatenate([recv_feat, mo_lo])   # (131,)
        upd_in_hi = np.concatenate([recv_feat, mo_hi])
        pu0_lo, pu0_hi = self._crown_fwd_ia(weights['upd_W0'], weights['upd_b0'], upd_in_lo, upd_in_hi)
        hu0_lo, hu0_hi = np.maximum(pu0_lo, 0), np.maximum(pu0_hi, 0)
        pu1_lo, pu1_hi = self._crown_fwd_ia(weights['upd_W1'], weights['upd_b1'], hu0_lo, hu0_hi)
        hu1_lo, hu1_hi = np.maximum(pu1_lo, 0), np.maximum(pu1_hi, 0)
        uo_lo,  uo_hi  = self._crown_fwd_ia(weights['upd_Wo'],  weights['upd_bo'],  hu1_lo, hu1_hi)

        pc0_lo, pc0_hi = self._crown_fwd_ia(weights['cbf_W0'], weights['cbf_b0'], uo_lo, uo_hi)
        hc0_lo, hc0_hi = np.maximum(pc0_lo, 0), np.maximum(pc0_hi, 0)
        pc1_lo, pc1_hi = self._crown_fwd_ia(weights['cbf_W1'], weights['cbf_b1'], hc0_lo, hc0_hi)
        hc1_lo, hc1_hi = np.maximum(pc1_lo, 0), np.maximum(pc1_hi, 0)
        pt_lo,  pt_hi  = self._crown_fwd_ia(weights['cbf_Wo'],  weights['cbf_bo'],  hc1_lo, hc1_hi)

        def _n_unstable(lo, hi): return int(np.sum((lo < 0) & (hi > 0)))
        n_unstable = (_n_unstable(pm0_lo, pm0_hi) + _n_unstable(pm1_lo, pm1_hi) +
                    _n_unstable(pu0_lo, pu0_hi) + _n_unstable(pu1_lo, pu1_hi) +
                    _n_unstable(pc0_lo, pc0_hi) + _n_unstable(pc1_lo, pc1_hi))

        print(f'  msg_L0: range=[{pm0_lo.min():.1f}, {pm0_hi.max():.1f}], unstable={_n_unstable(pm0_lo,pm0_hi)}')
        print(f'  msg_L1: range=[{pm1_lo.min():.1f}, {pm1_hi.max():.1f}], unstable={_n_unstable(pm1_lo,pm1_hi)}')
        print(f'  upd_L0: range=[{pu0_lo.min():.1f}, {pu0_hi.max():.1f}], unstable={_n_unstable(pu0_lo,pu0_hi)}')
        print(f'  upd_L1: range=[{pu1_lo.min():.1f}, {pu1_hi.max():.1f}], unstable={_n_unstable(pu1_lo,pu1_hi)}')

        # ── Phase 2: CROWN backward pass ────────────────────────────────
        # Maintain cert_lo (1, n) and cert_hi (1, n) — the lower/upper linear
        # certificate w.r.t. the current layer input, plus scalar biases.
        # Invariant: cert_lo @ z + b_lo <= pre_tanh <= cert_hi @ z + b_hi

        def _bkwd_linear(W, b_vec, cW_lo, cW_hi, cb_lo, cb_hi):
            # y = z @ W + b_vec  (exact linear, no uncertainty)
            # new cert w.r.t. z: (cW @ W.T), bias += cW @ b_vec
            Wt        = W.T                               # (out, in)
            new_cW_lo = cW_lo @ Wt                        # (1, in)
            new_cW_hi = cW_hi @ Wt
            new_b_lo  = cb_lo + float((cW_lo @ b_vec).squeeze())
            new_b_hi  = cb_hi + float((cW_hi @ b_vec).squeeze())
            return new_cW_lo, new_cW_hi, new_b_lo, new_b_hi

        def _bkwd_relu(pre_lo_r, pre_hi_r, cW_lo, cW_hi, cb_lo, cb_hi):
            # Backprop through ReLU with CROWN slopes.
            # Lower bound: where cW_lo[j]>=0 use lam_l (smaller), else lam_u.
            # Upper bound: where cW_hi[j]>=0 use lam_u (larger),  else lam_l.
            lam_l, mu_l, lam_u, mu_u = GCBFPlus._crown_relu(pre_lo_r, pre_hi_r)
            # lower cert
            s_lo      = np.where(cW_lo >= 0, lam_l[None,:], lam_u[None,:])
            m_lo      = np.where(cW_lo >= 0, mu_l[None,:],  mu_u[None,:])
            new_cW_lo = cW_lo * s_lo
            new_b_lo  = cb_lo + float((cW_lo * m_lo).sum())
            # upper cert
            s_hi      = np.where(cW_hi >= 0, lam_u[None,:], lam_l[None,:])
            m_hi      = np.where(cW_hi >= 0, mu_u[None,:],  mu_l[None,:])
            new_cW_hi = cW_hi * s_hi
            new_b_hi  = cb_hi + float((cW_hi * m_hi).sum())
            return new_cW_lo, new_cW_hi, new_b_lo, new_b_hi

        # Initialise: pre_tanh = hc1 @ cbf_Wo + cbf_bo
        # cert w.r.t. hc1: cW = cbf_Wo.T (1,256), b = cbf_bo[0]
        cW_lo = weights['cbf_Wo'].T.copy()    # (1, 256)
        cW_hi = weights['cbf_Wo'].T.copy()
        cb_lo = float(weights['cbf_bo'][0])
        cb_hi = float(weights['cbf_bo'][0])

        # cbf head (backward)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_relu(pc1_lo,pc1_hi, cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['cbf_W1'],weights['cbf_b1'], cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_relu(pc0_lo,pc0_hi, cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['cbf_W0'],weights['cbf_b0'], cW_lo,cW_hi,cb_lo,cb_hi)
        # cert now w.r.t. uo (128,)

        # upd head (backward)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['upd_Wo'],weights['upd_bo'], cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_relu(pu1_lo,pu1_hi, cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['upd_W1'],weights['upd_b1'], cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_relu(pu0_lo,pu0_hi, cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['upd_W0'],weights['upd_b0'], cW_lo,cW_hi,cb_lo,cb_hi)
        # cert now w.r.t. upd_in (131,) = [recv(3) | aggr(128)]
        # recv_feat dims [0:3] are constant -> absorb into bias
        cb_lo += float((cW_lo[0,:3] * recv_feat).sum())
        cb_hi += float((cW_hi[0,:3] * recv_feat).sum())
        cW_lo  = cW_lo[:,3:]    # (1, 128) cert w.r.t. aggr
        cW_hi  = cW_hi[:,3:]

        # msg head (backward)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['msg_Wo'],weights['msg_bo'], cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_relu(pm1_lo,pm1_hi, cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['msg_W1'],weights['msg_b1'], cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_relu(pm0_lo,pm0_hi, cW_lo,cW_hi,cb_lo,cb_hi)
        cW_lo,cW_hi,cb_lo,cb_hi = _bkwd_linear(weights['msg_W0'],weights['msg_b0'], cW_lo,cW_hi,cb_lo,cb_hi)
        # cert now w.r.t. msg_in (10,) = [edge(4) | sender(3) | recv(3)]
        # constant tail [4:10] -> absorb into bias
        cb_lo += float((cW_lo[0,4:] * const_tail).sum())
        cb_hi += float((cW_hi[0,4:] * const_tail).sum())

        A_lo = cW_lo[0,:4].copy()   # (4,) cert w.r.t. edge_feat
        A_hi = cW_hi[0,:4].copy()

        # ── Phase 3: box minimisation / maximisation ────────────────────
        x_worst = np.where(A_lo >= 0, edge_lo, edge_hi)
        x_best  = np.where(A_hi >= 0, edge_hi, edge_lo)
        pre_tanh_lo = float(A_lo @ x_worst) + cb_lo
        pre_tanh_hi = float(A_hi @ x_best)  + cb_hi
        # Tighten with IA (CROWN >= IA tightness, but take max/min as sanity)
        pre_tanh_lo = max(pre_tanh_lo, float(pt_lo[0]))
        pre_tanh_hi = min(pre_tanh_hi, float(pt_hi[0]))

        return {
            'pre_tanh_lo': pre_tanh_lo,
            'pre_tanh_hi': pre_tanh_hi,
            'A_lo': A_lo,
            'A_hi': A_hi,
            'b_lo': cb_lo,
            'b_hi': cb_hi,
            'n_unstable': n_unstable,
        }

    def _check_cbf_on_box(
        self,
        crown: dict,
        weights: dict,   # kept for API compat with _verify_agent_bnb
        edge_lo: np.ndarray,
        edge_hi: np.ndarray,
    ) -> bool:
        import math
        pre_lo = crown['pre_tanh_lo']
        pre_hi = crown['pre_tanh_hi']
        h_lo   = math.tanh(pre_lo)
        h_hi   = math.tanh(pre_hi)

        if h_hi < 0.0:
            return True   # no safe states in box

        A_lo = crown['A_lo']
        A_hi = crown['A_hi']

        abs_max = max(abs(pre_lo), abs(pre_hi))
        td_lo   = max(1.0 - math.tanh(abs_max)**2, 0.0)

        dh_lo = np.where(A_lo >= 0, td_lo * A_lo, 1.0 * A_lo)
        dh_hi = np.where(A_hi >= 0, 1.0 * A_hi,  td_lo * A_hi)

        def ip_lo(g_lo, g_hi, v_lo, v_hi):
            return min(g_lo*v_lo, g_lo*v_hi, g_hi*v_lo, g_hi*v_hi)

        Lf_lo = (ip_lo(float(dh_lo[0]), float(dh_hi[0]), edge_lo[2], edge_hi[2]) +
                ip_lo(float(dh_lo[1]), float(dh_hi[1]), edge_lo[3], edge_hi[3]))

        m = 0.1
        def abs_lower(g_lo_v, g_hi_v):
            if g_lo_v <= 0.0 <= g_hi_v: return 0.0
            return min(abs(g_lo_v), abs(g_hi_v))

        sup_Lgh_lo = (abs_lower(float(dh_lo[2]), float(dh_hi[2])) +
                    abs_lower(float(dh_lo[3]), float(dh_hi[3]))) / m

        alpha_eff  = float(self.alpha) * 0.1
        cbf_val_lo = Lf_lo + sup_Lgh_lo + alpha_eff * h_lo
        return cbf_val_lo >= 0.0

    def _verify_agent_bnb(
        self,
        weights: dict,
        recv_feat: np.ndarray,
        sender_feat: np.ndarray,
        X_full_lo: np.ndarray,
        X_full_hi: np.ndarray,
        max_depth: int = 30,
    ) -> bool:
        worklist   = [(X_full_lo.copy(), X_full_hi.copy(), 0)]
        n_boxes    = 0
        n_verified = 0

        while worklist:
            lo, hi, depth = worklist.pop()
            n_boxes += 1

            crown    = self._crown_propagate_box(weights, recv_feat, sender_feat, lo, hi)
            verified = self._check_cbf_on_box(crown, weights, lo, hi)

            if verified:
                n_verified += 1
                continue

            if depth >= max_depth:
                print(f'  [BnB/CROWN] Max depth {max_depth} reached — UNVERIFIED: '
                    f'n_unstable={crown["n_unstable"]} '
                    f'pre_tanh=[{crown["pre_tanh_lo"]:.3f},{crown["pre_tanh_hi"]:.3f}]')
                return False

            widths    = hi - lo
            split_dim = int(np.argmax(widths))
            mid       = (lo[split_dim] + hi[split_dim]) / 2.0

            lo_l = lo.copy(); hi_l = hi.copy(); hi_l[split_dim] = mid
            lo_r = lo.copy(); hi_r = hi.copy(); lo_r[split_dim] = mid
            worklist.append((lo_l, hi_l, depth + 1))
            worklist.append((lo_r, hi_r, depth + 1))

        print(f'  [BnB/CROWN] VERIFIED: {n_verified}/{n_boxes} boxes')
        return True

    def verify_cbf_full(self, graph, cbf_params,
                        area_size: float = 4.0,
                        v_max: float = 0.5,
                        max_depth: int = 30) -> np.ndarray:
        """
        Full formal verification of the CBF condition over the entire state space.

        X_full = { [dx, dy, vx, vy] : dx,dy ∈ [-area_size, area_size],
                                        vx,vy ∈ [-v_max, v_max] }

        For each agent with at least one neighbor, runs branch-and-bound
        to verify the CBF condition over all of X_full.

        The result is topology-cached: identical (senders, receivers) sets
        are verified only once.

        Returns: np.ndarray of shape (n_agents,), 1=verified, 0=not verified.
        """
        weights = self._extract_weights(cbf_params)

        import jax.tree_util as jtu
        graph_np = jtu.tree_map(lambda x: np.array(x), graph)
        nodes_raw = np.array(graph_np.nodes)
        if nodes_raw.ndim ==3:
            graph_t0 = jtu.tree_map(lambda x: x[0], graph_np)
        else:
            graph_t0 = graph_np

        nodes_array = np.array(graph_t0.nodes, dtype=np.float64)
        senders_list = np.array(graph_t0.senders, dtype=np.int32).tolist()
        receivers_arr = np.array(graph_t0.receivers, dtype=np.int32).flatten()

        #nodes = np.array(graph.nodes, dtype=np.float64)
        #senders = np.array(graph.senders, dtype=np.int32)
        #receivers = np.array(graph.receivers, dtype=np.int32)

        # Full state space box
        X_full_lo = np.array([-area_size, -area_size, -v_max, -v_max])
        X_full_hi = np.array([ area_size,  area_size,  v_max,  v_max])

        results = np.zeros(self.n_agents, dtype=np.int32)
        topo_cache = {}

        for agent_i in range(self.n_agents):
            nbr_edge_idxs = np.where(receivers_arr == agent_i)[0]

            if len(nbr_edge_idxs) == 0:
                print(f"[BnB] agent {agent_i}: isolated → VERIFIED (trivially)")
                results[agent_i] = 1
                continue

            recv_feat = nodes_array[agent_i]  # (3,) one-hot

            # For now: verify w.r.t. each neighbor independently (sound over-approximation)
            # If ALL neighbor relationships are verified → agent verified
            all_verified = True
            for k in nbr_edge_idxs:
                sender_i = int(np.asarray(senders_list[k]).item())
                sender_feat = nodes_array[sender_i]

                cache_key = (agent_i, sender_i,
                            tuple(recv_feat.tolist()),
                            tuple(sender_feat.tolist()))

                if cache_key in topo_cache:
                    ok = topo_cache[cache_key]
                else:
                    print(f"[BnB] agent {agent_i} ← sender {sender_i}: "
                        f"verifying over X_full...")
                    t0 = time.time()
                    ok = self._verify_agent_bnb(
                        weights, recv_feat, sender_feat,
                        X_full_lo, X_full_hi, max_depth=max_depth
                    )
                    print(f"[BnB] → {'VERIFIED' if ok else 'UNVERIFIED'} "
                        f"({time.time()-t0:.1f}s)")
                    topo_cache[cache_key] = ok

                if not ok:
                    all_verified = False
                    break

            results[agent_i] = 1 if all_verified else 0

        return results

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
