from typing import Optional, Tuple

import einops as ei
import ipdb
import jax
import jax.numpy as jnp
from jaxproxqp.jaxproxqp import JaxProxQP
from loguru import logger

from model.env.base import MultiAgentEnv
from model.trainer.data import Rollout
from model.utils.graph import GraphsTuple
from model.utils.typing import Action, Array, Params, PRNGKey, State
from model.utils.utils import mask2index

from .base import MultiAgentController
from .utils import get_cbf_fn


class DecShareCBF(MultiAgentController):
    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            alpha: float = 1.0,
            **kwargs
    ):
        super().__init__(env=env, node_dim=node_dim, edge_dim=edge_dim, action_dim=action_dim, n_agents=n_agents)

        if hasattr(env, "enable_stop"):
            env.enable_stop = False

        self.cbf_alpha = alpha
        self.cbf = get_cbf_fn(env)

    @property
    def config(self) -> dict:
        return {
            "alpha": self.cbf_alpha,
        }

    @property
    def actor_params(self) -> Params:
        raise NotImplementedError

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        raise NotImplementedError

    def get_cbf(self, graph: GraphsTuple) -> tuple[Array, Array]:
        return self.cbf(graph)

    def update(self, rollout: Rollout, step: int) -> dict:
        raise NotImplementedError

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        return self.get_qp_action(graph)[0]

    def get_qp_action(self, graph: GraphsTuple, relax_penalty: float = 1e3) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> tuple[Array, Array]:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            ah, aobs = self.get_cbf(new_graph)
            ah = ah.squeeze(-1)
            assert ah.shape == (self.n_agents,)
            assert aobs.shape == (self.n_agents,)
            return ah, aobs

        def h(new_agent_state: State) -> Array:
            return h_aug(new_agent_state)[0]

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        # (n_agents, )
        ah, aobs = h_aug(agent_state)
        # (n_agents | n_agents, nx)
        h_x = jax.jacfwd(h)(agent_state).squeeze()

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        aLf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i")
        aaLg_h: Array = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")

        def index_fn(idx: int):
            Lg_h = aaLg_h[idx, idx]
            assert Lg_h.shape == (self.action_dim,)
            return Lg_h

        aLg_h_self = jax.vmap(index_fn)(jnp.arange(self.n_agents))

        au_ref = self._env.u_ref(graph)
        assert au_ref.shape == (self.n_agents, self.action_dim)

        # (n_agents, ). 1 if agent-obs, 0.5 if agent-agent.
        aresp = jnp.where(aobs, 1.0, 0.5)

        # construct QP
        au_opt, ar = jax.vmap(self._solve_qp_single)(ah, aLf_h, aLg_h_self, au_ref, aresp)
        return au_opt, ar

    def _solve_qp_single(self, h, Lf_h, Lg_h, u_ref, responsibility: float, relax_penalty: float = 1e3):
        n_qp_x = self._env.action_dim + 1

        assert h.shape == tuple()
        assert Lf_h.shape == tuple()
        assert Lg_h.shape == (self._env.action_dim,)

        u_lb, u_ub = self._env.action_lim()
        assert u_lb.shape == u_ub.shape == (self.action_dim,)

        ###########

        H = jnp.eye(n_qp_x, dtype=jnp.float32)
        H = H.at[-1, -1].set(10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(1)], axis=0)
        assert g.shape == (n_qp_x,)

        C = -jnp.concatenate([Lg_h, jnp.ones(1)], axis=0)
        C = C[None]
        assert C.shape == (1, n_qp_x)

        # Responsibility is one if this is agent-obs, half if this is agent-agent.
        b = responsibility * (Lf_h + self.cbf_alpha * h)
        b = b[None]
        assert b.shape == (1,)

        r_lb = jnp.array([0.0], dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf], dtype=jnp.float32)

        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)
        assert l_box.shape == u_box.shape == (n_qp_x,)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 100
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim + 1,)
        u_opt, r = sol.x[: self.action_dim], sol.x[-1]

        return u_opt, r

    def save(self, save_dir: str, step: int):
        raise NotImplementedError

    def load(self, load_dir: str, step: int):
        raise NotImplementedError
