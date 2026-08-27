import jax.numpy as jnp
import jax
import einops as ei

from typing import Optional, Tuple
from jaxproxqp.jaxproxqp import JaxProxQP

from ..utils.typing import Action, Params, PRNGKey, Array, State
from ..utils.graph import GraphsTuple
from ..utils.utils import mask2index
from ..trainer.data import Rollout
from ..env.base import MultiAgentEnv
from .utils import get_cbf_fn
from .base import MultiAgentController
from .centralized_cbf import CentralizedCBF


class DecOptCBF(MultiAgentController):

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
        super(DecOptCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        self.alpha = alpha
        self.cbf = get_cbf_fn(env)

    @property
    def config(self) -> dict:
        return {
            'alpha': self.alpha,
        }

    @property
    def actor_params(self) -> Params:
        raise NotImplementedError

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        raise NotImplementedError

    def get_cbf(self, graph: GraphsTuple) -> Array:
        return self.cbf(graph)[0]

    def update(self, rollout: Rollout, step: int) -> dict:
        raise NotImplementedError

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        return self.get_qp_action(graph)[0]

    def get_qp_action(self, graph: GraphsTuple, relax_penalty: float = 1e3) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            return self.get_cbf(new_graph)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state).squeeze()
        h_x = jax.jacfwd(h_aug)(agent_state).squeeze()

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        Lf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i")
        Lg_h = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")
        Lg_h = Lg_h.reshape((self.n_agents, -1))

        u_lb, u_ub = self._env.action_lim()
        # u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        # u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ref = self._env.u_ref(graph)
        u_ref_flat = u_ref.reshape(-1)

        # construct QP
        u_opt, r = [], []
        for i in range(self.n_agents):
            H = jnp.eye(self._env.action_dim + 1, dtype=jnp.float32)
            H = H.at[-1, -1].set(H[-1, -1] * 10.0)
            g = jnp.concatenate([-u_ref[i], relax_penalty * jnp.ones(1)])
            C = -jnp.concatenate([Lg_h[i, i * self.action_dim: (i + 1) * self.action_dim], jnp.ones(1)], axis=0)[None, :]
            b = Lf_h[i] + self.alpha * h[i]
            # assume other agents follow the reference control
            b += jnp.sum(Lg_h[i, :self.action_dim * i] * u_ref_flat[:i * self.action_dim]) + \
                 jnp.sum(Lg_h[i, self.action_dim * (i + 1):] * u_ref_flat[(i + 1) * self.action_dim:])
            b = jnp.array([b], dtype=jnp.float32)

            r_lb = jnp.array([0.], dtype=jnp.float32)
            r_ub = jnp.array([jnp.inf], dtype=jnp.float32)

            l_box = jnp.concatenate([u_lb, r_lb], axis=0)
            u_box = jnp.concatenate([u_ub, r_ub], axis=0)

            qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
            settings = JaxProxQP.Settings.default()
            settings.max_iter = 100
            solver = JaxProxQP(qp, settings)
            sol = solver.solve()

            assert sol.x.shape == (self.action_dim + 1,)

            u_i, r_i = sol.x[:-1], sol.x[-1]
            u_opt.append(u_i)
            r.append(r_i)

        u_opt = jnp.stack(u_opt, axis=0)
        r = jnp.stack(r, axis=0)

        #
        # H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents, dtype=jnp.float32)
        # H = H.at[-self.n_agents:, -self.n_agents:].set(H[-self.n_agents:, -self.n_agents:] * 10.0)
        # g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents)])
        # C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents)], axis=1)
        # b = Lf_h + self.alpha * h
        #
        # r_lb = jnp.array([0.] * self.n_agents, dtype=jnp.float32)
        # r_ub = jnp.array([jnp.inf] * self.n_agents, dtype=jnp.float32)
        #
        # l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        # u_box = jnp.concatenate([u_ub, r_ub], axis=0)
        #
        # qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        # settings = JaxProxQP.Settings.default()
        # settings.max_iter = 100
        # solver = JaxProxQP(qp, settings)
        # sol = solver.solve()
        #
        # assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents,)
        # u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents:]
        # u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    def save(self, save_dir: str, step: int):
        raise NotImplementedError

    def load(self, load_dir: str, step: int):
        raise NotImplementedError
