import flax.linen as nn
import functools as ft

from typing import Type, Tuple
from ...nn.gnn import GNN
from ...nn.mlp import MLP
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params
from ...utils.graph import GraphsTuple


class CBFNet(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Array:
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))
        return x

##Adding this for supervised attention
    @nn.compact
    def call_with_attn(self, obs: GraphsTuple, n_agents: int) -> Tuple[Array, Array]:
        """
        Forward pass returning (h, attn_weights).
        attn_weights shape: (n_heads, num_edges) from last GNN layer.
        """
        x, attn_weights = self.gnn_cls().call_with_attn(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))
        return x, attn_weights

class CBF:

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, gnn_layers: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents

        self.cbf_gnn = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers,
            n_heads=4#Added
        )
        self.cbf_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='CBFHead'
        )
        self.net = CBFNet(
            gnn_cls=self.cbf_gnn,
            head_cls=self.cbf_head
        )

    def get_cbf(self, params: Params, obs: GraphsTuple) -> Array:
        return self.net.apply(params, obs, self.n_agents)

    #Again, added for the training with supw attention
    def get_cbf_with_attn(self, params: Params, obs: GraphsTuple) -> Tuple[Array, Array]:
       """
       Returns (h, attn_weights).
        attn_weights: (n_heads, num_edges) — use for radial velocity loss.
        """
       return self.net.apply(params, obs, self.n_agents, method=CBFNet.call_with_attn)
