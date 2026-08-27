import flax.linen as nn
import functools as ft

import ipdb
import jax.numpy as jnp
import jraph
import jax.tree_util as jtu
import numpy as np

from typing import Type, NamedTuple, Callable, Tuple

from jax._src.interpreters.batching import BatchTracer

from ..utils.typing import EdgeAttr, Node, Array
from ..utils.graph import GraphsTuple
from .mlp import MLP, default_nn_init
from .utils import safe_get
import os

save_attn = False

def save_set_attn(v):
    global save_attn
    save_attn = v

class GNNUpdate(NamedTuple):
    message: Callable[[EdgeAttr, Node, Node], Array]
    aggregate: Callable[[Array, Array, int], Array]
    update: Callable[[Node, Array], Array]

    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        assert graph.n_node.shape == tuple()
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, graph.senders), graph.nodes)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, graph.receivers), graph.nodes)

        # if save_attn:
        #     with open('send.csv', 'a') as f:
        #         for i in graph.senders:
        #             if i != graph.n_node - 1:
        #                 f.write(f'{i}, ')
        #         f.write('\n')
        #     with open('recv.csv', 'a') as f1:
        #         with open('dist.csv', 'a') as f2:
        #             for i in range(len(graph.receivers)):
        #                 if graph.receivers[i] != graph.n_node - 1:
        #                     f1.write(f'{graph.receivers[i]}, ')
        #                     f2.write(f'{jnp.linalg.norm(graph.edges[i, :2])}, ')
        #
        #             f1.write('\n')
        #             f2.write('\n')
            # with open('dist.csv', 'a') as f:
            #     for i in graph.receivers:


        # message passing
        edges = self.message(graph.edges, node_feats_send, node_feats_recv)

        # aggregate messages
        aggr_msg = jtu.tree_map(lambda edge: self.aggregate(edge, graph.receivers, graph.nodes.shape[0]), edges)

        # update nodes
        new_node_feats = self.update(graph.nodes, aggr_msg)

        return graph._replace(nodes=new_node_feats)


class GNNLayer(nn.Module):
    msg_net_cls: Type[nn.Module]
    aggr_net_cls: Type[nn.Module]
    update_net_cls: Type[nn.Module]
    msg_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, graph.senders), graph.nodes)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, graph.receivers), graph.nodes)

        feats = jnp.concatenate([graph.edges, node_feats_send, node_feats_recv], axis=-1)
        feats = self.msg_net_cls()(feats)
        edges = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)

        gate_feats = self.aggr_net_cls()(edges)
        gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
        attn = jraph.segment_softmax(gate_feats, segment_ids=graph.receivers, num_segments=graph.nodes.shape[0])
        aggr_msg = jraph.segment_sum(attn[:, None] * edges, segment_ids=graph.receivers, num_segments=graph.nodes.shape[0])   
        feats = jnp.concatenate([graph.nodes, aggr_msg], axis=-1)
        feats = self.update_net_cls()(feats)
        new_nodes = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)

        return graph._replace(nodes=new_nodes)

    @nn.compact
    def call_with_attn(self, graph: GraphsTuple) -> Tuple[GraphsTuple, Array]:
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, graph.senders), graph.nodes)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, graph.receivers), graph.nodes)

        feats = jnp.concatenate([graph.edges, node_feats_send, node_feats_recv], axis=-1)
        feats = self.msg_net_cls()(feats)
        edges = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)

        gate_feats = self.aggr_net_cls()(edges)
        gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
        attn = jraph.segment_softmax(gate_feats, segment_ids=graph.receivers, num_segments=graph.nodes.shape[0])
        aggr_msg = jraph.segment_sum(attn[:, None] * edges, segment_ids=graph.receivers, num_segments=graph.nodes.shape[0])

        feats = jnp.concatenate([graph.nodes, aggr_msg], axis=-1)
        feats = self.update_net_cls()(feats)
        new_nodes = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)

        attn_weights = attn[None, :]  # (1, num_edges)
        return graph._replace(nodes=new_nodes), attn_weights



class GNN(nn.Module):
    msg_dim: int
    hid_size_msg: Tuple[int, ...]
    hid_size_aggr: Tuple[int, ...]
    hid_size_update: Tuple[int, ...]
    out_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, node_type: int = None, n_type: int = None) -> Array:
        for i in range(self.n_layers):
            out_dim = self.out_dim if i == self.n_layers - 1 else self.msg_dim
            msg_net = ft.partial(MLP, hid_sizes=self.hid_size_msg, act=nn.relu, act_final=False, name="msg")
            attn_net = ft.partial(MLP, hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name="attn")
            update_net = ft.partial(MLP, hid_sizes=self.hid_size_update, act=nn.relu, act_final=False, name="update")
            gnn_layer = GNNLayer(
                msg_net_cls=msg_net,
                aggr_net_cls=attn_net,
                update_net_cls=update_net,
                msg_dim=self.msg_dim,
                out_dim=out_dim,
            )
            graph = gnn_layer(graph)
        if node_type is None:
            return graph.nodes
        else:
            return graph.type_nodes(node_type, n_type)

    @nn.compact
    def call_with_attn(self, graph: GraphsTuple, node_type: int = None, n_type: int = None) -> Tuple[Array, Array]:
        attn_weights_last = None
        for i in range(self.n_layers):
            out_dim = self.out_dim if i == self.n_layers - 1 else self.msg_dim
            msg_net = ft.partial(MLP, hid_sizes=self.hid_size_msg, act=nn.relu, act_final=False, name="msg")
            attn_net = ft.partial(MLP, hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name="attn")
            update_net = ft.partial(MLP, hid_sizes=self.hid_size_update, act=nn.relu, act_final=False, name="update")
            gnn_layer = GNNLayer(
                msg_net_cls=msg_net,
                aggr_net_cls=attn_net,
                update_net_cls=update_net,
                msg_dim=self.msg_dim,
                out_dim=out_dim,
            )
            graph, attn_weights_last = gnn_layer.call_with_attn(graph)
        nodes = graph.nodes if node_type is None else graph.type_nodes(node_type, n_type)
        return nodes, attn_weights_last  # attn_weights_last: (1, num_edges)
