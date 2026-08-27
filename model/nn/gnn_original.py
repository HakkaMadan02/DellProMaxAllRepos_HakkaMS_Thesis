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
        def message(edge_feats: EdgeAttr, sender_feats: Node, receiver_feats: Node) -> Array:
            feats = jnp.concatenate([edge_feats, sender_feats, receiver_feats], axis=-1)
            feats = self.msg_net_cls()(feats)
            feats = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)
            return feats

        def update(node_feats: Node, msgs: Array) -> Array:
            feats = jnp.concatenate([node_feats, msgs], axis=-1)
            feats = self.update_net_cls()(feats)
            feats = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)
            return feats

        def aggregate(msgs: Array, recv_idx: Array, num_segments: int) -> Array:
            gate_feats = self.aggr_net_cls()(msgs)
            gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
            attn = jraph.segment_softmax(gate_feats, segment_ids=recv_idx, num_segments=num_segments)
            assert attn.shape[0] == msgs.shape[0]

            # import jax
            # jax.debug.print('rev_idx: {}', recv_idx)
            # jax.debug.print('send_idx: {}', graph.senders)
            # jax.debug.print('attn_shape: {}', attn.shape)
            # jax.debug.print('attn: {}', attn)

            # if save_attn:
                # with open("attn.csv", "a") as f:
                #     # ipdb.set_trace()
                #     # f.write(f"{type(attn)},    {type(attn_val)}")
                #     # print(f'{attn_val.shape}')
                #     for i in np.array(recv_idx):
                #         if i != num_segments - 1:
                #             f.write(f'{attn[i]}, ')
                #     f.write("\n")
                # with open('send.csv', 'a') as f:
                #     for i in graph.senders:
                #         if i != graph.n_node - 1:
                #             f.write(f'{i}, ')
                #     f.write('\n')
                # with open('attn.csv', "a") as f_attn:
                #     with open('recv.csv', 'a') as f_recv:
                #         with open('dist.csv', 'a') as f_dist:
                #             with open('send.csv', 'a') as f_send:
                #                 for i in range(len(graph.receivers)):
                #                     if graph.receivers[i] != graph.n_node - 1:
                #                         f_recv.write(f'{graph.receivers[i]}, ')
                #                         f_send.write(f'{graph.senders[i]}, ')
                #                         f_dist.write(f'{jnp.linalg.norm(graph.edges[i, :2])}, ')
                #                         f_attn.write(f'{attn[i]}, ')
                #
                #
                #                 f_recv.write('\n')
                #                 f_send.write('\n')
                #                 f_dist.write('\n')
                #                 f_attn.write('\n')

            # if isinstance(attn, BatchTracer):
            #     attn_val = attn.val
            #     attn_val = np.array(attn_val)
            #
            #     with np.printoptions(threshold=np.inf):
            #         with open("attn.csv", "a") as f:
            #             # ipdb.set_trace()
            #             # f.write(f"{type(attn)},    {type(attn_val)}")
            #             # print(f'{attn_val.shape}')
            #             for i in np.array(recv_idx.val):
            #                 if i != num_segments:
            #                     f.write(f'{attn_val[i]}, ')
            #             f.write("\n")
            # with open("recv.csv", "a") as f:
            #     f.write(f"{recv_idx}\n")
            # with open("send.csv", "a") as f:
            #     f.write(f'{graph.senders}\n')

            aggr_msg = jraph.segment_sum(attn[:, None] * msgs, segment_ids=recv_idx, num_segments=num_segments)
            return aggr_msg

        update_fn = GNNUpdate(message, aggregate, update)
        return update_fn(graph)


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
