import flax.linen as nn
import functools as ft
import jax.numpy as jnp
import jraph
import jax.tree_util as jtu

from typing import Type, NamedTuple, Callable, Tuple

from ..utils.typing import EdgeAttr, Node, Array
from ..utils.graph import GraphsTuple
from .mlp import MLP, default_nn_init
from .utils import safe_get

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

            aggr_msg = jraph.segment_sum(attn[:, None] * msgs, segment_ids=recv_idx, num_segments=num_segments)
            return aggr_msg

        update_fn = GNNUpdate(message, aggregate, update)
        return update_fn(graph)

##Added CodeBlock
class MHAGNNLayer(nn.Module):
    """GNN layer with multi-head attention aggregation."""
    hid_size_msg: Tuple[int, ...]
    hid_size_aggr: Tuple[int, ...]
    hid_size_update: Tuple[int, ...]
    msg_dim: int
    out_dim: int
    n_heads: int = 4

    @nn.compact
    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        head_dim = self.msg_dim // self.n_heads  # 128 // 4 = 32

        def message(edge_feats: EdgeAttr, sender_feats: Node, receiver_feats: Node) -> Array:
            feats = jnp.concatenate([edge_feats, sender_feats, receiver_feats], axis=-1)
            feats = MLP(hid_sizes=self.hid_size_msg, act=nn.relu, act_final=False, name="msg")(feats)
            feats = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)
            return feats

        def update(node_feats: Node, msgs: Array) -> Array:
            feats = jnp.concatenate([node_feats, msgs], axis=-1)
            feats = MLP(hid_sizes=self.hid_size_update, act=nn.relu, act_final=False, name="update")(feats)
            feats = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)
            return feats

        def aggregate(msgs: Array, recv_idx: Array, num_segments: int) -> Array:
            head_dim = self.msg_dim // self.n_heads  # 128 // 4 = 32
            head_outputs = []
            for k in range(self.n_heads):
                # MLP2 copy — attention score for head k
                gate_feats = MLP(
                    hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name=f"attn_{k}"
                )(msgs)
                gate_feats = nn.Dense(
                    1, kernel_init=default_nn_init(), name=f"attn_score_{k}"
                )(gate_feats).squeeze(-1)
                attn = jraph.segment_softmax(gate_feats, segment_ids=recv_idx, num_segments=num_segments)

                # MLP3 copy — value projection for head k (full MLP, not just Dense)
                values = MLP(
                    hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name=f"value_mlp_{k}"
                )(msgs)
                values = nn.Dense(
                    head_dim, kernel_init=default_nn_init(), name=f"value_proj_{k}"
                )(values)  # (num_edges, 32)

                aggr = jraph.segment_sum(
                    attn[:, None] * values, segment_ids=recv_idx, num_segments=num_segments
                )  # (num_nodes, 32)
                head_outputs.append(aggr)

            # concat: (num_nodes, 128) → W_o → (num_nodes, 128)
            concat = jnp.concatenate(head_outputs, axis=-1)
            out = nn.Dense(self.msg_dim, kernel_init=default_nn_init(), name="mha_proj")(concat)
            return out


        update_fn = GNNUpdate(message, aggregate, update)
        return update_fn(graph)

##

class GNN(nn.Module):
    msg_dim: int
    hid_size_msg: Tuple[int, ...]
    hid_size_aggr: Tuple[int, ...]
    hid_size_update: Tuple[int, ...]
    out_dim: int
    n_layers: int
    #added
    n_heads: int = 4 #change this for number of heads.

    '''@nn.compact
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
            return graph.type_nodes(node_type, n_type)'''
    @nn.compact
    def __call__(self, graph: GraphsTuple, node_type: int = None, n_type: int = None) -> Array:
        for i in range(self.n_layers):
            out_dim = self.out_dim if i == self.n_layers - 1 else self.msg_dim

            if self.n_heads > 1:
                gnn_layer = MHAGNNLayer(
                    hid_size_msg=self.hid_size_msg,
                    hid_size_aggr=self.hid_size_aggr,
                    hid_size_update=self.hid_size_update,
                    msg_dim=self.msg_dim,
                    out_dim=out_dim,
                    n_heads=self.n_heads,
                )
            else:
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

