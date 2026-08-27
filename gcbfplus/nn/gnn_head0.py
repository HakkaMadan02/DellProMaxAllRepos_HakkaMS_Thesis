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


class GNNUpdate(NamedTuple):
    message: Callable[[EdgeAttr, Node, Node], Array]
    aggregate: Callable[[Array, Array, int], Array]
    update: Callable[[Node, Array], Array]

    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        assert graph.n_node.shape == tuple()
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, graph.senders), graph.nodes)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, graph.receivers), graph.nodes)

        edges = self.message(graph.edges, node_feats_send, node_feats_recv)
        aggr_msg = jtu.tree_map(lambda edge: self.aggregate(edge, graph.receivers, graph.nodes.shape[0]), edges)
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

        attn_weights = attn[None, :]  # (1, num_edges) — matches MHA format
        return graph._replace(nodes=new_nodes), attn_weights


class MHAGNNLayer(nn.Module):
    """GNN layer with multi-head attention aggregation."""
    hid_size_msg: Tuple[int, ...]
    hid_size_aggr: Tuple[int, ...]
    hid_size_update: Tuple[int, ...]
    msg_dim: int
    out_dim: int
    n_heads: int = 1

    @nn.compact
    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        """Standard forward pass — returns updated graph only."""
        updated_graph, _ = self._forward(graph)
        return updated_graph

    @nn.compact
    def call_with_attn(self, graph: GraphsTuple) -> Tuple[GraphsTuple, Array]:
        """Forward pass that also returns attention weights (n_heads, num_edges)."""
        return self._forward(graph)

    def _forward(self, graph: GraphsTuple) -> Tuple[GraphsTuple, Array]:
        head_dim = self.msg_dim // self.n_heads

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

        def aggregate(msgs: Array, recv_idx: Array, num_segments: int) -> Tuple[Array, Array]:
            head_dim = self.msg_dim // self.n_heads
            head_outputs = []
            all_attn = []

            for k in range(self.n_heads):
                gate_feats = MLP(
                    hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name=f"attn_{k}"
                )(msgs)
                gate_feats = nn.Dense(
                    1, kernel_init=default_nn_init(), name=f"attn_score_{k}"
                )(gate_feats).squeeze(-1)
                attn = jraph.segment_softmax(gate_feats, segment_ids=recv_idx, num_segments=num_segments)
                all_attn.append(attn)  # (num_edges,)

                values = MLP(
                    hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name=f"value_mlp_{k}"
                )(msgs)
                values = nn.Dense(
                    head_dim, kernel_init=default_nn_init(), name=f"value_proj_{k}"
                )(values)

                aggr = jraph.segment_sum(
                    attn[:, None] * values, segment_ids=recv_idx, num_segments=num_segments
                )
                head_outputs.append(aggr)

            attn_weights = jnp.stack(all_attn, axis=0)  # (n_heads, num_edges)
            concat = jnp.concatenate(head_outputs, axis=-1)
            out = nn.Dense(self.msg_dim, kernel_init=default_nn_init(), name="mha_proj")(concat)
            return out, attn_weights

        # --- message pass ---
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, graph.senders), graph.nodes)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, graph.receivers), graph.nodes)
        edges = message(graph.edges, node_feats_send, node_feats_recv)

        aggr_msg, attn_weights = aggregate(edges, graph.receivers, graph.nodes.shape[0])
        new_nodes = update(graph.nodes, aggr_msg)
        updated_graph = graph._replace(nodes=new_nodes)

        return updated_graph, attn_weights


class GNN(nn.Module):
    msg_dim: int
    hid_size_msg: Tuple[int, ...]
    hid_size_aggr: Tuple[int, ...]
    hid_size_update: Tuple[int, ...]
    out_dim: int
    n_layers: int
    n_heads: int = 1

    @nn.compact
    def __call__(self, graph: GraphsTuple, node_type: int = None, n_type: int = None) -> Array:
        """Standard forward — returns node features only. Unchanged behaviour."""
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
                graph = gnn_layer(graph)   # uses __call__, ignores attn
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

    @nn.compact
    def call_with_attn(self, graph: GraphsTuple, node_type: int = None, n_type: int = None) -> Tuple[Array, Array]:
        """
        Forward pass that returns (node_features, attn_weights).
        attn_weights shape: (n_heads, num_edges) from the LAST layer only.
        Only meaningful when n_heads > 1.
        """
        attn_weights_last = None
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
                graph, attn_weights_last = gnn_layer.call_with_attn(graph)
            else:
                # fallback for single head, no attn returned
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
        return nodes, attn_weights_last