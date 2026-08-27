# Quick test script — run this standalone
import jax
import jax.numpy as jnp
import jax.random as jr
from gcbfplus.nn.gnn import GNN
from gcbfplus.algo.module.cbf import CBF
from gcbfplus.utils.graph import GraphsTuple

# Minimal dummy graph
n_agents = 5
node_dim = 8
edge_dim = 4
key = jr.PRNGKey(0)

dummy_graph = GraphsTuple(
    nodes=jnp.zeros((n_agents, node_dim)),
    edges=jnp.zeros((n_agents, edge_dim)),
    states=jnp.zeros((n_agents, node_dim)),
    n_node=jnp.array(n_agents),
    n_edge=jnp.array(n_agents),
    senders=jnp.arange(n_agents),
    receivers=jnp.arange(n_agents),
    node_type=jnp.zeros((n_agents,)),
    env_states=jnp.zeros((n_agents,)),
)

cbf = CBF(node_dim=node_dim, edge_dim=edge_dim, n_agents=n_agents, gnn_layers=1)
params = cbf.net.init(key, dummy_graph, n_agents)

# Test standard path — must still work
h = cbf.get_cbf(params, dummy_graph)
print(f"h shape: {h.shape}")           # expect (5, 1)

# Test new path
h2, attn = cbf.get_cbf_with_attn(params, dummy_graph)
print(f"h2 shape: {h2.shape}")         # expect (5, 1)
print(f"attn shape: {attn.shape}")     # expect (2, 5) = (n_heads, num_edges)
print(f"attn sum per receiver: {attn}")