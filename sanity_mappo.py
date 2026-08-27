# sanity_mappo.py — run before launching training
import jax
import jax.numpy as jnp
import jax.random as jr
import jraph

from model.utils.graph import GraphsTuple
from model.algo.module.policy import PPOPolicy
from model.algo.mappo import compute_radial_loss

from model.algo.module.policy import TanhNormal

# ── Config ────────────────────────────────────────────────────────────────────
N_AGENTS   = 8
NODE_DIM   = 8    # adjust to match your env
EDGE_DIM   = 8    # adjust to match your env
STATE_DIM  = 4    # adjust to match your env
ACTION_DIM = 2
GNN_LAYERS = 1
KEY = jr.PRNGKey(42)

# ── Dummy graph ───────────────────────────────────────────────────────────────
senders   = jnp.array([i for i in range(N_AGENTS) for j in range(N_AGENTS) if i != j])
receivers = jnp.array([j for i in range(N_AGENTS) for j in range(N_AGENTS) if i != j])
num_edges = senders.shape[0]

key, subkey = jr.split(KEY)
dummy_graph = GraphsTuple(
    nodes=jr.normal(subkey, (N_AGENTS, NODE_DIM)),
    edges=jr.normal(subkey, (num_edges, EDGE_DIM)),
    states=jr.normal(subkey, (N_AGENTS, STATE_DIM)),
    n_node=jnp.array(N_AGENTS),
    n_edge=jnp.array(num_edges),
    senders=senders,
    receivers=receivers,
    node_type=jnp.zeros((N_AGENTS, )),
    env_states=jnp.zeros((N_AGENTS,)),
)


# ── Init policy ───────────────────────────────────────────────────────────────
policy = PPOPolicy(
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    n_agents=N_AGENTS,
    action_dim=ACTION_DIM,
    gnn_layers=GNN_LAYERS,
)

key, subkey = jr.split(key)
params = policy.dist.init(subkey, dummy_graph, N_AGENTS)
print(f"[OK] Policy initialised")

# ── Helper: call GNN's call_with_attn through TanhNormal.apply ───────────────
def get_attn(params, graph):
    _, attn_w = policy.dist.apply(
        params, graph, N_AGENTS,
        method=TanhNormal.call_with_attn   # ← class, not instance
    )
    return attn_w[0]

# ── Test 1: get_action ────────────────────────────────────────────────────────
action = policy.get_action(params, dummy_graph)
assert action.shape == (N_AGENTS, ACTION_DIM)
print(f"[OK] get_action shape: {action.shape}")

# ── Test 2: sample_action ─────────────────────────────────────────────────────
key, subkey = jr.split(key)
action2, log_pi = policy.sample_action(params, dummy_graph, subkey)
assert action2.shape == (N_AGENTS, ACTION_DIM)
assert log_pi.shape == (N_AGENTS,)
print(f"[OK] sample_action shape: {action2.shape}, log_pi shape: {log_pi.shape}")

# ── Test 3: eval_action ───────────────────────────────────────────────────────
key, subkey = jr.split(key)
lp, ent = policy.eval_action(params, dummy_graph, action2, subkey)
assert lp.shape == (N_AGENTS,)
assert ent.shape == (N_AGENTS,)
print(f"[OK] eval_action log_pi: {lp.shape}, entropy: {ent.shape}")

# ── Test 4: call_with_attn shape ──────────────────────────────────────────────
attn_flat = get_attn(params, dummy_graph)
assert attn_flat.shape == (num_edges,), f"Expected ({num_edges},), got {attn_flat.shape}"
print(f"[OK] call_with_attn attn shape: {attn_flat.shape}")

# ── Test 5: attn sums to 1 per receiver ──────────────────────────────────────
attn_sum = jraph.segment_sum(attn_flat, segment_ids=receivers, num_segments=N_AGENTS)
assert jnp.allclose(attn_sum, jnp.ones(N_AGENTS), atol=1e-5), f"Attn sum: {attn_sum}"
print(f"[OK] attn sums to 1 per receiver: {attn_sum}")

# ── Test 6: compute_radial_loss scalar ───────────────────────────────────────
loss = compute_radial_loss(attn_flat, dummy_graph.edges, dummy_graph.senders, dummy_graph.node_type)
assert loss.shape == ()
assert jnp.isfinite(loss)
print(f"[OK] compute_radial_loss: {loss:.6f}")

# ── Test 7: radial loss is differentiable ─────────────────────────────────────
def loss_fn(params):
    attn = get_attn(params, dummy_graph)
    return compute_radial_loss(attn, dummy_graph.edges, dummy_graph.senders, dummy_graph.node_type)

grad = jax.grad(loss_fn)(params)
grad_leaves = jax.tree_util.tree_leaves(grad)
assert all(jnp.isfinite(g).all() for g in grad_leaves), "NaN/Inf in gradients!"
grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
assert grad_norm > 0, "Grad norm is zero — no gradient flowing!"
print(f"[OK] Radial loss grad norm: {grad_norm:.6f}")

# ── Test 8: vmap over batch ───────────────────────────────────────────────────
batch_size = 4
batch_graph = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch_size), dummy_graph)
attn_batch = jax.vmap(lambda g: get_attn(params, g))(batch_graph)
assert attn_batch.shape == (batch_size, num_edges)
loss_batch = jnp.mean(jax.vmap(compute_radial_loss)(attn_batch, batch_graph.edges, batch_graph.senders, batch_graph.node_type))
assert loss_batch.shape == ()
print(f"[OK] Batched radial loss: {loss_batch:.6f}")

print("\n=== ALL SANITY CHECKS PASSED ===")