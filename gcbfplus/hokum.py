import os
import yaml
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import jax.random as jr

from gcbfplus.algo import make_algo, GCBFPlus
from gcbfplus.env import make_env
from gcbfplus.utils.utils import mask2index
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.utils import jax_jit_np, jax_vmap

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ---- CONFIG — change these to match your test command ----
PATH       = "/home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+"
ENV_ID     = "DoubleIntegrator"
N_AGENTS   = 8
N_OBS      = 8
AREA_SIZE  = 4.0
SEED       = 1234
# ----------------------------------------------------------

with open(os.path.join(PATH, "config.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

env = make_env(
    env_id=ENV_ID,
    num_agents=N_AGENTS,
    num_obs=N_OBS,
    area_size=AREA_SIZE,
    max_step=None,
    max_travel=None,
)

model_path = os.path.join(PATH, "models")
step = max([int(m) for m in os.listdir(model_path) if m.isdigit()])
print(f"Loading step {step}")

algo = make_algo(
    algo=config.algo,
    env=env,
    node_dim=env.node_dim,
    edge_dim=env.edge_dim,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    n_agents=env.num_agents,
    gnn_layers=config.gnn_layers,
    batch_size=config.batch_size,
    buffer_size=config.buffer_size,
    horizon=config.horizon,
    lr_actor=config.lr_actor,
    lr_cbf=config.lr_cbf,
    alpha=config.alpha,
    eps=0.02,
    inner_epoch=8,
    loss_action_coef=config.loss_action_coef,
    loss_unsafe_coef=config.loss_unsafe_coef,
    loss_safe_coef=config.loss_safe_coef,
    loss_h_dot_coef=config.loss_h_dot_coef,
    max_grad_norm=2.0,
    seed=config.seed,
)
algo.load(model_path, step)

# ---- Get one rollout, grab timestep t=0 ----
act_fn     = jax.jit(algo.act)
rollout_fn = jax_jit_np(env.rollout_fn(act_fn, None))
key        = jr.PRNGKey(SEED)
rollout    = rollout_fn(key)

# Slice timestep 0, convert back to JAX arrays
graph_t = jtu.tree_map(lambda x: jnp.array(x[0]), rollout.Tp1_graph)

# ---- Hessian diagnostic ----
agent_node_mask = graph_t.node_type == 0
agent_node_id   = mask2index(agent_node_mask, algo.n_agents)

def h_aug(new_agent_state):
    new_state = graph_t.states.at[agent_node_id].set(new_agent_state)
    new_graph = algo._env.add_edge_feats(graph_t, new_state)
    return algo.get_cbf(new_graph, params=algo.cbf_train_state.params)

agent_state = graph_t.type_states(type_idx=0, n_type=algo.n_agents)

print("Computing Jacobian...")
h_x    = jax.jacobian(h_aug)(agent_state).squeeze(1)  # (n,n,state_dim)

print("Computing Hessian (this may take ~30s)...")
h_xx_full = jax.hessian(h_aug)(agent_state).squeeze(1)
# shape: (n_agents, n_agents, state_dim, n_agents, state_dim)

print("\n--- Per-agent Hessian diagnostics ---")
print(f"{'Agent':>6} {'h_i':>10} {'||grad||':>10} {'||H||_F':>12} {'max|H|':>10} {'min_eig(H)':>12}")
for i in range(algo.n_agents):
    H_i      = np.array(h_xx_full[i, i, :, i, :])   # (state_dim, state_dim)
    grad_i   = np.array(h_x[i, i, :])
    h_i_val  = float(h_aug(agent_state).squeeze(-1)[i])
    norm_H   = np.linalg.norm(H_i, 'fro')
    max_H    = np.abs(H_i).max()
    eigvals  = np.linalg.eigvalsh(H_i)
    min_eig  = eigvals.min()
    norm_g   = np.linalg.norm(grad_i)
    print(f"{i:>6} {h_i_val:>10.4f} {norm_g:>10.4f} {norm_H:>12.2f} {max_H:>10.2f} {min_eig:>12.2f}")

print("\n--- What eps makes the quadratic term < 1% of h_i? ---")
for i in range(algo.n_agents):
    H_i     = np.array(h_xx_full[i, i, :, i, :])
    h_i_val = abs(float(h_aug(agent_state).squeeze(-1)[i]))
    norm_H  = np.linalg.norm(H_i, 'fro')
    if norm_H > 0 and h_i_val > 0:
        # 0.5 * ||H|| * eps^2 < 0.01 * h_i  =>  eps < sqrt(0.02 * h_i / ||H||)
        safe_eps = np.sqrt(0.02 * h_i_val / norm_H)
        print(f"  Agent {i}: safe eps for 1% quadratic contribution = {safe_eps:.6f}")
    else:
        print(f"  Agent {i}: h_i={h_i_val:.4f}, norm_H={norm_H:.2f} (degenerate)")