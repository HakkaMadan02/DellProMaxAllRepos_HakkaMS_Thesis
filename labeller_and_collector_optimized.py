import numpy as np
import time
from model.utils.graph import GraphsTuple

EPISTEMIC_PENALTIES = 0.9
TIMESTEPS = 256
NUM_OBSTACLES = 32

def safety_labeller(
    T_graph: GraphsTuple,
    num_agents: int,
    sensing_radius: float,
    agent_radius: float,
    actuator_bounds,
):
    start = time.perf_counter()

    # (T, 4480, 4) — one numpy copy, done
    all_edges = np.array(T_graph.edges[:TIMESTEPS])

    alphai    = float(actuator_bounds[0]) * (1 - EPISTEMIC_PENALTIES)
    alphaj_aa = float(actuator_bounds[1]) * (1 - EPISTEMIC_PENALTIES)
    alphaj_ao = float(actuator_bounds[2]) * (1 - EPISTEMIC_PENALTIES)

    # Block layout (matches your original block1/block2/block3):
    #   [0        : n*n]          agent-agent  (1600 edges)
    #   [n*n      : 2*n*n]        agent-goal   (1600 edges, skipped)
    #   [2*n*n    : 2*n*n + n*32] agent-obs    (1280 edges)

    aa_start = 0
    aa_end   = num_agents * num_agents
    ao_start = 2 * num_agents * num_agents
    ao_end   = ao_start + num_agents * NUM_OBSTACLES

    # Agent-Agent: (T, n*n, 4) -> (T, n, n, 4)
    aa = all_edges[:, aa_start:aa_end, :].reshape(TIMESTEPS, num_agents, num_agents, 4)
    Px, Py, Vx, Vy = aa[...,0], aa[...,1], aa[...,2], aa[...,3]
    dist_aa = np.sqrt(Px**2 + Py**2) + 1e-6
    rv_aa   = (Px*Vx + Py*Vy) / dist_aa
    d_aa    = dist_aa - 1e-6
    msd_aa  = rv_aa**2 / (2.0 * (alphai + alphaj_aa) + 1e-9)
    unsafe_pair_aa = (
        ~(d_aa >= sensing_radius) &
        ((rv_aa < 0) & (d_aa < 2.0*agent_radius + msd_aa) |
         (rv_aa == 0) & (d_aa < 2.0*agent_radius))
    )
    unsafe_pair_aa[:, np.eye(num_agents, dtype=bool)] = False
    unsafe_agent_aa = unsafe_pair_aa.any(axis=-1)           # (T, n)

    # Agent-Obstacle: (T, n*32, 4) -> (T, n, 32, 4)
    ao = all_edges[:, ao_start:ao_end, :].reshape(TIMESTEPS, num_agents, NUM_OBSTACLES, 4)
    Px_o, Py_o, Vx_o, Vy_o = ao[...,0], ao[...,1], ao[...,2], ao[...,3]
    dist_ao = np.sqrt(Px_o**2 + Py_o**2) + 1e-6
    rv_ao   = (Px_o*Vx_o + Py_o*Vy_o) / dist_ao
    d_ao    = dist_ao - 1e-6
    msd_ao  = rv_ao**2 / (2.0 * (alphai + alphaj_ao) + 1e-9)
    unsafe_pair_ao = (
        ~(d_ao >= sensing_radius) &
        ((rv_ao < 0) & (d_ao < agent_radius + msd_ao) |
         (rv_ao == 0) & (d_ao < agent_radius))
    )
    unsafe_agent_ao = unsafe_pair_ao.any(axis=-1)           # (T, n)

    # Combine
    unsafe = unsafe_agent_aa | unsafe_agent_ao
    labels = (~unsafe).astype(np.float32)                   # 1=safe, 0=unsafe

    features_array = {
        t: (T_graph, t)
        for t in range(TIMESTEPS)
        if unsafe[t].any()
    }

    elapsed = time.perf_counter() - start
    print(f"Labelling completed in {elapsed:.4f} seconds.")
    return labels, features_array