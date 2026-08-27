# clipped_logger.py

import os
import numpy as np
import jax.tree_util as jtu
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.trainer.utils import jax2np


def _env_states_to_np(env_states):
    """
    Convert env_states (EnvState NamedTuple) to a numpy object array
    that matches the batch format used by gcbfplus (jax2np on a GraphsTuple).

    The batch stores env_states as a numpy object array of shape (3,) containing:
        [0] agent  - np.ndarray float32  (n_agents, state_dim)
        [1] goal   - np.ndarray float32  (n_agents, state_dim)
        [2] obstacle - NamedTuple (Rectangle / Cuboid / Sphere) with numpy leaf arrays

    This is exactly what jtu.tree_map(np.array, env_states) would NOT produce
    (it converts leaves but keeps the NamedTuple wrapper). Instead, the batch .npy
    file is written by np.save which pickles the NamedTuple and np.load returns it
    as an object array of the NamedTuple's elements.

    We replicate that by converting every JAX leaf to numpy (via jax2np) and then
    keeping the resulting NamedTuple as-is so np.savez can pickle it correctly.
    """
    return jax2np(env_states)


def save_clipped_graphs_npz(clipped_data, num_agents, num_obstacles, 
                             out_dir="clipped_graphs_npz", compress=True):
    """
    Save all clipped graphs from one environment configuration into a single .npz file.
    
    The serialised format for every graph field is identical to the batch format
    produced by gcbfplus's training loop (Batch / Rollout) so that a clipped graph
    can be directly substituted into an existing batch.

    Args:
        clipped_data: List of tuples (timestep, failing_agent_id, clipped_graph, collision_mask, u_qp)
        num_agents: Original environment agent count (e.g., 30)
        num_obstacles: Number of obstacles (e.g., 8)
        out_dir: Directory to save files
        compress: Whether to use compressed .npz format
    
    Returns:
        Path to saved file
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Filename: n{original_agents}_obs{obstacles}.npz
    basename = f"n{int(num_agents)}_obs{int(num_obstacles)}.npz"
    path = os.path.join(out_dir, basename)
    
    # Check if file exists - we'll append to it
    if os.path.exists(path):
        # Load existing data
        existing = np.load(path, allow_pickle=True)
        serial = {k: existing[k] for k in existing.files}
        num_existing_samples = int(serial.get("num_samples", 0))
    else:
        serial = {}
        num_existing_samples = 0
    
    # Store metadata
    serial["num_agents_original"] = np.array(int(num_agents))
    serial["num_obstacles"] = np.array(int(num_obstacles))
    serial["num_agents_clipped"] = np.array(8)  # Always 8 for clipped graphs
    
    # Process each clipped graph tuple
    num_new_samples = len(clipped_data)
    
    for idx, (timestep, failing_agent_id, clipped_graph, collision_mask, u_qp) in enumerate(clipped_data):
        sample_idx = num_existing_samples + idx
        prefix = f"sample_{sample_idx:04d}"
        
        # Store tuple metadata
        serial[f"{prefix}__timestep"] = np.array(int(timestep))
        serial[f"{prefix}__failing_agent_id"] = np.array(int(failing_agent_id))
        
        # Store all graph fields — dtype / shape must match the batch format
        serial[f"{prefix}__graph_nodes"] = np.array(clipped_graph.nodes, dtype=np.float32)
        serial[f"{prefix}__graph_edges"] = np.array(clipped_graph.edges, dtype=np.float32)
        serial[f"{prefix}__graph_states"] = np.array(clipped_graph.states, dtype=np.float32)
        serial[f"{prefix}__graph_senders"] = np.array(clipped_graph.senders, dtype=np.int32)
        serial[f"{prefix}__graph_receivers"] = np.array(clipped_graph.receivers, dtype=np.int32)
        serial[f"{prefix}__graph_node_type"] = np.array(clipped_graph.node_type, dtype=np.int32)
        serial[f"{prefix}__graph_n_node"] = np.array(clipped_graph.n_node, dtype=np.int32)
        serial[f"{prefix}__graph_n_edge"] = np.array(clipped_graph.n_edge, dtype=np.int32)
        
        # env_states — must match batch format exactly.
        # In the batch, env_states is stored as a pickled EnvState NamedTuple
        # (numpy object array of shape (3,) = [agent, goal, obstacle]).
        # We store it the same way: convert JAX leaves to numpy, then let
        # np.savez pickle the NamedTuple via dtype=object.
        if clipped_graph.env_states is not None:
            env_states_np = _env_states_to_np(clipped_graph.env_states)
            serial[f"{prefix}__graph_env_states"] = np.array(env_states_np, dtype=object)
        else:
            # Even when None, store a sentinel so the loader knows.
            serial[f"{prefix}__graph_env_states"] = np.array(None, dtype=object)
        
        # Derive safe_mask and unsafe_mask from collision_mask
        # collision_mask: True = agent is colliding (unsafe)
        # safe_mask: agents that are NOT colliding
        # unsafe_mask: agents that ARE colliding
        unsafe_mask = collision_mask  # True for colliding agents
        safe_mask = ~collision_mask   # True for safe agents
        
        serial[f"{prefix}__safe_mask"] = np.array(safe_mask, dtype=bool)
        serial[f"{prefix}__unsafe_mask"] = np.array(unsafe_mask, dtype=bool)
        
        # Store u_qp (shape: 8, 2)
        serial[f"{prefix}__u_qp"] = np.array(u_qp, dtype=np.float32)
        
        # step and batch_indices - training metadata placeholders
        serial[f"{prefix}__step"] = np.array(0, dtype=np.int32)
        serial[f"{prefix}__batch_indices"] = np.array([-1], dtype=np.int32)
    
    # Update total sample count
    serial["num_samples"] = np.array(num_existing_samples + num_new_samples)
    
    # Save (allow_pickle is needed for env_states NamedTuples)
    if compress:
        np.savez_compressed(path, **serial)
    else:
        np.savez(path, **serial)
    
    print(f"✅ Saved {num_new_samples} clipped graphs to {path} (total: {serial['num_samples']})")
    return path


def load_clipped_graphs_npz(path):
    """
    Load clipped graphs from .npz file.
    
    Returns:
        dict with:
            - 'metadata': {num_agents_original, num_obstacles, num_agents_clipped, num_samples}
            - 'samples': list of dicts, each containing all fields for one sample
    """
    data = np.load(path, allow_pickle=True)
    
    # Extract metadata
    metadata = {
        'num_agents_original': int(data['num_agents_original']),
        'num_obstacles': int(data['num_obstacles']),
        'num_agents_clipped': int(data['num_agents_clipped']),
        'num_samples': int(data['num_samples'])
    }
    
    # Group samples by prefix
    samples = []
    for i in range(metadata['num_samples']):
        prefix = f"sample_{i:04d}"
        sample = {}
        
        for key in data.files:
            if key.startswith(prefix):
                field = key.replace(f"{prefix}__", "")
                sample[field] = data[key]
        
        if sample:  # Only add if we found data for this sample
            samples.append(sample)
    
    return {'metadata': metadata, 'samples': samples}


def reconstruct_graph_from_sample(sample):
    """
    Reconstruct a GraphsTuple from a loaded sample dict, matching the batch format
    used by gcbfplus so the graph can be directly inserted into a Batch.
    
    Args:
        sample: Dict containing all graph fields (from load_clipped_graphs_npz)
    
    Returns:
        GraphsTuple object with env_states preserved (EnvState NamedTuple or None)
    """
    # Recover env_states: stored as a pickled object — may be an EnvState
    # NamedTuple or a numpy scalar wrapping None.
    env_states_raw = sample.get('graph_env_states', None)
    if env_states_raw is not None:
        # np.load with allow_pickle wraps the NamedTuple in a 0-d object array;
        # extract the actual value.
        env_states = env_states_raw.item() if env_states_raw.ndim == 0 else env_states_raw
    else:
        env_states = None

    sample['graph_env_states'] = env_states  # Replace raw with extracted env_states

    return GraphsTuple(
        nodes=sample['graph_nodes'],
        edges=sample['graph_edges'],
        states=sample['graph_states'],
        senders=sample['graph_senders'],
        receivers=sample['graph_receivers'],
        node_type=sample['graph_node_type'],
        n_node=sample['graph_n_node'],
        n_edge=sample['graph_n_edge'],
        env_states=sample['graph_env_states']
    )
