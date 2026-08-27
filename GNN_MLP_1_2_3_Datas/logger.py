import os
import csv
import jax.numpy as jnp
import numpy as np
from gcbfplus.utils.graph import GraphsTuple
import json
import pickle


def log_unsafe_agents_to_csv(
    rollout_result,
    safety_labels,
    cbf_values,
    cbf_in_values,
    episode_num,
    num_agents,
    num_obstacles,
    csv_path="unsafe_agents_log.csv"
):
    
    
    # Convert JAX arrays to numpy for processing
    safety_labels_np = np.array(safety_labels)
    
    # Unpack cbf_values tuple
    Tb_x, Tb_y, Tbb_h = cbf_values
    Tbb_h_np = np.array(Tbb_h)  # Shape: (T, grid_x, grid_y) - with grid 1x1 = (T, 1, 1)
    
    # cbf_in_values shape: (T, grid_x, grid_y, 256) - with grid 1x1 = (T, 1, 1, 256)
    cbf_in_values_np = np.array(cbf_in_values)
    
    # Extract positions from graph STATES (not nodes!)
    # Tp1_graph.states shape: (T+1, num_agents, 4) where first 2 are [x, y] positions, last 2 are velocities
    states = np.array(rollout_result.Tp1_graph.states)  # Shape: (T+1, num_agents, 4)
    
    # Use first T timesteps to match cbf_values and safety_labels
    timesteps = safety_labels_np.shape[0]
    positions = states[:timesteps, :, :2]  # Shape: (T, num_agents, 2) - only x, y positions
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    # Prepare rows to write
    rows_to_write = []
    
    # With grid 1x1, we have only one grid position [0, 0]
    grid_x_idx = 0
    grid_y_idx = 0
    
    # Iterate through timesteps and agents
    for t in range(timesteps):
        for agent_id in range(num_agents):
            if safety_labels_np[t, agent_id] == 0:
                pos_x = positions[t, agent_id, 0]
                pos_y = positions[t, agent_id, 1]
                x_features = cbf_in_values_np[t, agent_id, 0, 0, :]  
                h_value = Tbb_h_np[t, agent_id, 0, 0]                
                row = [
                    episode_num, t, agent_id, num_agents, num_obstacles,
                    float(pos_x), float(pos_y)
                ] + x_features.tolist() + [float(h_value), 0]
                rows_to_write.append(row)
    """ for t in range(timesteps):
        for agent_id in range(num_agents):
            # Only log if safety_label is 0 (unsafe)
            if safety_labels_np[t, agent_id] == 0:
                # Extract data for this agent at this timestep from STATES
                pos_x = positions[t, agent_id, 0]
                pos_y = positions[t, agent_id, 1]
                
                # Extract X1...X256 features from the single grid position
                x_features = cbf_in_values_np[t, grid_x_idx, grid_y_idx, :]  # Shape: (256,)
                
                # Extract final h value from the single grid position
                h_value = Tbb_h_np[t, grid_x_idx, grid_y_idx]  # Scalar
                
                # Build row: episode, timestep, agent_id, N, O, pos_x, pos_y, X1...X256, h_value, safety_label
                row = [
                    episode_num,
                    t,
                    agent_id,
                    num_agents,
                    num_obstacles,
                    float(pos_x),
                    float(pos_y)
                ] + x_features.tolist() + [float(h_value), 0]  # safety_label is always 0 for logged entries
                
                rows_to_write.append(row) """
    
    # Write to CSV
    if len(rows_to_write) > 0:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file doesn't exist
            if not file_exists:
                header = ['episode', 'timestep', 'agent_id', 'N', 'O', 'pos_x', 'pos_y']
                header += [f'X{i+1}' for i in range(256)]
                header += ['h_value', 'safety_label']
                writer.writerow(header)
            
            # Write all unsafe agent rows
            writer.writerows(rows_to_write)
        
        print(f"Logged {len(rows_to_write)} unsafe agent entries to {csv_path}")
    else:
        print(f"No unsafe agents found in episode {episode_num}")


def log_balanced_safe_agents_to_csv(
    rollout_result,
    safety_labels,
    cbf_values,
    cbf_in_values,
    episode_num,
    num_agents,
    num_obstacles,
    num_unsafe_samples,
    csv_path="safe_agents_log.csv",
    random_seed=None
):
    """
    Log randomly selected safe agents (safety_label=1) to match the number of unsafe samples.
    This maintains a balanced dataset for training.
    
    Args:
        rollout_result: RolloutResult containing Tp1_graph with agent positions
        safety_labels: Array of shape (timesteps, num_agents) with binary labels (1=safe, 0=unsafe)
        cbf_values: Tuple (Tb_x, Tb_y, Tbb_h) from get_bb_cbf_fn()
        cbf_in_values: Tbb_h_in from get_bb_cbf_in_fn()
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O)
        num_unsafe_samples: Number of unsafe samples logged (to match this count)
        csv_path: Path to the CSV file (default: "unsafe_agents_log.csv")
        random_seed: Optional random seed for reproducibility
    
    Returns:
        int: Number of safe samples actually logged
    """
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert JAX arrays to numpy for processing
    safety_labels_np = np.array(safety_labels)
    
    # Unpack cbf_values tuple
    Tb_x, Tb_y, Tbb_h = cbf_values
    Tbb_h_np = np.array(Tbb_h)
    cbf_in_values_np = np.array(cbf_in_values)
    
    # Extract positions from graph STATES
    states = np.array(rollout_result.Tp1_graph.states)
    timesteps = safety_labels_np.shape[0]
    positions = states[:timesteps, :, :2]
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Grid indices for 1x1 grid
    grid_x_idx = 0
    grid_y_idx = 0
    
    # Collect ALL safe agent indices (timestep, agent_id)
    safe_indices = []
    for t in range(timesteps):
        for agent_id in range(num_agents):
            if safety_labels_np[t, agent_id] == 1:  # Safe agent
                safe_indices.append((t, agent_id))
    
    # Check if we have enough safe samples
    total_safe = len(safe_indices)
    if total_safe == 0:
        print(f"No safe agents found in episode {episode_num}")
        return 0
    
    # Randomly sample from safe indices to match unsafe count
    num_to_sample = min(num_unsafe_samples, total_safe)
    selected_indices = np.random.choice(len(safe_indices), size=num_to_sample, replace=False)
    
    # Prepare rows to write
    rows_to_write = []
    
    for idx in selected_indices:
        t, agent_id = safe_indices[idx]
        
        # Extract data for this agent at this timestep
        pos_x = positions[t, agent_id, 0]
        pos_y = positions[t, agent_id, 1]
        
        # Extract X1...X256 features from the single grid position
        x_features = cbf_in_values_np[t, agent_id, 0, 0, :]
        
        # Extract final h value from the single grid position
        h_value = Tbb_h_np[t, agent_id, 0, 0]
        
        # Build row with safety_label = 1 (safe)
        row = [
            episode_num,
            t,
            agent_id,
            num_agents,
            num_obstacles,
            float(pos_x),
            float(pos_y)
        ] + x_features.tolist() + [float(h_value), 1]  # safety_label is 1 for safe agents
        
        rows_to_write.append(row)
    
    # Write to CSV
    if len(rows_to_write) > 0:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file doesn't exist
            if not file_exists:
                header = ['episode', 'timestep', 'agent_id', 'N', 'O', 'pos_x', 'pos_y']
                header += [f'X{i+1}' for i in range(256)]
                header += ['h_value', 'safety_label']
                writer.writerow(header)
            
            # Write all selected safe agent rows
            writer.writerows(rows_to_write)
        
        print(f"Logged {len(rows_to_write)} safe agent entries (balanced with {num_unsafe_samples} unsafe) to {csv_path}")
    
    return len(rows_to_write)

def log_unsafe_agents_gnn_update(
    rollout_result,
    safety_labels,
    cbf_values,
    gnn_update_features,
    episode_num,
    num_agents,
    num_obstacles,
    csv_path="unsafe_agents_gnn_update.csv"
):
    """
    Log unsafe agents using Update MLP (node-based) for failed agents at unsafe timesteps.
    
    Args:
        rollout_result: RolloutResult containing Tp1_graph
        safety_labels: Array of shape (timesteps, num_agents) with binary labels (1=safe, 0=unsafe)
        cbf_values: Tuple (Tb_x, Tb_y, Tbb_h) from get_bb_cbf_fn()
        gnn_update_features: Update MLP intermediate output shape (timesteps, num_nodes, feature_dim)
                            First num_agents nodes correspond to agents
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O)
        csv_path: Path to the CSV file (default: "unsafe_agents_gnn_update.csv")
    """
    
    # Convert JAX arrays to numpy for processing
    safety_labels_np = np.array(safety_labels)
    
    # Unpack cbf_values tuple
    Tb_x, Tb_y, Tbb_h = cbf_values
    Tbb_h_np = np.array(Tbb_h)  # Shape: (T, num_agents, 1, 1)
    
    # GNN Update features shape: (T, num_nodes, feature_dim)
    gnn_update_np = np.array(gnn_update_features)
    feature_dim = gnn_update_np.shape[-1]
    
    timesteps = safety_labels_np.shape[0]
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    # Prepare rows to write
    rows_to_write = []
    
    # Iterate through timesteps and agents
    for t in range(timesteps):
        for agent_id in range(num_agents):
            if safety_labels_np[t, agent_id] == 0:  # Unsafe agent
                # Agent node is at index agent_id in the node list
                gnn_features = gnn_update_np[t, agent_id, :]  # Shape: (feature_dim,)
                h_value = Tbb_h_np[t, agent_id, 0, 0]
                
                # Build row: episode, timestep, agent_id, N, O, X1...Xn, h_value, safety_label
                row = [
                    episode_num,
                    t,
                    agent_id,
                    num_agents,
                    num_obstacles
                ] + gnn_features.tolist() + [float(h_value), 0]  # safety_label is 0 for unsafe
                
                rows_to_write.append(row)
    
    # Write to CSV
    if len(rows_to_write) > 0:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file doesn't exist
            if not file_exists:
                header = ['episode', 'timestep', 'agent_id', 'N', 'O']
                header += [f'X{i+1}' for i in range(feature_dim)]
                header += ['h_value', 'safety_label']
                writer.writerow(header)
            
            # Write all unsafe agent rows
            writer.writerows(rows_to_write)
        
        print(f"Logged {len(rows_to_write)} unsafe Update entries to {csv_path}")
    else:
        print(f"No unsafe agents found in episode {episode_num}")


def log_balanced_safe_agents_gnn_update(
    rollout_result,
    safety_labels,
    cbf_values,
    gnn_update_features,
    episode_num,
    num_agents,
    num_obstacles,
    num_unsafe_samples,
    csv_path="safe_agents_gnn_update.csv",
    random_seed=None
):
    """
    Log randomly selected safe agents using Update MLP to match the number of unsafe samples.
    This maintains a balanced dataset for training.
    
    Args:
        rollout_result: RolloutResult containing Tp1_graph
        safety_labels: Array of shape (timesteps, num_agents) with binary labels (1=safe, 0=unsafe)
        cbf_values: Tuple (Tb_x, Tb_y, Tbb_h) from get_bb_cbf_fn()
        gnn_update_features: Update MLP intermediate output shape (timesteps, num_nodes, feature_dim)
                            First num_agents nodes correspond to agents
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O)
        num_unsafe_samples: Number of unsafe samples logged (to match this count)
        csv_path: Path to the CSV file (default: "safe_agents_gnn_update.csv")
        random_seed: Optional random seed for reproducibility
    
    Returns:
        int: Number of safe samples actually logged
    """
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert JAX arrays to numpy for processing
    safety_labels_np = np.array(safety_labels)
    
    # Unpack cbf_values tuple
    Tb_x, Tb_y, Tbb_h = cbf_values
    Tbb_h_np = np.array(Tbb_h)
    
    # GNN Update features shape: (T, num_nodes, feature_dim)
    gnn_update_np = np.array(gnn_update_features)
    feature_dim = gnn_update_np.shape[-1]
    
    timesteps = safety_labels_np.shape[0]
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Collect ALL safe agent indices (timestep, agent_id)
    safe_indices = []
    for t in range(timesteps):
        for agent_id in range(num_agents):
            if safety_labels_np[t, agent_id] == 1:  # Safe agent
                safe_indices.append((t, agent_id))
    
    # Check if we have enough safe samples
    total_safe = len(safe_indices)
    if total_safe == 0:
        print(f"No safe agents found in episode {episode_num}")
        return 0
    
    # Randomly sample from safe indices to match unsafe count
    num_to_sample = min(num_unsafe_samples, total_safe)
    selected_indices = np.random.choice(len(safe_indices), size=num_to_sample, replace=False)
    
    # Prepare rows to write
    rows_to_write = []
    
    for idx in selected_indices:
        t, agent_id = safe_indices[idx]
        
        # Agent node is at index agent_id in the node list
        gnn_features = gnn_update_np[t, agent_id, :]  # Shape: (feature_dim,)
        h_value = Tbb_h_np[t, agent_id, 0, 0]
        
        # Build row with safety_label = 1 (safe)
        row = [
            episode_num,
            t,
            agent_id,
            num_agents,
            num_obstacles
        ] + gnn_features.tolist() + [float(h_value), 1]  # safety_label is 1 for safe
        
        rows_to_write.append(row)
    
    # Write to CSV
    if len(rows_to_write) > 0:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file doesn't exist
            if not file_exists:
                header = ['episode', 'timestep', 'agent_id', 'N', 'O']
                header += [f'X{i+1}' for i in range(feature_dim)]
                header += ['h_value', 'safety_label']
                writer.writerow(header)
            
            # Write all selected safe agent rows
            writer.writerows(rows_to_write)
        
        print(f"Logged {len(rows_to_write)} safe Update entries (balanced with {num_unsafe_samples} unsafe) to {csv_path}")
    
    return len(rows_to_write)

def log_unsafe_agents_gnn_msg(
    gnn_msg_features,
    unsafe_data_list,
    episode_num,
    num_agents,
    num_obstacles,
    csv_path="unsafe_agents_gnn_msg_MLP1.csv"
):
    """
    Log unsafe agents using Msg MLP (edge-based) using pre-collected failing edge pairs.
    
    Args:
        gnn_msg_features: Msg MLP intermediate output shape (timesteps, num_edges, feature_dim)
        unsafe_data_list: List of dicts with keys: timestep, agent_i, agent_j, collision_type, cbf_value
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O) - represents obstacles per agent block (32)
        csv_path: Path to the CSV file
    """
    
    # Convert JAX arrays to numpy
    gnn_msg_np = np.array(gnn_msg_features)  # Shape: (T, num_edges, feature_dim)
    feature_dim = gnn_msg_np.shape[-1]
    
    # Edge block structure
    block1_end = num_agents * num_agents  # Agent-Agent block ends here
    block2_start = num_agents * num_agents
    block2_end = block2_start + (num_agents * num_agents)  # Agent-Goal block ends here
    block3_start = block2_end  # Agent-Obstacle block starts here
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Prepare rows to write
    rows_to_write = []
    
    # Process each failing edge pair from unsafe_data_list
    for data_dict in unsafe_data_list:
        t = data_dict['timestep']
        agent_i = data_dict['agent_i']
        agent_j = data_dict['agent_j']
        collision_type = data_dict['collision_type']
        cbf_value = data_dict['cbf_value']
        
        # Map (agent_i, agent_j) to edge index based on block structure
        # collision_type: 0 = agent-agent, 1 = agent-obstacle
        
        if collision_type == 0:
            # Agent-Agent edge: edge_idx = agent_i * num_agents + agent_j
            edge_idx = agent_i * num_agents + agent_j
        else:
            # Agent-Obstacle edge: edge_idx = block3_start + agent_i * num_obstacles + (agent_j - num_agents)
            # Here num_obstacles is used as the block size (32 obstacles per agent)
            obstacle_idx = agent_j - num_agents
            edge_idx = block3_start + (agent_i * 32) + obstacle_idx
        
        # Extract feature vector for this SPECIFIC edge
        gnn_features = gnn_msg_np[t, edge_idx, :]  # Shape: (feature_dim,)
        
        # Build row: episode, timestep, agent_i, agent_j, N, O, edge_idx, collision_type, X1...Xn, cbf_value, safety_label
        row = [
            episode_num,
            t,
            agent_i,
            agent_j,
            num_agents,
            num_obstacles,
            edge_idx,
            int(collision_type)
        ] + gnn_features.tolist() + [float(cbf_value), 0]  # safety_label is 0 for unsafe
        
        rows_to_write.append(row)
    
    # Write to CSV
    if len(rows_to_write) > 0:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file doesn't exist
            if not file_exists:
                header = ['episode', 'timestep', 'agent_i', 'agent_j', 'N', 'O', 'edge_idx', 'collision_type']
                header += [f'X{i+1}' for i in range(feature_dim)]
                header += ['cbf_value', 'safety_label']
                writer.writerow(header)
            
            # Write all unsafe edge rows
            writer.writerows(rows_to_write)
        
        print(f"Logged {len(rows_to_write)} unsafe Msg entries (MLP1) to {csv_path}")
    else:
        print(f"No unsafe agents found in episode {episode_num}")

def log_safe_agents_gnn_msg(
    gnn_msg_features,
    unsafe_data_list,
    cbf_values,
    episode_num,
    num_agents,
    num_obstacles,
    csv_path="safe_agents_gnn_msg_MLP1.csv",
    random_seed=None
):
    """
    Log safe agents using Msg MLP (edge-based) by randomly sampling safe edges.
    
    Randomly samples edges and adds them to safe list if they are NOT in unsafe_data_list.
    Collects same number of safe samples as unsafe samples for balanced dataset.
    
    Args:
        gnn_msg_features: Msg MLP intermediate output shape (timesteps, num_edges, feature_dim)
        unsafe_data_list: List of dicts with unsafe edge pairs (timestep, agent_i, agent_j, collision_type, cbf_value)
        cbf_values: Tuple (Tb_x, Tb_y, Tbb_h) from get_bb_cbf_fn()
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles per agent (always 32)
        csv_path: Path to the CSV file
        random_seed: Optional random seed for reproducibility
    Returns:
        int: Number of safe samples actually logged
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert JAX arrays to numpy
    gnn_msg_np = np.array(gnn_msg_features)  # Shape: (T, num_edges, feature_dim)
    feature_dim = gnn_msg_np.shape[-1]
    
    # Extract CBF values
    Tb_x, Tb_y, Tbb_h = cbf_values
    Tbb_h_np = np.array(Tbb_h)  # Shape: (T, num_agents, 1, 1)
    
    # Edge block structure
    block1_end = num_agents * num_agents
    block2_start = num_agents * num_agents
    block2_end = block2_start + (num_agents * num_agents)
    block3_start = 2 * num_agents * num_agents
    
    # Total number of edges
    total_agent_agent_edges = num_agents * num_agents
    total_agent_goal_edges = num_agents * num_agents
    total_agent_obstacle_edges = num_agents * 32
    total_edges = total_agent_agent_edges + total_agent_goal_edges + total_agent_obstacle_edges
    
    # Get timesteps
    timesteps = gnn_msg_np.shape[0]
    
    # Create a set of unsafe (t, agent_i, agent_j) tuples for fast lookup
    # Note: collision_type is NOT stored because we only care about the (t, i, j) pair
    unsafe_set = set()
    for data_dict in unsafe_data_list:
        t = data_dict['timestep']
        agent_i = data_dict['agent_i']
        agent_j = data_dict['agent_j']
        unsafe_set.add((t, agent_i, agent_j))
    
    num_unsafe_samples = len(unsafe_data_list)
    
    # Collect safe edge pairs
    safe_edges = []
    max_attempts = num_unsafe_samples * 1500  # Prevent infinite loop
    attempts = 0
    
    while len(safe_edges) < num_unsafe_samples and attempts < max_attempts:
        # Randomly pick a timestep
        t = np.random.randint(0, timesteps)
        
        # Randomly pick an edge index
        edge_idx = np.random.randint(0, total_edges)
        
        # Convert edge_idx back to (agent_i, agent_j)
        if edge_idx < block1_end:
            # Agent-Agent edge
            agent_i = edge_idx // num_agents
            agent_j = edge_idx % num_agents
        elif edge_idx < block2_end:
            # Agent-Goal edge - SKIP these (we don't care about goals)
            attempts += 1
            continue
        else:
            # Agent-Obstacle edge
            obstacle_offset = edge_idx - block3_start
            agent_i = obstacle_offset // 32
            obstacle_idx = obstacle_offset % 32
            agent_j = num_agents + obstacle_idx
        
        # Check if this edge is NOT in unsafe set
        if (t, agent_i, agent_j) not in unsafe_set:
            safe_edges.append({
                'timestep': t,
                'agent_i': agent_i,
                'agent_j': agent_j,
                'edge_idx': edge_idx
            })
        
        attempts += 1
    
    # Check if we collected enough safe samples
    if len(safe_edges) < num_unsafe_samples:
        print(f"Warning: Could only collect {len(safe_edges)} safe samples out of {num_unsafe_samples} needed")
    
    # Write to CSV
    file_exists = os.path.exists(csv_path)
    rows_to_write = []
    
    for safe_dict in safe_edges:
        t = safe_dict['timestep']
        agent_i = safe_dict['agent_i']
        agent_j = safe_dict['agent_j']
        edge_idx = safe_dict['edge_idx']
        
        # Extract feature vector
        gnn_features = gnn_msg_np[t, edge_idx, :]
        
        # Extract CBF value for agent_i at timestep t
        cbf_value = float(Tbb_h_np[t, agent_i, 0, 0])
        
        # Build row with safety_label = 1 (safe)
        # NO collision_type column since safe agents don't collide
        row = [
            episode_num,
            t,
            agent_i,
            agent_j,
            num_agents,
            num_obstacles,
            edge_idx
        ] + gnn_features.tolist() + [cbf_value, 1]  # cbf_value extracted, safety_label=1 for safe
        
        rows_to_write.append(row)
    
    if len(rows_to_write) > 0:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file doesn't exist
            if not file_exists:
                header = ['episode', 'timestep', 'agent_i', 'agent_j', 'N', 'O', 'edge_idx']
                header += [f'X{i+1}' for i in range(feature_dim)]
                header += ['cbf_value', 'safety_label']
                writer.writerow(header)
            
            # Write all safe edge rows
            writer.writerows(rows_to_write)
        
        print(f"Logged {len(rows_to_write)} safe Msg entries (MLP1) to {csv_path}")
    else:
        print(f"No safe edges collected for episode {episode_num}")
    
    return len(rows_to_write)
