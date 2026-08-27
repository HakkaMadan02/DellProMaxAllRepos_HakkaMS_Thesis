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
    """
    Log only unsafe agents (safety_label=0) at unsafe timesteps to a single CSV file.
    
    Args:
        rollout_result: RolloutResult containing Tp1_graph with agent positions
        safety_labels: Array of shape (timesteps, num_agents) with binary labels (1=safe, 0=unsafe)
        cbf_values: Tuple (Tb_x, Tb_y, Tbb_h) from get_bb_cbf_fn() - shapes (T, grid_x, grid_y)
        cbf_in_values: Tbb_h_in from get_bb_cbf_in_fn() - shape (T, grid_x, grid_y, 256)
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O)
        csv_path: Path to the CSV file (default: "unsafe_agents_log.csv")
    """
    
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

def log_all_agents_to_csv(
    rollout_result,
    safety_labels,
    cbf_values,
    cbf_in_values,
    episode_num,
    num_agents,
    num_obstacles,
    csv_path="all_agents_log.csv"
):
    """
    Log ALL agents at all timesteps to a single CSV file.
    (Optional function in case you need to log everything later)
    
    Args:
        rollout_result: RolloutResult containing Tp1_graph with agent positions
        safety_labels: Array of shape (timesteps, num_agents) with binary labels (1=safe, 0=unsafe)
        cbf_values: Tuple (Tb_x, Tb_y, Tbb_h) from get_bb_cbf_fn()
        cbf_in_values: Tbb_h_in from get_bb_cbf_in_fn()
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O)
        csv_path: Path to the CSV file (default: "all_agents_log.csv")
    """
    
    # Convert JAX arrays to numpy
    safety_labels_np = np.array(safety_labels)
    
    # Unpack cbf_values tuple
    Tb_x, Tb_y, Tbb_h = cbf_values
    Tbb_h_np = np.array(Tbb_h)
    cbf_in_values_np = np.array(cbf_in_values)
    
    # Extract positions from STATES
    states = np.array(rollout_result.Tp1_graph.states)
    timesteps = safety_labels_np.shape[0]
    positions = states[:timesteps, :, :2]
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Prepare all rows
    rows_to_write = []
    
    # With grid 1x1
    grid_x_idx = 0
    grid_y_idx = 0
    
    for t in range(timesteps):
        for agent_id in range(num_agents):
            pos_x = positions[t, agent_id, 0]
            pos_y = positions[t, agent_id, 1]
            
            # Single grid position
            x_features = cbf_in_values_np[t, agent_id, 0, 0, :]
            h_value = Tbb_h_np[t, agent_id, 0, 0]
            safety_label = int(safety_labels_np[t, agent_id])
            
            row = [
                episode_num,
                t,
                agent_id,
                num_agents,
                num_obstacles,
                float(pos_x),
                float(pos_y)
            ] + x_features.tolist() + [float(h_value), safety_label]
            
            rows_to_write.append(row)
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            header = ['episode', 'timestep', 'agent_id', 'N', 'O', 'pos_x', 'pos_y']
            header += [f'X{i+1}' for i in range(256)]
            header += ['h_value', 'safety_label']
            writer.writerow(header)
        
        writer.writerows(rows_to_write)
    
    print(f"Logged {len(rows_to_write)} agent entries to {csv_path}")

def log_safety_features_dict_to_csv(
    SFT_features,
    episode_num,
    num_agents,
    num_obstacles,
    csv_path="safety_features_SFT_log.csv"
):

    file_exists = os.path.exists(csv_path)

    # Serialize the dict; convert non-JSON types to strings
    def default_serializer(obj):
        # GraphsTuple, JAX arrays, etc. will fall back to str()
        return str(obj)

    safety_features_str = json.dumps(SFT_features, default=default_serializer)

    row = [episode_num, num_agents, num_obstacles, safety_features_str]

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            header = ["episode", "N", "O", "safety_features_SFT"]
            writer.writerow(header)

        writer.writerow(row)

    print(f"Logged safety_features dict for episode {episode_num} to {csv_path}")

def log_safety_features_dict_to_pickle(
    SFT_features,
    episode_num,
    num_agents,
    num_obstacles,
    pickle_path="safety_features_SFT_log.pkl"
):
    """
    Log the safety_features dict to a pickle file (appending binary).
    Each call appends a new independent dictionary object to the file.
    
    Args:
        SFT_features: dictionary returned by safety_labeller
        episode_num: Current episode number
        num_agents: Number of agents (N)
        num_obstacles: Number of obstacles (O)
        pickle_path: Path to pickle file (default: "safety_features_SFT_log.pkl")
    """
    # Create a wrapper dictionary to keep metadata with the features
    data_entry = {
        "episode": episode_num,
        "N": num_agents,
        "O": num_obstacles,
        "safety_features_SFT": SFT_features
    }

    # 'ab' mode opens the file for Appending in Binary format
    with open(pickle_path, 'ab') as f:
        pickle.dump(data_entry, f)

    print(f"Pickled safety_features dict for episode {episode_num} to {pickle_path}")