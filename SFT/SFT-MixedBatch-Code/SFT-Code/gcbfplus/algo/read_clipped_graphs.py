"""
Quick script to read and inspect clipped graph NPZ files
"""

import numpy as np
import sys
import os
import glob
from typing import Dict

def load_and_merge_failed_batches(data_dir: str) -> Dict[str, np.ndarray]:
    """
    Reads all npz files from the data_dir, concatenates the samples,
    and shuffles them.
    
    Args:
        data_dir: Directory containing the .npz files.
        
    Returns:
        failed_batch: A dictionary containing concatenated and shuffled data.
                      Keys match the sample keys (without the sample_XXXX prefix).
    """
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return {}
    
    print(f"Found {len(npz_files)} .npz files.")
    
    all_samples = []
    
    # keys to extract for each sample
    # Based on read_clipped_graphs.py output
    keys_to_extract = [
        'graph_nodes', 'graph_edges', 'graph_states', 
        'graph_senders', 'graph_receivers', 'graph_node_type', 
        'graph_n_node', 'graph_n_edge', 'graph_env_states', 
        'safe_mask', 'unsafe_mask', 'u_qp', 'step', 'batch_indices'
    ]
    
    for fw in npz_files:
        print(f"Processing {fw}...")
        try:
            data = np.load(fw, allow_pickle=True)
            num_samples = int(data['num_samples'])
            
            for i in range(num_samples):
                sample_prefix = f"sample_{i:04d}"
                sample_data = {}
                
                # Check if this sample exists (it should based on num_samples)
                # We construct the dictionary for this sample
                for key in keys_to_extract:
                    full_key = f"{sample_prefix}__{key}"
                    if full_key in data:
                        sample_data[key] = data[full_key]
                    else:
                        # Handle missing keys if necessary, or warn
                        pass
                
                if sample_data:
                    all_samples.append(sample_data)
                    
        except Exception as e:
            print(f"Error reading {fw}: {e}")

    print(f"Total samples collected: {len(all_samples)}")
    
    if not all_samples:
        return {}

    # Concatenate
    # We want to stack them to create a batch dimension
    
    merged_batch = {}
    
    # Initialize lists for each key
    batch_lists = {key: [] for key in keys_to_extract}
    
    for sample in all_samples:
        for key in keys_to_extract:
            if key in sample:
                batch_lists[key].append(sample[key])
    
    # Stack and Shuffle
    # To shuffle consistently across valid keys, we generate indices
    num_total = len(all_samples)
    print(f"Shuffling {num_total} samples to ensure I.I.D. distribution...")
    perm_indices = np.random.permutation(num_total)
    
    failed_batch = {}
    
    for key, lst in batch_lists.items():
        if not lst:
            continue
            
        # Check if we can stack
        try:
            # We use np.stack to add a new batch dimension at axis 0
            # If arrays have different shapes, this will fail
            stacked = np.stack(lst, axis=0)
            
            # Shuffle
            failed_batch[key] = stacked[perm_indices]
            
        except ValueError as e:
            print(f"Could not stack key '{key}': {e}")
            # If variable size, maybe keep as object array or list
            failed_batch[key] = np.array(lst, dtype=object)[perm_indices]

    return failed_batch


def read_clipped_npz(filepath):
    """
    Read and display contents of a clipped graph NPZ file.
    
    Args:
        filepath: Path to the .npz file (e.g., 'clipped_graphs_npz/n30_obs8.npz')
    """
    # Load the data
    data = np.load(filepath, allow_pickle=True)
    
    print("="*70)
    print(f"CLIPPED GRAPHS DATA - {filepath}")
    print("="*70)
    
    # Extract metadata
    print("\n📋 Metadata:")
    print(f"  Original agents: {data['num_agents_original']}")
    print(f"  Obstacles: {data['num_obstacles']}")
    print(f"  Clipped agents: {data['num_agents_clipped']}")
    print(f"  Total samples: {data['num_samples']}")
    
    # Get all sample indices
    num_samples = int(data['num_samples'])
    
    print(f"\n📊 Data structure:")
    print(f"  Each sample has 14 keys matching training batch format")
    print(f"  Sample naming: sample_0000, sample_0001, ..., sample_{num_samples-1:04d}")
    
    # Show available keys for first sample
    if num_samples > 0:
        print(f"\n🔑 Keys for sample_0000:")
        sample_prefix = "sample_0000"
        sample_keys = [k.replace(f"{sample_prefix}__", "") for k in data.files if k.startswith(sample_prefix)]
        for key in sorted(sample_keys):
            print(f"    - {key}")
    
    # Show shapes for first sample
    if num_samples > 0:
        print(f"\n📐 Shapes for sample_0000:")
        prefix = "sample_0000"
        print(f"  Step: {data[f'{prefix}__step']}")
        print(f"  Batch indices shape: {data[f'{prefix}__batch_indices'].shape}")
        print(f"  Graph nodes shape: {data[f'{prefix}__graph_nodes'].shape}")
        print(f"  Graph edges shape: {data[f'{prefix}__graph_edges'].shape}")
        print(f"  Graph states shape: {data[f'{prefix}__graph_states'].shape}")
        print(f"  Graph senders shape: {data[f'{prefix}__graph_senders'].shape}")
        print(f"  Graph receivers shape: {data[f'{prefix}__graph_receivers'].shape}")
        print(f"  Graph node_type shape: {data[f'{prefix}__graph_node_type'].shape}")
        print(f"  Graph n_node shape: {data[f'{prefix}__graph_n_node'].shape}")
        print(f"  Graph n_edge shape: {data[f'{prefix}__graph_n_edge'].shape}")
        print(f"  Safe mask shape: {data[f'{prefix}__safe_mask'].shape}")
        print(f"  Unsafe mask shape: {data[f'{prefix}__unsafe_mask'].shape}")
        print(f"  QP actions shape: {data[f'{prefix}__u_qp'].shape}")
    
    # Statistics about collisions
    print("\n🎭 Collision statistics across all samples:")
    safe_total = 0
    unsafe_total = 0
    collision_counts = []
    
    for i in range(num_samples):
        prefix = f"sample_{i:04d}"
        safe_mask = data[f"{prefix}__safe_mask"]
        unsafe_mask = data[f"{prefix}__unsafe_mask"]
        
        safe_total += np.sum(safe_mask)
        unsafe_total += np.sum(unsafe_mask)
        collision_counts.append(np.sum(unsafe_mask))
    
    total_agents = num_samples * 8
    print(f"  Safe agent instances: {safe_total}/{total_agents} ({100*safe_total/total_agents:.1f}%)")
    print(f"  Unsafe agent instances: {unsafe_total}/{total_agents} ({100*unsafe_total/total_agents:.1f}%)")
    print(f"  Avg colliding agents per sample: {np.mean(collision_counts):.2f}")
    print(f"  Max colliding agents in a sample: {np.max(collision_counts)}")
    print(f"  Min colliding agents in a sample: {np.min(collision_counts)}")
    
    # Show first 3 samples in detail
    print(f"\n📍 First 3 samples (detailed):")
    for i in range(min(3, num_samples)):
        prefix = f"sample_{i:04d}"
        
        print(f"\n  Sample {i} ({prefix}):")
        print(f"    Timestep: {data[f'{prefix}__timestep']}")
        print(f"    Failing agent ID: {data[f'{prefix}__failing_agent_id']}")
        print(f"    Safe mask: {data[f'{prefix}__safe_mask']}")
        print(f"    Unsafe mask: {data[f'{prefix}__unsafe_mask']}")
        
        # Show first agent state
        states = data[f'{prefix}__graph_states']
        print(f"    First agent state (x,y,vx,vy): {states[0]}")
        print(f"    Action (u_qp): {data[f'{prefix}__u_qp'][:]}")  # First agent action
    
    # Sample distribution by timestep
    print(f"\n📈 Sample distribution:")
    timesteps = [int(data[f"sample_{i:04d}__timestep"]) for i in range(num_samples)]
    unique_timesteps = len(set(timesteps))
    print(f"  Unique timesteps with failures: {unique_timesteps}/256")
    print(f"  Timestep range: {min(timesteps)} - {max(timesteps)}")
    
    # Failing agent distribution
    failing_agents = [int(data[f"sample_{i:04d}__failing_agent_id"]) for i in range(num_samples)]
    unique_agents = len(set(failing_agents))
    print(f"  Unique failing agents: {unique_agents}/{data['num_agents_original']}")
    
    print("\n" + "="*70)
    print("✅ Done! Access data with: data['sample_XXXX__field_name']")
    print("="*70)
    
    return data


def extract_sample(data, sample_idx):
    """
    Extract a single sample as a dictionary.
    
    Args:
        data: Loaded NPZ data
        sample_idx: Sample index (0 to num_samples-1)
    
    Returns:
        Dictionary with all fields for that sample
    """
    prefix = f"sample_{sample_idx:04d}"
    sample = {}
    
    for key in data.files:
        if key.startswith(prefix):
            field = key.replace(f"{prefix}__", "")
            sample[field] = data[key]
    
    return sample


def main():
    """
    Main function to read clipped graph NPZ file.
    Usage: python read_clipped_graphs.py <path_to_npz_file>
    """
    if len(sys.argv) < 2:
        print("Usage: python read_clipped_graphs.py <path_to_npz_file>")
        print("Example: python read_clipped_graphs.py clipped_graphs_npz/n30_obs8.npz")
        return
    
    filepath = sys.argv[1]
    
    try:
        data = read_clipped_npz(filepath)
        
        # Optional: Extract and show a specific sample
        print("\n" + "="*70)
        print("EXAMPLE: Extracting sample 0 as dictionary")
        print("="*70)
        sample_0 = extract_sample(data, 0)
        print(f"\nSample 0 keys: {list(sample_0.keys())}")
        print(f"Timestep: {sample_0['timestep']}")
        print(f"Failing agent: {sample_0['failing_agent_id']}")
        print(f"Safe mask: {sample_0['safe_mask']}")
        print(f"Unsafe mask: {sample_0['unsafe_mask']}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
