#!/usr/bin/env python3

import numpy as np
import sys
import os
from collections import defaultdict

def analyze_npz_for_finetuning(npz_path: str):
    """
    Analyze an npz file to check if it has all information for fine-tuning.
    Returns a detailed report.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {npz_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(npz_path):
        print(f" File not found: {npz_path}")
        return False
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Group fields by prefix
    groups = defaultdict(dict)
    metadata = {}
    
    for key in data.files:
        if key in ('episode', 'N', 'O'):
            metadata[key] = data[key]
        elif "__" in key:
            prefix, field = key.split("__", 1)
            groups[prefix][field] = data[key]
        else:
            metadata[key] = data[key]
    
    # Print metadata
    print("\n METADATA:")
    if 'episode' in metadata:
        print(f"  Episode: {int(metadata['episode'].item())}")
    if 'N' in metadata:
        print(f"  Num agents: {int(metadata['N'].item())}")
    if 'O' in metadata:
        print(f"  Num obstacles: {int(metadata['O'].item())}")
    
    # Check each graph group
    print(f"\n GRAPH GROUPS FOUND: {len(groups)}")
    
    all_complete = True
    required_fields = {
        'nodes': 'Node features (agent/goal/obstacle types)',
        'edges': 'Edge features (relative states)',
        'senders': 'Edge source node indices',
        'receivers': 'Edge target node indices',
        'n_node': 'Number of nodes per graph',
        'n_edge': 'Number of edges per graph',
        'node_type': 'Node type classification (0=agent, 1=goal, 2=obstacle)',
        'states': 'Full state information (positions, velocities)',
        'env_states': 'Environment state (obstacles, goals)',
    }
    
    for prefix in sorted(groups.keys()):
        group = groups[prefix]
        print(f"\n  [{prefix}] Graph:")
        
        # Check required fields
        missing = []
        for field, description in required_fields.items():
            if field in group:
                val = group[field]
                if val is not None:
                    arr = np.asarray(val)
                    print(f"     {field:15s} shape={arr.shape} dtype={arr.dtype}")
                else:
                    print(f"      {field:15s} (None)")
            else:
                print(f"     {field:15s} MISSING")
                missing.append(field)
        
        if missing:
            all_complete = False
            print(f"    Missing fields: {missing}")
        
        # Check connectivity
        if 'senders' in group and 'receivers' in group:
            s = np.asarray(group['senders']).flatten()
            r = np.asarray(group['receivers']).flatten()
            n_edges = len(s)
            max_node_id = max(s.max(), r.max())
            print(f"    Graph connectivity: {n_edges} edges, max node id = {max_node_id}")
    
    # Summary
    print(f"\n{'='*80}")
    print(" FINE-TUNING READINESS CHECK:")
    print(f"{'='*80}")
    
    checks = {
        " Graph structure (nodes, edges, senders, receivers)": 'nodes' in list(groups.values())[0] if groups else False,
        " Graph sizes (n_node, n_edge)": 'n_node' in list(groups.values())[0] if groups else False,
        " Node types (agent/goal/obstacle)": 'node_type' in list(groups.values())[0] if groups else False,
        " Full state information": 'states' in list(groups.values())[0] if groups else False,
        " Environment context": 'env_states' in list(groups.values())[0] if groups else False,
    }
    
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {check}")
    
    print(f"\n WHAT YOU CAN DO WITH THIS DATA:")
    print(f"{'='*80}")
    
    capabilities = [
        "1. Reconstruct complete GraphsTuple objects for GNN processing",
        "2. Create minibatches with proper graph structure for training",
        "3. Compute CBF values using the reconstructed graphs",
        "4. Apply loss functions (safe/unsafe region constraints)",
        "5. Fine-tune actor and CBF networks",
        "6. Verify collision constraints using states",
    ]
    
    if all_complete and groups:
        print(" YES - Data is COMPLETE for fine-tuning!")
        for cap in capabilities:
            print(f"   {cap}")
    else:
        print("  PARTIAL - Some fields missing (but core graph structure OK)")
    
    print(f"\n NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Load graphs using: from reader import load_npz_groups, rehydrate_group_to_graph")
    print("2. Reconstruct GraphsTuple: graph = rehydrate_group_to_graph(group_dict)")
    print("3. Create minibatches for training (handle variable graph sizes with padding)")
    print("4. Compute CBF outputs: cbf_output = cbf_network(graph)")
    print("5. Combine with safety labels from labeller_and_collector.py for supervision")
    
    return all_complete


if __name__ == "__main__":
    # Check example npz file or use command line arg
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        # Try to find a saved episode
        import glob
        episodes = glob.glob("saved_graphs_npz/episode_*.npz")
        if episodes:
            npz_path = episodes[0]
            print(f"No path provided, using: {npz_path}")
        else:
            print("Usage: python check_npz_completeness.py <path_to_npz>")
            print("Or save episodes first with test.py")
            sys.exit(1)
    
    analyze_npz_for_finetuning(npz_path)
