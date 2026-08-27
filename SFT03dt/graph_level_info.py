#!/usr/bin/env python3
"""
graph_level_info.py

Extract and reconstruct graph-level information from NPZ files for CBF fine-tuning.

This module provides functions to:
1. Extract graph metadata (n_agents, n_obstacles, n_goals)
2. Compute graph-level statistics (node/edge feature aggregations)
3. Parse environment states
4. Create graph context vectors for CBF network input
5. Organize multiple graphs for minibatching with consistent graph-level info

Usage:
    from graph_level_info import extract_graph_context, load_all_graph_contexts
    
    contexts = load_all_graph_contexts("path/to/episode.npz")
    for graph_id, context in contexts.items():
        print(f"Graph {graph_id}: {context.n_agents} agents, {context.n_obstacles} obstacles")
        print(f"  Node feature mean: {context.node_feature_mean}")
        print(f"  Graph context vector: {context.context_vector}")
"""

import numpy as np
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass
import re
from reader import load_npz_groups, rehydrate_group_to_graph, parse_envstate_string


@dataclass
class GraphLevelInfo:
    """Container for graph-level information extracted from a single graph in NPZ."""
    
    graph_id: str  # Prefix used in NPZ (e.g., "0", "29", "42")
    
    # Graph structure metadata
    n_agents: int  # Agents (node_type == 0)
    n_obstacles: int  # Obstacles (node_type == 2)
    n_goals: int  # Goals (node_type == 1)
    n_total_nodes: int  # Total nodes
    n_edges: int  # Total edges
    
    # Node feature statistics
    node_feature_mean: np.ndarray  # Mean of node features (shape: n_node_features)
    node_feature_std: np.ndarray  # Std of node features
    node_feature_min: np.ndarray
    node_feature_max: np.ndarray
    
    # Edge feature statistics
    edge_feature_mean: np.ndarray  # Mean of edge features
    edge_feature_std: np.ndarray
    edge_feature_min: np.ndarray
    edge_feature_max: np.ndarray
    
    # State statistics (agent positions/velocities)
    states_mean: Optional[np.ndarray] = None  # Mean of all states
    states_std: Optional[np.ndarray] = None
    agent_positions_mean: Optional[np.ndarray] = None  # Mean of first state_dim//2 (positions)
    agent_velocities_mean: Optional[np.ndarray] = None  # Mean of second half (velocities)
    
    # Environment context
    env_context: Dict[str, Any] = None  # Parsed env_states
    
    # Aggregated context vector for CBF (concatenates key stats)
    context_vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize env_context if not provided."""
        if self.env_context is None:
            self.env_context = {}


def extract_graph_context(group_dict: Dict[str, Any], graph_id: str) -> GraphLevelInfo:
    """
    Extract graph-level information from a single graph group in NPZ.
    
    Args:
        group_dict: Dictionary with keys like 'nodes', 'edges', 'senders', 'receivers', 
                   'n_node', 'n_edge', 'node_type', 'states', 'env_states'
        graph_id: Identifier for this graph (the prefix from NPZ)
    
    Returns:
        GraphLevelInfo object with all extracted metadata and statistics
    """
    
    # Extract arrays
    nodes = np.asarray(group_dict.get("nodes", []))  # shape: (n_nodes, n_node_features)
    edges = np.asarray(group_dict.get("edges", []))  # shape: (n_edges, n_edge_features)
    node_type = np.asarray(group_dict.get("node_type", []))  # shape: (n_nodes,)
    states = group_dict.get("states")  # shape: (n_nodes, state_dim) or None
    env_states = group_dict.get("env_states")  # string or None
    
    # Get counts
    n_total_nodes = len(node_type) if len(node_type) > 0 else nodes.shape[0]
    n_edges = edges.shape[0] if edges.size > 0 else 0
    
    # Count by type (0=agent, 1=goal, 2=obstacle)
    n_agents = int(np.sum(node_type == 0)) if node_type.size > 0 else 0
    n_goals = int(np.sum(node_type == 1)) if node_type.size > 0 else 0
    n_obstacles = int(np.sum(node_type == 2)) if node_type.size > 0 else 0
    
    # Compute node feature statistics
    if nodes.size > 0:
        node_feature_mean = np.mean(nodes, axis=0)
        node_feature_std = np.std(nodes, axis=0)
        node_feature_min = np.min(nodes, axis=0)
        node_feature_max = np.max(nodes, axis=0)
    else:
        node_feature_mean = np.array([])
        node_feature_std = np.array([])
        node_feature_min = np.array([])
        node_feature_max = np.array([])
    
    # Compute edge feature statistics
    if edges.size > 0:
        edge_feature_mean = np.mean(edges, axis=0)
        edge_feature_std = np.std(edges, axis=0)
        edge_feature_min = np.min(edges, axis=0)
        edge_feature_max = np.max(edges, axis=0)
    else:
        edge_feature_mean = np.array([])
        edge_feature_std = np.array([])
        edge_feature_min = np.array([])
        edge_feature_max = np.array([])
    
    # Parse states (agent/goal/obstacle positions + velocities)
    states_mean = None
    states_std = None
    agent_positions_mean = None
    agent_velocities_mean = None
    if states is not None:
        states = np.asarray(states)
        if states.size > 0:
            states_mean = np.mean(states, axis=0)
            states_std = np.std(states, axis=0)
            
            # Assuming state_dim = 2*spatial_dim (pos, vel)
            state_dim = states.shape[1]
            if state_dim > 0:
                spatial_dim = state_dim // 2
                agent_positions_mean = np.mean(states[:, :spatial_dim], axis=0)
                agent_velocities_mean = np.mean(states[:, spatial_dim:], axis=0)
    
    # Parse environment context
    env_context = {}
    if env_states is not None:
        try:
            env_str = str(env_states)
            parsed = parse_envstate_string(env_str)
            env_context = parsed
        except Exception as e:
            print(f"Warning: could not parse env_states for graph {graph_id}: {e}")
    
    # Create graph-level context vector: concatenate key statistics
    # This can be fed into a "graph encoder" layer of the CBF network
    context_parts = []
    context_parts.append(np.array([n_agents, n_obstacles, n_goals, n_total_nodes, n_edges]))
    context_parts.append(node_feature_mean)
    context_parts.append(node_feature_std)
    if edge_feature_mean.size > 0:
        context_parts.append(edge_feature_mean)
    if agent_positions_mean is not None:
        context_parts.append(agent_positions_mean)
    if agent_velocities_mean is not None:
        context_parts.append(agent_velocities_mean)
    
    context_vector = np.concatenate(context_parts) if context_parts else np.array([])
    
    info = GraphLevelInfo(
        graph_id=graph_id,
        n_agents=n_agents,
        n_obstacles=n_obstacles,
        n_goals=n_goals,
        n_total_nodes=n_total_nodes,
        n_edges=n_edges,
        node_feature_mean=node_feature_mean,
        node_feature_std=node_feature_std,
        node_feature_min=node_feature_min,
        node_feature_max=node_feature_max,
        edge_feature_mean=edge_feature_mean,
        edge_feature_std=edge_feature_std,
        edge_feature_min=edge_feature_min,
        edge_feature_max=edge_feature_max,
        states_mean=states_mean,
        states_std=states_std,
        agent_positions_mean=agent_positions_mean,
        agent_velocities_mean=agent_velocities_mean,
        env_context=env_context,
        context_vector=context_vector,
    )
    
    return info


def load_all_graph_contexts(npz_path: str) -> Dict[str, GraphLevelInfo]:
    """
    Load all graphs from an NPZ file and extract graph-level information for each.
    
    Returns:
        Dictionary mapping graph_id (prefix) -> GraphLevelInfo
    """
    data, groups, top_level = load_npz_groups(npz_path)
    
    contexts = {}
    for graph_id, group_dict in sorted(groups.items()):
        try:
            context = extract_graph_context(group_dict, graph_id)
            contexts[graph_id] = context
        except Exception as e:
            print(f"Warning: could not extract context for graph {graph_id}: {e}")
    
    return contexts


def print_graph_context(info: GraphLevelInfo):
    """Pretty-print graph-level information."""
    print(f"\n{'='*70}")
    print(f"Graph-Level Information: {info.graph_id}")
    print(f"{'='*70}")
    
    print(f"\nStructure Metadata:")
    print(f"  Agents:     {info.n_agents}")
    print(f"  Goals:      {info.n_goals}")
    print(f"  Obstacles:  {info.n_obstacles}")
    print(f"  Total Nodes: {info.n_total_nodes}")
    print(f"  Total Edges: {info.n_edges}")
    
    print(f"\nNode Feature Statistics (shape: {info.node_feature_mean.shape}):")
    print(f"  Mean: {info.node_feature_mean}")
    print(f"  Std:  {info.node_feature_std}")
    print(f"  Min:  {info.node_feature_min}")
    print(f"  Max:  {info.node_feature_max}")
    
    if info.edge_feature_mean.size > 0:
        print(f"\nEdge Feature Statistics (shape: {info.edge_feature_mean.shape}):")
        print(f"  Mean: {info.edge_feature_mean}")
        print(f"  Std:  {info.edge_feature_std}")
        print(f"  Min:  {info.edge_feature_min}")
        print(f"  Max:  {info.edge_feature_max}")
    
    if info.states_mean is not None:
        print(f"\nState Statistics (shape: {info.states_mean.shape}):")
        print(f"  Mean: {info.states_mean}")
        print(f"  Std:  {info.states_std}")
    
    if info.agent_positions_mean is not None:
        print(f"\nAgent Position Mean: {info.agent_positions_mean}")
    
    if info.agent_velocities_mean is not None:
        print(f"Agent Velocity Mean: {info.agent_velocities_mean}")
    
    print(f"\nGraph Context Vector (dim={len(info.context_vector)}):")
    print(f"  {info.context_vector}")
    
    if info.env_context:
        print(f"\nEnvironment Context:")
        for key, val in info.env_context.items():
            if isinstance(val, np.ndarray):
                print(f"  {key}: array shape {val.shape}")
            else:
                print(f"  {key}: {val}")


def create_minibatch_with_contexts(graph_ids: List[str], contexts_dict: Dict[str, GraphLevelInfo]) -> Dict[str, Any]:
    """
    Prepare minibatch metadata that includes graph-level information for all graphs.
    Useful when processing multiple graphs together in the CBF fine-tuning loop.
    
    Returns:
        Dictionary with:
          - 'n_agents_per_graph': list of agent counts
          - 'context_vectors': stacked context vectors (for graph encoder input)
          - 'graph_indices': indices mapping nodes/edges back to graphs
          - 'metadata': full GraphLevelInfo objects
    """
    n_agents = []
    context_vectors = []
    metadata = []
    
    for gid in graph_ids:
        if gid in contexts_dict:
            info = contexts_dict[gid]
            n_agents.append(info.n_agents)
            context_vectors.append(info.context_vector)
            metadata.append(info)
    
    batch = {
        'n_agents_per_graph': n_agents,
        'context_vectors': np.stack(context_vectors) if context_vectors else np.array([]),
        'metadata': metadata,
        'graph_ids': graph_ids,
    }
    
    return batch


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graph_level_info.py <path/to/episode.npz>")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    print(f"Loading graph-level information from: {npz_path}\n")
    
    contexts = load_all_graph_contexts(npz_path)
    
    for graph_id in sorted(contexts.keys()):
        info = contexts[graph_id]
        print_graph_context(info)
    
    print(f"\n{'='*70}")
    print(f"Summary: Found {len(contexts)} graphs")
    print(f"{'='*70}")
    
    # Example: create minibatch with context vectors
    if contexts:
        graph_ids = sorted(contexts.keys())
        batch = create_minibatch_with_contexts(graph_ids, contexts)
        print(f"\nMinibatch context shape: {batch['context_vectors'].shape}")
        print(f"Agents per graph: {batch['n_agents_per_graph']}")
