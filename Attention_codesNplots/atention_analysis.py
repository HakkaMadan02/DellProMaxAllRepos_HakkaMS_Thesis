"""
Attention Analysis Script - FINAL VERSION with Distance + Velocity plots
Plots attention weights received by an agent (segment_softmax normalized per receiver)
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import tree_index
from pathlib import Path


plt.rcParams.update({
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})


def plot_attention_vs_distance_single_agent(
    graph: GraphsTuple,
    attn_weights: np.ndarray,
    agent_id: int,
    n_agents: int,
    sensing_radius: float = 0.5,
    timestep: int = 0,
    save_path: str = None
):
    """
    Plot attention weights vs distance AND velocity for edges received by a single agent
    Uses sparse graph with receiver-based attention (segment_softmax)
    """
    
    # Convert to numpy
    edges = np.array(graph.edges)
    senders = np.array(graph.senders)
    receivers = np.array(graph.receivers)
    node_types = np.array(graph.node_type)
    attn_weights = np.array(attn_weights)
    
    print("\n" + "="*80)
    print(f"Analyzing Agent {agent_id} - RECEIVER-BASED ATTENTION")
    print("="*80)
    
    # Padding node is the LAST node
    padding_node_id = len(node_types) - 1
    
    print(f"Padding node ID: {padding_node_id}")
    print(f"Total nodes: {len(node_types)}")
    print(f"Total edges in graph: {len(edges)}")
    
    # Filter out padding edges
    non_padding_mask = (senders != padding_node_id) & (receivers != padding_node_id)
    real_edges = edges[non_padding_mask]
    real_senders = senders[non_padding_mask]
    real_receivers = receivers[non_padding_mask]
    real_attn = attn_weights[non_padding_mask]
    
    print(f"Non-padding edges: {len(real_edges)}")
    
    # Find edges where this agent is the RECEIVER
    agent_receiver_mask = (real_receivers == agent_id)
    agent_edges = real_edges[agent_receiver_mask]
    agent_senders = real_senders[agent_receiver_mask]
    agent_attn = real_attn[agent_receiver_mask]
    
    print(f"\nEdges TO agent {agent_id} (received from neighbors): {len(agent_edges)}")
    print(f"Senders: {agent_senders}")
    
    if len(agent_edges) == 0:
        print(f"WARNING: No edges found to agent {agent_id}!")
        return None, None, None, None
    
    # Determine edge types based on SENDER node types
    edge_type_labels = []
    
    print(f"\nDEBUG: Determining edge types")
    print(f"  agent_senders: {agent_senders}")
    print(f"  node_types[agent_senders]: {node_types[agent_senders]}")
    
    for i, sender_id in enumerate(agent_senders):
        sender_type_id = node_types[sender_id]
        
        if sender_type_id == 0:  # Agent
            edge_type_labels.append('agent')
        elif sender_type_id == 1:  # Goal
            edge_type_labels.append('goal')
        elif sender_type_id == 2:  # Obstacle
            edge_type_labels.append('obstacle')
        else:
            edge_type_labels.append('unknown')
    
    print(f"  Edge type labels: {edge_type_labels}")
    
    # Compute distances and velocities from edge features
    rel_px = agent_edges[:, 0]  # x-component of relative position
    rel_py = agent_edges[:, 1]  # y-component of relative position
    rel_vx = agent_edges[:, 2]  # x-component of relative velocity
    rel_vy = agent_edges[:, 3]  # y-component of relative velocity

    distances = np.sqrt(rel_px**2 + rel_py**2)
    velocities = np.sqrt(rel_vx**2 + rel_vy**2)

    denom = distances + 1e-6  # Avoid division by zero
    radial_velocities = (rel_px * rel_vx + rel_py * rel_vy) / denom

    # Compute minimum stopping distance for each edge
    minimum_stopping_distances = np.zeros_like(radial_velocities)

    for i, sender_id in enumerate(agent_senders):
        sender_type_id = node_types[sender_id]
        
        if radial_velocities[i] < 0:  # Moving towards each other
            if sender_type_id == 0:  # Agent-agent interaction
                # Both can brake: alphai + alphaj
                minimum_stopping_distances[i] = (radial_velocities[i] ** 2) / (2 * (1.414 + 1.414))
            else:  # Agent-obstacle or agent-goal interaction
                # Only agent can brake: alphai only
                minimum_stopping_distances[i] = (radial_velocities[i] ** 2) / (2 * 1.414)
        else:  # Moving away (radial_velocity >= 0)
            minimum_stopping_distances[i] = -0.075
    zero_velocity = 0.0

    
    # Print statistics
    print(f"\n" + "="*80)
    print(f"Agent {agent_id} at Timestep {timestep} - RESULTS")
    print("="*80)
    print(f"Total incoming edges: {len(distances)}")
    print(f"  From agents: {sum(1 for x in edge_type_labels if x == 'agent')}")
    print(f"  From goals: {sum(1 for x in edge_type_labels if x == 'goal')}")
    print(f"  From obstacles: {sum(1 for x in edge_type_labels if x == 'obstacle')}")
    
    print(f"\nDistance statistics:")
    print(f"  Range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  Mean: {distances.mean():.4f}")
    
    print(f"\nVelocity statistics:")
    print(f"  Range: [{velocities.min():.4f}, {velocities.max():.4f}]")
    print(f"  Mean: {velocities.mean():.4f}")
    
    print(f"\nAttention weight statistics (should sum to 1.0):")
    print(f"  Range: [{agent_attn.min():.6f}, {agent_attn.max():.6f}]")
    print(f"  Sum: {agent_attn.sum():.6f}")
    print(f"  Mean: {agent_attn.mean():.6f}")
    
    # Print by edge type
    for edge_type in ['agent', 'goal', 'obstacle']:
        mask = np.array(edge_type_labels) == edge_type
        if np.any(mask):
            type_attn = agent_attn[mask]
            type_dist = distances[mask]
            type_vel = velocities[mask]
            print(f"\n  {edge_type.capitalize()}: {np.sum(mask)} edges")
            print(f"    Attention: min={type_attn.min():.6f}, max={type_attn.max():.6f}, mean={type_attn.mean():.6f}, sum={type_attn.sum():.6f}")
            print(f"    Distance: min={type_dist.min():.4f}, max={type_dist.max():.4f}, mean={type_dist.mean():.4f}")
            print(f"    Velocity: min={type_vel.min():.4f}, max={type_vel.max():.4f}, mean={type_vel.mean():.4f}")
    
    # Create THREE subplots side by side
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40, 10))
    
    colors = {'bloodred': '#880808', 'jetblack': '#0a0a0a', 'gungrey': '#818589'}

    plot_params = {
        'agent': {'color': colors['jetblack'], 'marker': 'o', 'label': 'Agent', 'size': 100},
        'goal': {'color': colors['gungrey'], 'marker': 'o', 'label': 'Goal', 'size': 100},
        'obstacle': {'color': colors['bloodred'], 'marker': 'o', 'label': 'Obstacle', 'size': 100}
    }
    
    # ============ LEFT PLOT: Distance vs Attention ============
    for edge_type, params in plot_params.items():
        mask = np.array(edge_type_labels) == edge_type
        if np.any(mask):
            ax1.scatter(distances[mask], agent_attn[mask],
                       alpha=0.8, s=params['size'], c=params['color'],
                       marker=params['marker'], label=params['label'],
                       edgecolors='black', linewidth=0.8)
    
    # Add sensing radius line
    ax1.axvline(x=sensing_radius, color='black', linestyle='--',
               linewidth=2.0, label=f'R = {sensing_radius}')
    
    # Formatting left plot
    ax1.set_xlabel('Distance $d_{ij}$', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Attention Weight $w_{ij}$', fontsize=14, fontweight='bold')
    ax1.set_title('Attention vs Distance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    ax1.tick_params(axis='both', labelsize=20)
    
    # Set axis limits for distance plot
    max_dist = 3
    max_attn = max(agent_attn.max() * 1.15, 0.1) if len(agent_attn) > 0 else 1.0
    ax1.set_xlim(-0.1, max_dist)
    ax1.set_ylim(-0.1, max_attn)
    
    # ============ RIGHT PLOT: Velocity vs Attention ============
    for edge_type, params in plot_params.items():
        mask = np.array(edge_type_labels) == edge_type
        if np.any(mask):
            ax2.scatter(velocities[mask], agent_attn[mask],
                       alpha=0.8, s=params['size'], c=params['color'],
                       marker=params['marker'], label=params['label'],
                       edgecolors='black', linewidth=0.8)
    
    # Formatting right plot
    ax2.set_xlabel('Relative Velocity $|v_{ij}|$', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Attention Weight $w_{ij}$', fontsize=14, fontweight='bold')
    ax2.set_title('Attention vs Velocity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.tick_params(axis='both', labelsize=20)

    
    # Set axis limits for velocity plot
    max_vel = max(velocities.max() * 1.15, 0.5) if len(velocities) > 0 else 1.0
    ax2.set_xlim(-0.1, max_vel)
    ax2.set_ylim(-0.1, max_attn)

    # ============ 3rd PLOT : Radial Velocity vs Attention ============
    for edge_type, params in plot_params.items():
        mask = np.array(edge_type_labels) == edge_type
        if np.any(mask):
            ax3.scatter(radial_velocities[mask], agent_attn[mask],
                       alpha=0.8, s=params['size'], c=params['color'],
                       marker=params['marker'], label=params['label'],
                       edgecolors='black', linewidth=0.8)
    # Add sensing radius line
    #ax3.axvline(x=0, color='black', linestyle='--',
    #           linewidth=2.0, label=f'R = {zero_velocity}')
    
    # Formatting left plot
    ax3.set_xlabel('Radial Velocity $r_{ij}$', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Attention Weight $w_{ij}$', fontsize=14, fontweight='bold')
    ax3.set_title('Attention vs Radial Velocity', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax3.tick_params(axis='both', labelsize=20)
    
    # Set axis limits for distance plot
    max_vel = 0.75
    min_vel = -0.75
    max_attn = max(agent_attn.max() * 1.15, 0.1) if len(agent_attn) > 0 else 1.0
    ax3.set_xlim(min_vel, max_vel)
    ax3.set_ylim(-0.1, max_attn)

    # ============ 4th PLOT : Minimum Stopping Distance vs Attention ============
    for edge_type, params in plot_params.items():
        mask = np.array(edge_type_labels) == edge_type
        if np.any(mask):
            ax4.scatter(minimum_stopping_distances[mask], agent_attn[mask],
                       alpha=0.8, s=params['size'], c=params['color'],
                       marker=params['marker'], label=params['label'],
                       edgecolors='black', linewidth=0.8)
    
    ax4.axvline(x=0, color='black', linestyle='--',
               linewidth=2.0, label=f'R = {zero_velocity}')

    # Formatting left plot
    ax4.set_xlabel('Minimum Stopping Distance', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Attention Weight $w_{ij}$', fontsize=14, fontweight='bold')
    ax4.set_title('Attention vs Minimum Stopping Distance', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Set axis limits for distance plot
    max_stop_dis = 0.3
    min_stop_dis = -0.1
    max_attn = max(agent_attn.max() * 1.15, 0.1) if len(agent_attn) > 0 else 1.0
    ax4.set_xlim(min_stop_dis, max_stop_dis)
    ax4.set_ylim(-0.1, max_attn)
    
    # Overall title
    fig.suptitle(f'Attention Analysis for Agent {agent_id} at Timestep {timestep}',
                fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
    
    plt.close(fig)
    
    return distances, velocities, agent_attn, edge_type_labels


def analyze_attention_for_agent(
    rollout_result,
    algo,
    agent_id: int,
    timestep: int,
    n_agents: int,
    sensing_radius: float = 0.5,
    save_path: str = None
):
    """
    Main function to analyze attention for one agent at one timestep
    """
    
    print(f"\n{'='*80}")
    print(f"Analyzing Attention for Agent {agent_id} at Timestep {timestep}")
    print(f"{'='*80}")
    
    # Extract graph at the specified timestep
    graph_t = tree_index(rollout_result.Tp1_graph, timestep)
    
    print(f"Graph extracted successfully")
    print(f"  Number of nodes: {len(graph_t.nodes)}")
    print(f"  Number of edges: {len(graph_t.edges)}")
    
    # Get attention weights from GNN
    attn_weights = algo.get_attention_weights(graph_t)
    attn_weights = np.array(attn_weights)
    
    print(f"  Total attention weights: {len(attn_weights)}")
    
    # Plot and analyze
    distances, velocities, weights, edge_types = plot_attention_vs_distance_single_agent(
        graph=graph_t,
        attn_weights=attn_weights,
        agent_id=agent_id,
        n_agents=n_agents,
        sensing_radius=sensing_radius,
        timestep=timestep,
        save_path=save_path
    )
    
    return distances, velocities, weights, edge_types

def plot_attention_vs_distance_mha_heads(
    graph: GraphsTuple,
    attn_weights_per_head: np.ndarray,   # shape: (n_heads, num_edges)
    agent_id: int,
    n_agents: int,
    sensing_radius: float = 0.5,
    timestep: int = 0,
    save_path: str = None
):
    """
    Plot Distance vs Attention for each head side by side.
    4 subplots = 4 heads. Tests for head collapse.
    """
    edges      = np.array(graph.edges)
    senders    = np.array(graph.senders)
    receivers  = np.array(graph.receivers)
    node_types = np.array(graph.node_type)
    attn_per_head = np.array(attn_weights_per_head)  # (n_heads, num_edges)
    n_heads = attn_per_head.shape[0]

    # Filter padding
    padding_node_id = len(node_types) - 1
    non_padding_mask = (senders != padding_node_id) & (receivers != padding_node_id)
    real_edges    = edges[non_padding_mask]
    real_senders  = senders[non_padding_mask]
    real_receivers = receivers[non_padding_mask]
    # apply mask per head
    real_attn_per_head = attn_per_head[:, non_padding_mask]  # (n_heads, real_edges)

    # Edges received by this agent
    agent_mask   = (real_receivers == agent_id)
    agent_edges  = real_edges[agent_mask]
    agent_senders = real_senders[agent_mask]
    agent_attn_per_head = real_attn_per_head[:, agent_mask]  # (n_heads, agent_edges)

    if len(agent_edges) == 0:
        print(f"WARNING: No edges to agent {agent_id}")
        return

    # Edge type labels
    edge_type_labels = []
    for sid in agent_senders:
        t = node_types[sid]
        if t == 0: edge_type_labels.append('agent')
        elif t == 1: edge_type_labels.append('goal')
        elif t == 2: edge_type_labels.append('obstacle')
        else: edge_type_labels.append('unknown')
    edge_type_labels = np.array(edge_type_labels)

    # Distance from edge features
    rel_px, rel_py = agent_edges[:, 0], agent_edges[:, 1]
    distances = np.sqrt(rel_px**2 + rel_py**2)

    # Plot
    colors = {'agent': '#0a0a0a', 'goal': '#818589', 'obstacle': '#880808'}
    markers = {'agent': 'o', 'goal': 'o', 'obstacle': 'o'}

    fig, axes = plt.subplots(1, n_heads, figsize=(10 * n_heads, 10), sharey=False)
    if n_heads == 1:
        axes = [axes]

    max_attn_global = agent_attn_per_head.max() * 1.15

    for k, ax in enumerate(axes):
        head_attn = agent_attn_per_head[k]  # (agent_edges,)
        for etype in ['agent', 'goal', 'obstacle']:
            mask = (edge_type_labels == etype)
            if np.any(mask):
                ax.scatter(distances[mask], head_attn[mask],
                           alpha=0.8, s=200, c=colors[etype],
                           marker=markers[etype], label=etype.capitalize(),
                           edgecolors='black', linewidth=0.8)

        ax.axvline(x=sensing_radius, color='black', linestyle='--',
                   linewidth=2.0, label=f'R = {sensing_radius}')
        ax.set_xlabel('Distance $d_{ij}$', fontsize=25, fontweight='bold')
        ax.set_ylabel('Attention Weight $w_{ij}$', fontsize=25, fontweight='bold')
        ax.set_title(f'Head {k} — Attn vs Distance', fontsize=30, fontweight='bold')
        ax.legend(fontsize=25, loc='upper right', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_xlim(-0.1, 0.7)
        ax.set_ylim(-0.1, max(max_attn_global, 0.1))

        # Print stats per head
        print(f"\nHead {k}: attn range [{head_attn.min():.6f}, {head_attn.max():.6f}], "
              f"sum={head_attn.sum():.4f}")

    fig.suptitle(f'MHA — Agent {agent_id} at Timestep {timestep} | {n_heads} Heads',
                 fontsize=30, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved to: {save_path}")
    plt.close(fig)


def analyze_attention_mha(
    rollout_result,
    algo,
    agent_id: int,
    timestep: int,
    n_agents: int,
    sensing_radius: float = 0.5,
    save_path: str = None
):
    """Drop-in replacement for analyze_attention_for_agent, for MHA models."""
    from gcbfplus.utils.utils import tree_index
    graph_t = tree_index(rollout_result.Tp1_graph, timestep)

    # Returns (n_heads, num_edges)
    attn_per_head = np.array(algo.get_attention_weights(graph_t))
    print(f"attn_per_head shape: {attn_per_head.shape}")

    plot_attention_vs_distance_mha_heads(
        graph=graph_t,
        attn_weights_per_head=attn_per_head,
        agent_id=agent_id,
        n_agents=n_agents,
        sensing_radius=sensing_radius,
        timestep=timestep,
        save_path=save_path,
    )

def plot_attention_mha_all_features(
    graph: GraphsTuple,
    attn_weights_per_head: np.ndarray,  # (n_heads, num_edges)
    agent_id: int,
    n_agents: int,
    n_obstacles: int,
    sensing_radius: float = 0.5,
    timestep: int = 0,
    save_path: str = None
):
    """
    Unified MHA grid: rows = heads, columns = features.
    For 4 heads x 4 features → 16 subplots in one image.
    Features: Distance | Speed | Radial Velocity | Min Stopping Distance
    """
    edges      = np.array(graph.edges)
    senders    = np.array(graph.senders)
    receivers  = np.array(graph.receivers)
    node_types = np.array(graph.node_type)
    attn_per_head = np.array(attn_weights_per_head)
    n_heads = attn_per_head.shape[0]

    # ── Filter padding ────────────────────────────────────────────────────────
    padding_node_id = len(node_types) - 1
    non_padding_mask = (senders != padding_node_id) & (receivers != padding_node_id)
    real_edges    = edges[non_padding_mask]
    real_senders  = senders[non_padding_mask]
    real_receivers = receivers[non_padding_mask]
    real_attn_per_head = attn_per_head[:, non_padding_mask]

    # ── Filter to this agent as receiver ─────────────────────────────────────
    agent_mask = (real_receivers == agent_id)
    agent_edges = real_edges[agent_mask]
    agent_senders = real_senders[agent_mask]
    agent_attn_per_head = real_attn_per_head[:, agent_mask]

    if len(agent_edges) == 0:
        print(f"WARNING: No edges to agent {agent_id}")
        return

    # ── Edge type labels ──────────────────────────────────────────────────────
    edge_type_labels = []
    for sid in agent_senders:
        t = node_types[sid]
        if t == 0:   edge_type_labels.append('agent')
        elif t == 1: edge_type_labels.append('goal')
        elif t == 2: edge_type_labels.append('obstacle')
        else:        edge_type_labels.append('unknown')
    edge_type_labels = np.array(edge_type_labels)

    # ── Compute all features (identical formulas to single-head function) ─────
    rel_px = agent_edges[:, 0]
    rel_py = agent_edges[:, 1]
    rel_vx = agent_edges[:, 2]
    rel_vy = agent_edges[:, 3]

    distances         = np.sqrt(rel_px**2 + rel_py**2)
    velocities        = np.sqrt(rel_vx**2 + rel_vy**2)
    denom             = distances + 1e-6
    radial_velocities = (rel_px * rel_vx + rel_py * rel_vy) / denom

    minimum_stopping_distances = np.zeros_like(radial_velocities)
    for i, sender_id in enumerate(agent_senders):
        sender_type_id = node_types[sender_id]
        if radial_velocities[i] < 0:  # approaching
            if sender_type_id == 0:   # agent-agent: both brake
                minimum_stopping_distances[i] = (radial_velocities[i] ** 2) / (2 * (1.414 + 1.414))
            else:                     # agent-obstacle/goal: only ego brakes
                minimum_stopping_distances[i] = (radial_velocities[i] ** 2) / (2 * 1.414)
        else:                         # moving away
            minimum_stopping_distances[i] = -0.075
    zero_velocity = 0.0

    # ── Feature columns config (mirrors single-head subplot order) ────────────
    features = [
        {
            'data':    distances,
            'xlabel':  'Distance $d_{ij}$',
            'title':   'Attention vs Distance',
            'xlim':    (-0.05, 0.75),
            'axvline': sensing_radius,
            'vline_label': f'Sensing Radius',
        },
        #{
        #    'data':    velocities,
        #    'xlabel':  'Relative Velocity $|v_{ij}|$',
        #    'title':   'Attn vs Velocity',
        #    'xlim':    (-0.1, max(velocities.max() * 1.15, 0.5)),
        #    'axvline': None,
        #    'vline_label': None,
        #},
        {
            'data':    radial_velocities,
            'xlabel':  'Radial Velocity $r_{ij}$',
            'title':   'Attention vs Radial Velocity',
            'xlim':    (-0.75, 0.75),
            'axvline': None,
            'vline_label': f'R = {zero_velocity}',
        },
        #{
        #    'data':    minimum_stopping_distances,
        #    'xlabel':  'Minimum Stopping Distance',
        #    'title':   'Attn vs Min Stopping Dist',
        #    'xlim':    (-0.1, 0.3),
        #    'axvline': 0,
        #    'vline_label': f'R = {zero_velocity}',
        #},
    ]
    n_features = len(features)

    # ── Formatting (identical to original) ───────────────────────────────────
    colors = {'agent': '#0a0a0a', 'goal': '#0a0a0a', 'obstacle': '#880808'}
    plot_params = {
        'agent':    {'color': colors['agent'],    'marker': 'o', 'label': 'Agent',    'size': 200},
        'goal':     {'color': colors['goal'],     'marker': '*', 'label': 'Goal',     'size': 250},
        'obstacle': {'color': colors['obstacle'], 'marker': 'o', 'label': 'Obstacle', 'size': 200},
    }

    # ── Create grid: rows=heads, cols=features ────────────────────────────────
    
    # Normalise axes to always be 2D array
    #if n_heads == 1:
    #    axes = axes[np.newaxis, :]
    
    for f_idx, feat in enumerate(features):
        for k in range(n_heads):
            fig, ax = plt.subplots(
                figsize=(10, 9))

            head_attn  = agent_attn_per_head[k]
            max_attn   = max(head_attn.max() * 1.15, 0.1) if len(head_attn) > 0 else 1.0
            feat_data = feat['data']

            for etype, params in plot_params.items():
                mask = (edge_type_labels == etype)
                if np.any(mask):
                    ax.scatter(
                        feat_data[mask], head_attn[mask],
                        alpha=0.8, s=params['size'], c=params['color'],
                        marker=params['marker'], label=params['label'],
                        edgecolors='black', linewidth=0.8
                    )

            if feat['axvline'] is not None:
                ax.axvline(
                    x=feat['axvline'], color='black', linestyle='--',
                    linewidth=2.0, label=feat['vline_label']
                )

            ax.set_xlabel(feat['xlabel'],           fontsize=30, fontweight='normal')
            ax.set_ylabel('Attention Weight $w_{ij}$', fontsize=30, fontweight='normal')
            #ax.set_title(f'Head {k} — {feat["title"]}\n', fontsize=35, fontweight='normal')
            ax.set_title("")
            ax.text(
                0.5, 1.11,
                f'Head {k} — {feat["title"]}',
                transform=ax.transAxes,
                fontsize=30, fontweight='normal', ha='center'
            )
            ax.text(
                0.5, 1.04,
                f'Env: ({n_agents} agents, {n_obstacles} obstacles) | Timestep: {timestep} | Agent: {agent_id}',
                transform=ax.transAxes,
                fontsize=22, fontweight='normal', ha='center'
            )
            '''ax.text(
                0.5, 1.00,
                f'Timestep: {timestep}, Agent :{agent_id}',
                transform=ax.transAxes,
                fontsize=23, fontweight='normal', ha='center'
            )'''

            ax.legend(fontsize=23, loc='upper right', framealpha=0.75, edgecolor='black')
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_xlim(feat['xlim'])
            ax.set_ylim(-0.05, 0.30)

            #fig.subplots_adjust(top=0.80)

            print(f"\nHead {k}: attn range [{head_attn.min():.6f}, {head_attn.max():.6f}], "
                f"sum={head_attn.sum():.4f}")

            plt.tight_layout(rect=[0, 0, 1, 0.85])
            #plt.tight_layout()

            if save_path:
                out_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_"
                f"{feat['title'].lower().replace(' ', '_')}_"
                f"head{k+1}"
                f"{Path(save_path).suffix}"
            )
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {out_path}")
            plt.close(fig)


def analyze_attention_mha_all_features(
    rollout_result,
    algo,
    agent_id: int,
    timestep: int,
    n_agents: int,
    n_obstacles: int,
    sensing_radius: float = 0.5,
    save_path: str = None,
):
    """Drop-in wrapper: calls plot_attention_mha_all_features for one rollout."""
    from gcbfplus.utils.utils import tree_index
    graph_t = tree_index(rollout_result.Tp1_graph, timestep)
    attn_per_head = np.array(algo.get_attention_weights(graph_t))
    print(f"attn_per_head shape: {attn_per_head.shape}")
    plot_attention_mha_all_features(
        graph=graph_t,
        attn_weights_per_head=attn_per_head,
        agent_id=agent_id,
        n_agents=n_agents,
        n_obstacles=n_obstacles,
        sensing_radius=sensing_radius,
        timestep=timestep,
        save_path=save_path,
    )
