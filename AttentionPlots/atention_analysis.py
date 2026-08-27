"""
Attention Analysis Script - FINAL VERSION with Distance + Velocity plots
Plots attention weights received by an agent (segment_softmax normalized per receiver)
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import tree_index


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
    ax3.axvline(x=0, color='black', linestyle='--',
               linewidth=2.0, label=f'R = {zero_velocity}')
    
    # Formatting left plot
    ax3.set_xlabel('Radial Velocity $r_{ij}$', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Attention Weight $w_{ij}$', fontsize=14, fontweight='bold')
    ax3.set_title('Attention vs Radial Velocity', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
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
