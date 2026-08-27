import sys
import jax.numpy as jnp
import numpy as np
import jax
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import tree_index, np2jax
from gcbfplus.trainer.utils import jax2np


def compute_qp_values(graph: GraphsTuple, algo):
    """
    Compute QP values for a single timestep graph.
    
    Args:
        graph: Single timestep GraphsTuple (not batched)
        algo: The algorithm instance (GCBFPlus or similar) with get_qp_action method
        
    Returns:
        u_qp: QP action values, shape (n_agents, action_dim)
    """
    # Ensure graph is single (not batched)
    if not graph.is_single:
        raise ValueError("Graph must be a single timestep graph (not batched)")
    
    # Convert graph from numpy to JAX arrays
    # (tree_index returns numpy arrays, but get_qp_action expects JAX arrays)
    graph_jax = np2jax(graph)
    
    # Get QP action using the algorithm's method
    # get_qp_action returns (u_opt, relaxation)
    u_qp, relaxation = algo.get_qp_action(graph_jax, cbf_params=algo.cbf_tgt.params)
    
    # Convert from JAX array to numpy
    u_qp = jax2np(u_qp)
    return u_qp


def clip_agent_centric_graphs(T_graph: GraphsTuple, num_agents: int, sensing_radius: int, agent_radius: int, actuator_bounds: jnp.ndarray, algo=None):
    """
    Creates agent-centric 8-agent clipped graphs for supervised fine-tuning.
    
    Args:
        T_graph: Full trajectory graph with 256 timesteps
        num_agents: Number of agents in the original environment
        sensing_radius: Sensing radius for safety checking
        agent_radius: Agent collision radius
        actuator_bounds: [agent_accel_x, agent_accel_y, obstacle_accel]
        algo: (Optional) Algorithm instance for computing QP values
    
    Returns:
        List of tuples: (timestep, failing_agent_id, clipped_graph, collision_mask, u_qp)
    """
    
    epistemic_penalties = 0.9
    num_obstacles = 32
    timesteps = 256
    avg_obstacle = 0
    target_agents = 8  # Target number of agents in clipped graph
    
    clipped_data = []
    
    # Iterate over all timesteps
    for t in range(timesteps):
        graph = tree_index(T_graph, t)

        agent_collisions = {}  # agent_id -> list of colliding agent_ids
        agent_obstacle_collision = {}  # agent_id -> bool
        unsafe_found = False
        
        # Iterate over all agents to detect collisions
        for i in range(num_agents):
            agent_collisions[i] = []
            agent_obstacle_collision[i] = False
            
            # Agent-Agent collision detection
            for j in range(num_agents):
                if i == j:
                    continue
                
                edge_ij = graph.edges[i * num_agents + j]
                Px_ij = float(edge_ij[0])
                Py_ij = float(edge_ij[1])
                Vx_ij = float(edge_ij[2])
                Vy_ij = float(edge_ij[3])
                
                denominator = np.sqrt(Px_ij**2 + Py_ij**2) + 1e-6
                radial_velocity = (Px_ij*Vx_ij + Py_ij*Vy_ij) / denominator
                real_remaining_distance = np.sqrt(Px_ij**2 + Py_ij**2)
                
                # Safety check for agent-agent pair
                is_unsafe = False
                if real_remaining_distance < sensing_radius:
                    if radial_velocity > 0:
                        is_unsafe = False
                    elif radial_velocity == 0 and real_remaining_distance >= 2.0*agent_radius:
                        is_unsafe = False
                    elif radial_velocity == 0 and real_remaining_distance < 2.0*agent_radius:
                        is_unsafe = True
                    elif radial_velocity < 0:
                        alphai = actuator_bounds[0] * (1 - epistemic_penalties)
                        alphaj = actuator_bounds[1] * (1 - epistemic_penalties)
                        minimum_stopping_distance = (radial_velocity**2) / (2*(alphai+alphaj))
                        if real_remaining_distance < (2.0*agent_radius) + minimum_stopping_distance:
                            unsafe_found = True
                            is_unsafe = True

                            

                
                if is_unsafe:
                    agent_collisions[i].append(j)
            
            # Agent-Obstacle collision detection
            block1 = num_agents * num_agents
            block2 = num_agents * num_agents
            block3_starting_index = block1 + block2
            
            for k in range(num_obstacles):
                edge_ik = graph.edges[block3_starting_index + i*num_obstacles + k]
                Px_ik = float(edge_ik[0])
                Py_ik = float(edge_ik[1])
                Vx_ik = float(edge_ik[2])
                Vy_ik = float(edge_ik[3])
                
                denominator = np.sqrt(Px_ik**2 + Py_ik**2) + 1e-6
                radial_velocity = (Px_ik*Vx_ik + Py_ik*Vy_ik) / denominator
                real_remaining_distance = np.sqrt(Px_ik**2 + Py_ik**2)
                
                # Safety check for agent-obstacle pair
                is_unsafe = False
                if real_remaining_distance < sensing_radius:
                    if radial_velocity > 0:
                        is_unsafe = False
                    elif radial_velocity == 0 and real_remaining_distance >= (1.0*agent_radius):
                        is_unsafe = False
                    elif radial_velocity == 0 and real_remaining_distance < ((1.0*agent_radius) + avg_obstacle):
                        is_unsafe = True
                    elif radial_velocity < 0:
                        alphai = actuator_bounds[0] * (1 - epistemic_penalties)
                        alphaj = actuator_bounds[2] * (1 - epistemic_penalties)
                        minimum_stopping_distance = (radial_velocity**2) / (2*(alphai+alphaj))
                        if real_remaining_distance < (1.0*agent_radius) + avg_obstacle + minimum_stopping_distance:
                            is_unsafe = True
                            unsafe_found = True
                if is_unsafe:
                    agent_obstacle_collision[i] = True
                    unsafe_found = True
                    break  # One obstacle collision is enough to mark as unsafe
        
        if unsafe_found and algo is not None:
            u_qp_full = compute_qp_values(graph, algo)
     
        # Now process each failing agent
        for i in range(num_agents):
            # Check if agent i is failing (either agent or obstacle collision)
            is_failing = len(agent_collisions[i]) > 0 or agent_obstacle_collision[i]
            
            if not is_failing:
                continue
            
            # Agent i is failing - create agent-centric graph
            colliding_agents = agent_collisions[i]
            x = len(colliding_agents)
            
            # Select agents for clipped graph
            if x >= 7:
                # Take only 7 colliding agents (closest ones if more than 7)
                if x > 7:
                    # Get distances to all colliding agents
                    distances = []
                    for j in colliding_agents:
                        edge_ij = graph.edges[i * num_agents + j]
                        dist = np.sqrt(edge_ij[0]**2 + edge_ij[1]**2)
                        distances.append((dist, j))
                    distances.sort()
                    selected_agents = [i] + [j for _, j in distances[:7]]
                else:
                    selected_agents = [i] + colliding_agents
            else:
                # Need to add (7-x) nearest safe agents
                safe_agents = [j for j in range(num_agents) if j != i and j not in colliding_agents]
                
                # Calculate distances to all safe agents
                distances = []
                for j in safe_agents:
                    edge_ij = graph.edges[i * num_agents + j]
                    dist = np.sqrt(edge_ij[0]**2 + edge_ij[1]**2)
                    distances.append((dist, j))
                
                # Sort by distance and take closest (7-x)
                distances.sort()
                nearest_safe = [j for _, j in distances[:7-x]]
                
                selected_agents = [i] + colliding_agents + nearest_safe
            
            # Create collision mask for selected agents (8 agents)
            collision_mask = np.zeros(target_agents, dtype=bool)
            collision_mask[0] = True  # Self agent is failing
            for idx, agent_id in enumerate(selected_agents[1:], start=1):
                if agent_id in colliding_agents:
                    collision_mask[idx] = True
            
            # Now extract clipped graph
            clipped_graph = extract_subgraph(graph, selected_agents, num_agents, num_obstacles)
            
            # Extract QP values for selected agents
            if u_qp_full is not None:
                # Extract u_qp values for the selected agents
                u_qp = u_qp_full[selected_agents]  # Shape: (8, action_dim)
                print(f"Extracted QP values for selected agents: {u_qp}" )
            else:
                # If no algo provided, initialize with zeros
                u_qp = np.zeros((8, 2), dtype=np.float32)  # 8 agents, 2D action
            print(f"Extracted QP values for selected agents: {u_qp}" )

            # Store tuple
            clipped_data.append((t, i, clipped_graph, collision_mask, u_qp))
    
    return clipped_data


def extract_subgraph(graph, selected_agents, num_agents_original, num_obstacles):
    """
    Extract subgraph for selected 8 agents.
    
    Args:
        graph: Single timestep graph
        selected_agents: List of 8 agent IDs (global indices)
        num_agents_original: Original number of agents
        num_obstacles: Number of obstacle rays per agent (32)
    
    Returns:
        Clipped GraphsTuple with 8 agents
    """
    num_agents_clipped = len(selected_agents)
    assert num_agents_clipped == 8, "Must select exactly 8 agents"
    
    # Extract nodes
    # Original: [agents(n), goals(n), obstacles(32n), padding(1)]
    # Clipped: [agents(8), goals(8), obstacles(256), padding(1)]
    
    agent_nodes = []
    goal_nodes = []
    obstacle_nodes = []
    
    for global_id in selected_agents:
        agent_nodes.append(graph.nodes[global_id])
        goal_nodes.append(graph.nodes[num_agents_original + global_id])
        
        # Extract 32 obstacle nodes for this agent
        obs_start = 2 * num_agents_original + global_id * num_obstacles
        for k in range(num_obstacles):
            obstacle_nodes.append(graph.nodes[obs_start + k])
    
    padding_node = graph.nodes[-1]
    
    clipped_nodes = np.array(agent_nodes + goal_nodes + obstacle_nodes + [padding_node], dtype=np.float32)
    
    # Extract states
    agent_states = []
    goal_states = []
    obstacle_states = []
    
    for global_id in selected_agents:
        agent_states.append(graph.states[global_id])
        goal_states.append(graph.states[num_agents_original + global_id])
        
        # Extract 32 obstacle states for this agent
        obs_start = 2 * num_agents_original + global_id * num_obstacles
        for k in range(num_obstacles):
            obstacle_states.append(graph.states[obs_start + k])
    
    padding_state = graph.states[-1]
    
    clipped_states = np.array(agent_states + goal_states + obstacle_states + [padding_state], dtype=np.float32)
    
    # Extract edges
    # Block 1: Agent-Agent edges (64 = 8*8)
    # Block 2: Agent-Goal edges (64 = 8*8)
    # Block 3: Agent-Obstacle edges (256 = 8*32)
    
    agent_agent_edges = []
    for local_i, global_i in enumerate(selected_agents):
        for local_j, global_j in enumerate(selected_agents):
            edge_idx = global_i * num_agents_original + global_j
            agent_agent_edges.append(graph.edges[edge_idx])
    
    agent_goal_edges = []
    block2_start = num_agents_original * num_agents_original
    for local_i, global_i in enumerate(selected_agents):
        for local_j, global_j in enumerate(selected_agents):
            edge_idx = block2_start + global_i * num_agents_original + global_j
            agent_goal_edges.append(graph.edges[edge_idx])
    
    agent_obstacle_edges = []
    block3_start = 2 * num_agents_original * num_agents_original
    for global_i in selected_agents:
        for k in range(num_obstacles):
            edge_idx = block3_start + global_i * num_obstacles + k
            agent_obstacle_edges.append(graph.edges[edge_idx])
    
    clipped_edges = np.array(agent_agent_edges + agent_goal_edges + agent_obstacle_edges, dtype=np.float32)
    
    # Create senders and receivers for clipped graph
    # Agent-Agent block
    senders = []
    receivers = []
    for i in range(num_agents_clipped):
        for j in range(num_agents_clipped):
            senders.append(i)
            receivers.append(j)
    
    # Agent-Goal block
    for i in range(num_agents_clipped):
        for j in range(num_agents_clipped):
            senders.append(i)
            receivers.append(num_agents_clipped + j)
    
    # Agent-Obstacle block
    for i in range(num_agents_clipped):
        for k in range(num_obstacles):
            senders.append(i)
            receivers.append(2 * num_agents_clipped + i * num_obstacles + k)
    
    senders = np.array(senders, dtype=np.int32)
    receivers = np.array(receivers, dtype=np.int32)
    
    # Create node_type
    node_type = np.zeros(len(clipped_nodes), dtype=np.int32)
    node_type[num_agents_clipped:2*num_agents_clipped] = 1  # Goals
    node_type[2*num_agents_clipped:-1] = 2  # Obstacles
    
    # Extract env_states for selected agents
    # EnvState is a NamedTuple(agent, goal, obstacle)
    # agent: shape [n_agents, state_dim] -> slice to selected agents
    # goal: shape [n_agents, state_dim] -> slice to selected agents
    # obstacle: Obstacle object (shared, keep as-is)
    clipped_env_states = None
    if graph.env_states is not None:
        selected_indices = np.array(selected_agents)
        clipped_agent = graph.env_states.agent[selected_indices]
        clipped_goal = graph.env_states.goal[selected_indices]
        clipped_env_states = graph.env_states._replace(
            agent=clipped_agent,
            goal=clipped_goal,
        )

    
    # Create clipped GraphsTuple
    clipped_graph = GraphsTuple(
        nodes=clipped_nodes,
        edges=clipped_edges,
        senders=senders,
        receivers=receivers,
        states=clipped_states,
        node_type=node_type,
        env_states=clipped_env_states,
        n_node=np.array([len(clipped_nodes)]),
        n_edge=np.array([len(clipped_edges)])
    )
    
    return clipped_graph
