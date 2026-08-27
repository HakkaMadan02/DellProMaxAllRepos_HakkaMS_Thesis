import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml
import numpy as np
#from model.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from model.env import make_env
from model.env.base import RolloutResult
#from gcbfplus.trainer.utils import get_bb_cbf_all_agents, get_bb_cbf_in_all_agents
from model.utils.graph import GraphsTuple
from model.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap

epistemic_penalties = 0.9  ## Set to 1 for unrealistic, perfect, environment.

def safety_labeller(T_graph: GraphsTuple, num_agents: int, sensing_radius: int, agent_radius: int, actuator_bounds: jnp.ndarray):
    """ORIGINAL LABELLER - returns only safety labels"""
    timesteps = 256
    labels = jnp.ones((timesteps, num_agents))
    
    features_array2 = {}

    worst_obstacle_length = 0.5
    avg_obstacle_length = 0.25
    worst_obstacle = jnp.sqrt(2*((worst_obstacle_length/2)**2))
    avg_obstacle = 0

    for t in range(timesteps):
        graph = tree_index(T_graph, t)
        
        for i in range(num_agents):
            pos_i = graph.nodes[i, 0:2]
            vel_i = graph.nodes[i, 2:4]
            local_label_agent_agent = 1.0
            local_label_agent_obstacle = 1.0
            unsafeFlag = False
            
            # Agent-Agent Safety Flag
            for j in range(num_agents):
                if i == j:
                    continue
                
                radial_velocity = 0
                edge_ij = graph.edges[i*num_agents + j]
                Px_ij = edge_ij[0]
                Py_ij = edge_ij[1]
                Vx_ij = edge_ij[2]
                Vy_ij = edge_ij[3]
                
                denominator = jnp.sqrt(Px_ij**2 + Py_ij**2) + 1e-6
                radial_velocity = (Px_ij*Vx_ij + Py_ij*Vy_ij)/denominator
                real_remaining_distance = jnp.sqrt(Px_ij**2 + Py_ij**2)
                
                if real_remaining_distance >= sensing_radius:
                    local_label_agent_agent = 1.0
                else:
                    if radial_velocity > 0:
                        local_label_agent_agent = 1.0
                    elif radial_velocity == 0 and real_remaining_distance >= 2.0*agent_radius:
                        local_label_agent_agent = 1.0
                    elif radial_velocity == 0 and real_remaining_distance < 2.0*agent_radius:
                        local_label_agent_agent = 0.0
                        labels = labels.at[t, i].set(0.0)
                        unsafeFlag = True
                        break
                    elif radial_velocity < 0:
                        alphai = actuator_bounds[0]*(1-epistemic_penalties) #Needs to be multiplied with 0.1 since normalizing with mass
                        alphaj = actuator_bounds[1]*(1-epistemic_penalties)
                        minimum_stopping_distance = (radial_velocity**2) / (2*(alphai+alphaj))
                        if real_remaining_distance >= (2.0*agent_radius) + minimum_stopping_distance:
                            local_label_agent_agent = 1.0
                        else:
                            local_label_agent_agent = 0.0
                            labels = labels.at[t, i].set(0.0)
                            unsafeFlag = True
                            break


                if unsafeFlag:
                    #features_array[(t)] = (T_graph, t)
                    continue
            
            # Agent-Obstacle Safety Flag
            num_obstacles = 32
            block1 = num_agents * num_agents
            block2 = num_agents * num_agents
            block3_starting_index = block1 + block2
            
            for j in range(num_obstacles):
                radial_velocity = 0
                edge_ij = graph.edges[block3_starting_index + i*num_obstacles + j]
                Px_ij = edge_ij[0]
                Py_ij = edge_ij[1]
                Vx_ij = edge_ij[2]
                Vy_ij = edge_ij[3]
                
                denominator = jnp.sqrt(Px_ij**2 + Py_ij**2) + 1e-6
                radial_velocity = (Px_ij*Vx_ij + Py_ij*Vy_ij)/denominator
                real_remaining_distance = jnp.sqrt(Px_ij**2 + Py_ij**2)
                
                if real_remaining_distance >= sensing_radius:
                    local_label_agent_obstacle = 1.0
                else:
                    if radial_velocity > 0:
                        local_label_agent_obstacle = 1.0
                    elif radial_velocity == 0 and real_remaining_distance >= (1.0*agent_radius):
                        local_label_agent_obstacle = 1.0
                    elif radial_velocity == 0 and real_remaining_distance < ((1.0*agent_radius) + avg_obstacle):
                        local_label_agent_obstacle = 0.0
                        labels = labels.at[t, i].set(0.0)
                        unsafeFlag = True
                        break
                    elif radial_velocity < 0:
                        alphai = actuator_bounds[0]*(1-epistemic_penalties)
                        alphaj = actuator_bounds[2]*(1-epistemic_penalties)
                        minimum_stopping_distance = (radial_velocity**2) / (2*(alphai+alphaj))
                        if real_remaining_distance >= (1.0*agent_radius) + avg_obstacle + minimum_stopping_distance:
                            local_label_agent_obstacle = 1.0
                        else:
                            local_label_agent_obstacle = 0.0
                            labels = labels.at[t, i].set(0.0)
                            unsafeFlag = True
                            break
                
                if unsafeFlag:
                    #features_array[(t)] = (T_graph, t)
                    continue
            
            if local_label_agent_obstacle == 1.0 and local_label_agent_agent == 1.0:
                labels = labels.at[t, i].set(1.0)

        # Here if any agent in the timestep is unsafe, collect graph.state for the timestep only once and then check for next timestep
        if labels[t].min() == 0.0:
            features_array2[(t)] = (T_graph, t)

    return labels , features_array2

def safety_labeller_with_features(T_graph: GraphsTuple, num_agents: int, sensing_radius: int, agent_radius: int, actuator_bounds: jnp.ndarray):
    
    #Collects ALL pairwise features for agent-agent and agent-obstacle interactions.
    #Mirrors original labeller structure but continues checking all pairs (no early break).
    
    #Returns: features_array of shape (timesteps, num_agents, num_agents + num_obstacles, 8)
    #Each entry: [Px, Py, Vx, Vy, pos_norm, vel_norm, radial_vel, interaction_label]
    
    epistemic_penalties = 0.9
    num_obstacles = 32
    timesteps = 256
    avg_obstacle = 0
    
    # Shape: (timesteps, num_agents, num_agents+num_obstacles, 8)
    features_array = np.zeros((timesteps, num_agents, num_agents + num_obstacles, 9), dtype=np.float32)
    
    # Iterate over all timesteps
    for t in range(timesteps):
        graph = tree_index(T_graph, t)
        
        # Iterate over all agents
        for i in range(num_agents):
            # Agent-Agent Safety Flag - NO EARLY BREAK
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
                vel_norm = np.sqrt(Vx_ij**2 + Vy_ij**2)
                
                # Safety check for agent-agent pair
                label = 1.0
                if real_remaining_distance < sensing_radius:
                    if radial_velocity > 0:
                        label = 1.0
                    elif radial_velocity == 0 and real_remaining_distance >= 2.0*agent_radius:
                        label = 1.0
                    elif radial_velocity == 0 and real_remaining_distance < 2.0*agent_radius:
                        label = 0.0
                    elif radial_velocity < 0:
                        alphai = actuator_bounds[0] * (1 - epistemic_penalties)
                        alphaj = actuator_bounds[1] * (1 - epistemic_penalties)
                        minimum_stopping_distance = (radial_velocity**2) / (2*(alphai+alphaj))
                        if real_remaining_distance < (2.0*agent_radius) + minimum_stopping_distance:
                            label = 0.0
                
                # Store all 8 features for agent-agent pair
                features_array[t, i, j, :] = [Px_ij, Py_ij, Vx_ij, Vy_ij, real_remaining_distance, vel_norm, radial_velocity, label, 0]
                ##Here 0 means that it is agent agent interaction
            
            # Agent-Obstacle Safety Flag - NO EARLY BREAK
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
                vel_norm = np.sqrt(Vx_ik**2 + Vy_ik**2)
                
                # Safety check for agent-obstacle pair
                label = 1.0
                if real_remaining_distance < sensing_radius:
                    if radial_velocity > 0:
                        label = 1.0
                    elif radial_velocity == 0 and real_remaining_distance >= (1.0*agent_radius):
                        label = 1.0
                    elif radial_velocity == 0 and real_remaining_distance < ((1.0*agent_radius) + avg_obstacle):
                        label = 0.0
                    elif radial_velocity < 0:
                        alphai = actuator_bounds[0] * (1 - epistemic_penalties)
                        alphaj = actuator_bounds[2] * (1 - epistemic_penalties)
                        minimum_stopping_distance = (radial_velocity**2) / (2*(alphai+alphaj))
                        if real_remaining_distance < (1.0*agent_radius) + avg_obstacle + minimum_stopping_distance:
                            label = 0.0
                
                # Store all 8 features for agent-obstacle pair
                features_array[t, i, num_agents + k, :] = [Px_ik, Py_ik, Vx_ik, Vy_ik, real_remaining_distance, vel_norm, radial_velocity, label, 1]
                ##Here 1 means that it is agent obstacle interaction
    
    return features_array 
