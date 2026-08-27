## only an init labeller function
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

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.trainer.utils import get_bb_cbf_all_agents, get_bb_cbf_in_all_agents
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap

epistemic_penalties = 0.9 ## Set to 1 for unrealistic, perfect, environment.
#Logic-Only
def safety_labeller(T_graph: GraphsTuple, num_agents: int, sensing_radius: int, agent_radius: int, actuator_bounds: jnp.ndarray):
    timesteps = 256
    labels = jnp.ones((timesteps, num_agents))
    worst_obstacle_length = 0.5
    avg_obstacle_length = 0.25
    worst_obstacle = jnp.sqrt(2*((worst_obstacle_length/2)**2))
    avg_obstacle = 0 #jnp.sqrt(2*((avg_obstacle_length/2)**2))
    
    #Iterate over all tiesteps
    for t in range(timesteps):
        graph = tree_index(T_graph, t)
        #iterate over all agents
        for i in range(num_agents):
            #Retrieve its position and velocity only for logging purpose, wont be used anywhere
            pos_i = graph.nodes[i, 0:2] # Gets Px and Py
            vel_i = graph.nodes[i, 2:4] # Gets Vx and Vy
            
            local_label_agent_agent = 1.0 #Safe by default
            local_label_agent_obstacle = 1.0 #Safe by default
            unsafeFlag = False

            #Iterate over all agent-agent pairs for agent i in Edges
            #Agent-Agent Safety Flag
            for j in range(num_agents):
                if i == j:
                    continue
                radial_velocity= 0
                #Not self-loop
                edge_ij = graph.edges[i*num_agents + j]
                Px_ij = edge_ij[0]
                Py_ij = edge_ij[1]
                Vx_ij = edge_ij[2]
                Vy_ij = edge_ij[3]
                denominator = jnp.sqrt(Px_ij**2 + Py_ij**2) + 1e-6 # To avoid nan
                radial_velocity = (Px_ij*Vx_ij + Py_ij*Vy_ij)/denominator
                real_remaining_distance = jnp.sqrt(Px_ij**2 + Py_ij**2)
                
                #Create notion of sensing radius
                if real_remaining_distance >= sensing_radius:
                    local_label_agent_agent = 1.0 #Safe by default
                else:
                    if radial_velocity > 0: #Going away - always safe
                        local_label_agent_agent = 1.0
                    elif radial_velocity == 0 and real_remaining_distance >= 2.0*agent_radius:
                        local_label_agent_agent = 1.0
                    elif radial_velocity == 0 and real_remaining_distance < 2.0*agent_radius:
                        local_label_agent_agent = 0.0
                        labels = labels.at[t, i].set(0.0)
                        unsafeFlag = True
                        break
                    elif radial_velocity < 0: #Going closer
                        alphai = actuator_bounds[0]*(1-epistemic_penalties)
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
                continue #Skip to next agent

            #End of agent-agent pairs for agent i

            #Agent-Obstacle Safety Flag
            num_obstacles = 32
            block1 = num_agents * num_agents
            block2 = num_agents * num_agents ## Because of hidden/masked edges in original code
            block3_starting_index = block1 + block2  #(To get it to 0 based indexing) 51st edge will be edge[50] 
            for j in range(num_obstacles):
                radial_velocity= 0
                edge_ij = graph.edges[block3_starting_index + i*num_obstacles + j]
                Px_ij = edge_ij[0]
                Py_ij = edge_ij[1]
                Vx_ij = edge_ij[2]
                Vy_ij = edge_ij[3]
                denominator = jnp.sqrt(Px_ij**2 + Py_ij**2) + 1e-6 # To avoid nan
                radial_velocity = (Px_ij*Vx_ij + Py_ij*Vy_ij)/denominator
                real_remaining_distance = jnp.sqrt(Px_ij**2 + Py_ij**2)

                #Create notion of sensing radius
                #This will worrk even in case of obstacles, since worst case obstacle size is 0.25*sqrt(2) = 0.3533
                #Max a is 10
                #Put everything in equation --> Max stopping distance = (0.707^2)/(2*10) = 0.003
                #0.5> 0.003 + 0.05 + 0.3533
                # But this only works for this hard coded param of double integrator (mass hard coded, velocity hard coded, acceleration hard coded, obstacle size and drone size hard coded)
                # We can generalise  by branching into another if condition where sensing radius doesnt matter, but no need for our logging purposes - since math is shown to work 
                if real_remaining_distance >= sensing_radius:
                    local_label_agent_obstacle = 1.0 #Safe by default
                else:
                    if radial_velocity > 0: #Going away - always safe
                        local_label_agent_obstacle = 1.0
                    elif radial_velocity == 0 and real_remaining_distance >= (1.0*agent_radius):
                        local_label_agent_obstacle = 1.0
                    elif radial_velocity == 0 and real_remaining_distance < ((1.0*agent_radius) + avg_obstacle):
                        local_label_agent_obstacle = 0.0
                        labels = labels.at[t, i].set(0.0)
                        unsafeFlag = True
                        break
                    elif radial_velocity < 0: #Going closer
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
            continue
                #End of agent-obstacle pairs for agent i
        if local_label_agent_obstacle == 1.0 and local_label_agent_agent == 1.0:
            labels = labels.at[t, i].set(1.0)
    return labels
