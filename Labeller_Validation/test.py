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
import pandas as pd
import sys
import matplotlib.pyplot as plt

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.trainer.utils import get_bb_cbf_all_agents, get_bb_cbf_in_all_agents
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap
from labeller_and_collector import safety_labeller, safety_labeller_with_features
from logger import log_unsafe_agents_to_csv , log_balanced_safe_agents_to_csv, log_all_agents_to_csv
#from unsafe_generator import generate_multiple_unsafe_scenarios, analyze_unsafe_scenarios
#np.set_printoptions(threshold=sys.maxsize)

def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    print("number of agents to log: ", num_agents)
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    if not args.u_ref:
        if args.path is not None:
            path = args.path
            model_path = os.path.join(path, "models")
            if args.step is None:
                models = os.listdir(model_path)
                step = max([int(model) for model in models if model.isdigit()])
            else:
                step = args.step
            print("step: ", step)

            algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                gnn_layers=config.gnn_layers,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                horizon=config.horizon,
                lr_actor=config.lr_actor,
                lr_cbf=config.lr_cbf,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed
            )
            algo.load(model_path, step)
            act_fn = jax.jit(algo.act)
        else:
            algo = make_algo(
                algo=args.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
            )
            act_fn = jax.jit(algo.act)
            path = os.path.join(f"./logs/{args.env}/{args.algo}")
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(os.path.join("./logs", args.env)):
            os.mkdir(os.path.join("./logs", args.env))
        if not os.path.exists(path):
            os.mkdir(path)
        algo = None
        act_fn = jax.jit(env.u_ref)
        step = 0

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecShareCBF))

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf_all_agents, algo.get_cbf, env, agent_id = None, x_dim=0, y_dim=1, num_agents= num_agents)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)
        get_bb_cbf_in_fn_ = ft.partial(get_bb_cbf_in_all_agents, algo.get_cbf_inner_layer, env, agent_id = None,  x_dim=0, y_dim=1, num_agents= num_agents)
        get_bb_cbf_in_fn_ = jax_jit_np(get_bb_cbf_in_fn_)

        """ def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h """
        
        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
        
        def get_bb_cbf_in_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            
            #print("T_graph", T_graph)
            #print("state, node, edge lengths",len(T_graph.states), len(T_graph.nodes), len(T_graph.edges))
            #print("T", T)
            N = 256
            Timestep1 = T_graph.states[0]
            Timestep_Nplus1 = T_graph.states[N]
            #print("Timestep1", Timestep1)
            #print("TimestepN", Timestep_Nplus1)
            #print("Shape Timestep1", jnp.shape(Timestep1))
            #print("Nodes at Timestep1", T_graph.nodes[0] , "with shape", jnp.shape(T_graph.nodes[0]))
            #print("Nodes at TimestepN", T_graph.nodes[N] , "with shape", jnp.shape(T_graph.nodes[N]))
            #print("Edges at Timestep1", T_graph.edges[0], "with shape", jnp.shape(T_graph.edges[0]))
            #print("Edges at TimestepN", T_graph.edges[N], "with shape", jnp.shape(T_graph.edges[N]))
            #print("Checking for velocities at TimestepN", T_graph.states[N])
            """"SELF-
            Tracker to see if edges between agents are actually updating"""
            graph_N = tree_index(T_graph, N)
            mask_N = (graph_N.node_type[graph_N.senders] == 0) & (graph_N.node_type[graph_N.receivers] == 0)

            total_edges_StepN = len(T_graph.edges[N])
            agent_goal_edges = num_agents
            agent_agent_edges = int(jnp.sum(mask_N))
            agent_obstacle_edges = total_edges_StepN - agent_goal_edges - agent_agent_edges
            #print("Total edges at timestep N:", total_edges_StepN)
            #print("Agent-goal edges at timestep N:", agent_goal_edges)
            #print("Agent-agent edges COUNT at timestep N:", agent_agent_edges)
            #print("Agent-agent edges:", graph_N.edges[:agent_agent_edges])
            #print("VerrifierBlock:", T_graph.edges[N][:agent_agent_edges])
            #mask = (T_graph.node_type[T_graph.senders] == 0) & (T_graph.node_type[T_graph.receivers] == 0)
            #aa_edges = jnp.where(mask)[0]
            #print("Agent-agent edge indices at timestep {N}:",  T_graph.edges[N][aa_edges])
            ### END OF TRACKER ###

            outs = [get_bb_cbf_in_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tbb_h_in = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tbb_h_in
    else:
        get_bb_cbf_in_fn = None
        get_bb_cbf_fn = None
        cbf_fn = None

    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")
        rollout_fn = env.rollout_fn_jitstep(act_fn, args.max_step, noedge=True, nograph=args.no_video)

        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")
        rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step))

        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))

    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    rates = []
    cbfs = []
    cbfs_in = []
    actuator_bounds= [] # index 0 is agent accn max in x, index 1 is agent accn max in y, index 2 is obstacle accn max

    #SELF-
    F_lower, F_Upper = env.action_lim()
    mass_agent = env.params['m']
    #F_lower[0], F_lower[1] = jnp.sqrt(F_lower[0]**2 + F_lower[1]**2)
    #F_Upper[0], F_Upper[1] = jnp.sqrt(F_Upper[0]**2 + F_Upper[1]**2)
    actuator_bounds.append(jnp.sqrt(F_Upper[0]**2 + F_Upper[1]**2)/mass_agent)
    actuator_bounds.append(jnp.sqrt(F_Upper[0]**2 + F_Upper[1]**2)/mass_agent)
    actuator_bounds.append(0.0)
    print("Agent acceleration bounds alphai and alphaj", actuator_bounds[0], actuator_bounds[1])
    print("Obstacle acceleration bounds", actuator_bounds[2])

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        if args.nojit_rollout:
            rollout: RolloutResult
            rollout, is_unsafe, is_finish = rollout_fn(key_x0)
            # if not jnp.isnan(rollout.T_reward).any():
            is_unsafes.append(is_unsafe)
            is_finishes.append(is_finish)
        else:
            rollout: RolloutResult = rollout_fn(key_x0)
            # if not jnp.isnan(rollout.T_reward).any():
            is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
            is_finishes.append(is_finish_fn(rollout.Tp1_graph))

        """ obstacle_info = env.get_obstacle_info()
        print(f"\nObstacle Info for Episode {i_epi}:")
        for obs_id, info in obstacle_info.items():
            print(f"  Obstacle {obs_id}: Pos({info['x_center']:.3f}, {info['y_center']:.3f}), "
                f"Size({info['length']:.3f}x{info['breadth']:.3f}), Angle({info['rotation_angle']:.3f})")
 """
        
        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
            cbfs_in.append(get_bb_cbf_in_fn(rollout.Tp1_graph))
            #print("cbf_in", get_bb_cbf_in_fn(rollout.Tp1_graph))
            #print(type(get_bb_cbf_in_fn(rollout.Tp1_graph)))
            #print("Shape intermediate layer output", jnp.shape(get_bb_cbf_in_fn(rollout.Tp1_graph)))
            #print("cbf", cbfs) , #print("cbf_in", cbfs_in)
            #print(type(get_bb_cbf_fn(rollout.Tp1_graph)))
            #print("Shape main cbf output", len(get_bb_cbf_fn(rollout.Tp1_graph)))

            #Safety labeller function call"""
            print(f"\n--- Safety Labeling for Episode {i_epi} ---")
            sensing_radius = env.params['comm_radius']  # 0.5
            agent_radius = env.params['car_radius']     # 0.05
            # Call safety labeler - Original
            safety_labels = safety_labeller(
                T_graph=rollout.Tp1_graph,
                num_agents=num_agents,
                sensing_radius=sensing_radius,
                agent_radius=agent_radius,
                actuator_bounds=jnp.array(actuator_bounds)
            )
            df = pd.DataFrame(np.array(safety_labels))
            print(df.head(10))
            print("Safety labels shape", jnp.shape(safety_labels))
            print(f"Safety labels shape: {safety_labels.shape}")  # Should be (256, num_agents)
            print(f"    Total safe agents across max tiestep: {jnp.sum(safety_labels)}")
            print(f"    Total unsafe agents across max timesteps: {jnp.sum(1 - safety_labels)}")
            print(f"\n--- --------------------------------------")

            print(f"\nExtracting agent interaction data for Episode {i_epi}...")

            features_array = safety_labeller_with_features(
                T_graph=rollout.Tp1_graph,
                num_agents=num_agents,
                sensing_radius=sensing_radius,
                agent_radius=agent_radius,
                actuator_bounds=jnp.array(actuator_bounds)
            )
            # Extract unsafe and safe cases
            print(f"\n=== DEBUG: Analyzing data structure ===")
            print(f"features_array shape: {features_array.shape}")
            print(f"num_agents: {num_agents}")
            print(f"Agent-agent range: j from 0 to {num_agents-1}")
            print(f"Agent-obstacle range: j from {num_agents} to {num_agents + 32 - 1}")
            print(f"Extraction loop iterates j from 0 to {features_array.shape[2]-1}")
            print(f"Features at j={num_agents}: {features_array[0, 0, num_agents, :]}")
            print(f"Features at j={num_agents+5}: {features_array[0, 0, num_agents+5, :]}")
            print("=== END DEBUG ===\n")
            
            unsafe_data_list = []

            for t in range(features_array.shape[0]):
                for i in range(features_array.shape[1]):
                    for j in range(features_array.shape[2]):
                        if i == j:
                            continue
                        
                        rel_pos_mag = features_array[t, i, j, 4]
                        rel_vel_mag = features_array[t, i, j, 5]
                        radial_vel = features_array[t, i, j, 6]
                        safety_label = features_array[t, i, j, 7]
                        collision_type = features_array[t, i, j, 8]  # 0 for agent-agent, 1 for agent-obstacle
                        
                        # ONLY collect if TRULY UNSAFE and non-zero
                        if safety_label == 0 and (rel_pos_mag > 1e-6 or rel_vel_mag > 1e-6):
                            unsafe_data_list.append({
                                'timestep': t,
                                'agent_i': i,
                                'agent_j': j,
                                'rel_pos_mag': rel_pos_mag,
                                'rel_vel_mag': rel_vel_mag,
                                'radial_vel': radial_vel,
                                'collision_type': collision_type,
                            })
            """ for t in range(features_array.shape[0]):
                for i in range(features_array.shape[1]):
                    if safety_labels[t,i]==0:
                        for j in range(features_array.shape[2]):
                            if i == j:
                                continue
                        
                        rel_pos_mag = features_array[t, i, j, 4]
                        rel_vel_mag = features_array[t, i, j, 5]
                        radial_vel = features_array[t, i, j, 6]
                        
                        # SKIP zero-valued pairs
                        if rel_pos_mag < 1e-6 and rel_vel_mag < 1e-6:
                            continue
                        
                        # RE-CHECK safety logic using stored data
                        is_unsafe_pair = False
                        if rel_pos_mag < sensing_radius:
                            if radial_vel < 0:  # Approaching
                                alphai = actuator_bounds[0] * (1 - 0.9)
                                alphaj = actuator_bounds[1] * (1 - 0.9)
                                min_stop_dist = (radial_vel**2) / (2*(alphai+alphaj))
                                if rel_pos_mag < (2*agent_radius) + min_stop_dist:
                                    is_unsafe_pair = True
                            elif radial_vel == 0 and rel_pos_mag < 2*agent_radius:
                                is_unsafe_pair = True
                        
                        if is_unsafe_pair:
                            unsafe_data_list.append({
                                'timestep': t,
                                'agent_i': i,
                                'agent_j': j,
                                'rel_pos_mag': rel_pos_mag,
                                'rel_vel_mag': rel_vel_mag,
                                'radial_vel': radial_vel,
                            })

            # DEBUG: Count what's actually in features_array before extraction
            print("\n=== DEBUG: Analyzing features_array ===")
            agent_agent_unsafe_count = 0
            agent_obstacle_unsafe_count = 0
            zero_pairs_count = 0
            total_pairs = 0

            for t in range(features_array.shape[0]):
                for i in range(features_array.shape[1]):
                    for j in range(features_array.shape[2]):
                        if i == j:
                            continue
                        
                        total_pairs += 1
                        rel_pos_mag = features_array[t, i, j, 4]
                        rel_vel_mag = features_array[t, i, j, 5]
                        radial_vel = features_array[t, i, j, 6]
                        stored_label = features_array[t, i, j, 7]
                        
                        if rel_pos_mag < 1e-6 and rel_vel_mag < 1e-6:
                            zero_pairs_count += 1
                        else:
                            if stored_label == 0:  # What the labeller stored
                                if j < num_agents:  # Agent-agent
                                    agent_agent_unsafe_count += 1
                                else:  # Agent-obstacle (conceptually)
                                    agent_obstacle_unsafe_count += 1

            print(f"Total pairs examined: {total_pairs}")
            print(f"Zero-valued pairs: {zero_pairs_count}")
            print(f"Agent-agent pairs with label=0: {agent_agent_unsafe_count}")
            print(f"Agent-obstacle pairs with label=0: {agent_obstacle_unsafe_count}")

            # DEBUG: Now count what extraction finds
            print("\n=== DEBUG: After extraction ===")
            print(f"Unsafe interactions extracted: {len(unsafe_data_list)}")

            # Show first 20 of them
            if len(unsafe_data_list) > 0:
                print("\nFirst entries:")
                for entry in unsafe_data_list[:20]:
                    print(f"  t={entry['timestep']}, i={entry['agent_i']}, j={entry['agent_j']}, pos={entry['rel_pos_mag']:.4f}, vel={entry['rel_vel_mag']:.4f}, radial={entry['radial_vel']:.4f}")
 """

            # Save unsafe interactions to CSV
            if len(unsafe_data_list) > 0:
                unsafe_df = pd.DataFrame(unsafe_data_list)
                unsafe_csv_path = f"unsafe_interactions_epi{i_epi}_n{num_agents}.csv"
                unsafe_df.to_csv(unsafe_csv_path, index=False)
                #print(f"✓ Saved {len(unsafe_data_list)} unsafe interactions to {unsafe_csv_path}")
            else:
                #print(f"⚠ No unsafe interactions found in Episode {i_epi}")
                unsafe_df = None

            # Calculate decision boundaries
            
            area_size = args.area_size if hasattr(args, 'area_size') else 10  # Default 10
            alphai = actuator_bounds[0] * (1 - 0.9) #Normalizing with mass since we divided it in force
            alphaj = actuator_bounds[1] * (1 - 0.9)
            total_alpha1 = alphai + alphaj
            total_alpha2 = alphai + actuator_bounds[2]

            print(f"total acceleration for agent-agent interactions: {total_alpha1}")
            print(f"total acceleration for agent-obstacle interactions: {total_alpha2}")
            # Create position range from 0 to area_size at 0.001 increments. If not fine enough then discontinuouspots
            pos_range = np.arange(0, area_size, 0.001)

            
            critical_radial_vel1 = []
            for pos in pos_range:
                if pos <= 2*agent_radius:
                    # Within collision radius - always unsafe
                    critical_radial_vel1.append(0)  # Any radial_vel <= 0 is unsafe
                else:
                    # Beyond collision radius - calculate stopping distance boundary
                    critical_vel = -np.sqrt(2 * total_alpha1 * (pos - 2*agent_radius))
                    critical_radial_vel1.append(critical_vel)

            critical_radial_vel2 = []
            for pos in pos_range:
                if pos <= agent_radius:
                    critical_radial_vel2.append(0)  # Any radial_vel <= 0 is unsafe
                else:
                    critical_vel = -np.sqrt(2 * total_alpha2 * (pos - agent_radius))
                    critical_radial_vel2.append(critical_vel)

            critical_radial_vel1 = np.array(critical_radial_vel1)
            critical_radial_vel2 = np.array(critical_radial_vel2)
            agent_obstacle_radius = 1.0 * agent_radius  
            agent_agent_radius = 2.0 * agent_radius
            agent_agent = 2*agent_radius
            agent_obstacle = agent_radius
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # ========== PLOT 1: Radial Velocity vs Relative Position with Boundary ==========

            colors = {'bloodred': '#880808', 'jetblack': '#0A0A0A', 'gungrey': '#818589', 'silver': '#C0C0C0', 'bluegrey': '#7393B3'}

            if unsafe_df is not None and len(unsafe_df) > 0:
                
                agent_agent_df = unsafe_df[unsafe_df['collision_type'] == 0]
                agent_obstacle_df = unsafe_df[unsafe_df['collision_type'] == 1]
                
                # Plot agent-agent collisions
                if len(agent_agent_df) > 0:
                    axes[0].scatter(agent_agent_df['rel_pos_mag'], agent_agent_df['radial_vel'], 
                                alpha=0.75, s=30, color='white', edgecolors=colors['jetblack'], 
                                linewidth=0.6, label='Unsafe (A-A)', zorder=3)
                
                # Plot agent-obstacle collisions
                if len(agent_obstacle_df) > 0:
                    axes[0].scatter(agent_obstacle_df['rel_pos_mag'], agent_obstacle_df['radial_vel'], 
                                alpha=0.75, s=30, color=colors['bloodred'], edgecolors=colors['jetblack'], 
                                linewidth=0.6, label='Unsafe (A-O)', zorder=3)

            # Plot the decision boundary
            axes[0].plot(pos_range, critical_radial_vel1, 
                        color=colors['bluegrey'], linestyle='--', linewidth=2, 
                        label='A-A Dec Boundary', zorder=2)
            axes[0].plot(pos_range, critical_radial_vel2, 
                        color=colors['jetblack'], linestyle='--', linewidth=2, 
                        label='A-O Dec Boundary', zorder=2)

            # Shade the unsafe region
            #axes[0].fill_between(pos_range, critical_radial_vel, -2, 
            #                    alpha=0.2, color=colors['gungrey'], label='Theoretical unsafe zone')
            # Agent-obstacle unsafe zone (0 to agent_obstacle_radius)
            mask_obs = pos_range <= agent_agent_radius
            if mask_obs.sum() > 0:
                pos_obs = pos_range[mask_obs]
                vel_obs = critical_radial_vel1[mask_obs]
                # Remove any NaN/inf values
                valid_mask = np.isfinite(vel_obs)
                axes[0].fill_between(pos_obs[valid_mask], vel_obs[valid_mask], -2, 
                                    alpha=0.5, color=colors['bluegrey'], label='A-A, A-O unsafe zone')
            mask_obs = pos_range <= agent_agent_radius
            if mask_obs.sum() > 0:
                pos_obs = pos_range[mask_obs]
                vel_obs = critical_radial_vel2[mask_obs]
                # Remove any NaN/inf values
                valid_mask = np.isfinite(vel_obs)
                axes[0].fill_between(pos_obs[valid_mask], vel_obs[valid_mask], -2, 
                                    alpha=0.95, color=colors['bluegrey'], label='A-A, A-O unsafe zone')
            # Rest of unsafe zone (beyond agent_agent_radius)
            mask_rest = pos_range > agent_agent_radius
            if mask_rest.sum() > 0:
                pos_rest = pos_range[mask_rest]
                vel_rest = critical_radial_vel1[mask_rest]
                # Remove any NaN/inf values
                valid_mask = np.isfinite(vel_rest)
                axes[0].fill_between(pos_rest[valid_mask], vel_rest[valid_mask], -2, 
                                    alpha=0.4, color=colors['gungrey'], label='General unsafe zone')
            mask_rest = pos_range > agent_agent_radius
            if mask_rest.sum() > 0:
                pos_rest = pos_range[mask_rest]
                vel_rest = critical_radial_vel2[mask_rest]
                # Remove any NaN/inf values
                valid_mask = np.isfinite(vel_rest)
                axes[0].fill_between(pos_rest[valid_mask], vel_rest[valid_mask], -2, 
                                    alpha=0.2, color=colors['gungrey'], label='General unsafe zone')



            axes[0].axvline(x=sensing_radius, color=colors['bloodred'], linestyle=':', linewidth=2, 
                        alpha=0.8, label=f'Sensing Radius ({sensing_radius})')
            axes[0].axvline(x=agent_agent, color=colors['jetblack'], linestyle=':', linewidth=2, 
                        alpha=0.8, label=f'A-A Always Unsafe')
            axes[0].axvline(x=agent_obstacle, color=colors['jetblack'], linestyle=':', linewidth=2, 
                        alpha=0.8, label=f'A-O Always Unsafe')


            axes[0].set_xlabel('Relative Position Magnitude ||p_ij|| 0th norm', fontsize=12, fontweight='normal')
            axes[0].set_ylabel('Radial Velocity (negative=approaching)', fontsize=12, fontweight='normal')
            axes[0].set_title(f'Radial Velocity vs Position with Decision Boundary\nEpisode {i_epi}, N={num_agents} agents', 
                            fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].legend(fontsize=9, loc='upper right')
            axes[0].set_xlim(0, min(area_size, sensing_radius*1.5))
            axes[0].set_ylim(-2, 0.5)

            # ========== PLOT 2: Relative Velocity Magnitude vs Position ==========
            if unsafe_df is not None and len(unsafe_df) > 0:
                axes[1].scatter(unsafe_df['rel_pos_mag'], unsafe_df['rel_vel_mag'], 
                            alpha=0.75, s=30, color=colors['bloodred'], edgecolors=colors['jetblack'], 
                            linewidth=0.6, label='Unsafe (labeller)')

            # Add reference regions
            axes[1].fill_between([0, 2*agent_radius], 0, 3, alpha=0.8, color=colors['bluegrey'], 
                                label='A-A, A-O unsafe zone')

            axes[1].axvline(x=sensing_radius, color=colors['bloodred'], linestyle=':', linewidth=2, 
                        alpha=0.8, label=f'Sensing Radius ({sensing_radius})')
            axes[1].axvline(x=agent_agent, color=colors['jetblack'], linestyle=':', linewidth=2, 
                        alpha=0.8, label=f'A-A Always Unsafe')
            axes[1].axvline(x=agent_obstacle, color=colors['jetblack'], linestyle=':', linewidth=2, 
                        alpha=0.8, label=f'A-O Always Unsafe')

            axes[1].set_xlabel('Relative Position Magnitude ||p_ij|| 0th norm', fontsize=12, fontweight='normal')
            axes[1].set_ylabel('Relative Velocity Magnitude ||v_ij|| 0th norm', fontsize=12, fontweight='normal')
            axes[1].set_title(f'Relative Velocity Magnitude vs Relative Position\nEpisode {i_epi}, N={num_agents} agents', 
                            fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend(fontsize=9)
            axes[1].set_xlim(0, min(area_size, sensing_radius*1.5))
            axes[1].set_ylim(0, 2.5)

            plt.tight_layout()
            plot_path = f"diagnostic_plots_epi{i_epi}_n{num_agents}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            # ============================================================================
            # END DIAGNOSTIC PLOTTING CODE
            # ============================================================================



            # Log unsafe agents to CSV
            log_unsafe_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=cbfs[i_epi],
                cbf_in_values=cbfs_in[i_epi],
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                csv_path=f"unsafe_agents_log{num_agents}.csv")
            # Log balanced safe agents to CSV
            num_unsafe_samples = int(jnp.sum(1 - safety_labels))
            log_balanced_safe_agents_to_csv(rollout_result=rollout,
                                            safety_labels=safety_labels,
                cbf_values=cbfs[i_epi],
                cbf_in_values=cbfs_in[i_epi],
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path=f"safe_agents_log{num_agents}.csv",
                random_seed=args.seed + i_epi)  # Different seed per episode
            # Log all agents to CSV
            log_all_agents_to_csv(rollout_result=rollout,
                                  safety_labels=safety_labels,
                cbf_values=cbfs[i_epi],
                cbf_in_values=cbfs_in[i_epi],
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                csv_path=f"all_agents_log{num_agents}.csv")
            
        else:
            cbfs.append(None)
            cbfs_in.append(None)
        if len(is_unsafes) == 0:
            continue
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        finish_rate = is_finishes[-1].max(axis=0).mean()
        success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
              f"safe rate: {safe_rate * 100:.3f}%,"
              f"finish rate: {finish_rate * 100:.3f}%, "
              f"success rate: {success_rate * 100:.3f}%")

        rates.append(np.array([safe_rate, finish_rate, success_rate]))
    
    #print("1", len(cbfs), "2", len(cbfs_in))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    is_finish = np.max(np.stack(is_finishes), axis=1)

    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    finish_mean, finish_std = is_finish.mean(), is_finish.std()
    success_mean, success_std = ((1 - is_unsafe) * is_finish).mean(), ((1 - is_unsafe) * is_finish).std()

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, "
        f"finish_rate: {finish_mean * 100:.3f}%, "
        f"success_rate: {success_mean * 100:.3f}%"
    )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f},"
                    f"{finish_mean * 100:.3f},{finish_std * 100:.3f},"
                    f"{success_mean * 100:.3f},{success_std * 100:.3f}\n")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe, cbf) in enumerate(zip(rollouts, is_unsafes, cbfs)):
        if algo_is_cbf:
            safe_rate, finish_rate, success_rate = rates[ii] * 100
            video_name = f"n{num_agents}_epi{ii:02}_sr{safe_rate:.0f}_fr{finish_rate:.0f}_sr{success_rate:.0f}"
        else:
            video_name = f"n{num_agents}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"

        viz_opts = {}
        if args.cbf is not None:
            video_name += f"_cbf{args.cbf}"
            viz_opts["cbf"] = [*cbf, args.cbf]

        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
