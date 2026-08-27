import argparse
import datetime
import os
import pathlib
import pickle
import sys

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.algo import make_algo
from model.env import make_env
from model.env.base import RolloutResult
from model.utils.utils import jax_vmap, tree_index, jax_jit_np

from labeller_and_collector import safety_labeller, safety_labeller_with_features
from logger import (
    log_unsafe_agents_gnn_update,
    log_balanced_safe_agents_gnn_update,
    log_unsafe_agents_gnn_msg,
    log_safe_agents_gnn_msg,
    log_unsafe_agents_to_csv,
    log_balanced_safe_agents_to_csv,
    log_all_agents_to_csv,
)


def test(args):
    print(f"> Running test_mappo.py {args}")
    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    config = None
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    if config is not None:
        env_id = config.env if args.env is None else args.env
        num_agents = config.num_agents if args.num_agents is None else args.num_agents
    else:
        assert args.env is not None, "Provide --env when --path is not given."
        assert args.num_agents is not None, "Provide --num-agents when --path is not given."
        env_id = args.env
        num_agents = args.num_agents

    print("number of agents to log:", num_agents)

    env = make_env(
        env_id=env_id,
        num_agents=num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
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
            print("step:", step)

            algo = make_algo(
                algo="mappo",
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
                lr_critic=config.lr_critic if hasattr(config, "lr_critic") else 3e-5,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed,
                cost_weight=config.cost_weight if hasattr(config, "cost_weight") else 1.0,
            )

            ckpt_path = os.path.join(model_path, str(step))
            with open(os.path.join(ckpt_path, "actor.pkl"), "rb") as f:
                actor_params = pickle.load(f)
            with open(os.path.join(ckpt_path, "critic.pkl"), "rb") as f:
                critic_params = pickle.load(f)

            algo.policy_train_state = algo.policy_train_state.replace(params=actor_params)
            algo.critic_train_state = algo.critic_train_state.replace(params=critic_params)

            algo.policy.params = algo.policy_train_state.params  # Already set above
            algo.critic.params = algo.critic_train_state.params  # Already set above

            act_fn = jax.jit(algo.act)
        else:
            algo = make_algo(
                algo="mappo",
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
                lr_critic=args.lr_critic,
                cost_weight=args.cost_weight,
                gnn_layers=args.gnn_layers,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                horizon=args.horizon,
                lr_actor=args.lr_actor,
                lr_cbf=args.lr_cbf,
                loss_action_coef=args.loss_action_coef,
                loss_unsafe_coef=args.loss_unsafe_coef,
                loss_safe_coef=args.loss_safe_coef,
                loss_h_dot_coef=args.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=args.seed,
            )
            act_fn = jax.jit(algo.act)
            path = os.path.join(f"./logs/{args.env}/mappo")
            os.makedirs(path, exist_ok=True)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        os.makedirs(path, exist_ok=True)
        algo = None
        act_fn = jax.jit(env.u_ref)
        step = 0

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1000)[: args.epi]
    test_keys = test_keys[args.offset:]

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

    #gnn_msg_intermediates = []
    #gnn_aggr_intermediates = []
    #gnn_update_intermediates = []
    #policy_head_intermediates = []
    critic_L0_msg_intermediates = []
    critic_L0_aggr_intermediates = []
    critic_L0_update_intermediates = []
    critic_L1_msg_intermediates = [] 
    critic_L1_aggr_intermediates = [] 
    critic_L1_update_intermediates = []
    critic_value_attn_intermediates = []

    F_lower, F_upper = env.action_lim()
    mass_agent = env.params["m"]
    actuator_bounds = [
        jnp.sqrt(F_upper[0] ** 2 + F_upper[1] ** 2) / mass_agent,
        jnp.sqrt(F_upper[0] ** 2 + F_upper[1] ** 2) / mass_agent,
        0.0,
    ]

    '''def get_gnn_msg_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_gnn_msg_intermediate(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_gnn_aggr_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_gnn_aggr_intermediate(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_gnn_update_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_gnn_update_intermediate(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_policy_head_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_policy_head_intermediate(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)'''

    def get_critic_L0_msg_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_L0_msg(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_critic_L0_aggr_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_L0_aggr(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_critic_L0_update_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_L0_update(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_critic_value_attn_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_value_attn(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_critic_L1_msg_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_L1_msg(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_critic_L1_aggr_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_L1_aggr(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    def get_critic_L1_update_fn(T_graph):
        T = min(T_graph.states.shape[0] - 1, 256)
        outs = [algo.get_critic_L1_update(tree_index(T_graph, kk)) for kk in range(T)]
        return jnp.stack(outs, axis=0)

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        if args.nojit_rollout:
            rollout: RolloutResult
            rollout, is_unsafe, is_finish = rollout_fn(key_x0)
            is_unsafes.append(is_unsafe)
            is_finishes.append(is_finish)
        else:
            rollout: RolloutResult = rollout_fn(key_x0)
            is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
            is_finishes.append(is_finish_fn(rollout.Tp1_graph))

        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        if algo is not None:
            critic_L0_msg   = get_critic_L0_msg_fn(rollout.Tp1_graph)
            critic_L0_aggr  = get_critic_L0_aggr_fn(rollout.Tp1_graph)
            critic_L0_update = get_critic_L0_update_fn(rollout.Tp1_graph)
            critic_value_attn = get_critic_value_attn_fn(rollout.Tp1_graph)
            critic_L1_msg   = get_critic_L1_msg_fn(rollout.Tp1_graph)
            critic_L1_aggr  = get_critic_L1_aggr_fn(rollout.Tp1_graph)
            critic_L1_update = get_critic_L1_update_fn(rollout.Tp1_graph)

            # store them exactly like the policy lists did
            critic_L0_msg_intermediates.append(critic_L0_msg)
            critic_L0_aggr_intermediates.append(critic_L0_aggr)
            critic_L0_update_intermediates.append(critic_L0_update)
            critic_value_attn_intermediates.append(critic_value_attn)
            critic_L1_msg_intermediates.append(critic_L1_msg)
            critic_L1_aggr_intermediates.append(critic_L1_aggr)
            critic_L1_update_intermediates.append(critic_L1_update)
            
            print("L0 msg shape:", critic_L0_msg.shape)
            print("L0 aggr shape:", critic_L0_aggr.shape)
            print("L0 update shape:", critic_L0_update.shape)
            print("Value attn shape:", critic_value_attn.shape)
            print("L1 msg shape:", critic_L1_msg.shape)
            print("L1 aggr shape:", critic_L1_aggr.shape)
            print("L1 update shape:", critic_L1_update.shape)

        sensing_radius = env.params["comm_radius"]
        agent_radius = env.params["car_radius"]

        safety_labels, _ = safety_labeller(
            T_graph=rollout.Tp1_graph,
            num_agents=num_agents,
            sensing_radius=sensing_radius,
            agent_radius=agent_radius,
            actuator_bounds=jnp.array(actuator_bounds),
        )

        features_array = safety_labeller_with_features(
            T_graph=rollout.Tp1_graph,
            num_agents=num_agents,
            sensing_radius=sensing_radius,
            agent_radius=agent_radius,
            actuator_bounds=jnp.array(actuator_bounds),
        )

        print("Safety labels shape:", safety_labels.shape)
        print("Total safe agents:", jnp.sum(safety_labels))
        print("Total unsafe agents:", jnp.sum(1 - safety_labels))

        unsafe_data_list = []
        T_used = min(features_array.shape[0], 256)

        for t in range(T_used):
            for i in range(features_array.shape[1]):
                for j in range(features_array.shape[2]):
                    if j < num_agents and i == j:
                        continue

                    rel_pos_mag = features_array[t, i, j, 4]
                    rel_vel_mag = features_array[t, i, j, 5]
                    safety_label = features_array[t, i, j, 7]
                    collision_type = int(features_array[t, i, j, 8])

                    if safety_label == 0 and (rel_pos_mag > 1e-6 or rel_vel_mag > 1e-6):
                        unsafe_data_list.append(
                            {
                                "timestep": t,
                                "agent_i": i,
                                "agent_j": j,
                                "collision_type": collision_type,
                                "cbf_value": 0.0,
                            }
                        )

        num_unsafe_samples = int(jnp.sum(1 - safety_labels))

        if algo is not None:
            timesteps = int(safety_labels.shape[0])
            dummy_h_values = np.zeros((timesteps, num_agents, 1, 1), dtype=np.float32)

            # L0 msg (edge-level)
            log_unsafe_agents_gnn_msg(
                gnn_msg_features=critic_L0_msg_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_L0_msg.csv",
            )
            log_safe_agents_gnn_msg(
                gnn_msg_features=critic_L0_msg_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                cbf_values=(None, None, dummy_h_values),
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="safe_critic_L0_msg.csv", random_seed=args.seed + i_epi,
            )

            # L0 aggr (edge-level)
            log_unsafe_agents_gnn_msg(
                gnn_msg_features=critic_L0_aggr_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_L0_aggr.csv",
            )
            log_safe_agents_gnn_msg(
                gnn_msg_features=critic_L0_aggr_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                cbf_values=(None, None, dummy_h_values),
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="safe_critic_L0_aggr.csv", random_seed=args.seed + i_epi,
            )

            # L0 update (node-level)
            log_unsafe_agents_gnn_update(
                rollout_result=rollout, safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=critic_L0_update_intermediates[i_epi],
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_L0_update.csv",
            )
            log_balanced_safe_agents_gnn_update(
                rollout_result=rollout, safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=critic_L0_update_intermediates[i_epi],
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_critic_L0_update_balanced.csv", random_seed=args.seed + i_epi,
            )

            # ValueAttn (node-level)
            log_unsafe_agents_gnn_update(
                rollout_result=rollout, safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=critic_value_attn_intermediates[i_epi],
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_ValueAttn.csv",
            )
            log_balanced_safe_agents_gnn_update(
                rollout_result=rollout, safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=critic_value_attn_intermediates[i_epi],
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_critic_ValueAttn_balanced.csv", random_seed=args.seed + i_epi,
            )

            # L1 msg (edge-level)
            log_unsafe_agents_gnn_msg(
                gnn_msg_features=critic_L1_msg_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_L1_msg.csv",
            )
            log_safe_agents_gnn_msg(
                gnn_msg_features=critic_L1_msg_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                cbf_values=(None, None, dummy_h_values),
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="safe_critic_L1_msg.csv", random_seed=args.seed + i_epi,
            )
            # L1 aggr (edge-level)
            log_unsafe_agents_gnn_msg(
                gnn_msg_features=critic_L1_aggr_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_L1_aggr.csv",
            )
            log_safe_agents_gnn_msg(
                gnn_msg_features=critic_L1_aggr_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                cbf_values=(None, None, dummy_h_values),
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="safe_critic_L1_aggr.csv", random_seed=args.seed + i_epi,
            )
            # L1 update (node-level)
            log_unsafe_agents_gnn_update(
                rollout_result=rollout, safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=critic_L1_update_intermediates[i_epi],
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                csv_path="unsafe_critic_L1_update.csv",
            )
            log_balanced_safe_agents_gnn_update(
                rollout_result=rollout, safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=critic_L1_update_intermediates[i_epi],
                episode_num=i_epi, num_agents=num_agents, num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_critic_L1_update_balanced.csv", random_seed=args.seed + i_epi,
            )
            
            # 1
            '''log_unsafe_agents_gnn_update(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=gnn_update_intermediates[i_epi],
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="unsafe_agents_gnn_update_MLP3.csv",
            )

            # 2
            log_balanced_safe_agents_gnn_update(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=gnn_update_intermediates[i_epi],
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_agents_gnn_update_balanced_MLP3.csv",
                random_seed=args.seed + i_epi,
            )

            # 3
            log_unsafe_agents_gnn_msg(
                gnn_msg_features=gnn_msg_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="unsafe_agents_gnn_msg_MLP1.csv",
            )

            # 4
            log_safe_agents_gnn_msg(
                gnn_msg_features=gnn_msg_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                cbf_values=(None, None, dummy_h_values),
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="safe_agents_gnn_msg_MLP1.csv",
                random_seed=args.seed + i_epi,
            )

            # 5
            log_unsafe_agents_gnn_msg(
                gnn_msg_features=gnn_aggr_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="unsafe_agents_gnn_aggr_MLP2.csv",
            )

            # 6
            log_safe_agents_gnn_msg(
                gnn_msg_features=gnn_aggr_intermediates[i_epi],
                unsafe_data_list=unsafe_data_list,
                cbf_values=(None, None, dummy_h_values),
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="safe_agents_gnn_aggr_MLP2.csv",
                random_seed=args.seed + i_epi,
            )

            head_features = np.array(policy_head_intermediates[i_epi])[:, :, None, None, :]

            # 7
            log_unsafe_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                cbf_in_values=head_features,
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="unsafe_agents_policy_head.csv",
            )

            # 8
            log_balanced_safe_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                cbf_in_values=head_features,
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_agents_policy_head_balanced.csv",
                random_seed=args.seed + i_epi,
            )'''

            # 9
            '''log_all_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                cbf_in_values=head_features,
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="all_agents_policy_head.csv",
            )'''

            # 10 - GNN output (64-d)
            '''log_unsafe_agents_gnn_update(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=policy_head_intermediates[i_epi],
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                csv_path="unsafe_agents_gnn_output_64d.csv",
            )

            # 11- balanced safe
            log_balanced_safe_agents_gnn_update(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=(None, None, dummy_h_values),
                gnn_update_features=policy_head_intermediates[i_epi],
                episode_num=i_epi,
                num_agents=num_agents,
                num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_agents_gnn_output_64d_balanced.csv",
                random_seed=args.seed + i_epi,
            )'''

        if len(is_unsafes) == 0:
            continue

        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        finish_rate = is_finishes[-1].max(axis=0).mean()
        success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()

        print(
            f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
            f"safe rate: {safe_rate * 100:.3f}%, "
            f"finish rate: {finish_rate * 100:.3f}%, "
            f"success rate: {success_rate * 100:.3f}%"
        )

        rates.append(np.array([safe_rate, finish_rate, success_rate]))

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

    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(
                f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                f"{env.area_size},{env.params['n_obs']},"
                f"{safe_mean * 100:.3f},{safe_std * 100:.3f},"
                f"{finish_mean * 100:.3f},{finish_std * 100:.3f},"
                f"{success_mean * 100:.3f},{success_std * 100:.3f}\n"
            )

    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        video_name = f"n{num_agents}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"
        viz_opts = {}
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

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default="mappo")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)

    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr-actor", type=float, default=1e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--lr-critic", type=float, default=3e-5)
    parser.add_argument("--loss-action-coef", type=float, default=1e-4)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef", type=float, default=1e-2)
    parser.add_argument("--cost-weight", type=float, default=1.0)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()