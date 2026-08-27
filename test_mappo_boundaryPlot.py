import argparse
import datetime
import os
import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import yaml
import pickle
import matplotlib.pyplot as plt

from model.algo import make_algo
from model.env import make_env
from model.env.base import RolloutResult
from model.utils.utils import jax_jit_np, jax_vmap
from model.utils.graph import GraphsTuple
from labeller_and_collector import safety_labeller, safety_labeller_with_features


def build_actuator_bounds(env):
    actuator_bounds = []

    F_lower, F_Upper = env.action_lim()
    mass_agent = env.params['m']

    actuator_bounds.append(jnp.sqrt(F_Upper[0] ** 2 + F_Upper[1] ** 2) / mass_agent)
    actuator_bounds.append(jnp.sqrt(F_Upper[0] ** 2 + F_Upper[1] ** 2) / mass_agent)
    actuator_bounds.append(0.0)

    return actuator_bounds


def print_gcbf_style_unsafe_label_stats(env, rollout, num_agents, i_epi, actuator_bounds):
    print(f"\n--- Safety Labeling for Episode {i_epi} ---")
    sensing_radius = env.params['comm_radius']
    agent_radius = env.params['car_radius']

    safety_labels, _ = safety_labeller(
        T_graph=rollout.Tp1_graph,
        num_agents=num_agents,
        sensing_radius=sensing_radius,
        agent_radius=agent_radius,
        actuator_bounds=jnp.array(actuator_bounds)
    )

    labeled_unsafe_count = int(jnp.sum(1 - safety_labels))

    print("Safety labels shape", jnp.shape(safety_labels))
    print(f"Safety labels shape: {safety_labels.shape}")
    print(f" Total safe agents across max timestep: {jnp.sum(safety_labels)}")
    print(f" Total unsafe agents across max timesteps: {jnp.sum(1 - safety_labels)}")
    print(f" Total unsafe labeled interactions: {labeled_unsafe_count}")
    print(f"\n--- --------------------------------------")

    return labeled_unsafe_count, safety_labels


def prepare_gcbf_style_plot_data(
    env,
    rollout,
    num_agents,
    actuator_bounds,
    i_epi,
    path,
    agent_unsafe_mask_episode,
    area_size,
):
    labeled_unsafe_count, safety_labels = print_gcbf_style_unsafe_label_stats(
        env=env,
        rollout=rollout,
        num_agents=num_agents,
        i_epi=i_epi,
        actuator_bounds=actuator_bounds,
    )

    sensing_radius = env.params['comm_radius']
    agent_radius = env.params['car_radius']

    print(f"\nExtracting agent interaction data for Episode {i_epi}...")

    features_array = safety_labeller_with_features(
        T_graph=rollout.Tp1_graph,
        num_agents=num_agents,
        sensing_radius=sensing_radius,
        agent_radius=agent_radius,
        actuator_bounds=jnp.array(actuator_bounds)
    )

    # print(f"\n=== DEBUG: Analyzing data structure ===")
    # print(f"features_array shape: {features_array.shape}")
    # print(f"num_agents: {num_agents}")
    # print(f"Agent-agent range: j from 0 to {num_agents - 1}")
    # print(f"Agent-obstacle range: j from {num_agents} to {num_agents + 32 - 1}")
    # print(f"Extraction loop iterates j from 0 to {features_array.shape[2] - 1}")
    # print(f"Features at j={num_agents}: {features_array[0, 0, num_agents, :]}")
    # print(f"Features at j={num_agents + 5}: {features_array[0, 0, num_agents + 5, :]}")
    # print("=== END DEBUG ===\n")

    unsafe_data_list = []
    all_data_list = []

    for t in range(features_array.shape[0]):
        for i in range(features_array.shape[1]):
            for j in range(features_array.shape[2]):
                if i == j:
                    continue

                rel_pos_mag = float(features_array[t, i, j, 4])
                rel_vel_mag = float(features_array[t, i, j, 5])
                radial_vel = float(features_array[t, i, j, 6])
                safety_label = int(features_array[t, i, j, 7])
                collision_type = int(features_array[t, i, j, 8])

                if rel_pos_mag > 1e-6 or rel_vel_mag > 1e-6:
                    all_data_list.append({
                        'timestep': t,
                        'agent_i': i,
                        'agent_j': j,
                        'rel_pos_mag': rel_pos_mag,
                        'rel_vel_mag': rel_vel_mag,
                        'radial_vel': radial_vel,
                        'collision_type': collision_type,
                        'safety_label': safety_label
                    })

                if safety_label == 0 and (rel_pos_mag > 1e-6 or rel_vel_mag > 1e-6):
                    unsafe_data_list.append({
                        'timestep': t,
                        'agent_i': i,
                        'agent_j': j,
                        'rel_pos_mag': rel_pos_mag,
                        'rel_vel_mag': rel_vel_mag,
                        'radial_vel': radial_vel,
                        'collision_type': collision_type,
                        'safety_label': safety_label
                    })

    if len(unsafe_data_list) > 0:
        unsafe_df = pd.DataFrame(unsafe_data_list)
        unsafe_csv_path = os.path.join(path, f"unsafe_interactions_epi{i_epi}_n{num_agents}.csv")
        unsafe_df.to_csv(unsafe_csv_path, index=False)
    else:
        unsafe_df = None

    if len(all_data_list) > 0:
        all_df = pd.DataFrame(all_data_list)
    else:
        all_df = None

    alphai = actuator_bounds[0] * (1 - 0.9)
    alphaj = actuator_bounds[1] * (1 - 0.9)
    total_alpha1 = alphai + alphaj
    total_alpha2 = alphai + actuator_bounds[2]

    print(f"total acceleration for agent-agent interactions: {total_alpha1}")
    print(f"total acceleration for agent-obstacle interactions: {total_alpha2}")

    pos_range = np.arange(0, area_size, 0.001)

    critical_radial_vel1 = []
    for pos in pos_range:
        if pos <= 2 * agent_radius:
            critical_radial_vel1.append(0)
        else:
            critical_vel = -np.sqrt(2 * total_alpha1 * (pos - 2 * agent_radius))
            critical_radial_vel1.append(critical_vel)

    critical_radial_vel2 = []
    for pos in pos_range:
        if pos <= agent_radius:
            critical_radial_vel2.append(0)
        else:
            critical_vel = -np.sqrt(2 * total_alpha2 * (pos - agent_radius))
            critical_radial_vel2.append(critical_vel)

    critical_radial_vel1 = np.array(critical_radial_vel1)
    critical_radial_vel2 = np.array(critical_radial_vel2)

    agent_obstacle_radius = 1.0 * agent_radius
    agent_agent_radius = 2.0 * agent_radius
    agent_agent = 2 * agent_radius
    agent_obstacle = agent_radius

    return {
        "labeled_unsafe_count": labeled_unsafe_count,
        "safety_labels": safety_labels,
        "features_array": features_array,
        "unsafe_df": unsafe_df,
        "all_df": all_df,
        "pos_range": pos_range,
        "critical_radial_vel1": critical_radial_vel1,
        "critical_radial_vel2": critical_radial_vel2,
        "sensing_radius": sensing_radius,
        "agent_radius": agent_radius,
        "agent_obstacle_radius": agent_obstacle_radius,
        "agent_agent_radius": agent_agent_radius,
        "agent_agent": agent_agent,
        "agent_obstacle": agent_obstacle,
    }


def make_gcbf_style_plot(plot_data, i_epi, path):
    unsafe_df = plot_data["unsafe_df"]
    pos_range = plot_data["pos_range"]
    critical_radial_vel1 = plot_data["critical_radial_vel1"]
    critical_radial_vel2 = plot_data["critical_radial_vel2"]
    sensing_radius = plot_data["sensing_radius"]
    agent_agent_radius = plot_data["agent_agent_radius"]
    agent_agent = plot_data["agent_agent"]
    agent_obstacle = plot_data["agent_obstacle"]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    plt.subplots_adjust(wspace=0.04)

    colors = {
        'bloodred': '#880808',
        'crimson': '#cc1515',
        'jetblack': '#0A0A0A',
        'gungrey': '#818589',
        'silver': '#C0C0C0',
        'bluegrey': '#7393B3',
        'vermelion': '#d55e00',
        'white': '#FFFFFF',
        'beige': '#7f6b00'
    }

    # ========== PLOT 1: Radial Velocity vs Relative Position with Boundary ==========
    if unsafe_df is not None and len(unsafe_df) > 0:
        agent_agent_df = unsafe_df[unsafe_df['collision_type'] == 0]
        agent_obstacle_df = unsafe_df[unsafe_df['collision_type'] == 1]

        if len(agent_agent_df) > 0:
            axes[0].scatter(
                agent_agent_df['rel_pos_mag'],
                agent_agent_df['radial_vel'],
                alpha=0.9,
                s=250,
                color='white',
                edgecolors=colors['jetblack'],
                linewidth=0.75,
                label='Unsafe (AA)',
                zorder=3
            )

        if len(agent_obstacle_df) > 0:
            axes[0].scatter(
                agent_obstacle_df['rel_pos_mag'],
                agent_obstacle_df['radial_vel'],
                alpha=0.9,
                s=250,
                color=colors['jetblack'],
                edgecolors=colors['white'],
                linewidth=0.75,
                label='Unsafe (AO)',
                zorder=3
            )

    axes[0].plot(
        pos_range,
        critical_radial_vel1,
        color=colors['bluegrey'],
        linestyle='--',
        linewidth=2,
        zorder=2
    )
    axes[0].plot(
        pos_range,
        critical_radial_vel2,
        color=colors['jetblack'],
        linestyle='--',
        linewidth=2,
        zorder=2
    )

    mask_obs = pos_range <= agent_agent_radius
    if mask_obs.sum() > 0:
        pos_obs = pos_range[mask_obs]
        vel_obs = critical_radial_vel1[mask_obs]
        valid_mask = np.isfinite(vel_obs)
        axes[0].fill_between(
            pos_obs[valid_mask],
            vel_obs[valid_mask],
            -2,
            alpha=0.5,
            color=colors['bluegrey']
        )

    mask_obs = pos_range <= agent_agent_radius
    if mask_obs.sum() > 0:
        pos_obs = pos_range[mask_obs]
        vel_obs = critical_radial_vel2[mask_obs]
        valid_mask = np.isfinite(vel_obs)
        axes[0].fill_between(
            pos_obs[valid_mask],
            vel_obs[valid_mask],
            -2,
            alpha=0.95,
            color=colors['bluegrey']
        )

    mask_rest = pos_range > agent_agent_radius
    if mask_rest.sum() > 0:
        pos_rest = pos_range[mask_rest]
        vel_rest = critical_radial_vel1[mask_rest]
        valid_mask = np.isfinite(vel_rest)
        axes[0].fill_between(
            pos_rest[valid_mask],
            vel_rest[valid_mask],
            -2,
            alpha=0.4,
            color=colors['gungrey'],
            label='Unsafe Backward\nReachability Set'
        )

    mask_rest = pos_range > agent_agent_radius
    if mask_rest.sum() > 0:
        pos_rest = pos_range[mask_rest]
        vel_rest = critical_radial_vel2[mask_rest]
        valid_mask = np.isfinite(vel_rest)
        axes[0].fill_between(
            pos_rest[valid_mask],
            vel_rest[valid_mask],
            -2,
            alpha=0.2,
            color=colors['gungrey']
        )

    axes[0].axvline(
        x=sensing_radius,
        color=colors['bloodred'],
        linestyle=':',
        linewidth=2,
        alpha=0.8,
        label=r'$R_{{ij}}$'
    )
    axes[0].axvline(
        x=agent_agent,
        color=colors['jetblack'],
        linestyle=':',
        linewidth=2,
        alpha=0.8
    )
    axes[0].axvline(
        x=agent_obstacle,
        color=colors['jetblack'],
        linestyle=':',
        linewidth=2,
        alpha=0.8
    )

    axes[0].text(
        0.03, -1.45, "AO Unsafe Zone",
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        color=colors['jetblack'],
        fontsize=30,
        fontweight='normal'
    )

    axes[0].text(
        0.08, -1.45, "AA Unsafe Zone",
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        color=colors['jetblack'],
        fontsize=30,
        fontweight='normal'
    )

    axes[1].text(
        0.03, -1.45, "AO Unsafe Zone",
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        color=colors['jetblack'],
        fontsize=30,
        fontweight='normal'
    )

    axes[1].text(
        0.08, -1.45, "AA Unsafe Zone",
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        color=colors['jetblack'],
        fontsize=30,
        fontweight='normal'
    )

    axes[0].set_xlabel(r'$||p_{ij}||$', fontsize=40, fontweight='normal')
    axes[0].set_ylabel(r'$r_{ij}$ (negative = approaching)', fontsize=40, fontweight='normal')
    axes[0].tick_params(axis='both', which='major', labelsize=35, pad=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=35, loc='upper right')
    axes[0].set_xlim(0, 0.75)
    axes[0].set_ylim(-2, 0.5)

    # ==========================================
    # PLOT 2: Uncaptured failures
    # ==========================================
    axes[1].plot(
        pos_range,
        critical_radial_vel1,
        color=colors['bluegrey'],
        linestyle='--',
        linewidth=2
    )
    axes[1].plot(
        pos_range,
        critical_radial_vel2,
        color=colors['jetblack'],
        linestyle='--',
        linewidth=2
    )

    mask_obs = pos_range <= agent_agent_radius
    if mask_obs.sum() > 0:
        pos_obs = pos_range[mask_obs]
        vel_obs = critical_radial_vel1[mask_obs]
        valid_mask = np.isfinite(vel_obs)
        axes[1].fill_between(
            pos_obs[valid_mask],
            vel_obs[valid_mask],
            -2,
            alpha=0.5,
            color=colors['bluegrey']
        )

    mask_obs = pos_range <= agent_agent_radius
    if mask_obs.sum() > 0:
        pos_obs = pos_range[mask_obs]
        vel_obs = critical_radial_vel2[mask_obs]
        valid_mask = np.isfinite(vel_obs)
        axes[1].fill_between(
            pos_obs[valid_mask],
            vel_obs[valid_mask],
            -2,
            alpha=0.95,
            color=colors['bluegrey']
        )

    mask_rest = pos_range > agent_agent_radius
    if mask_rest.sum() > 0:
        pos_rest = pos_range[mask_rest]
        vel_rest = critical_radial_vel1[mask_rest]
        valid_mask = np.isfinite(vel_rest)
        axes[1].fill_between(
            pos_rest[valid_mask],
            vel_rest[valid_mask],
            -2,
            alpha=0.4,
            color=colors['gungrey']
        )

    mask_rest = pos_range > agent_agent_radius
    if mask_rest.sum() > 0:
        pos_rest = pos_range[mask_rest]
        vel_rest = critical_radial_vel2[mask_rest]
        valid_mask = np.isfinite(vel_rest)
        axes[1].fill_between(
            pos_rest[valid_mask],
            vel_rest[valid_mask],
            -2,
            alpha=0.2,
            color=colors['gungrey']
        )

    axes[1].axvline(
        x=sensing_radius,
        color=colors['bloodred'],
        linestyle=':',
        linewidth=2,
        alpha=0.8
    )
    axes[1].axvline(
        x=agent_agent,
        color=colors['jetblack'],
        linestyle=':',
        linewidth=2,
        alpha=0.8
    )
    axes[1].axvline(
        x=agent_obstacle,
        color=colors['jetblack'],
        linestyle=':',
        linewidth=2,
        alpha=0.8
    )

    if unsafe_df is not None and len(unsafe_df) > 0:
        uncaptured_aa = unsafe_df[
            (unsafe_df['collision_type'] == 0) &
            (unsafe_df['rel_pos_mag'] > 0.1)
        ]

        uncaptured_ao = unsafe_df[
            (unsafe_df['collision_type'] == 1) &
            (unsafe_df['rel_pos_mag'] > 0.05)
        ]

        if len(uncaptured_aa) > 0:
            axes[1].scatter(
                uncaptured_aa['rel_pos_mag'],
                uncaptured_aa['radial_vel'],
                alpha=0.9,
                s=250,
                color='white',
                edgecolors=colors['jetblack'],
                linewidth=0.75,
                label='Uncaptured Failures (AA)',
                zorder=3
            )

        if len(uncaptured_ao) > 0:
            axes[1].scatter(
                uncaptured_ao['rel_pos_mag'],
                uncaptured_ao['radial_vel'],
                alpha=0.9,
                s=250,
                color=colors['jetblack'],
                edgecolors=colors['white'],
                linewidth=0.75,
                label='Uncaptured Failures (AO)',
                zorder=3
            )

    axes[1].set_xlabel(r'$||p_{ij}||$', fontsize=40, fontweight='normal')
    axes[1].tick_params(axis='both', which='major', labelsize=35, pad=9)
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=35, loc='upper right')
    axes[1].set_xlim(0, 0.75)
    axes[1].set_ylim(-2, 0.5)

    for ax, label in zip(axes, ['(a)', '(b)']):
        ax.text(
            0.5, 1.02, label,
            transform=ax.transAxes,
            fontsize=40,
            fontweight='normal',
            va='bottom',
            ha='center'
        )

    plt.tight_layout()

    plot_path1 = os.path.join(path, f"LabelerAnalysis_extendee2_epi{i_epi}.png")
    plot_path2 = os.path.join(path, f"LabelerAnalysis_extended2_epi{i_epi}.pdf")
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()


def test(args):
    print(f"> Running test_mappo.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
        num_agents = config.num_agents if args.num_agents is None else args.num_agents
        env_id = config.env if args.env is None else args.env
    else:
        num_agents = args.num_agents
        env_id = args.env

    assert num_agents is not None, "num_agents must be provided if config.yaml is not loaded."
    assert env_id is not None, "env must be provided if config.yaml is not loaded."

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
            print("step: ", step)

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
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
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
    # episode_metrics = []

    actuator_bounds = build_actuator_bounds(env)
    print("Agent acceleration bounds alphai and alphaj", actuator_bounds[0], actuator_bounds[1])
    print("Obstacle acceleration bounds", actuator_bounds[2])

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

        plot_data = prepare_gcbf_style_plot_data(
            env=env,
            rollout=rollout,
            num_agents=num_agents,
            actuator_bounds=actuator_bounds,
            i_epi=i_epi,
            path=path,
            agent_unsafe_mask_episode=is_unsafes[-1],
            area_size=args.area_size if args.area_size is not None else 10.0,
        )

        labeled_unsafe_count = plot_data["labeled_unsafe_count"]

        make_gcbf_style_plot(
            plot_data=plot_data,
            i_epi=i_epi,
            path=path
        )

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

        # episode_metrics.append({
        #     "episode": i_epi,
        #     "seed": args.seed + args.offset + i_epi,
        #     "labeled_unsafe_count": labeled_unsafe_count,
        #     "finish_rate": float(finish_rate),
        # })

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

    # episode_metrics_df = pd.DataFrame(episode_metrics)
    # episode_metrics_path = os.path.join(script_dir, f"episode_metrics_MAPPO_Attention_{num_agents}.csv")
    # episode_metrics_df.to_csv(episode_metrics_path, index=False)
    # print(f"Saved per-episode metrics to {episode_metrics_path}")

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