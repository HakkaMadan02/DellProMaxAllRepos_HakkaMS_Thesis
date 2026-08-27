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
from matplotlib.patches import Polygon, Patch, Circle
from matplotlib.collections import PatchCollection

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.trainer.utils import get_bb_cbf_all_agents, get_bb_cbf_in_all_agents
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap

colors = {'bloodred': '#880808', 'jetblack': '#0A0A0A', 'gungrey': '#818589', 'silver': '#C0C0C0', 'bluegrey': '#7393B3'}

def plot_collision_sequence(T_graph, env, start_timestep, end_timestep, 
                           agent_i, agent_j, num_agents):
    """
    Plot a sequence of timesteps showing collision sequence.
    agent_i and agent_j highlighted in blood red, all others blue, obstacles black.
    """
    
    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")
    
    timesteps = list(range(start_timestep, end_timestep + 1))
    num_plots = len(timesteps)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Extract obstacle data (static across all timesteps)
    graph_t0 = tree_index(T_graph, 0)
    obstacles = graph_t0.env_states.obstacle
    n_obs = len(obstacles.center)
    obs_polys = [Polygon(obstacles.points[ii]) for ii in range(n_obs)]
    
    # Get agent and environment parameters
    car_radius = env.params["car_radius"]
    area_size = env.area_size
    
    # Plot each timestep
    for idx, t in enumerate(timesteps):
        ax = axes[idx]
        
        # Extract graph at timestep t
        graph = tree_index(T_graph, t)
        
        # Get agent positions
        agent_states = graph.type_states(type_idx=0, n_type=num_agents)
        agent_positions = agent_states[:, :2]
        
        # Set axis limits and properties
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('X Position', fontsize=9, fontweight='normal')
        ax.set_ylabel('Y Position', fontsize=9, fontweight='normal')
        ax.set_title(f'Timestep {t}', fontsize=10, fontweight='bold')
        
        # Plot obstacles (black polygons)
        obs_col = PatchCollection(obs_polys, color=colors['jetblack'], alpha=0.8, zorder=5,
                                 edgecolor='black', linewidth=1.5)
        ax.add_collection(obs_col)
        
        # Plot agents as circles with colors
        agent_circles = []
        for a_idx, pos in enumerate(agent_positions):
            if a_idx == agent_i or a_idx == agent_j:
                color = colors['bloodred']  # Blood red
                alpha = 1
                zorder = 10
            else:
                color = colors['bluegrey']  # Blue grey
                alpha = 1
                zorder = 7
            
            circle = Circle(
                xy=(float(pos[0]), float(pos[1])),
                radius=car_radius,
                facecolor=color,
                edgecolor=colors['gungrey'],
                linewidth=1.0,
                alpha=alpha,
                zorder=zorder
            )
            agent_circles.append(circle)
        
        # Add all agent circles
        agent_col = PatchCollection(agent_circles, match_original=True, alpha=1.0, zorder=8)
        ax.add_collection(agent_col)
        
    
    plt.tight_layout()
    
    # Save figure
    output_dir = pathlib.Path(f"./collision_visualizations/{num_agents}_agents")
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_name = f"{stamp_str}_collision_t{start_timestep}-{end_timestep}_agents_{agent_i}_{agent_j}.png"
    fig_path = output_dir / fig_name
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Car Radius: {car_radius}, Area Size: {area_size}")
    
    plt.show()

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
            
            
            N = 256
            Timestep1 = T_graph.states[0]
            Timestep_Nplus1 = T_graph.states[N]
            
            graph_N = tree_index(T_graph, N)
            mask_N = (graph_N.node_type[graph_N.senders] == 0) & (graph_N.node_type[graph_N.receivers] == 0)

            total_edges_StepN = len(T_graph.edges[N])
            agent_goal_edges = num_agents
            agent_agent_edges = int(jnp.sum(mask_N))
            agent_obstacle_edges = total_edges_StepN - agent_goal_edges - agent_agent_edges
            

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

    plot_collision_sequence(
    T_graph=rollout.Tp1_graph,
    env=env,
    start_timestep=26,  # Change to your timestep
    end_timestep=36,    # Change to your timestep
    agent_i=20,          # Change to agent i
    agent_j=20,          # Change to agent j
    num_agents=num_agents
)

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
