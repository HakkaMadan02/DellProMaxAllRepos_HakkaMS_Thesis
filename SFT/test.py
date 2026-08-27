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
#from gcbfplus.trainer.utils import get_bb_cbf_all_agents, get_bb_cbf_in_all_agents
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap
#from labeller_and_collector import safety_labeller
from labeller_and_collector_optimized import safety_labeller
#from logger import log_unsafe_agents_to_csv , log_balanced_safe_agents_to_csv, log_all_agents_to_csv, log_safety_features_dict_to_csv, log_safety_features_dict_to_pickle
#from unsafe_generator import generate_multiple_unsafe_scenarios, analyze_unsafe_scenarios
#np.set_printoptions(threshold=sys.maxsize)

def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
            print("state, node, edge lengths",len(T_graph.states), len(T_graph.nodes), len(T_graph.edges))
            #print("T", T)
            N = 200
            Timestep1 = T_graph.states[0]
            TimestepN = T_graph.states[N]
            """ print(f"Nth timestep edge counts: {len(T_graph.edges[N])}")
            print(f"Nth timestep node counts: {len(T_graph.nodes[N])}")
            print(f"Nth timestep state counts: {len(T_graph.states[N])}")
            print(f"Nth timestep state shape: {jnp.shape(T_graph.states[N])}")
            print(f"node types at TimestepN: {type(T_graph.nodes[N])}, edge types at TimestepN: {type(T_graph.edges[N])}, state types at TimestepN: {type(T_graph.states[N])}")
            print(f"First 12 nodes at TimestepN: {T_graph.nodes[N][:12]}")
            print(f"First 12 states at TimestepN: {T_graph.states[N][:12]}")
            print(f"Last state and last node at TimestepN: {T_graph.states[N][-1], T_graph.nodes[N][-1]}") """
        
            

            #print("Timestep1", Timestep1)
            #print("TimestepN", Timestep_Nplus1)
            #print("Shape Timestep1", jnp.shape(Timestep1))
            #print("Nodes at Timestep1", T_graph.nodes[0] , "with shape", jnp.shape(T_graph.nodes[0]))
            #print("Nodes at TimestepN", T_graph.nodes[N] , "with shape", jnp.shape(T_graph.nodes[N]))
            #print("Edges at Timestep1", T_graph.edges[0], "with shape", jnp.shape(T_graph.edges[0]))
            #print("Edges at TimestepN", T_graph.edges[N], "with shape", jnp.shape(T_graph.edges[N]))
            #print("Checking for velocities at TimestepN", T_graph.states[N])
            """"SELF-Hakka
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
            #print("VerrifierBlockHakka:", T_graph.edges[N][:agent_agent_edges])
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
    episode_metrics = []
    actuator_bounds= [] # index 0 is agent accn max in x, index 1 is agent accn max in y, index 2 is obstacle accn max

    #SELF-Hakka
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
            # Call safety labeler - Hakka Original
            safety_labels, SFT_features = safety_labeller(
                T_graph=rollout.Tp1_graph,
                num_agents=num_agents,
                sensing_radius=sensing_radius,
                agent_radius=agent_radius,
                actuator_bounds=jnp.array(actuator_bounds)
            ) # Gives labels and features both


            df_lablels = pd.DataFrame(np.array(safety_labels))
            #print(df_lablels.head(10))
            print("Safety labels shape", jnp.shape(safety_labels))
            print(f"Safety labels shape: {safety_labels.shape}")  # Should be (256, num_agents)
            print(f"    Total safe agents across max tiestep: {jnp.sum(safety_labels)}")
            print(f"    Total unsafe agents across max timesteps: {jnp.sum(1 - safety_labels)}")
            print(f"\n--- --------------------------------------")

            print("Inspecting number of failing timesteps", len(SFT_features))
            #print(SFT_features)

            labeled_safe_count = int(jnp.sum(safety_labels))
            labeled_unsafe_count = int(jnp.sum(1 - safety_labels))
            labeled_total = labeled_safe_count + labeled_unsafe_count
            labeled_safe_ratio = labeled_safe_count / labeled_total if labeled_total > 0 else np.nan
            labeled_unsafe_ratio = labeled_unsafe_count / labeled_total if labeled_total > 0 else np.nan
            

            """  #SFT LOGGING
            # Log unsafe agents to CSV
            log_unsafe_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=cbfs[i_epi],
                cbf_in_values=cbfs_in[i_epi],
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                csv_path=f"unsafe_agents_log.csv")
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
                csv_path=f"safe_agents_log.csv",
                random_seed=args.seed + i_epi)  # Different seed per episode  """
            
            """ #SFT features logging
            log_safety_features_dict_to_csv(
                SFT_features=SFT_features,
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                csv_path=f"safety_features_SFT_log.csv") """
            
            '''log_safety_features_dict_to_pickle(
            SFT_features=SFT_features, 
            episode_num=i_epi,  
            num_agents=args.num_agents,
            num_obstacles=args.obs,   
            pickle_path="safety_features_SFT_log.pkl")'''

        else:
            cbfs.append(None)
            cbfs_in.append(None)
            labeled_safe_count = np.nan
            labeled_unsafe_count = np.nan
            labeled_total = np.nan
            labeled_safe_ratio = np.nan
            labeled_unsafe_ratio = np.nan
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
        episode_metrics.append({
            "episode": i_epi,
            "env_episode": i_epi,
            "seed": args.seed + args.offset + i_epi,
            "reward": float(epi_reward),
            "cost": float(epi_cost),
            "labeled_safe_count": labeled_safe_count,
            "labeled_unsafe_count": labeled_unsafe_count,
            "labeled_total": labeled_total,
            "labeled_safe_ratio": labeled_safe_ratio,
            "labeled_unsafe_ratio": labeled_unsafe_ratio,
            "safe_rate": float(safe_rate),
            "finish_rate": float(finish_rate),
            "success_rate": float(success_rate),
        })
    
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
        episode_metrics_df = pd.DataFrame(episode_metrics)
        episode_metrics_path = os.path.join(script_dir, f"GCBF_ExtendedFinishRates_{num_agents}.csv")
        episode_metrics_df.to_csv(episode_metrics_path, index=False)
        print(f"Saved per-episode metrics to {episode_metrics_path}")

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
