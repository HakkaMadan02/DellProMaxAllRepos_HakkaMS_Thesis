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
from gcbfplus.trainer.utils import get_bb_cbf_all_agents, get_bb_cbf_in_all_agents , get_gnn_msg_intermediate_all_agents, get_gnn_aggr_intermediate_all_agents, get_gnn_update_intermediate_all_agents
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap
from labeller_and_collector import safety_labeller, safety_labeller_with_features
from logger import log_unsafe_agents_to_csv , log_balanced_safe_agents_to_csv
from logger import log_unsafe_agents_gnn_update , log_balanced_safe_agents_gnn_update, log_unsafe_agents_gnn_msg, log_safe_agents_gnn_msg
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

### All The Layers Log Their Data From Here - Add layer-wise variables Here if you must probe any other layer

        get_gnn_msg_intermediate_fn_ = ft.partial(
            get_gnn_msg_intermediate_all_agents,
            algo.get_gnn_msg_intermediate,
            env,
            num_agents=num_agents
        )
        get_gnn_msg_intermediate_fn_ = jax_jit_np(get_gnn_msg_intermediate_fn_)

        get_gnn_aggr_intermediate_fn_ = ft.partial(
            get_gnn_aggr_intermediate_all_agents,
            algo.get_gnn_aggr_intermediate,
            env,
            num_agents=num_agents
        )
        get_gnn_aggr_intermediate_fn_ = jax_jit_np(get_gnn_aggr_intermediate_fn_)

        get_gnn_update_intermediate_fn_ = ft.partial(
            get_gnn_update_intermediate_all_agents,
            algo.get_gnn_update_intermediate,
            env,
            num_agents=num_agents
        )
        get_gnn_update_intermediate_fn_ = jax_jit_np(get_gnn_update_intermediate_fn_)
        
        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
        
        def get_bb_cbf_in_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)

            outs = [get_bb_cbf_in_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tbb_h_in = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tbb_h_in
    else:
        get_bb_cbf_in_fn = None
        get_bb_cbf_fn = None
        cbf_fn = None
        get_cbf_gnn_aggr_intermediate = None
        get_cbf_gnn_message_intermediate = None
        get_cbf_gnn_update_intermediate = None


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
    gnn_msg_intermediates = []
    gnn_aggr_intermediates = []
    gnn_update_intermediates = []
    actuator_bounds= [] 

    F_lower, F_Upper = env.action_lim()
    mass_agent = env.params['m']
    actuator_bounds.append(jnp.sqrt(F_Upper[0]**2 + F_Upper[1]**2)/mass_agent)
    actuator_bounds.append(jnp.sqrt(F_Upper[0]**2 + F_Upper[1]**2)/mass_agent)
    actuator_bounds.append(0.0)

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

        

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
            cbfs_in.append(get_bb_cbf_in_fn(rollout.Tp1_graph))
            """ #print("cbf_in", get_bb_cbf_in_fn(rollout.Tp1_graph))
            print("TYPE inner CBFs", type(get_bb_cbf_in_fn(rollout.Tp1_graph)))
            print("Shape intermediate layer MLP 4 output", jnp.shape(get_bb_cbf_in_fn(rollout.Tp1_graph)))
            #print("cbf", cbfs) , #print("cbf_in", cbfs_in)
            print("TYPE main cbfs", type(get_bb_cbf_fn(rollout.Tp1_graph)))
            print("Shape main cbf output", len(get_bb_cbf_fn(rollout.Tp1_graph))) """

            gnn_msg_intermediates.append(get_gnn_msg_intermediate_fn_(rollout.Tp1_graph))
            gnn_aggr_intermediates.append(get_gnn_aggr_intermediate_fn_(rollout.Tp1_graph))
            gnn_update_intermediates.append(get_gnn_update_intermediate_fn_(rollout.Tp1_graph))

            print("TYPE Message", type(get_gnn_msg_intermediate_fn_(rollout.Tp1_graph)))
            print("Shape intermediate layer MLP 3 output", np.shape(get_gnn_msg_intermediate_fn_(rollout.Tp1_graph)))
            print("TYPE Aggregate", type(get_gnn_aggr_intermediate_fn_(rollout.Tp1_graph)))
            print("Shape intermediate layer MLP 2 output", np.shape(get_gnn_aggr_intermediate_fn_(rollout.Tp1_graph)))
            print("TYPE Update", type(get_gnn_update_intermediate_fn_(rollout.Tp1_graph)))
            print("Shape intermediate layer MLP 1 output", np.shape(get_gnn_update_intermediate_fn_(rollout.Tp1_graph)))


            """ curr_msg = gnn_msg_intermediates[-1]
            curr_aggr = gnn_aggr_intermediates[-1]
            curr_upd = gnn_update_intermediates[-1]
            
            print(f"\n--- Episode {i_epi} Data Shapes ---")
            print(f"Msg (Edge?):   {curr_msg.shape}")    # e.g. (T, 2050, 256)
            print(f"Aggr (Edge?):  {curr_aggr.shape}")   # e.g. (T, 2050, 128)
            print(f"Update (Node): {curr_upd.shape}")    # e.g. (T, 851, 256)
            print(f"Num Agents:    {num_agents}")        # e.g. 25
            print("----------------------------------\n") """

             #Logging all agents data

            #Safety labeller function call
            print(f"\n--- Safety Labeling for Episode {i_epi} ---")
            sensing_radius = env.params['comm_radius']  # 0.5
            agent_radius = env.params['car_radius']     # 0.05
            # Call safety labeler 
            safety_labels, SFT_features = safety_labeller(
                T_graph=rollout.Tp1_graph,
                num_agents=num_agents,
                sensing_radius=sensing_radius,
                agent_radius=agent_radius,
                actuator_bounds=jnp.array(actuator_bounds)
            ) 
            
            # Gives labels and features both - Use the optimized labeler if you do not require the later
            features_array = safety_labeller_with_features(
                T_graph = rollout.Tp1_graph,
                num_agents = num_agents,
                sensing_radius = sensing_radius,
                agent_radius = agent_radius,
                actuator_bounds = jnp.array(actuator_bounds)
            )


            df_lablels = pd.DataFrame(np.array(safety_labels))
            print(df_lablels.head(10))
            print("Safety labels shape", jnp.shape(safety_labels))
            print(f"Safety labels shape: {safety_labels.shape}")  # Should be (256, num_agents)
            print(f"    Total safe agents across max tiestep: {jnp.sum(safety_labels)}")
            print(f"    Total unsafe agents across max timesteps: {jnp.sum(1 - safety_labels)}")
            print(f"\n--- --------------------------------------")

            print("Inspecting number of failing timesteps", len(SFT_features))
            
            # INSPECTION: Count actual failing interactions
            
            unsafe_data_list = []

            if args.cbf is not None:
                cbf_tuple = cbfs[-1]
                Tb_x, Tb_y, Tbb_h = cbf_tuple
                Tbb_h_np = np.array(Tbb_h)

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

                        cbf_value = float(Tbb_h_np[t, i, 0, 0]) if args.cbf is not None else None

                        
                        # ONLY collect if TRULY UNSAFE and non-zero
                        if safety_label == 0 and (rel_pos_mag > 1e-6 or rel_vel_mag > 1e-6):
                            unsafe_data_list.append({
                                'timestep': t,
                                'agent_i': i,
                                'agent_j': j,
                                'collision_type': collision_type,
                                'cbf_value':cbf_value
                            })

            if len(unsafe_data_list) > 0:
                unsafe_df = pd.DataFrame(unsafe_data_list)
                unsafe_csv_path = "EDGE_DEBUGGER.csv"
                unsafe_df.to_csv(unsafe_csv_path, mode = "a", header=not os.path.exists(unsafe_csv_path), index=False)
            else:
                unsafe_df = None

            log_unsafe_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=cbfs[i_epi],
                cbf_in_values=cbfs_in[i_epi],
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                csv_path=f"unsafe_agents_log_MLP4_OOD.csv"
            )
            
            num_unsafe_samples = int(jnp.sum(1 - safety_labels))
            log_balanced_safe_agents_to_csv(
                rollout_result=rollout,
                safety_labels=safety_labels,
                cbf_values=cbfs[i_epi],
                cbf_in_values=cbfs_in[i_epi],
                episode_num=i_epi,
                num_agents=args.num_agents,
                num_obstacles=args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path=f"safe_agents_log_MLP4_OOD.csv",
                random_seed=args.seed + i_epi) 
            
            # ============================================================================
            # 2. GNN UPDATE MLP LOGGERS (Node-based GNN Update features)
            # ============================================================================
            num_unsafe_samples = int(jnp.sum(1 - safety_labels))
            # Log unsafe agents (Update MLP features)
            log_unsafe_agents_gnn_update(
                rollout,
                safety_labels,
                cbfs[i_epi],
                gnn_update_intermediates[i_epi],
                i_epi,
                args.num_agents,
                args.obs,
                csv_path="unsafe_agents_gnn_update_MLP3_OOD.csv"
            )

            # Log balanced safe agents (Update MLP features)
            log_balanced_safe_agents_gnn_update(
                rollout,
                safety_labels,
                cbfs[i_epi],
                gnn_update_intermediates[i_epi],
                i_epi,
                args.num_agents,
                args.obs,
                num_unsafe_samples=num_unsafe_samples,
                csv_path="safe_agents_gnn_update_balanced_MLP3_OOD.csv",
                random_seed=args.seed + i_epi
            )

             # ============================================================================
            # 3. GNN MSG MLP LOGGERS (Edge-based GNN Msg features)
            # ============================================================================

            # Log unsafe agents (Msg MLP features - exact failing edge)
            log_unsafe_agents_gnn_msg(
            gnn_msg_intermediates[i_epi],
            unsafe_data_list,
            i_epi,
            args.num_agents,
            args.obs,
            csv_path="unsafe_agents_gnn_msg_MLP1_OOD.csv"
            )

            log_safe_agents_gnn_msg(
            gnn_msg_intermediates[i_epi],
            unsafe_data_list,
            cbfs[i_epi],
            i_epi,
            args.num_agents,
            args.obs,
            csv_path="safe_agents_gnn_msg_MLP1_OOD.csv",
            random_seed=args.seed + i_epi
            )

            # Log unsafe agents (AGGR MLP features - Reusing the function)
            log_unsafe_agents_gnn_msg(
            gnn_aggr_intermediates[i_epi],
            unsafe_data_list,
            i_epi,
            args.num_agents,
            args.obs,
            csv_path="unsafe_agents_gnn_aggr_MLP2_OOD.csv"
            )

            log_safe_agents_gnn_msg(
            gnn_aggr_intermediates[i_epi],
            unsafe_data_list,
            cbfs[i_epi],
            i_epi,
            args.num_agents,
            args.obs,
            csv_path="safe_agents_gnn_aggr_MLP2_OOD.csv",
            random_seed=args.seed + i_epi
            ) 

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
        with open(os.path.join(path, "test_log_OOD.csv"), "a") as f:
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
