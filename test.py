import argparse
import datetime
import functools as ft
import os
import pathlib
import cv2
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import trange

from model.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBFK, DecShareCBF, DecOptCBF
from model.env import make_env
from model.env.base import RolloutResult
from model.nn.gnn import save_set_attn
from model.trainer.utils import get_bb_cbf, plot_cbf
from model.utils.graph import GraphsTuple
from model.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap

def test(args):
    print(f"> Running test_llc.py {args}")

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
                algo=config.algo,  # shared
                env=env,  # shared
                node_dim=env.node_dim,  # shared
                edge_dim=env.edge_dim,  # shared
                state_dim=env.state_dim,  # shared
                action_dim=env.action_dim,  # shared
                n_agents=env.num_agents,  # shared
                gnn_layers=config.gnn_layers,  # gcbf, gcbf+, mappo
                batch_size=config.batch_size,  # gcbf, gcbf+, mappo
                buffer_size=config.buffer_size,  # gcbf, gcbf+
                horizon=config.horizon,  # gcbf+
                lr_actor=config.lr_actor,  # gcbf, gcbf+, mappo
                lr_cbf=config.lr_cbf,  # gcbf, gcbf+
                lr_critic=config.lr_critic if hasattr(config, 'lr_critic') else None,  # mappo
                alpha=config.alpha,  # gcbf, gcbf+
                eps=0.02,  # gcbf, gcbf+
                inner_epoch=8,  # gcbf, gcbf+
                loss_action_coef=config.loss_action_coef,  # gcbf, gcbf+
                loss_unsafe_coef=config.loss_unsafe_coef,  # gcbf, gcbf+
                loss_safe_coef=config.loss_safe_coef,  # gcbf, gcbf+
                loss_h_dot_coef=config.loss_h_dot_coef,  # gcbf, gcbf+
                max_grad_norm=2.0,  # gcbf, gcbf+, mappo
                seed=config.seed,  # shared
                cost_weight=config.cost_weight if hasattr(config, 'cost_weight') else None,  # mappo
                online_pol_refine=args.online_refine  # gcbf
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
    # step_fn = jax.jit(env.step)
    test_keys = test_keys[args.offset:]

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecOptCBF, DecShareCBF, DecShareCBFK))

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        # cbf_fn = jax.jit(algo.get_cbf)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h

    else:
        cbf_fn = None

    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")
        rollout_fn = env.rollout_fn_jitstep(act_fn, args.max_step, noedge=True, nograph=args.no_video)

        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")
        rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step))

        is_unsafe_fn = jax_jit_np(jax.vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax.vmap(env.finish_mask))

    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    rates = []
    cbfs = []
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        if args.nojit_rollout:
            rollout: RolloutResult
            rollout, is_unsafe, is_finish = rollout_fn(key_x0)
            if not jnp.isnan(rollout.T_reward).any():
                is_unsafes.append(is_unsafe)
                is_finishes.append(is_finish)
        else:
            rollout: RolloutResult = rollout_fn(key_x0)
            # print("Calling is_unsafe_fn")
            if not jnp.isnan(rollout.T_reward).any():
                is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
                # print("Calling is_finish_fn")
                is_finishes.append(is_finish_fn(rollout.Tp1_graph))

        # plot h
        if args.show_h:
            from jax.experimental import io_callback

            save_set_attn(False)
            if os.path.exists('send.csv'):
                os.remove('send.csv')
            if os.path.exists('recv.csv'):
                os.remove('recv.csv')

            if os.path.exists('dist.csv'):
                os.remove('dist.csv')
            if os.path.exists('attn.csv'):
                os.remove('attn.csv')
            from model.utils.utils import np2jax

            graphs = jtu.tree_map(lambda x: x[:-1], rollout.Tp1_graph)
            # hs = jax.vmap(algo.get_cbf)(graphs)

            for tt in range(env.max_episode_steps):
                graph_t = tree_index(graphs, tt)
                graph_t = np2jax(graph_t)
                algo.get_cbf(graph_t)
            save_set_attn(False)
            #
            # @jax.jit
            # def tmp(hs_, graphs_):
            #     io_callback(jax.vmap(algo.get_cbf), hs_, graphs_)

            # io_callback(jax.vmap(algo.get_cbf), hs, graphs)

            # tmp(hs, graphs)

            # # print('graph: ', graphs.states[1])
            # hs = merge01(hs.transpose((1, 2, 0)))
            # import matplotlib.pyplot as plt
            # plt.figure()
            # for iii in range(hs.shape[0]):
            #     plt.plot(hs[iii])
            # plt.savefig(f'{path}/h_{i_epi}.png')
            # next_graph1 = jtu.tree_map(lambda x: x[1:], rollout.Tp1_graph)
            # # print('next_graph1: ', next_graph1.states[0])
            # # next_graph = jax.vmap(env.forward_graph)(rollout.Tp1_graph, rollout.T_action)
            # next_graph2 = jax.vmap(env.forward_graph)(graphs, rollout.T_action)
            # # print('next_graph2: ', next_graph2.states[0])
            #
            # h_next1 = jax.vmap(algo.get_cbf)(next_graph1)
            # h_next2 = jax.vmap(algo.get_cbf)(next_graph2)
            # h_next1 = merge01(h_next1.transpose((1, 2, 0)))
            # h_next2 = merge01(h_next2.transpose((1, 2, 0)))
            # h_dot1 = (h_next1 - hs) / env.dt
            # h_dot2 = (h_next2 - hs) / env.dt
            # plt.figure()
            # for iii in range(h_dot1.shape[0]):
            #     plt.plot(h_dot1[iii] + algo.alpha * hs[iii], label=f'agent {iii}')
            # plt.legend()
            # plt.savefig(f'{path}/h_dot_gt_{i_epi}.png')
            # plt.figure()
            # for iii in range(h_dot2.shape[0]):
            #     plt.plot(h_dot2[iii] + algo.alpha * hs[iii], label=f'agent {iii}')
            # plt.legend()
            # plt.savefig(f'{path}/h_dot_fw_{i_epi}.png')

        # If there are any unsafes, then print out who is unsafe.
        # (T, n_agents)
        # Ta_is_unsafes = is_unsafes[-1]
        # a_is_unsafes = Ta_is_unsafes.max(axis=0)
        # if a_is_unsafes.any():
        #     for ii in range(env.num_agents):
        #         if a_is_unsafes[ii]:
        #             print("Agent {} unsafe at kk={}".format(ii, np.argmax(Ta_is_unsafes[:, ii])))

        # print("after rollout. sum reward")
        epi_reward = rollout.T_reward.sum()
        # print("sum cost")
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
        else:
            cbfs.append(None)
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
    is_unsafes = np.max(np.stack(is_unsafes), axis=1)
    finish_time = np.argmax(np.stack(is_finishes), axis=1)
    is_finishes = np.max(np.stack(is_finishes), axis=1)

    safe_mean, safe_std = (1 - is_unsafes).mean(), (1 - is_unsafes).std()
    finish_mean, finish_std = is_finishes.mean(), is_finishes.std()
    success_mean, success_std = ((1 - is_unsafes) * is_finishes).mean(), ((1 - is_unsafes) * is_finishes).std()

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, "
        f"finish_rate: {finish_mean * 100:.3f}%, "
        f"success_rate: {success_mean * 100:.3f}%"
    )

    # save reach time
    if args.reach_time:
        np.save(os.path.join(path, "reach_time.npy"), finish_time)

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
    for ii, (rollout, Ta_is_safe, cbf) in enumerate(zip(rollouts, is_unsafes, cbfs)):
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
        # if isinstance(algo, GCBFPlus):
        #     u_qp = jax_vmap(algo.get_qp_action)(rollout.Tp1_graph)[0]
        # else:
        #     u_qp = None
        # env.render(rollout, u_qp)
        env.render_video(rollout, video_path, Ta_is_safe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbf", type=int, default=None)
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--online-refine", action="store_true", default=False)
    parser.add_argument("--reach-time", action="store_true", default=False)

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
    parser.add_argument('--show-h', action="store_true", default=False)

    parser.add_argument("--dpi", type=int, default=80)
    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
