import argparse
import datetime
import functools as ft
import os
import pathlib
import time
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
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap


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
                seed=config.seed,
            )

            algo.load(model_path, step)

            # ── DEBUG: inspect param tree structure ──────────────────────────
            '''import jax.tree_util as jtu
            param_shapes = jtu.tree_map(lambda x: x.shape, algo.cbf_train_state.params)
            import pprint
            pprint.pprint(param_shapes)'''
            # ─────────────────────────────────────────────────────────────────
            # ── SANITY CHECK 1: weight extraction ─────────────────────────────
            '''weights = algo._extract_weights(algo.cbf_train_state.params)
            print("Weight keys:", list(weights.keys()))
            print("msg_W0 shape:", weights['msg_W0'].shape)   # expect (10, 256)
            print("cbf_Wo shape:", weights['cbf_Wo'].shape)   # expect (256, 1)'''
            # ──────────────────────────────────────────────────────────────────

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

    # ------------------------------------------------------------------ #
    # Determine if we can run Z3 verification:
    # Only GCBFPlus has verify_cbf_z3 and get_cbf; skip for nominal/u_ref.
    # ------------------------------------------------------------------ #
    run_z3 = args.verify_z3 and isinstance(algo, GCBFPlus)
    if args.verify_z3 and not run_z3:
        print("[Z3] Warning: --verify-z3 requested but algo is not GCBFPlus. Skipping Z3.")

    run_dreal = args.verify_dreal and isinstance(algo, GCBFPlus)
    if args.verify_dreal and not run_dreal:
        print("[dReal] Warning: --verify-dreal requested but algo is not GCBFPlus. Skipping dReal.")

    run_dreal_ball = args.verify_dreal_ball and isinstance(algo, GCBFPlus)
    if args.verify_dreal_ball and not run_dreal_ball:
        print("[dReal-Ball] Warning: --verify-dreal-ball requested but algo is not GCBFPlus. Skipping.")

        # ------------------------------------------------------------------ #
    # Determine if we can run SEEV verification.
    # ------------------------------------------------------------------ #
    run_seev = args.verify_seev and isinstance(algo, GCBFPlus)
    if args.verify_seev and not run_seev:
        print("[SEEV] Warning: --verify-seev requested but algo is not GCBFPlus. Skipping.")

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
    else:
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

    z3_sat_all_epis = [] 
    z3_times = []          
    dreal_sat_all_epis = []
    dreal_times = []      
    dreal_ball_sat_all_epis = []
    dreal_ball_times = []
    seev_sat_all_epis = []
    seev_times = []
    seev_tier_stats_all = []

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

        # ── SANITY CHECK 2: interval propagation (first episode only) ──
        '''if i_epi == 0:
            import jax.tree_util as jtu
            g0 = jtu.tree_map(lambda x: x[0], rollout.Tp1_graph)
            ia = algo._interval_propagate(g0, agent_i=0, weights=weights, eps=0.0)
            print("n_unstable (eps=0.0) :", ia['n_unstable'])    # expect 0
            print("pre_tanh_lo          :", ia['pre_tanh_lo'])
            print("pre_tanh_hi          :", ia['pre_tanh_hi'])   # should == lo
            ia_ball = algo._interval_propagate(g0, agent_i=0, weights=weights, eps=0.05)
            print("n_unstable (eps=0.05):", ia_ball['n_unstable'])  # expect > 0'''
        # ──────────────────────────────────────────────────────────────

        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
        else:
            cbfs.append(None)

        # -------------------------------------------------------------- #
        # Z3 VERIFICATION PASS
        # Iterates over every timestep in this episode's trajectory and
        # calls verify_cbf_z3 for each single-timestep graph.
        #
        # rollout.Tp1_graph has shape (T+1, ...) — we use all T+1 frames.
        # T_sat shape: (T+1, n_agents), values in {0, 1}
        # -------------------------------------------------------------- #
        if run_z3:
            t0_z3 = time.time()
            T_sat = algo.verify_cbf_z3_trajectory(
                rollout.Tp1_graph,
                cbf_params=algo.cbf_train_state.params,
            )
            z3_elapsed = time.time() - t0_z3
            z3_sat_all_epis.append(T_sat)
            z3_times.append(z3_elapsed)

            # Per-episode Z3 stats
            z3_sat_rate = T_sat.mean()          # fraction of (t, agent) pairs that are SAT
            z3_unsat_any = (T_sat == 0).any(axis=1)  # timesteps where ANY agent is UNSAT
            z3_unsafe_timestep_rate = z3_unsat_any.mean()

            print(
                f"  [Z3] epi {i_epi}: SAT rate={z3_sat_rate * 100:.2f}%  "
                f"| UNSAT timestep rate={z3_unsafe_timestep_rate * 100:.2f}%  "
                f"| time={z3_elapsed:.2f}s"
            )

        if run_dreal:
            t0_dreal = time.time()
            T_sat_dreal = algo.verify_cbf_dreal_trajectory(
                rollout.Tp1_graph,
                cbf_params=algo.cbf_train_state.params,
            )
            dreal_elapsed = time.time() - t0_dreal
            dreal_sat_all_epis.append(T_sat_dreal)
            dreal_times.append(dreal_elapsed)

            dreal_sat_rate = float(T_sat_dreal.mean())
            dreal_unsat_any = (T_sat_dreal == 0).any(axis=1)
            dreal_unsafe_timestep_rate = float(dreal_unsat_any.mean())

            print(
                f"  [dReal] epi {i_epi}: SAT rate={dreal_sat_rate * 100:.2f}%  "
                f"| UNSAT timestep rate={dreal_unsafe_timestep_rate * 100:.2f}%  "
                f"| time={dreal_elapsed:.2f}s"
            )

        '''if run_dreal:
            t0_dreal = time.time()
            T_sat_dreal = algo.verify_cbf_full(
                rollout.Tp1_graph,
                cbf_params=algo.cbf_train_state.params,
                area_size=env.area_size
            )
            dreal_elapsed = time.time() - t0_dreal
            dreal_sat_all_epis.append(T_sat_dreal)
            dreal_times.append(dreal_elapsed)

            dreal_sat_rate            = float(T_sat_dreal.mean())
            dreal_unsat_any           = bool((T_sat_dreal == 0).any())
            dreal_unsafe_timestep_rate = float(1-dreal_sat_rate)

            print(
                f"  [dReal] epi {i_epi}: SAT rate={dreal_sat_rate * 100:.2f}%  "
                f"| UNSAT timestep rate={dreal_unsafe_timestep_rate * 100:.2f}%  "
                f"| time={dreal_elapsed:.2f}s")'''

        if run_dreal_ball:
            t0_db = time.time()
            T_sat_db = algo.verify_cbf_dreal_ball_trajectory(
                rollout.Tp1_graph,
                cbf_params=algo.cbf_train_state.params,
            )
            db_elapsed = time.time() - t0_db
            dreal_ball_sat_all_epis.append(T_sat_db)
            dreal_ball_times.append(db_elapsed)

            db_sat_rate              = T_sat_db.mean()
            db_unsat_any             = (T_sat_db == 0).any(axis=1)
            db_unsafe_timestep_rate  = db_unsat_any.mean()

            print(
                f"  [dReal-Ball] epi {i_epi}: robust SAT rate={db_sat_rate * 100:.2f}%  "
                f"| non-robust timestep rate={db_unsafe_timestep_rate * 100:.2f}%  "
                f"| time={db_elapsed:.2f}s"
            )

        if run_seev:
            t0_seev = time.time()
            T_sat_seev, seev_tier_stats = algo.verify_cbf_seev_trajectory(
                rollout.Tp1_graph,
                cbf_params=algo.cbf_train_state.params,
                k_grid=args.seev_k_grid,
            )
            seev_elapsed = time.time() - t0_seev
            seev_sat_all_epis.append(T_sat_seev)
            seev_times.append(seev_elapsed)
            seev_tier_stats_all.append(seev_tier_stats)

            seev_cert_rate          = T_sat_seev.mean()
            seev_uncert_any         = (T_sat_seev == 0).any(axis=1)
            seev_uncert_timestep_rate = seev_uncert_any.mean()

            print(
                f" [SEEV] epi {i_epi}: cert rate={seev_cert_rate * 100:.2f}%"
                f" | uncert timestep rate={seev_uncert_timestep_rate * 100:.2f}%"
                f" | time={seev_elapsed:.2f}s"
            )

        if len(is_unsafes) == 0:
            continue

        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        finish_rate = is_finishes[-1].max(axis=0).mean()
        success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()
        print(
            f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
            f"safe rate: {safe_rate * 100:.3f}%,"
            f"finish rate: {finish_rate * 100:.3f}%, "
            f"success rate: {success_rate * 100:.3f}%"
        )

        rates.append(np.array([safe_rate, finish_rate, success_rate]))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    is_finish = np.max(np.stack(is_finishes), axis=1)

    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    finish_mean, finish_std = is_finish.mean(), is_finish.std()
    success_mean, success_std = (
        ((1 - is_unsafe) * is_finish).mean(),
        ((1 - is_unsafe) * is_finish).std(),
    )

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, "
        f"finish_rate: {finish_mean * 100:.3f}%, "
        f"success_rate: {success_mean * 100:.3f}%"
    )

    # ------------------------------------------------------------------ #
    # Z3 SUMMARY across all episodes
    # ------------------------------------------------------------------ #
    if run_z3 and len(z3_sat_all_epis) > 0:
        all_sat = np.concatenate(z3_sat_all_epis, axis=0)  # (total_T, n_agents)
        z3_sat_mean = all_sat.mean()
        z3_unsat_mean = 1.0 - z3_sat_mean
        z3_sat_per_agent = all_sat.mean(axis=0)             # (n_agents,)
        total_z3_time = sum(z3_times)

        print("\n========== Z3 CBF Verification Summary ==========")
        print(f"  Episodes        : {args.epi}")
        print(f"  Total (t,agent) pairs checked : {all_sat.size}")
        print(f"  Overall SAT rate              : {z3_sat_mean * 100:.2f}%")
        print(f"  Overall UNSAT rate            : {z3_unsat_mean * 100:.2f}%")
        print(f"  Per-agent SAT rate            : {np.array2string(z3_sat_per_agent * 100, precision=2)}%")
        print(f"  Total Z3 wall time            : {total_z3_time:.2f}s")
        print(f"  Avg Z3 time per episode       : {total_z3_time / args.epi:.2f}s")
        print("=================================================\n")

        if args.log:
            z3_log_path = os.path.join(path, "z3_verification_log.csv")
            with open(z3_log_path, "a") as f:
                f.write(
                    f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{z3_sat_mean * 100:.3f},{z3_unsat_mean * 100:.3f},"
                    f"{total_z3_time:.3f}\n"
                )
    '''
    This is for the full formal verification file, not point-wise 

    '''
    
    if run_dreal and len(dreal_sat_all_epis) > 0:
        all_sat_dreal = np.concatenate(dreal_sat_all_epis, axis=0)  # (total_T, n_agents)
        dreal_sat_mean = all_sat_dreal.mean()
        dreal_unsat_mean = 1.0 - dreal_sat_mean
        dreal_sat_per_agent = all_sat_dreal.mean(axis=0)             # (n_agents,)
        total_dreal_time = sum(dreal_times)

        print("\n========== dReal CBF Verification Summary ==========")
        print(f"  Episodes        : {args.epi}")
        print(f"  Total (t,agent) pairs checked : {all_sat_dreal.size}")
        print(f"  Overall SAT rate              : {dreal_sat_mean * 100:.2f}%")
        print(f"  Overall UNSAT rate            : {dreal_unsat_mean * 100:.2f}%")
        print(f"  Per-agent SAT rate            : {np.array2string(dreal_sat_per_agent * 100, precision=2)}%")
        print(f"  Total dReal wall time         : {total_dreal_time:.2f}s")
        print(f"  Avg dReal time per episode    : {total_dreal_time / args.epi:.2f}s")
        print("=================================================\n")

        if args.log:
            dreal_log_path = os.path.join(path, "dreal_verification_log.csv")
            with open(dreal_log_path, "a") as f:
                f.write(
                    f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{dreal_sat_mean * 100:.3f},{dreal_unsat_mean * 100:.3f},"
                    f"{total_dreal_time:.3f}\n"
                )


    if run_dreal_ball and len(dreal_ball_sat_all_epis) > 0:
        all_sat_db       = np.concatenate(dreal_ball_sat_all_epis, axis=0)
        db_sat_mean      = all_sat_db.mean()
        db_unsat_mean    = 1.0 - db_sat_mean
        db_sat_per_agent = all_sat_db.mean(axis=0)
        total_db_time    = sum(dreal_ball_times)

        print("\n========== dReal Ball CBF Verification Summary (eps=0.1) ==========")
        print(f"  Episodes                      : {args.epi}")
        print(f"  Total (t,agent) pairs checked : {all_sat_db.size}")
        print(f"  Overall robust SAT rate       : {db_sat_mean * 100:.2f}%")
        print(f"  Overall non-robust rate       : {db_unsat_mean * 100:.2f}%")
        print(f"  Per-agent robust SAT rate     : {np.array2string(db_sat_per_agent * 100, precision=2)}%")
        print(f"  Total dReal-Ball wall time    : {total_db_time:.2f}s")
        print(f"  Avg dReal-Ball time/episode   : {total_db_time / args.epi:.2f}s")
        print("====================================================================\n")

        if args.log:
            db_log_path = os.path.join(path, "dreal_ball_verification_log.csv")
            with open(db_log_path, "a") as f:
                f.write(
                    f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{db_sat_mean * 100:.3f},{db_unsat_mean * 100:.3f},"
                    f"{total_db_time:.3f}\n"
                )

    if run_seev and len(seev_sat_all_epis) > 0:
        all_sat_seev        = np.concatenate(seev_sat_all_epis, axis=0)  # (total_T, n_agents)
        seev_cert_mean      = all_sat_seev.mean()
        seev_uncert_mean    = 1.0 - seev_cert_mean
        seev_cert_per_agent = all_sat_seev.mean(axis=0)                  # (n_agents,)
        total_seev_time     = sum(seev_times)
        n_grid_pts          = args.seev_k_grid ** env.action_dim

        print("\n========== SEEV CBF Verification Summary (Option B) ==========")
        print(f"  Grid density k               : {args.seev_k_grid}^{env.action_dim} = {n_grid_pts} points")
        print(f"  Episodes                     : {args.epi}")
        print(f"  Total (t,agent) pairs checked: {all_sat_seev.size}")
        print(f"  Overall CERT rate            : {seev_cert_mean * 100:.2f}%")
        print(f"  Overall UNCERT rate          : {seev_uncert_mean * 100:.2f}%")
        print(f"  Per-agent CERT rate          : {np.array2string(seev_cert_per_agent * 100, precision=2)}%")
        print(f"  Total SEEV wall time         : {total_seev_time:.2f}s")
        print(f"  Avg SEEV time per episode    : {total_seev_time / args.epi:.2f}s")
        # In the summary block — add after existing prints:
        total_t1_fail = sum(s['tier1_fail'] for s in seev_tier_stats_all)
        total_t2_fail = sum(s['tier2_fail'] for s in seev_tier_stats_all)
        total_t1_pass = sum(s['tier1_pass'] for s in seev_tier_stats_all)
        total_checked = sum(s['total_checked'] for s in seev_tier_stats_all)

        print(f"  --- Tier breakdown (of {all_sat_seev.size} agent-timestep pairs) ---")
        print(f"  Tier 1 failures (base point)    : {total_t1_fail} ({total_t1_fail/total_checked*100:.2f}%)")
        print(f"  Tier 2 failures (T1 passed, T2 failed): {total_t2_fail} ({total_t2_fail/total_checked*100:.2f}%)")
        print(f"  Tier 2 fail rate given T1 pass  : {total_t2_fail/total_t1_pass*100:.2f}%")
        print("==============================================================\n")

        if args.log:
            seev_log_path = os.path.join(path, "seev_verification_log.csv")
            with open(seev_log_path, "a") as f:
                f.write(
                    f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{args.seev_k_grid},{n_grid_pts},"
                    f"{seev_cert_mean * 100:.3f},{seev_uncert_mean * 100:.3f},"
                    f"{total_seev_time:.3f}\n"
                )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(
                f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                f"{env.area_size},{env.params['n_obs']},"
                f"{safe_mean * 100:.3f},{safe_std * 100:.3f},"
                f"{finish_mean * 100:.3f},{finish_std * 100:.3f},"
                f"{success_mean * 100:.3f},{success_std * 100:.3f}\n"
            )

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

    # ------------------------------------------------------------------ #
    # NEW: Z3 verification flag
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--verify-z3",
        action="store_true",
        default=False,
        help="Run Z3 CBF feasibility verification over the rollout trajectory.",
    )
    parser.add_argument(
        "--verify-dreal",
        action="store_true",
        default=False,
        help="Run dReal CBF feasibility verification over the rollout trajectory.",
    )
    parser.add_argument(
        "--verify-dreal-ball",
        action="store_true",
        default=False,
        help="Run dReal robust ball CBF verification (eps=0.1) over the rollout trajectory.",
    )
    parser.add_argument(
    "--verify-seev",
    action="store_true",
    default=False,
    help=(
        "Run SEEV Option-B CBF neighborhood verification. "
        "Tier 1: analytical sufficient condition (no solver). "
        "Tier 2: k^action_dim grid over one-step reachable set."
    ),
    )
    parser.add_argument(
        "--seev-k-grid",
        type=int,
        default=3,
        help=(
            "Grid density per action dimension for SEEV Tier 2 reachable check. "
            "Default 3 → 3^2 = 9 grid points for 2D double integrator action."
        ),
    )

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
