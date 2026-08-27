import os
import sys

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import time
import pathlib
import ipdb
import numpy as np
import yaml
import pickle
import argparse
import datetime

from model.algo import make_algo
from model.env import make_env
from model.env.base import RolloutResult
from model.utils.utils import jax_jit_np, jax_vmap
from model.utils.graph import GraphsTuple


def test(args):
    print(f"> Running verify_mappo_main.py {args}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    np.random.seed(args.seed)

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import jax.tree_util as jtu

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # ------------------------------------------------------------------ #
    # Load config + build env
    # ------------------------------------------------------------------ #
    with open(os.path.join(args.path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    # ------------------------------------------------------------------ #
    # Load checkpoint
    # ------------------------------------------------------------------ #
    model_path = os.path.join(args.path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(m) for m in models if m.isdigit()])
    else:
        step = args.step
    print(f"Loading step: {step}")

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

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step))
    is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
    is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1000)[: args.epi]

    # ------------------------------------------------------------------ #
    # Which verification methods to run
    # ------------------------------------------------------------------ #
    run_z3 = args.verify_z3
    run_dreal = args.verify_dreal
    run_dreal_ball = args.verify_dreal_ball
    run_seev = args.verify_seev
    runfull = args.verify_full


    if not (run_z3 or run_dreal or run_dreal_ball or run_seev or runfull):
        print("[Warning] No verification flag selected.")
        print("Use one or more of: --verify-z3 --verify-dreal --verify-dreal-ball --verify-seev --verify-full")
        return

    # ------------------------------------------------------------------ #
    # Per-episode verification containers
    # ------------------------------------------------------------------ #
    all_safe_rates = []
    all_rewards = []
    all_costs = []

    all_z3_times = []
    all_z3_sat = []

    all_dreal_times = []
    all_dreal_sat = []

    all_dreal_ball_times = []
    all_dreal_ball_sat = []

    all_seev_times = []
    all_seev_sat = []
    all_seev_tier_stats = []

    fullsatallepis = []
    fulltimes = []

    # ------------------------------------------------------------------ #
    # Per-episode rollout + verification
    # ------------------------------------------------------------------ #
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        rollout: RolloutResult = rollout_fn(key_x0)

        print("ROLLOUT FIELDS:", rollout._fields)


        is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
        is_finish = is_finish_fn(rollout.Tp1_graph)

        epi_reward = float(rollout.T_reward.sum())
        epi_cost = float(rollout.T_cost.sum())
        safe_rate = float(1 - is_unsafe.max(axis=0).mean())

        all_rewards.append(epi_reward)
        all_costs.append(epi_cost)
        all_safe_rates.append(safe_rate)

        T = rollout.Tp1_graph.states.shape[0]
        print(
            f"\n[epi {i_epi}] T={T} reward={epi_reward:.3f} "
            f"cost={epi_cost:.3f} safe_rate={safe_rate*100:.1f}%"
        )

        # --- Z3 verification over trajectory ---
        if run_z3:
            print(f" Running Z3 verification over {T} timesteps ...")
            t0 = time.time()
            T_sat = algo.verify_v_z3_trajectory(
                rollout.Tp1_graph,
                alpha=args.alpha,
            )
            z3_time = time.time() - t0

            all_z3_times.append(z3_time)
            all_z3_sat.append(T_sat)

            z3_safe_rate = float(T_sat.mean())
            print(f" Z3 safe rate : {z3_safe_rate * 100:.2f}%")
            print(
                f" Z3 time : {z3_time:.4f}s "
                f"({z3_time / T * 1000:.2f} ms/timestep)"
            )

        # --- dReal verification over trajectory ---
        if run_dreal:
            print(f" Running dReal verification over {T} timesteps ...")
            t0 = time.time()
            T_sat = algo.verify_v_dreal_trajectory(
                rollout.Tp1_graph,
                alpha=args.alpha,
                delta=args.delta,
            )
            dreal_time = time.time() - t0

            all_dreal_times.append(dreal_time)
            all_dreal_sat.append(T_sat)

            dreal_safe_rate = float(T_sat.mean())
            print(f" dReal safe rate : {dreal_safe_rate * 100:.2f}%")
            print(
                f" dReal time : {dreal_time:.4f}s "
                f"({dreal_time / T * 1000:.2f} ms/timestep)"
            )

        # --- dReal-ball verification over trajectory ---
        if run_dreal_ball:
            print(f" Running dReal-ball verification over {T} timesteps ...")
            t0 = time.time()
            T_sat = algo.verify_v_dreal_ball_trajectory(
                rollout.Tp1_graph,
                alpha=args.alpha,
                delta=args.delta,
                eps=args.eps,
            )
            dreal_ball_time = time.time() - t0

            all_dreal_ball_times.append(dreal_ball_time)
            all_dreal_ball_sat.append(T_sat)

            dreal_ball_safe_rate = float(T_sat.mean())
            print(f" dReal-ball robust safe rate : {dreal_ball_safe_rate * 100:.2f}%")
            print(
                f" dReal-ball time : {dreal_ball_time:.4f}s "
                f"({dreal_ball_time / T * 1000:.2f} ms/timestep)"
            )

        if run_seev:
            t0_seev = time.time()
            T_sat_seev, seev_tier_stats = algo.verify_v_seev_trajectory(
                rollout.Tp1_graph,
                alpha=args.alpha,
                kgrid=args.seev_k_grid,
            )
            seev_elapsed = time.time() - t0_seev

            all_seev_times.append(seev_elapsed)
            all_seev_sat.append(T_sat_seev)
            all_seev_tier_stats.append(seev_tier_stats)

            seev_cert_rate = float(T_sat_seev.mean())
            seev_uncert_any = (T_sat_seev == 0).any(axis=1)
            seev_uncert_timestep_rate = float(seev_uncert_any.mean())

            print(
                f" [SEEV] epi {i_epi}: CERT rate={seev_cert_rate * 100:.2f}% "
                f"| UNCERT timestep rate={seev_uncert_timestep_rate * 100:.2f}% "
                f"| time={seev_elapsed:.2f}s"
            )

        
        if runfull:
            t0full = time.time()
            Tsatfull = algo.verify_v_full(
                rollout.Tp1_graph,
                critic_params=algo.critic_train_state.params,
                area_size=args.area_size,
                v_max=args.full_vmax,
                v_threshold=args.full_v_threshold,
                alpha=args.alpha,
                mass=args.full_mass,
                max_depth=args.full_max_depth,
            )
            fullelapsed = time.time() - t0full

            fullsatallepis.append(Tsatfull)
            fulltimes.append(fullelapsed)

            fullsatrate = float(Tsatfull.mean())
            fullunsafemean = float(1.0 - fullsatrate)

            print(
            f" Full SAT rate {fullsatrate * 100:.2f} "
            f"UNSAT rate {fullunsafemean * 100:.2f} "
            f"time {fullelapsed:.2f}s"
            )
    # ------------------------------------------------------------------ #
    # Aggregate summary — Z3
    # ------------------------------------------------------------------ #
    if run_z3 and len(all_z3_sat) > 0:
        T_sat_all = np.concatenate(all_z3_sat, axis=0)
        total_T = T_sat_all.shape[0]
        total_agent_timestep_pairs = T_sat_all.size

        z3_total_time = sum(all_z3_times)
        z3_mean_per_epi = np.mean(all_z3_times)
        z3_per_step = z3_total_time / total_T * 1000
        z3_per_agent = z3_total_time / total_agent_timestep_pairs * 1000

        print("\n" + "=" * 65)
        print("Z3 VERIFICATION SUMMARY — MAPPO")
        print("=" * 65)
        print(f" Episodes : {args.epi}")
        print(f" Agents : {num_agents}")
        print(f" Env : {config.env if args.env is None else args.env}")
        print(f" Checkpoint step : {step}")
        print()
        print(f" {'Metric':<45} {'Value':>12}")
        print(" " + "-" * 58)
        print(f" {'Total timesteps verified':<45} {total_T:>12}")
        print(f" {'Total (t, agent) pairs checked':<45} {total_agent_timestep_pairs:>12}")
        print(f" {'Agent-agent collision pairs / timestep':<45} {num_agents*(num_agents-1)//2:>12}")
        print()
        print(f" {'Z3 total wall time (s)':<45} {z3_total_time:>12.4f}")
        print(f" {'Z3 mean time / episode (s)':<45} {z3_mean_per_epi:>12.4f}")
        print(f" {'Z3 mean time / timestep (ms)':<45} {z3_per_step:>12.4f}")
        print(f" {'Z3 mean time / (t,agent) pair (ms)':<45} {z3_per_agent:>12.6f}")
        print()
        print(f" {'Z3 SAT rate (all (t,agent) pairs)':<45} {T_sat_all.mean()*100:>11.2f}%")
        print(f" {'Z3 UNSAT rate':<45} {(1 - T_sat_all.mean())*100:>11.2f}%")
        print(f" {'Per-agent SAT rate':<45}")
        for i, r in enumerate(T_sat_all.mean(axis=0)):
            print(f"   agent {i:02d}: {r*100:.1f}%")
        print()
        print(f" {'Rollout safe rate (mean over episodes)':<45} {np.mean(all_safe_rates)*100:>11.2f}%")
        print(f" {'Mean reward':<45} {np.mean(all_rewards):>12.3f}")
        print(f" {'Mean cost':<45} {np.mean(all_costs):>12.3f}")
        print("=" * 65)

    # ------------------------------------------------------------------ #
    # Aggregate summary — dReal
    # ------------------------------------------------------------------ #
    if run_dreal and len(all_dreal_sat) > 0:
        T_sat_all = np.concatenate(all_dreal_sat, axis=0)
        total_T = T_sat_all.shape[0]
        total_agent_timestep_pairs = T_sat_all.size

        dreal_total_time = sum(all_dreal_times)
        dreal_mean_per_epi = np.mean(all_dreal_times)
        dreal_per_step = dreal_total_time / total_T * 1000
        dreal_per_agent = dreal_total_time / total_agent_timestep_pairs * 1000

        print("\n" + "=" * 65)
        print("DREAL VERIFICATION SUMMARY — MAPPO")
        print("=" * 65)
        print(f" Episodes : {args.epi}")
        print(f" Agents : {num_agents}")
        print(f" Env : {config.env if args.env is None else args.env}")
        print(f" Checkpoint step : {step}")
        print()
        print(f" {'Metric':<45} {'Value':>12}")
        print(" " + "-" * 58)
        print(f" {'Total timesteps verified':<45} {total_T:>12}")
        print(f" {'Total (t, agent) pairs checked':<45} {total_agent_timestep_pairs:>12}")
        print(f" {'Agent-agent collision pairs / timestep':<45} {num_agents*(num_agents-1)//2:>12}")
        print()
        print(f" {'dReal total wall time (s)':<45} {dreal_total_time:>12.4f}")
        print(f" {'dReal mean time / episode (s)':<45} {dreal_mean_per_epi:>12.4f}")
        print(f" {'dReal mean time / timestep (ms)':<45} {dreal_per_step:>12.4f}")
        print(f" {'dReal mean time / (t,agent) pair (ms)':<45} {dreal_per_agent:>12.6f}")
        print()
        print(f" {'dReal SAT rate (all (t,agent) pairs)':<45} {T_sat_all.mean()*100:>11.2f}%")
        print(f" {'dReal UNSAT rate':<45} {(1 - T_sat_all.mean())*100:>11.2f}%")
        print(f" {'Per-agent SAT rate':<45}")
        for i, r in enumerate(T_sat_all.mean(axis=0)):
            print(f"   agent {i:02d}: {r*100:.1f}%")
        print()
        print(f" {'Rollout safe rate (mean over episodes)':<45} {np.mean(all_safe_rates)*100:>11.2f}%")
        print(f" {'Mean reward':<45} {np.mean(all_rewards):>12.3f}")
        print(f" {'Mean cost':<45} {np.mean(all_costs):>12.3f}")
        print("=" * 65)

    # ------------------------------------------------------------------ #
    # Aggregate summary — dReal-ball
    # ------------------------------------------------------------------ #
    if run_dreal_ball and len(all_dreal_ball_sat) > 0:
        T_sat_all = np.concatenate(all_dreal_ball_sat, axis=0)
        total_T = T_sat_all.shape[0]
        total_agent_timestep_pairs = T_sat_all.size

        dreal_ball_total_time = sum(all_dreal_ball_times)
        dreal_ball_mean_per_epi = np.mean(all_dreal_ball_times)
        dreal_ball_per_step = dreal_ball_total_time / total_T * 1000
        dreal_ball_per_agent = dreal_ball_total_time / total_agent_timestep_pairs * 1000

        print("\n" + "=" * 65)
        print("DREAL-BALL VERIFICATION SUMMARY — MAPPO")
        print("=" * 65)
        print(f" Episodes : {args.epi}")
        print(f" Agents : {num_agents}")
        print(f" Env : {config.env if args.env is None else args.env}")
        print(f" Checkpoint step : {step}")
        print(f" eps : {args.eps}")
        print()
        print(f" {'Metric':<45} {'Value':>12}")
        print(" " + "-" * 58)
        print(f" {'Total timesteps verified':<45} {total_T:>12}")
        print(f" {'Total (t, agent) pairs checked':<45} {total_agent_timestep_pairs:>12}")
        print(f" {'Agent-agent collision pairs / timestep':<45} {num_agents*(num_agents-1)//2:>12}")
        print()
        print(f" {'dReal-ball total wall time (s)':<45} {dreal_ball_total_time:>12.4f}")
        print(f" {'dReal-ball mean time / episode (s)':<45} {dreal_ball_mean_per_epi:>12.4f}")
        print(f" {'dReal-ball mean time / timestep (ms)':<45} {dreal_ball_per_step:>12.4f}")
        print(f" {'dReal-ball mean time / (t,agent) pair (ms)':<45} {dreal_ball_per_agent:>12.6f}")
        print()
        print(f" {'dReal-ball robust SAT rate':<45} {T_sat_all.mean()*100:>11.2f}%")
        print(f" {'dReal-ball non-robust rate':<45} {(1 - T_sat_all.mean())*100:>11.2f}%")
        print(f" {'Per-agent robust SAT rate':<45}")
        for i, r in enumerate(T_sat_all.mean(axis=0)):
            print(f"   agent {i:02d}: {r*100:.1f}%")
        print()
        print(f" {'Rollout safe rate (mean over episodes)':<45} {np.mean(all_safe_rates)*100:>11.2f}%")
        print(f" {'Mean reward':<45} {np.mean(all_rewards):>12.3f}")
        print(f" {'Mean cost':<45} {np.mean(all_costs):>12.3f}")
        print("=" * 65)

    if run_seev and len(all_seev_sat) > 0:
        all_sat_seev = np.concatenate(all_seev_sat, axis=0)
        seev_cert_mean = float(all_sat_seev.mean())
        seev_uncert_mean = 1.0 - seev_cert_mean
        seev_cert_per_agent = all_sat_seev.mean(axis=0)
        total_seev_time = sum(all_seev_times)
        n_grid_pts = args.seev_k_grid ** env.action_dim

        total_tier1fail = sum(s["tier1fail"] for s in all_seev_tier_stats)
        total_tier1pass = sum(s["tier1pass"] for s in all_seev_tier_stats)
        total_tier2fail = sum(s["tier2fail"] for s in all_seev_tier_stats)
        total_checked = sum(s["totalchecked"] for s in all_seev_tier_stats)

        print("\n========== SEEV VERIFICATION SUMMARY — MAPPO ==========")
        print(f" Grid density k : {args.seev_k_grid}^{env.action_dim} = {n_grid_pts} points")
        print(f" Episodes : {args.epi}")
        print(f" Total (t,agent) pairs checked : {all_sat_seev.size}")
        print(f" Overall CERT rate : {seev_cert_mean * 100:.2f}%")
        print(f" Overall UNCERT rate : {seev_uncert_mean * 100:.2f}%")
        print(f" Per-agent CERT rate : {np.array2string(seev_cert_per_agent * 100, precision=2)}%")
        print(f" Total SEEV wall time : {total_seev_time:.2f}s")
        print(f" Avg SEEV time per episode : {total_seev_time / args.epi:.2f}s")
        print("-------------------------------------------------------")
        print(
            f" Tier 1 failures (base point) : "
            f"{total_tier1fail} ({100.0 * total_tier1fail / total_checked:.2f}%)"
        )
        print(
            f" Tier 2 failures (given Tier 1 pass) : "
            f"{total_tier2fail} ({100.0 * total_tier2fail / total_checked:.2f}%)"
        )
        print(
            f" Tier 2 fail rate | Tier 1 pass : "
            f"{(100.0 * total_tier2fail / total_tier1pass) if total_tier1pass > 0 else 0.0:.2f}%"
        )
        print("=======================================================\n")


    if runfull and len(fullsatallepis) > 0:
        allsatfull = np.stack(fullsatallepis, axis=0)   # (epi, n_agents)
        fullsatmean = float(allsatfull.mean())
        fullunsatmean = 1.0 - fullsatmean
        fullsatperagent = allsatfull.mean(axis=0)
        totalfulltime = sum(fulltimes)

        print("Full Verification Summary MAPPO")
        print(f"Episodes {args.epi}")
        print(f"Total agent checks {allsatfull.size}")
        print(f"Overall SAT rate {fullsatmean * 100:.2f}")
        print(f"Overall UNSAT rate {fullunsatmean * 100:.2f}")
        print(f"Per-agent SAT rate {np.array2string(fullsatperagent * 100, precision=2)}")
        print(f"Total Full wall time {totalfulltime:.2f}s")
        print(f"Avg Full time per episode {totalfulltime / args.epi:.2f}s")
        print()
    # ------------------------------------------------------------------ #
    # Optionally save results
    # ------------------------------------------------------------------ #
    if args.log:
        save_dict = {
            "safe_rates": np.array(all_safe_rates),
            "rewards": np.array(all_rewards),
            "costs": np.array(all_costs),
        }

        if run_z3 and len(all_z3_sat) > 0:
            save_dict["z3_T_sat"] = np.concatenate(all_z3_sat, axis=0)
            save_dict["z3_times"] = np.array(all_z3_times)

        if run_dreal and len(all_dreal_sat) > 0:
            save_dict["dreal_T_sat"] = np.concatenate(all_dreal_sat, axis=0)
            save_dict["dreal_times"] = np.array(all_dreal_times)

        if run_dreal_ball and len(all_dreal_ball_sat) > 0:
            save_dict["dreal_ball_T_sat"] = np.concatenate(all_dreal_ball_sat, axis=0)
            save_dict["dreal_ball_times"] = np.array(all_dreal_ball_times)

        log_path = os.path.join(args.path, "mappo_verification_results.npz")
        np.savez(log_path, **save_dict)
        print(f"Results saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser()

    # --- same args as test_mappo.py / test_z3_mappo.py ---
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.1)

    parser.add_argument(
        "--verify-z3",
        action="store_true",
        default=False,
        help="Run Z3 surrogate value-CBF verification over the rollout trajectory.",
    )
    parser.add_argument(
        "--verify-dreal",
        action="store_true",
        default=False,
        help="Run dReal surrogate value-CBF verification over the rollout trajectory.",
    )
    parser.add_argument(
        "--verify-dreal-ball",
        action="store_true",
        default=False,
        help="Run dReal robust ball surrogate verification over the rollout trajectory.",
    )

    parser.add_argument(
        "--verify-seev",
        action="store_true",
        default=False,
        help="Run SEEV surrogate value-neighborhood verification over the rollout trajectory.",
    )
    parser.add_argument(
        "--seev-k-grid",
        type=int,
        default=3,
        help="Grid density per action dimension for SEEV Tier 2 reachable check.",
    )

    # parser
    parser.add_argument(
        "--verify-full",
        action="store_true",
        default=False,
        help="Run full formal surrogate-value verification over the state box.",
    )
    parser.add_argument("--full-v-threshold", type=float, default=0.0)
    parser.add_argument("--full-vmax", type=float, default=0.5)
    parser.add_argument("--full-max-depth", type=int, default=30)
    parser.add_argument("--full-mass", type=float, default=0.1)


    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()