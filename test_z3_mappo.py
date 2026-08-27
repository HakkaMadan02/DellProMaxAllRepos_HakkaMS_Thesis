import os
import sys

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import time
import pathlib
import ipdb
#import jax
#import jax.numpy as jnp
#import jax.random as jr
#import jax.tree_util as jtu
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


def test_z3(args):
    print(f"> Running test_z3_mappo.py {args}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #if args.cpu:
    #    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import jax.tree_util as jtu
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
    test_keys = jr.split(test_key, 1_000)[:args.epi]

    # ------------------------------------------------------------------ #
    # Per-episode Z3 verification + timing
    # ------------------------------------------------------------------ #
    all_z3_times   = []   # wall-clock seconds per trajectory
    all_z3_sat     = []   # (T, n_agents) per episode
    all_safe_rates = []
    all_rewards    = []
    all_costs      = []

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        # --- standard rollout ---
        rollout: RolloutResult = rollout_fn(key_x0)
        is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
        is_finish = is_finish_fn(rollout.Tp1_graph)

        epi_reward = float(rollout.T_reward.sum())
        epi_cost   = float(rollout.T_cost.sum())
        safe_rate  = float(1 - is_unsafe.max(axis=0).mean())
        all_rewards.append(epi_reward)
        all_costs.append(epi_cost)
        all_safe_rates.append(safe_rate)

        T = rollout.Tp1_graph.states.shape[0]
        print(f"\n[epi {i_epi}] T={T}  reward={epi_reward:.3f}  "
              f"cost={epi_cost:.3f}  safe_rate={safe_rate*100:.1f}%")

        # --- Z3 verification over trajectory ---
        print(f"  Running Z3 verification over {T} timesteps ...")
        t0 = time.time()
        #T_sat = algo.verify_v_z3_trajectory(rollout.Tp1_graph, alpha=args.alpha, )
        #T_sat = algo.verify_v_dreal_trajectory(rollout.Tp1_graph, alpha=args.alpha, delta=args.delta)
        T_sat = algo.verify_v_dreal_ball_trajectory(rollout.Tp1_graph, alpha=args.alpha, delta=args.delta, eps = args.eps)
        z3_time = time.time() - t0

        all_z3_times.append(z3_time)
        all_z3_sat.append(T_sat)

        z3_safe_rate = float(T_sat.mean())
        print(f"  Z3 safe rate : {z3_safe_rate * 100:.2f}%")
        print(f"  Z3 time      : {z3_time:.4f}s  "
              f"({z3_time / T * 1000:.2f} ms/timestep)")

        # ------------------------------------------------------------------ #
        # Aggregate summary  (mirrors GCBF+ runtime table format exactly)
        # ------------------------------------------------------------------ #
        T_sat_all  = np.concatenate(all_z3_sat, axis=0)   # (total_T, n_agents)
        total_T    = T_sat_all.shape[0]
        total_agent_timestep_pairs = T_sat_all.size        # total_T * n_agents  ← matches GCBF+

        z3_total_time   = sum(all_z3_times)
        z3_mean_per_epi = np.mean(all_z3_times)
        z3_per_step     = z3_total_time / total_T * 1000
        z3_per_agent    = z3_total_time / total_agent_timestep_pairs * 1000

        print("\n" + "=" * 65)
        print("Z3 VERIFICATION SUMMARY — MAPPO")
        print("=" * 65)
        print(f"  Episodes               : {args.epi}")
        print(f"  Agents                 : {num_agents}")
        print(f"  Env                    : {config.env if args.env is None else args.env}")
        print(f"  Checkpoint step        : {step}")
        print()
        print(f"  {'Metric':<45} {'Value':>12}")
        print("  " + "-" * 58)
        print(f"  {'Total timesteps verified':<45} {total_T:>12}")
        print(f"  {'Total (t, agent) pairs checked':<45} {total_agent_timestep_pairs:>12}")
        print(f"  {'Agent-agent collision pairs / timestep':<45} {num_agents*(num_agents-1)//2:>12}")
        print()
        print(f"  {'Z3 total wall time (s)':<45} {z3_total_time:>12.4f}")
        print(f"  {'Z3 mean time / episode (s)':<45} {z3_mean_per_epi:>12.4f}")
        print(f"  {'Z3 mean time / timestep (ms)':<45} {z3_per_step:>12.4f}")
        print(f"  {'Z3 mean time / (t,agent) pair (ms)':<45} {z3_per_agent:>12.6f}")
        print()
        print(f"  {'Z3 SAT rate (all (t,agent) pairs)':<45} {T_sat_all.mean()*100:>11.2f}%")
        print(f"  {'Z3 UNSAT rate':<45} {(1-T_sat_all.mean())*100:>11.2f}%")
        print(f"  {'Per-agent SAT rate':<45}")
        #for i, r in enumerate(T_sat_all.mean(axis=0)):
        #    print(f"    agent {i:02d}: {r*100:.1f}%")
        print()
        print(f"  {'Rollout safe rate (mean over episodes)':<45} {np.mean(all_safe_rates)*100:>11.2f}%")
        print(f"  {'Mean reward':<45} {np.mean(all_rewards):>12.3f}")
        print(f"  {'Mean cost':<45} {np.mean(all_costs):>12.3f}")
        print("=" * 65)

    # ------------------------------------------------------------------ #
    # Optionally save results
    # ------------------------------------------------------------------ #
    if args.log:
        log_path = os.path.join(args.path, "z3_results.npz")
        np.savez(
            log_path,
            T_sat=T_sat_all,
            z3_times=np.array(all_z3_times),
            safe_rates=np.array(all_safe_rates),
            rewards=np.array(all_rewards),
            costs=np.array(all_costs),
        )
        print(f"Results saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    # --- same args as test_mappo.py ---
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

    args = parser.parse_args()
    test_z3(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()