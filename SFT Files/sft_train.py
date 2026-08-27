import argparse
import datetime
import os

import ipdb
import numpy as np
import wandb
import yaml

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from model.algo import make_algo
from model.env import make_env
from model.trainer.trainer import Trainer
from model.trainer.utils import is_connected


def train(args):
    print(f"> Running train.py {args}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"

    np.random.seed(args.seed)

    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
    )

    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        gnn_layers=args.gnn_layers,
        gamma=args.gamma,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        epoch_ppo=args.epoch_ppo,
        epoch_criticnet=args.epoch_criticnet,
        clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda,
        coef_ent=args.coef_ent,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    if args.pretrained_path is not None:
        print(f"> Loading pretrained MAPPO checkpoint from {args.pretrained_path}/{args.pretrained_step}")
        algo.load(args.pretrained_path, args.pretrained_step)
        print("> Pretrained actor and critic loaded successfully.")

    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}", exist_ok=True)

    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"
    run_name = f"{args.algo}_{args.env}_{start_time}" if args.name is None else args.name

    effective_steps = args.sft_steps if args.run_sft else args.steps
    train_params = {
        "run_name": run_name,
        "training_steps": effective_steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
    }

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
    )

    wandb.config.update(vars(args))
    wandb.config.update(algo.config)

    if not args.debug:
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(vars(args), f)
            yaml.dump(algo.config, f)

    if args.run_sft:
        print(
            f"> Starting MAPPO SFT ({args.sft_mode}) for {args.sft_steps} steps, "
            f"batch_size={args.sft_batch_size}, critic_lr={args.sft_lr_critic}"
        )
        algo.sft_setup(
            sft_lr_critic=args.sft_lr_critic,
            sft_batch_size=args.sft_batch_size,
            freeze_actor=not args.sft_update_actor,
        )
        trainer.sft_train(sft_mode=args.sft_mode)
        print("> MAPPO SFT completed.")
        wandb.finish()
        return

    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-agents", type=int, default=8)
    parser.add_argument("--algo", type=str, default="mappo")
    parser.add_argument("--env", type=str, default="SimpleCar")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=15000)# for pre-training - remove for SFT
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--obs", type=int, default=8)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, required=True)

    parser.add_argument("--cost-weight", type=float, default=100.0)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-actor", type=float, default=1e-5)
    parser.add_argument("--lr-critic", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epoch-ppo", type=int, default=10)
    parser.add_argument("--epoch-criticnet", type=int, default=3)
    parser.add_argument("--clip-eps", type=float, default=0.25)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)

    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--pretrained-step", type=int, default=15000)

    parser.add_argument("--run-sft", action="store_true", default=False)
    parser.add_argument("--sft-steps", type=int, default=2000)
    parser.add_argument("--sft-lr-critic", type=float, default=15e-7)
    parser.add_argument("--sft-batch-size", type=int, default=256)
    parser.add_argument("--sft-mode", type=str, default="mixed", choices=["mixed", "npz_only"])
    parser.add_argument("--sft-update-actor", action="store_true", default=False)

    parser.add_argument("--n-env-train", type=int, default=32)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()