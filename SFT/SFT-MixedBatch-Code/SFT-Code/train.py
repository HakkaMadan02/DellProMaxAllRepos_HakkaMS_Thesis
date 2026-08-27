# import argparse
# import datetime
# import os
# import ipdb
# import numpy as np
# import wandb
# import yaml
# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
# os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Allocate 80% of GPU memory (16GB out of 20GB)

# from gcbfplus.algo import make_algo
# from gcbfplus.env import make_env
# from gcbfplus.trainer.trainer import Trainer
# from gcbfplus.trainer.utils import is_connected


# def train(args):
#     print(f"> Running train.py {args}")

#     # set up environment variables and seed
#     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#     if not is_connected():
#         os.environ["WANDB_MODE"] = "offline"
#     np.random.seed(args.seed)
#     if args.debug:
#         os.environ["WANDB_MODE"] = "disabled"
#         os.environ["JAX_DISABLE_JIT"] = "True"

#     # create environments
#     env = make_env(
#         env_id=args.env,
#         num_agents=args.num_agents,
#         num_obs=args.obs,
#         n_rays=args.n_rays,
#         area_size=args.area_size
#     )
#     env_test = make_env(
#         env_id=args.env,
#         num_agents=args.num_agents,
#         num_obs=args.obs,
#         n_rays=args.n_rays,
#         area_size=args.area_size
#     )

#     # create low level controller
#     algo = make_algo(
#         algo=args.algo,
#         env=env,
#         node_dim=env.node_dim,
#         edge_dim=env.edge_dim,
#         state_dim=env.state_dim,
#         action_dim=env.action_dim,
#         n_agents=env.num_agents,
#         gnn_layers=args.gnn_layers,
#         batch_size=256,
#         buffer_size=args.buffer_size,
#         horizon=args.horizon,
#         lr_actor=args.lr_actor,
#         lr_cbf=args.lr_cbf,
#         alpha=args.alpha,
#         eps=0.02,
#         inner_epoch=8,
#         loss_action_coef=args.loss_action_coef,
#         loss_unsafe_coef=args.loss_unsafe_coef,
#         loss_safe_coef=args.loss_safe_coef,
#         loss_h_dot_coef=args.loss_h_dot_coef,
#         max_grad_norm=2.0,
#         seed=args.seed,
#     )

#     # load pretrained model if specified
#     if args.pretrained_path is not None:
#         print(f"> Loading pretrained model from {args.pretrained_path}")
#         import pickle
#         actor_path = os.path.join(args.pretrained_path, 'actor.pkl')
#         cbf_path = os.path.join(args.pretrained_path, 'cbf.pkl')
#         algo.actor_train_state = algo.actor_train_state.replace(
#             params=pickle.load(open(actor_path, 'rb'))
#         )
#         algo.cbf_train_state = algo.cbf_train_state.replace(
#             params=pickle.load(open(cbf_path, 'rb'))
#         )
#         print("> Pretrained actor and CBF loaded successfully.")
#         # also update the CBF target network with loaded params
#         if hasattr(algo, 'cbf_tgt'):
#             from gcbfplus.trainer.utils import tree_copy
#             algo.cbf_tgt = algo.cbf_tgt.replace(params=tree_copy(algo.cbf_train_state.params))
#             print("> CBF target network synced with loaded CBF.")

#     # set up logger
#     start_time = datetime.datetime.now()
#     start_time = start_time.strftime("%Y%m%d%H%M%S")
#     if not os.path.exists(args.log_dir):
#         os.makedirs(args.log_dir)
#     if not os.path.exists(f"{args.log_dir}/{args.env}"):
#         os.makedirs(f"{args.log_dir}/{args.env}")
#     if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
#         os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}")
#     log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"
#     run_name = f"{args.algo}_{args.env}_{start_time}" if args.name is None else args.name

#     # get training parameters
#     # When running SFT, use sft_steps as the training step count
#     effective_steps = args.sft_steps if (args.pretrained_path is not None and args.sft_steps > 0) else args.steps
#     train_params = {
#         "run_name": run_name,
#         "training_steps": effective_steps,
#         "eval_interval": args.eval_interval,
#         "eval_epi": args.eval_epi,
#         "save_interval": args.save_interval,
#     }

#     # create trainer
#     trainer = Trainer(
#         env=env,
#         env_test=env_test,
#         algo=algo,
#         log_dir=log_dir,
#         n_env_train=args.n_env_train,
#         n_env_test=args.n_env_test,
#         seed=args.seed,
#         params=train_params,
#         save_log=not args.debug,
#     )

#     # save config
#     wandb.config.update(args)
#     wandb.config.update(algo.config)
#     if not args.debug:
#         with open(f"{log_dir}/config.yaml", "w") as f:
#             yaml.dump(args, f)
#             yaml.dump(algo.config, f)

#     # run SFT on failed data (this is the only training when using pretrained weights)
#     if args.pretrained_path is not None and args.sft_steps > 0:
#         print(f"> Starting SFT on failed data for {args.sft_steps} steps with CBF lr={args.sft_lr_cbf}")
#         algo.sft_setup(sft_lr_cbf=args.sft_lr_cbf)
#         trainer.sft_train()
#         print("> SFT completed. Exiting.")
#         wandb.finish()
#         return

#     # start normal training (only when no pretrained path is specified)
#     trainer.train()


# def main():
#     parser = argparse.ArgumentParser()

#     # custom arguments
#     parser.add_argument("-n", "--num-agents", type=int, default=8)
#     parser.add_argument("--algo", type=str, default="gcbf+")
#     parser.add_argument("--env", type=str, default="SimpleCar")
#     parser.add_argument("--seed", type=int, default=2)
#     parser.add_argument("--steps", type=int, default=1250)
#     parser.add_argument("--name", type=str, default=None)
#     parser.add_argument("--debug", action="store_true", default=False)
#     parser.add_argument("--obs", type=int, default=8)
#     parser.add_argument("--n-rays", type=int, default=32)
#     parser.add_argument("--area-size", type=float, required=True)

#     # gcbf / gcbf+ arguments
#     parser.add_argument("--gnn-layers", type=int, default=1)
#     parser.add_argument("--alpha", type=float, default=1.0)
#     parser.add_argument("--horizon", type=int, default=32)
#     parser.add_argument("--lr-actor", type=float, default=1e-5)
#     parser.add_argument("--lr-cbf", type=float, default=1e-5)
#     parser.add_argument("--loss-action-coef", type=float, default=0.0001)
#     parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
#     parser.add_argument("--loss-safe-coef", type=float, default=1.0)
#     parser.add_argument("--loss-h-dot-coef", type=float, default=0.01)
#     parser.add_argument("--buffer-size", type=int, default=512)

#     # SFT / pretrained arguments
#     parser.add_argument("--pretrained-path", type=str, default=None,
#                         help="Path to pretrained model directory (e.g., pretrained/DoubleIntegrator/gcbf+/models/1000)")
#     parser.add_argument("--sft-steps", type=int, default=250,
#                         help="Number of SFT steps on failed data")
#     parser.add_argument("--sft-lr-cbf", type=float, default=5e-7,
#                         help="CBF learning rate for SFT")

#     # default arguments
#     parser.add_argument("--n-env-train", type=int, default=16)
#     parser.add_argument("--n-env-test", type=int, default=32)
#     parser.add_argument("--log-dir", type=str, default="./logs")
#     parser.add_argument("--eval-interval", type=int, default=1)
#     parser.add_argument("--eval-epi", type=int, default=1)
#     parser.add_argument("--save-interval", type=int, default=10)

#     args = parser.parse_args()
#     train(args)


# if __name__ == "__main__":
#     with ipdb.launch_ipdb_on_exception():
#         main()
import argparse
import datetime
import os
import ipdb
import numpy as np
import wandb
import yaml
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Allocate 80% of GPU memory (16GB out of 20GB)

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.trainer.trainer import Trainer
from gcbfplus.trainer.utils import is_connected


def train(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size
    )

    # create low level controller
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=args.gnn_layers,
        batch_size=256,
        buffer_size=args.buffer_size,
        horizon=args.horizon,
        lr_actor=args.lr_actor,
        lr_cbf=args.lr_cbf,
        alpha=args.alpha,
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=args.loss_action_coef,
        loss_unsafe_coef=args.loss_unsafe_coef,
        loss_safe_coef=args.loss_safe_coef,
        loss_h_dot_coef=args.loss_h_dot_coef,
        max_grad_norm=2.0,
        seed=args.seed,
    )

    # load pretrained model if specified
    if args.pretrained_path is not None:
        print(f"> Loading pretrained model from {args.pretrained_path}")
        import pickle
        actor_path = os.path.join(args.pretrained_path, 'actor.pkl')
        cbf_path = os.path.join(args.pretrained_path, 'cbf.pkl')
        algo.actor_train_state = algo.actor_train_state.replace(
            params=pickle.load(open(actor_path, 'rb'))
        )
        algo.cbf_train_state = algo.cbf_train_state.replace(
            params=pickle.load(open(cbf_path, 'rb'))
        )
        print("> Pretrained actor and CBF loaded successfully.")
        # also update the CBF target network with loaded params
        if hasattr(algo, 'cbf_tgt'):
            from gcbfplus.trainer.utils import tree_copy
            algo.cbf_tgt = algo.cbf_tgt.replace(params=tree_copy(algo.cbf_train_state.params))
            print("> CBF target network synced with loaded CBF.")

    # set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%Y%m%d%H%M%S")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(f"{args.log_dir}/{args.env}"):
        os.makedirs(f"{args.log_dir}/{args.env}")
    if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
        os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}")
    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"
    run_name = f"{args.algo}_{args.env}_{start_time}" if args.name is None else args.name

    # get training parameters
    # When running SFT, use sft_steps as the training step count
    sft_steps = args.sft_steps if (args.pretrained_path is not None and args.sft_steps > 0) else args.steps
    train_params = {
        "run_name": run_name,
        "training_steps": sft_steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
    }

    # create trainer
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

    # save config
    wandb.config.update(args)
    wandb.config.update(algo.config)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)

    # run SFT on failed data (this is the only training when using pretrained weights)
    if args.pretrained_path is not None and args.sft_steps > 0:
        print(f"> Starting SFT ({args.sft_mode}) for {args.sft_steps} steps, "
              f"batch_size={args.sft_batch_size}, CBF lr={args.sft_lr_cbf}")
        algo.sft_setup(sft_lr_cbf=args.sft_lr_cbf, sft_batch_size=args.sft_batch_size)
        trainer.sft_train(sft_mode=args.sft_mode)
        print("> SFT completed. Exiting.")
        wandb.finish()
        return

    # start normal training (only when no pretrained path is specified)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    # custom arguments
    parser.add_argument("-n", "--num-agents", type=int, default=8)
    parser.add_argument("--algo", type=str, default="gcbf+")
    parser.add_argument("--env", type=str, default="SimpleCar")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1250)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--obs", type=int, default=8)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, required=True)

    # gcbf / gcbf+ arguments
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr-actor", type=float, default=1e-5)
    parser.add_argument("--lr-cbf", type=float, default=1e-5)
    parser.add_argument("--loss-action-coef", type=float, default=0.0001)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef", type=float, default=0.01)
    parser.add_argument("--buffer-size", type=int, default=512)

    # SFT / pretrained arguments
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to pretrained model directory (e.g., pretrained/DoubleIntegrator/gcbf+/models/1000)")
    parser.add_argument("--sft-steps", type=int, default=150,
                        help="Number of SFT steps on failed data")
    parser.add_argument("--sft-lr-cbf", type=float, default=1e-5,
                        help="CBF learning rate for SFT")
    parser.add_argument("--sft-batch-size", type=int, default=256,
                        help="Mini-batch size for SFT (number of NPZ samples per mini-batch)")
    parser.add_argument("--sft-mode", type=str, default="exp1", choices=["exp1", "exp2", "exp3"],
                        help="SFT experiment: exp1=NPZ-only, exp2=mixed(NPZ+on-policy), exp3=DAgger(NPZ+QP-labeled on-policy)")

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=16)
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