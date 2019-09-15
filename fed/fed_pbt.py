import gym
import math
import random
import numpy as np
#from easydict import EasyDict
import argparse

import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.atari_wrappers import is_atari, wrap_deepmind

from utils import gen_policy_graphs, make_fed_env, fed_pbt_train

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

LR = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
GAMMA = [0.997,0.995,0.99,0.98,0.97,0.95,0.9,0.85,0.8]
ENTROPY_COEFF = [0.0, 0.001, 0.01]
lr_idx = np.random.randint(len(LR))
lr = LR[lr_idx]
gamma_idx = np.random.randint(len(GAMMA))
gamma = GAMMA[gamma_idx]
entropy_idx = np.random.randint(len(ENTROPY_COEFF))
entropy_coeff = ENTROPY_COEFF[entropy_idx]
def fed_pbt_wrapper(args):
    ray.init(ignore_reinit_error=True, redis_password="panda")
    policy_graphs = gen_policy_graphs(args)
    multienv_name = make_fed_env(args)
    callbacks = fed_pbt_train(args)
    if args.divide_batches:
        batch_size = args.train_batch_size/args.num_agents
    else:
        batch_size = args.train_batch_size
    tune.run(
        args.algo,
        name=f"{args.env}-{args.algo}-{args.num_agents}-{args.temp}-{args.interval}",
        #stop={"timesteps_total": args.max_steps},
        #resume=True,
        config={
                #"output": "fed-out", "output_max_file_size": 5000000,
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(lambda agent_id: f'agent_{agent_id}'),
                },
                "callbacks":{
                    "on_train_result": tune.function(callbacks),
                    # "on_train_result": tune.grid_search([tune.function(callback) for callback in callbacks]),
                },
                "num_workers": args.num_workers,
                "num_gpus": args.gpus,
                "env": multienv_name,
                # can't access args.lambda, it's a syntax error
                "lambda": args.Lambda,
                "gamma": args.gamma,
                #"gamma": gamma,
                "kl_coeff": args.kl_coeff,
                "clip_rewards": args.clip_rewards,
                "clip_param": args.clip_param,
                "vf_clip_param": args.vf_clip_param,
                "vf_loss_coeff": args.vf_loss_coeff,
                "entropy_coeff": args.entropy_coeff,
                #"entropy_coeff": entropy_coeff,
                "num_sgd_iter": args.num_sgd_iter,
                "num_envs_per_worker": args.num_envs_per_worker,
                "sgd_minibatch_size": args.sgd_minibatch_size,
                "sample_batch_size": args.sample_batch_size,
                # divide batch between agents, because we'll replicate it later
                "train_batch_size": batch_size, 
                "model": {
                    "free_log_std": args.free_log_std,
                    "dim": args.dim,
                },
                "use_gae": args.use_gae,
                "batch_mode": args.batch_mode,
                "vf_share_layers": args.vf_share_layers,
                "observation_filter": args.observation_filter,
                "grad_clip": args.grad_clip,
            },
        # resources_per_trial={
        #     "gpu": args.gpus,
        #     "cpu": args.cpus,
        # }
        checkpoint_at_end=True
    )

parser = argparse.ArgumentParser(
    description='Run Federated Population Based Training')
parser.add_argument("--env", type=str, 
    choices=['HalfCheetah-v2', 'Humanoid-v2', 'Hopper-v2',
    "BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "QbertNoFrameskip-v4", "BeamRiderNoFrameskip-v4", 
    "SpaceInvadersNoFrameskip-v4", "MountainCarContinuous-v0"], default="MountainCarContinuous-v0")
parser.add_argument("--tune", type=bool, default=True)
parser.add_argument("--pbt", type=bool, default=False)
parser.add_argument("--divide_batches", action="store_true")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--gpus", type=int, default=0)
# parser.add_argument("--cpus", type=int, default=1)
parser.add_argument("--num_agents", type=int, default=1)
parser.add_argument("--interval", type=int, default=8)
parser.add_argument("--temp", type=float, default=1.5)
parser.add_argument("--temp_decay", type=float, default=0.0)
parser.add_argument("--quantile", type=float, default=0.25)
parser.add_argument("--resample_probability", type=float, default=0.25)
parser.add_argument("--algo", type=str, default='PPO')
#parser.add_argument("--lr", type=list, default=[1e-2, 1e-3, 1e-4])
#parser.add_argument("--gammas", type=list, default=[0.995, 0.99, 0.95, 0.9])
parser.add_argument("--entropy_coeffs", type=list, default=[0.001, 0.01, 0.0])
parser.add_argument("--lr", type=list, default=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6])
parser.add_argument("--gammas", type=list, default=[0.997, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8])
# parser.add_argument("--entropy_coeffs", type=list, default=[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])
parser.add_argument("--explore_params", type=list, default=["lr", "gamma", "entropy_coeff"])
args = parser.parse_args()
parser.add_argument("--beta", type=float, default=args.temp)
#import pdb; pdb.set_trace()

if args.env in ["BreakoutNoFrameskip-v4", "QbertNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"]:
    parser.add_argument("--Lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_coeff", type=float, default=0.5)
    parser.add_argument("--clip_rewards", type=bool, default=True)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--num_sgd_iter", type=int, default=10)
    parser.add_argument("--sgd_minibatch_size", type=int, default=500)
    parser.add_argument("--sample_batch_size", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=6000)
    parser.add_argument("--free_log_std", type=bool, default=False)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--batch_mode", type=str, default="truncate_episodes")
    parser.add_argument("--vf_share_layers", type=bool, default=False)
    parser.add_argument("--observation_filter", type=str, default="NoFilter")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--is_atari", type=bool, default=True)
    parser.add_argument("--num_envs_per_worker", type=int, default=5)
    parser.add_argument("--dim", type=int, default=84)
    parser.add_argument("--max_steps", type=int, default=1e8)

elif args.env=='HalfCheetah-v2':
    parser.add_argument("--Lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--clip_rewards", type=bool, default=False)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    parser.add_argument("--num_sgd_iter", type=int, default=32)
    parser.add_argument("--sgd_minibatch_size", type=int, default=4096)
    parser.add_argument("--sample_batch_size", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=65536)
    parser.add_argument("--free_log_std", type=bool, default=False)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--batch_mode", type=str, default="truncate_episodes")
    parser.add_argument("--vf_share_layers", type=bool, default=False)
    parser.add_argument("--observation_filter", type=str, default="MeanStdFilter")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--is_atari", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=84)
    parser.add_argument("--max_steps", type=int, default=1e8)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)

elif args.env=='Humanoid-v2':
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.955)
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--clip_rewards", type=bool, default=False)
    parser.add_argument("--clip_param", type=float, default=0.3)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    parser.add_argument("--num_sgd_iter", type=int, default=20)
    parser.add_argument("--sgd_minibatch_size", type=int, default=32678)
    parser.add_argument("--sample_batch_size", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=320000)
    parser.add_argument("--free_log_std", type=bool, default=True)
    parser.add_argument("--use_gae", type=bool, default=False)
    parser.add_argument("--batch_mode", type=str, default="complete_episodes")
    parser.add_argument("--vf_share_layers", type=bool, default=False)
    parser.add_argument("--observation_filter", type=str, default="MeanStdFilter")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--is_atari", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=84)
    parser.add_argument("--max_steps", type=int, default=1e8)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)

elif args.env=="MountainCarContinuous-v0":
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_coeff", type=float, default=0.2)
    parser.add_argument("--clip_rewards", type=bool, default=False)
    parser.add_argument("--clip_param", type=float, default=0.3)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    parser.add_argument("--num_sgd_iter", type=int, default=30)
    parser.add_argument("--sgd_minibatch_size", type=int, default=128)
    parser.add_argument("--sample_batch_size", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=4000)
    parser.add_argument("--free_log_std", type=bool, default=False)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--batch_mode", type=str, default="truncate_episodes")
    parser.add_argument("--vf_share_layers", type=bool, default=False)
    parser.add_argument("--observation_filter", type=str, default="NoFilter")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--is_atari", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=84)
    parser.add_argument("--max_steps", type=int, default=1e5)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)

elif args.env=='Hopper-v2':
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--clip_rewards", type=bool, default=False)
    parser.add_argument("--clip_param", type=float, default=0.3)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    parser.add_argument("--num_sgd_iter", type=int, default=20)
    parser.add_argument("--sgd_minibatch_size", type=int, default=32678)
    parser.add_argument("--sample_batch_size", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=160000)
    parser.add_argument("--free_log_std", type=bool, default=False)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--batch_mode", type=str, default="complete_episodes")
    parser.add_argument("--vf_share_layers", type=bool, default=False)
    parser.add_argument("--observation_filter", type=str, default="MeanStdFilter")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--is_atari", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=84)
    parser.add_argument("--max_steps", type=int, default=2e7)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)

elif args.env=="PongNoFrameskip-v4":
    parser.add_argument("--Lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl_coeff", type=float, default=0.5)
    parser.add_argument("--clip_rewards", type=bool, default=True)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--vf_clip_param", type=float, default=10.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--num_sgd_iter", type=int, default=10)
    parser.add_argument("--sgd_minibatch_size", type=int, default=500)
    parser.add_argument("--sample_batch_size", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=5000)
    parser.add_argument("--free_log_std", type=bool, default=False)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--batch_mode", type=str, default="truncate_episodes")
    parser.add_argument("--vf_share_layers", type=bool, default=True)
    parser.add_argument("--observation_filter", type=str, default="NoFilter")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--is_atari", type=bool, default=True)
    parser.add_argument("--dim", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=3e6)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)
args = parser.parse_args()
# train
fed_pbt_wrapper(args)
