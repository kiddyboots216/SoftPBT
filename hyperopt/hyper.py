from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import ray
from ray.tune import run, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

if __name__ == "__main__":
    ray.init(redis_password="hyperopt")
    space = {
        "lr": hp.choice("lr", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]),
        "gamma": hp.choice("gamma", [0.997, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8]),
        "entropy_coeff": hp.choice("entropy_coeff", [0.0, 0.001, 0.01])
    }
    current_best_params = [ 
        {
            "lr": 4,
            "gamma": 4,
            "entropy_coeff": 1,
        }
    ] 
    config = {
        "num_samples": 1000,
        "config": {
            "lambda": 0.95,
            "clip_rewards": True,
            "clip_param": 0.1,
            "vf_clip_param": 10.0,
            "kl_coeff": 0.5,
            "train_batch_size": 5000,
            "sample_batch_size": 100,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 15,
            "num_envs_per_worker": 5,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "vf_share_layers": True,
            "num_gpus": 1,
            "env": "BreakoutNoFrameskip-v4",
            # TODO: Are these necessary?
            #"lr": 1e-3,
            #"gamma": 0.99,
            #"entropy_coeff": 0.001,
        },
        "stop": {
            "timesteps_total": 50000000,
        },
    }
    algo = HyperOptSearch(
        space,
        max_concurrent=4,
        metric="episode_reward_mean",
        mode="max",
        #points_to_evaluate=current_best_params
    )
    scheduler = AsyncHyperBandScheduler(metric="episode_reward_mean", mode="max")
    run(
        "PPO",
        name="hyperopt_breakout_50m",
        search_alg=algo,
        #scheduler=scheduler,
        **config
    )
