#!/usr/bin/env python
"""Example of using PBT with RLlib.
Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import ray
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining

if __name__ == "__main__":

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        reward_attr="episode_reward_mean",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            # "lambda": lambda: random.uniform(0.9, 1.0),
            # "clip_param": lambda: random.uniform(0.01, 0.5),
            # "lr": [1e-3, 5e-5, 1e-4, 5e-4],
            "lr": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
            "gamma": [0.997,0.995,0.99,0.98,0.97,0.95,0.9,0.85,0.8],
            "entropy_coeff": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0],
            # "num_sgd_iter": lambda: random.randint(1, 30),
            # "sgd_minibatch_size": lambda: random.randint(128, 16384),
            # "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore)

    ray.init()
    run(
        "PPO",
        name="pbt_breakout",
        scheduler=pbt,
        **{
            "num_samples": 8,
            "config": {
                "lambda": 0.95, #0.5 to 0.99
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "gamma": 0.97,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01, #0 to 0.2
                "train_batch_size": 5000,
                "sample_batch_size": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "num_workers": 5,
                "num_envs_per_worker": 5,
                "batch_mode": "truncate_episodes",
                "observation_filter": "NoFilter",
                "vf_share_layers": True,
                "num_gpus": 1,
                "env": "BreakoutNoFrameskip-v4",
                # "env": multienv_name,
                # These params are tuned from a fixed starting value.
                # "lambda": 0.95,
                # "clip_param": 0.2,
                "lr": 1e-4, #1e-2 to 1e-5
                # These params start off randomly drawn from a set.
            },
        })
