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
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            # "num_sgd_iter": lambda: random.randint(1, 30),
            # "sgd_minibatch_size": lambda: random.randint(128, 16384),
            # "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore)

    ray.init()
    run(
        "PPO",
        name="pbt_humanoid",
        scheduler=pbt,
        **{
            "num_samples": 4,
            "config": {
                "num_workers": 15,
                "num_gpus": 1,
                "gamma": 0.995,
                "kl_coeff": 1.0,
                "num_sgd_iter": 20,
                "lr": 5e-5,
                # "vf_loss_coeff": 0.5,
                # "clip_param": 0.2,
                "sgd_minibatch_size": 32768,
                "train_batch_size": 320000,
                # "grad_clip": 0.5,
                "batch_mode": "complete_episodes",
                "observation_filter": "MeanStdFilter",
                "env": "Humanoid-v2",
                # "env": multienv_name,
                # These params are tuned from a fixed starting value.
                # "lambda": 0.95,
                # "clip_param": 0.2,
                # "lr": 1e-4,
                # These params start off randomly drawn from a set.
            },
        })