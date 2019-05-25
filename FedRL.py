import gym
import math
import random
import numpy as np
from easydict import EasyDict

import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.atari_wrappers import is_atari, wrap_deepmind

PBT_QUANTILE = 0.25
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
def make_multiagent(args):
    class MultiEnv(MultiAgentEnv):
        def __init__(self):
            self.agents = [gym.make(env) for _ in range(args.num_agents)]
            if is_atari(self.agents[0]):
                self.agents = [wrap_deepmind(env) for env in self.agents]
            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space

        def reset(self):
            self.dones = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

    return MultiEnv

def make_fed_env(args):   
    FedEnv = make_multiagent(args)
    env_name = "multienv_FedRL"
    register_env(env_name, lambda _: FedEnv())
    return env_name

def gen_policy_graphs(args):
    single_env = gym.make(args.env)
    if is_atari(single_env):
        single_env = wrap_deepmind(single_env)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    policy_graphs = {f'agent_{i}': (None, obs_space, act_space, {}) 
         for i in range(args.num_agents)}
    return policy_graphs

def policy_mapping_fn(agent_id):
    return f'agent_{agent_id}'
def change_weights(weights, i):
    """
    Helper function for FedQ-Learning
    """
    dct = {}
    for key, val in weights.items():
        # new_key = key
        still_here = key[:6]
        there_after = key[7:]
        # new_key[6] = i
        new_key = still_here + str(i) + there_after
        dct[new_key] = val
    # print(dct.keys())
    return dct

def synchronize(agent, weights, num_agents):
    """
    Helper function to synchronize weights of the multiagent
    """
    weights_to_set = {f'agent_{i}': weights 
         for i in range(num_agents)}
    # weights_to_set = {f'agent_{i}': change_weights(weights, i) 
    #    for i in range(num_agents)}
    agent.set_weights(weights_to_set)

def uniform_initialize(agent, num_agents):
    """
    Helper function for uniform initialization
    """
    new_weights = agent.get_weights(["agent_0"]).get("agent_0")
    # print(new_weights.keys())
    synchronize(agent, new_weights, num_agents)

def compute_softmax_weighted_avg(weights, alphas, num_agents, temperature=1):
    """
    Helper function to compute weighted avg of weights weighted by alphas
    Weights and alphas must have same keys. Uses softmax.
    params:
        weights - dictionary
        alphas - dictionary
    returns:
        new_weights - array
    """
    def softmax(x, beta=temperature, length=num_agents):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(beta * (x - np.max(x)))
        return (e_x / e_x.sum()).reshape(length, 1)
    
    alpha_vals = np.array(list(alphas.values()))
    soft = softmax(alpha_vals)
    weight_vals = np.array(list(weights.values()))
    new_weights = sum(np.multiply(weight_vals, soft))
    return new_weights

def compute_reward_weighted_avg(weights, alphas, num_agents):
    alpha_vals = np.array(list(alphas.values()))
    weight_vals = np.array(list(weights.values()))
    soft = (alpha_vals/alpha_vals.sum()).reshape(num_agents, 1)
    new_weights = sum(np.multiply(weight_vals, soft))
    return new_weights

def reward_weighted_update(agent, result, num_agents):
    """
    Helper function to synchronize weights of multiagent via
    reward-weighted avg of weights
    """
    all_weights = agent.get_weights()
    policy_reward_mean = result['policy_reward_mean']
    if policy_reward_mean:
        new_weights = compute_reward_weighted_avg(all_weights, policy_reward_mean, num_agents)
        synchronize(agent, new_weights, num_agents)

def softmax_reward_weighted_update(agent, result, num_agents, temperature=1, 
        resample_probability=0.25, explore_dict={"lr": [1e-3, 5e-5, 1e-4, 5e-4]}):
    """
    Helper function to synchronize weights of multiagent via
    softmax reward-weighted avg of weights with specific temperature
    """
    all_weights = agent.get_weights()
    policy_reward_mean = result['policy_reward_mean']
    if policy_reward_mean:
        new_weights = compute_softmax_weighted_avg(all_weights, policy_reward_mean, num_agents, temperature=temperature)
        synchronize(agent, new_weights, num_agents)
        explore(agent, policy_reward_mean, explore_dict, num_agents, resample_probability)

def population_based_train(agent, result, num_agents):
    """
    Helper function to implement population based training
    """
    all_weights = agent.get_weights()
    agents = [f'agent_{id}' for id in range(num_agents)]
    policy_reward_mean = result['policy_reward_mean']
    if policy_reward_mean:
        # import pdb; pdb.set_trace()
        sorted_rewards = sorted(policy_reward_mean.items(), key=lambda kv: kv[1])
        upper_quantile = [kv[0] for kv in sorted_rewards[int(math.floor(PBT_QUANTILE * -num_agents)):]]
        lower_quantile = [kv[0] for kv in sorted_rewards[:int(math.ceil(PBT_QUANTILE * num_agents))]]
        new_weights = {agent_id: all_weights[agent_id] if agent_id not in lower_quantile else all_weights[random.choice(upper_quantile)] 
         for agent_id in agents}
        agent.set_weights(new_weights)
        # explore(agent, lower_quantile)

def explore(agent, policy_reward_mean, explore_dict, num_agents, resample_probability):
    """
    Helper function to explore hyperparams (currently just lr)
    """
    from ray.rllib.utils.schedules import ConstantSchedule
    sorted_rewards = sorted(policy_reward_mean.items(), key=lambda kv: kv[1])
    upper_quantile = [kv[0] for kv in sorted_rewards[int(math.floor(PBT_QUANTILE * -num_agents)):]]
    lower_quantile = [kv[0] for kv in sorted_rewards[:int(math.ceil(PBT_QUANTILE * num_agents))]]
    #print(lower_quantile)
    for agent_id in lower_quantile:
        policy_graph = agent.get_policy(agent_id)
        # old_lr = policy_graph._sess.run(policy_graph.cur_lr)
        new_policy_graph = agent.get_policy(random.choice(upper_quantile))
        if "lr" in explore_dict:
            exemplar_lr = new_policy_graph.cur_lr
            distribution = explore_dict["lr"]
            if random.random() < resample_probability or \
                    exemplar_lr not in distribution:
                new_val = random.choice(distribution)
                # policy_graph.cur_lr.load(
                #     new_val,
                #     session=policy_graph._sess)
                policy_graph.lr_schedule = ConstantSchedule(new_val)
            elif random.random() > 0.5:
                new_val = distribution[max(
                    0,
                    distribution.index(exemplar_lr) - 1)]
                # policy_graph.cur_lr.load(
                #     new_val,
                #     session=policy_graph._sess)
                policy_graph.lr_schedule = ConstantSchedule(new_val)
            else:
                new_val = distribution[min(
                    len(distribution) - 1,
                    distribution.index(exemplar_lr) + 1)]
                # policy_graph.cur_lr.load(
                #     new_val,
                #     session=policy_graph._sess)
                policy_graph.lr_schedule = ConstantSchedule(new_val)
        # print(f"Changed lr from {old_lr} to {policy_graph._sess.run(policy_graph.cur_lr)} which is {new_val}")
        # if "lr" in explore_dict:
        #     if random.random() < resample_probability:
        #         policy_graph.cur_lr = explore_dict["lr"]()
        #     elif random.random() > 0.5:
        #         policy_graph.cur_lr = new_policy_graph.cur_lr * 1.2
        #     else:
        #         policy_graph.cur_lr = new_policy_graph.cur_lr * 0.8

def fed_train(args):
    num_agents = args.num_agents
    if args.tune:
        temp_schedule = args.temp_schedule
        init_temp = temp_schedule[0]
        init_temp_1, init_temp_2, init_temp_3, init_temp_4 = init_temp
        hotter_temp = temp_schedule[1]
        hotter_temp_1, hotter_temp_2, hotter_temp_3, hotter_temp_4 = hotter_temp
        temp_shift = temp_schedule[2]
        fed_schedule = args.fed_schedule
        init_iters = fed_schedule[0]
        init_iters_1, init_iters_2, init_iters_3, init_iters_4 = init_iters
        increased_iters = fed_schedule[1]
        increased_iters_1, increased_iters_2, increased_iters_3, increased_iters_4 = increased_iters
        fed_shift = fed_schedule[2]
        def fed_learn_1(info):
            return
    #       get stuff out of info
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            num_iters = init_iters_1
            temperature = init_temp_1
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            # potentially scale up
            if result['timesteps_total'] > fed_shift:
                num_iters = increased_iters_1
            if result['timesteps_total'] > temp_shift:
                temperature = hotter_temp_1
            # correct result reporting
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            result['federated'] = "No federation"
            if result['training_iteration'] == 1:
                uniform_initialize(agent, num_agents)
            elif result['training_iteration'] % num_iters == 0:
                result['federated'] = f"Federation with {temperature}"
                # update weights
                #reward_weighted_update(agent, result, num_agents)
                softmax_reward_weighted_update(agent, result, num_agents, temperature, explore_dict={"lr": args.lr})
                # clear buffer, don't want smoothing here
                optimizer.episode_history = []
        def fed_learn_2(info):
    #       get stuff out of info
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            num_iters = init_iters_2
            temperature = init_temp_2
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            # potentially scale up
            if result['timesteps_total'] > fed_shift:
                num_iters = increased_iters_2
            if result['timesteps_total'] > temp_shift:
                temperature = hotter_temp_2
            # correct result reporting
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            result['federated'] = "No federation"
            if result['training_iteration'] == 1:
                uniform_initialize(agent, num_agents)
            elif result['training_iteration'] % num_iters == 0:
                result['federated'] = f"Federation with {temperature}"
                # update weights
                #reward_weighted_update(agent, result, num_agents)
                softmax_reward_weighted_update(agent, result, num_agents, temperature, explore_dict={"lr": args.lr})
                # clear buffer, don't want smoothing here
                optimizer.episode_history = []
        def fed_learn_3(info):
    #       get stuff out of info
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            num_iters = init_iters_3
            temperature = init_temp_3
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            # potentially scale up
            if result['timesteps_total'] > fed_shift:
                num_iters = increased_iters_3
            if result['timesteps_total'] > temp_shift:
                temperature = hotter_temp_3
            # correct result reporting
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            result['federated'] = "No federation"
            if result['training_iteration'] == 1:
                uniform_initialize(agent, num_agents)
            elif result['training_iteration'] % num_iters == 0:
                result['federated'] = f"Federation with {temperature}"
                # update weights
                #reward_weighted_update(agent, result, num_agents)
                softmax_reward_weighted_update(agent, result, num_agents, temperature, explore_dict={"lr": args.lr})
                # clear buffer, don't want smoothing here
                optimizer.episode_history = []
        def fed_learn_4(info):
    #       get stuff out of info
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            num_iters = init_iters_4
            temperature = init_temp_4
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            # potentially scale up
            if result['timesteps_total'] > fed_shift:
                num_iters = increased_iters_4
            if result['timesteps_total'] > temp_shift:
                temperature = hotter_temp_4
            # correct result reporting
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            result['federated'] = "No federation"
            if result['training_iteration'] == 1:
                uniform_initialize(agent, num_agents)
            elif result['training_iteration'] % num_iters == 0:
                result['federated'] = f"Federation with {temperature}"
                # update weights
                #reward_weighted_update(agent, result, num_agents)
                softmax_reward_weighted_update(agent, result, num_agents, temperature, explore_dict={"lr": args.lr})
                # clear buffer, don't want smoothing here
                optimizer.episode_history = []
        return fed_learn_1, fed_learn_2, fed_learn_3, fed_learn_4
    elif args.pbt:
        iters = args.num_iters
        num_iters_1, num_iters_2, num_iters_3, num_iters_4 = iters
        def pbt_1(info):
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            if result['training_iteration'] % num_iters_1 == 0:
                population_based_train(agent, result, num_agents)
                optimizer.episode_history = []
        def pbt_2(info):
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            if result['training_iteration'] % num_iters_2 == 0:
                population_based_train(agent, result, num_agents)
                optimizer.episode_history = []
        def pbt_3(info):
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            if result['training_iteration'] % num_iters_3 == 0:
                population_based_train(agent, result, num_agents)
                optimizer.episode_history = []
        def pbt_4(info):
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            if result['training_iteration'] % num_iters_4 == 0:
                population_based_train(agent, result, num_agents)
                optimizer.episode_history = []
        return pbt_1, pbt_2, pbt_3, pbt_4
    else:
        temperature = args.temperature
        num_iters = args.num_iters
        def fed_learn(info):
    #       get stuff out of info
            result = info["result"]
            agent = info["trainer"]
            optimizer = agent.optimizer
            result['timesteps_total'] = result['timesteps_total'] * num_agents
            # correct result reporting
            result['episode_reward_mean'] = result['episode_reward_mean']/num_agents
            result['episode_reward_max'] = result['episode_reward_max']/num_agents
            result['episode_reward_min'] = result['episode_reward_min']/num_agents
            result['federated'] = "No federation"
            if result['training_iteration'] == 1:
                uniform_initialize(agent, num_agents)
            elif result['training_iteration'] % num_iters == 0:
                result['federated'] = f"Federation with {temperature}"
                # update weights
                reward_weighted_update(agent, result, num_agents)
                # softmax_reward_weighted_update(agent, result, num_agents, temperature)
                # clear buffer, don't want smoothing here
                optimizer.episode_history = []
        return fed_learn

def fedrl(args):
    ray.init(ignore_reinit_error=True)
    policy_graphs = gen_policy_graphs(args)
    multienv_name = make_fed_env(args)
   # callbacks = fed_train(args)
    tune.run(
        #'PPO',
        args.algo,
        name=f"{args.env}-{args.algo}-{args.num_agents}",
        stop={"timesteps_total": 1e6},
        config={
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(lambda agent_id: f'agent_{agent_id}'),
                },
                #"env": args.env,
                "lambda": 0.95, #0.5 to 0.99
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01, #0 to 0.2
                "train_batch_size": tune.grid_search([5000, 25000, 50000, 75000]),
                "sample_batch_size": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "num_envs_per_worker": 1,
                "batch_mode": "truncate_episodes",
                "observation_filter": "NoFilter",
                "vf_share_layers": True,
                "num_gpus": 1,
                #"lr": 1e-4,
                #"callbacks":{
                 #   "on_train_result": tune.function(callbacks[0]),
                    #"on_train_result": tune.grid_search([tune.function(callback) for callback in callbacks]),
                #},
                "num_workers": 15,
                "env": multienv_name,
            },
        # resources_per_trial={
        #     "gpu": 1,
        #     "cpu": 16,
        # }
        checkpoint_at_end=True
    )


args = EasyDict({
    'num_agents': 1,
    'num_workers': 15,
    'fed_schedule': [(100,0,4,8), (2,3,4,5), 2e7],
    'temp_schedule': [(0, 2, 2, 2), (0.5, 1, 1.5, 2), 2e7],
    'tune': True,
    'pbt': False,
    'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    # 'temperature': 1,
    # 'num_iters': [2, 4, 8, 16],
    # 'timesteps': 1e7,
    # 'lr': 5e-4,
    # 'lrs': [5e-5, 5e-4, 5e-3],
    # 'episodes': 150,
#     'num_iters': 100,
    'env': "BreakoutNoFrameskip-v4",
    #'env': 'HalfCheetah-v2',
    'name': 'fed_experiment',
    'algo': 'PPO',
    # 'hyperparams': {
    #     "gamma": 0.99,
    #     "lambda": 0.95,
    #     "kl_coeff": 1.0,
    #     "num_sgd_iter": 32,
    #     "lr": .0003,
    #     "vf_loss_coeff": 0.5,
    #     "clip_param": 0.2,
    #     "sgd_minibatch_size": 128,
    #     "train_batch_size": 256,
    #     "grad_clip": 0.5,
    #     "batch_mode": "truncate_episodes",
    #     "observation_filter": "MeanStdFilter",
    # }
})
# train
fedrl(args)
# eval
