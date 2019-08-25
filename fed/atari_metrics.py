def _fetch_atari_metrics(base_env):
    """
    Fix atari metrics reporting for multiagentenv
    """
    unwrapped = base_env.get_unwrapped()
    if not unwrapped:
        return None
    atari_out = []	
    try:
        agent_rewards = {f'{id}': 0 for id in range(len(unwrapped[0].agents))}
        unwrapped = unwrapped[0].agents
        for i, u in enumerate(unwrapped):
            monitor = get_wrapper_by_cls(u, MonitorEnv)
            if not monitor:
                return None
            for eps_rew, eps_len in monitor.next_episode_results():
                agent_rewards[f'{i}'] += eps_rew
                atari_out.append(RolloutMetrics(eps_len, eps_rew, {}, {}, {}))
    except:
        agent_rewards, unwrapped = {}, unwrapped
        for u in unwrapped:
            monitor = get_wrapper_by_cls(u, MonitorEnv)
            if not monitor:
                return None
            for eps_rew, eps_len in monitor.next_episode_results():
                atari_out.append(RolloutMetrics(eps_len, eps_rew, {}, {}, {}))
    return atari_out, agent_rewards

def _process_observations(base_env, policies, batch_builder_pool,
                          active_episodes, unfiltered_obs, rewards, dones,
                          infos, off_policy_actions, horizon, preprocessors,
                          obs_filters, unroll_length, pack, callbacks,
                          soft_horizon):
    """Record new data from the environment and prepare for policy evaluation.

    Returns:
        active_envs: set of non-terminated env ids
        to_eval: map of policy_id to list of agent PolicyEvalData
        outputs: list of metrics and samples to return from the sampler
    """

    active_envs = set()
    to_eval = defaultdict(list)
    outputs = []

    # For each environment
    for env_id, agent_obs in unfiltered_obs.items():
        new_episode = env_id not in active_episodes
        episode = active_episodes[env_id]
        if not new_episode:
            episode.length += 1
            episode.batch_builder.count += 1
            #episode._add_agent_rewards(rewards[env_id])

        if (episode.batch_builder.total() > max(1000, unroll_length * 10)
                and log_once("large_batch_warning")):
            logger.warning(
                "More than {} observations for {} env steps ".format(
                    episode.batch_builder.total(),
                    episode.batch_builder.count) + "are buffered in "
                "the sampler. If this is more than you expected, check that "
                "that you set a horizon on your environment correctly. Note "
                "that in multi-agent environments, `sample_batch_size` sets "
                "the batch size based on environment steps, not the steps of "
                "individual agents, which can result in unexpectedly large "
                "batches.")

        # Check episode termination conditions
        if dones[env_id]["__all__"] or episode.length >= horizon:
            hit_horizon = (episode.length >= horizon
                           and not dones[env_id]["__all__"])
            all_done = True
            atari_metrics_and_rewards = _fetch_atari_metrics(base_env)
            if atari_metrics_and_rewards is not None:
                atari_metrics, agent_rewards = atari_metrics_and_rewards
                if not new_episode:
                    episode._add_agent_rewards(agent_rewards)
                for m in atari_metrics:
                    outputs.append(
                        m._replace(custom_metrics=episode.custom_metrics,
                        agent_rewards=episode.agent_rewards
			))
#            atari_metrics = _fetch_atari_metrics(base_env)
#            if atari_metrics is not None:
#                for m in atari_metrics:
#                    outputs.append(
#                        m._replace(custom_metrics=episode.custom_metrics))
            else:
                outputs.append(
                    RolloutMetrics(episode.length, episode.total_reward,
                                   dict(episode.agent_rewards),
                                   episode.custom_metrics, {}))
        else:
            hit_horizon = False
            all_done = False
            active_envs.add(env_id)
