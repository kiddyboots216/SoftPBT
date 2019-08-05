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
	except:
		unwrapped = unwrapped
	for i, u in enumerate(unwrapped):
		monitor = get_wrapper_by_cls(u, MonitorEnv)
		if not monitor:
			return None
		for eps_rew, eps_len in monitor.next_episode_results():
			agent_rewards[f{'i'}] += eps_rew
			atari_out.append(RolloutMetrics(eps_len, eps_rew, {}, {}, {}))
	return atari_out, agent_rewards

if not new_episode:
	episode.length += 1
	episode.batch_builder.count += 1

atari_metrics, agent_rewards = _fetch_atari_metrics(base_env)
if atari_metrics is not None:
	if not new_episode:
		episode._add_agent_rewards(agent_rewards)
	for m in atari_metrics:
		outputs.append(
			m._replace(custom_metrics=episode.custom_metrics,
				agent_rewards=episode.agent_rewards
				))