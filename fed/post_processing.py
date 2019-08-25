def postprocess_ppo_fed(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    if not other_agent_batches:
        return postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)
    other_agent_batches['this_agent'] = ('this_agent', sample_batch)
    new_batch = postprocess_ppo_gae(policy, sample_batch)
    other_batches = [postprocess_ppo_gae(policy, other_agent_batches[agent][1], None, episode) for agent in other_agent_batches]
    import numpy as np
    for i in new_batch:
        global_batch = np.concatenate([other_batch[i] for other_batch in other_batches])
        new_batch[i] = global_batch
    return new_batch

PPOTFPolicy = build_tf_policy(
    name="PPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    #postprocess_fn=postprocess_ppo_gae,
    postprocess_fn=postprocess_ppo_fed,
    gradients_fn=clip_gradients,
    before_loss_init=setup_mixins,
    mixins=[LearningRateSchedule, KLCoeffMixin, ValueNetworkMixin])
