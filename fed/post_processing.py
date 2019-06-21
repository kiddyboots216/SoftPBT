def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(len(policy.model.state_in)):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch

def postprocess_ppo_fed(policy,
                        sample_batch,
                        other_agent_batches,
                        episode=None):
    other_agent_batches['this_agent'] = ('this_agent', sample_batch)
    other_batches = [postprocess_ppo_gae(other_agent_batches[agent][1] for agent in other_agent_batches)]
    import numpy as np
    for i in sample_batch:
        global_batch = np.concatenate([other_batch[i] for other_batch in other_batches])
        sample_batch[i] = global_batch
    return sample_batch

PPOTFPolicy = build_tf_policy(
    name="PPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_ppo_fed,
    gradients_fn=clip_gradients,
    before_loss_init=setup_mixins,
    mixins=[LearningRateSchedule, KLCoeffMixin, ValueNetworkMixin])