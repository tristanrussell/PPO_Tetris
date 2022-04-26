import numpy as np

import tensorflow as tf

from PPO import PPO

eps = np.finfo(np.float32).eps.item()


class TRGPPO(PPO):
    def __init__(self,
                 num_actions,
                 observation_dimensions,
                 policy_optimizer,
                 value_optimizer,
                 steps_per_episode,
                 train_policy_iterations,
                 train_value_iterations,
                 target_kl,
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 entropy_beta=0.001,
                 critic_discount=1.0,
                 delta=0.001,
                 batch_size=0):
        super().__init__(
            num_actions,
            observation_dimensions,
            policy_optimizer,
            value_optimizer,
            steps_per_episode,
            train_policy_iterations,
            train_value_iterations,
            target_kl,
            gamma,
            lam,
            clip_ratio,
            entropy_beta,
            critic_discount,
            batch_size
        )
        self.delta = delta

    def train(self, actor, critic, states, actions, logprobs, advantages, returns, train_policy, train_value):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            if tf.cast(train_policy, tf.bool):
                tape.watch(actor.trainable_variables)
            if tf.cast(train_value, tf.bool):
                tape.watch(critic.trainable_variables)

            logits = actor(states)

            new_logprob_all = tf.nn.log_softmax(logits)
            new_logprob = tf.reduce_sum(tf.multiply(tf.one_hot(actions, self.num_actions), new_logprob_all), axis=1)

            kls = tf.multiply(tf.math.exp(new_logprob), (new_logprob - logprobs))
            kls_mask = tf.less(kls, self.delta)
            kls = tf.boolean_mask(kls, kls_mask)
            low_clip = tf.reduce_min(kls)
            low_clip = tf.minimum(low_clip, 1 - self.epsilon)
            high_clip = tf.reduce_max(kls)
            high_clip = tf.maximum(high_clip, 1 + self.epsilon)

            ratio = tf.exp(new_logprob - logprobs)
            min_advantage = tf.where(
                advantages > 0,
                high_clip * advantages,
                low_clip * advantages,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, min_advantage)
            )

            value_loss = tf.reduce_mean(tf.square(returns - critic(states)))

            entropy = -tf.reduce_sum(tf.multiply(tf.math.exp(new_logprob_all), new_logprob_all), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)

            loss = policy_loss + self.critic_discount * value_loss - self.entropy_beta * entropy

        if tf.cast(train_policy, tf.bool):
            policy_grads = tape.gradient(loss, actor.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

        if tf.cast(train_value, tf.bool):
            value_grads = tape.gradient(loss, critic.trainable_variables)
            self.value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

        del tape

        logits = actor(states)

        new_logprob_all = tf.nn.log_softmax(logits)
        new_logprob = tf.reduce_sum(tf.multiply(tf.one_hot(actions, self.num_actions), new_logprob_all), axis=1)

        kl = tf.reduce_mean(logprobs - new_logprob)
        kl = tf.reduce_sum(kl)
        return kl
