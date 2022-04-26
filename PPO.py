import numpy as np

import tensorflow as tf
from typing import Tuple

from EnvWrapper import EnvWrapper
from gaes import gaes

eps = np.finfo(np.float32).eps.item()


class PPO:
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
                 entropy_beta=0.0,
                 critic_discount=1.0,
                 batch_size=0):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.lamda = lam
        self.epsilon = clip_ratio
        self.entropy_beta = entropy_beta
        self.critic_discount = critic_discount

        self.num_actions = num_actions
        self.state_shape = observation_dimensions

        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.batch_size = steps_per_episode if batch_size == 0 else batch_size
        self.max_steps = steps_per_episode
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.normalize = True
        self.target_kl = target_kl

    def train(self, actor, critic, states, actions, logprobs, advantages, returns, train_policy, train_value):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            if tf.cast(train_policy, tf.bool):
                tape.watch(actor.trainable_variables)
            if tf.cast(train_value, tf.bool):
                tape.watch(critic.trainable_variables)

            logits = actor(states)

            new_logprob_all = tf.nn.log_softmax(logits)
            new_logprob = tf.reduce_sum(tf.multiply(tf.one_hot(actions, self.num_actions), new_logprob_all), axis=1)

            ratio = tf.exp(new_logprob - logprobs)
            min_advantage = tf.where(
                advantages > 0,
                (1 + self.epsilon) * advantages,
                (1 - self.epsilon) * advantages,
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

    def train_minibatch(self, actor, critic, states, actions, logprobs, advantages, returns, train_policy, train_value):
        indices = tf.range(self.max_steps)
        indices = tf.random.shuffle(indices)

        kl = tf.constant(0.0)

        for i in tf.range(0, self.max_steps, self.batch_size):
            curr_ind = indices[i:tf.minimum(self.max_steps, i + self.batch_size)]

            kl = self.train(
                actor,
                critic,
                tf.gather(states, curr_ind),
                tf.gather(actions, curr_ind),
                tf.gather(logprobs, curr_ind),
                tf.gather(advantages, curr_ind),
                tf.gather(returns, curr_ind),
                train_policy,
                train_value
            )

        return kl

    def update(self, actor, critic, states, actions, rewards, values, logprobs, masks):
        advantages, returns = gaes(
            rewards=rewards,
            masks=masks,
            values=values[:-1],
            next_values=values[1:],
            gamma=self.gamma,
            lamda=self.lamda,
            normalize=self.normalize
        )

        train_policy = tf.constant(1)
        train_value = tf.constant(1)

        for t in tf.range(tf.maximum(self.train_policy_iterations, self.train_value_iterations)):
            if tf.equal(t, self.train_policy_iterations):
                train_policy = tf.constant(0)
            if tf.equal(t, self.train_value_iterations):
                train_value = tf.constant(0)

            kl = self.train_minibatch(
                actor, critic, states, actions, logprobs, advantages, returns, train_policy, train_value
            )

            if tf.cast(train_policy, tf.bool):
                if tf.greater(self.target_kl, 0.0):
                    if tf.greater(kl, 1.5 * self.target_kl):
                        train_policy = tf.constant(0)
                        if tf.logical_not(tf.cast(train_value, tf.bool)):
                            break

    @tf.function
    def run(self,
            env: EnvWrapper,
            initial_state: tf.Tensor,
            actor: tf.keras.Model,
            critic: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        state = initial_state

        episode_return = tf.constant(0)
        episode_length = tf.constant(0)
        episode_info = tf.constant(0)

        sum_return = tf.constant(0)
        sum_length = tf.constant(0)
        sum_info = tf.constant(0)
        num_episodes = tf.constant(0)

        const_shape = sum_return.shape

        state_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=self.state_shape)
        action_buffer = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=())
        reward_buffer = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=())
        value_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=())
        logprob_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=())
        mask_buffer = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=())

        for t in tf.range(self.max_steps):
            observation = tf.reshape(state, shape=(1, *self.state_shape))
            logits = actor(observation)
            action = tf.random.categorical(logits, 1, dtype=tf.int32)[0, 0]

            next_state, reward, done, info = env.tf_env_step(action)
            episode_return = tf.add(episode_return, reward)
            episode_return.set_shape(const_shape)
            episode_length = tf.add(episode_length, 1)
            episode_length.set_shape(const_shape)
            episode_info = info
            episode_info.set_shape(const_shape)

            value = critic(observation)

            logprob_all = tf.nn.log_softmax(logits)
            logprob = tf.reduce_sum(tf.multiply(tf.one_hot(action, self.num_actions), logprob_all))

            state_buffer = state_buffer.write(t, state)
            action_buffer = action_buffer.write(t, action)
            reward_buffer = reward_buffer.write(t, reward)
            value_buffer = value_buffer.write(t, value)
            logprob_buffer = logprob_buffer.write(t, logprob)
            mask_buffer = mask_buffer.write(t, 1 - done)

            state = next_state
            state.set_shape(self.state_shape)

            if tf.equal(t, self.max_steps - 1):
                value_buffer = value_buffer.write(t + 1, tf.cond(tf.cast(done, tf.bool), lambda: tf.constant(0.0),
                                                                 lambda: critic(
                                                                     tf.reshape(state, shape=(1, *self.state_shape)))))
                sum_return = tf.add(sum_return, episode_return)
                sum_return.set_shape(const_shape)
                sum_length = tf.add(sum_length, episode_length)
                sum_length.set_shape(const_shape)
                sum_info = tf.add(sum_info, episode_info)
                sum_info.set_shape(const_shape)
                num_episodes = tf.add(num_episodes, 1)
                num_episodes.set_shape(const_shape)
                break

            if tf.cast(done, tf.bool):
                sum_return = tf.add(sum_return, episode_return)
                sum_return.set_shape(const_shape)
                sum_length = tf.add(sum_length, episode_length)
                sum_length.set_shape(const_shape)
                sum_info = tf.add(sum_info, episode_info)
                sum_info.set_shape(const_shape)
                num_episodes = tf.add(num_episodes, 1)
                num_episodes.set_shape(const_shape)
                state = env.tf_env_reset()
                state.set_shape(self.state_shape)
                episode_return, episode_length, episode_info = tf.constant(0), tf.constant(0), tf.constant(0)
                episode_return.set_shape(const_shape)
                episode_length.set_shape(const_shape)
                episode_info.set_shape(const_shape)

        state_buffer = state_buffer.stack()
        action_buffer = action_buffer.stack()
        reward_buffer = reward_buffer.stack()
        value_buffer = value_buffer.stack()
        logprob_buffer = logprob_buffer.stack()
        mask_buffer = mask_buffer.stack()

        self.update(actor, critic, state_buffer, action_buffer, reward_buffer, value_buffer, logprob_buffer,
                    mask_buffer)

        return sum_return, sum_length, sum_info, num_episodes
