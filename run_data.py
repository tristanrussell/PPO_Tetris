import datetime
import os

import imageio
import gym
import gym_simpletetris
import numpy as np
import tqdm
import argparse
import configparser
import json

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D
from tensorflow.keras import Model

from gaes import gaes

eps = np.finfo(np.float32).eps.item()

parser = argparse.ArgumentParser()

parser.add_argument('--names', nargs='*', required=True,
                    help="A list of tests to run, separated by commas.")

parser.add_argument('-s', '--start', type=int, default=0,
                    help="The training iteration to start on (default is 0).")

parser.add_argument('-e', '--end', type=int, default=0,
                    help="The training iteration to finish on (default is the latest checkpoint).")

parser.add_argument('-g', '--gif', type=int, default=0,
                    help="Enable gifs.")

parser.add_argument('-l', '--logging', action='store_true',
                    help="Enable logs.")

args = parser.parse_args()

enable_logging = args.logging


def mlp(x, sizes, activation='relu', output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = Dense(units=size, activation=activation)(x)
    return Dense(units=sizes[-1], activation=output_activation)(x)


def build_conv(x, layers, flatten=True, activation='relu'):
    conv = x

    for layer in layers:
        conv = Conv2D(layer[0], layer[1], strides=layer[2], activation=activation)(conv)

    if flatten:
        return Flatten()(conv)
    else:
        return conv


@tf.function
def log_values(actor: tf.keras.Model,
               critic: tf.keras.Model,
               states: tf.Tensor,
               actions: tf.Tensor,
               rewards: tf.Tensor,
               values: tf.Tensor,
               logprobs: tf.Tensor,
               masks: tf.Tensor,
               num_actions: tf.Tensor,
               step: tf.Tensor,
               agent:tf.Tensor=tf.constant('PPO'),
               gamma:tf.Tensor=tf.constant(0.99),
               lamda:tf.Tensor=tf.constant(0.95),
               clip_ratio:tf.Tensor=tf.constant(0.2),
               entropy_beta:tf.Tensor=tf.constant(0.0),
               critic_discount:tf.Tensor=tf.constant(1.0),
               delta:tf.Tensor=tf.constant(0.001)):
    advantages, returns = gaes(
        rewards=rewards,
        masks=masks,
        values=values[:-1],
        next_values=values[1:],
        gamma=gamma,
        lamda=lamda,
        normalize=True
    )

    logits = actor(states)

    new_logprob_all = tf.nn.log_softmax(logits)
    new_logprob = tf.reduce_sum(tf.multiply(tf.one_hot(actions, num_actions), new_logprob_all), axis=1)

    ratio = tf.constant(0.0)
    min_advantage = tf.constant(0.0)

    if tf.equal(agent, tf.constant('PPO')):
        ratio = tf.exp(new_logprob - logprobs)
        min_advantage = tf.where(
            advantages > 0,
            (1 + clip_ratio) * advantages,
            (1 - clip_ratio) * advantages,
        )

    if tf.equal(agent, tf.constant('TRGPPO')):
        kls = tf.multiply(tf.math.exp(new_logprob), (new_logprob - logprobs))
        kls_mask = tf.less(kls, delta)
        kls = tf.boolean_mask(kls, kls_mask)
        low_clip = tf.reduce_min(kls)
        low_clip = tf.minimum(low_clip, 1 - clip_ratio)
        high_clip = tf.reduce_max(kls)
        high_clip = tf.maximum(high_clip, 1 + clip_ratio)

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

    loss = policy_loss + critic_discount * value_loss - entropy_beta * entropy

    with summary_writer.as_default():
        tf.summary.scalar('policy_loss', policy_loss, step=step)
        tf.summary.scalar('value_loss', value_loss, step=step)
        tf.summary.scalar('entropy', entropy, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('time', time, step=step)
        tf.summary.scalar('score', time + (100 * lines_cleared), step=step)
        tf.summary.scalar('lines_cleared', lines_cleared, step=step)


for name in args.names:
    checkpoint_path = name + "/weights/{model}-{epoch:09d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    gif_path = name + "/gifs/{epoch:09d}.gif"
    gif_dir = os.path.dirname(gif_path)
    results_path = name + "/results.csv"
    results_dir = os.path.dirname(results_path)
    config_path = name + "/config.ini"
    config_dir = os.path.dirname(config_path)

    config = configparser.ConfigParser()
    config.read(config_path)

    environment_config = config['environment']
    agent_config = config['agent']

    reward_step = environment_config.getboolean('reward_step')
    penalise_height = environment_config.getboolean('penalise_height')
    penalise_height_increase = environment_config.getboolean('penalise_height_increase')
    advanced_clears = environment_config.getboolean('advanced_clears')
    high_scoring = environment_config.getboolean('high_scoring')
    penalise_holes = environment_config.getboolean('penalise_holes')
    penalise_holes_increase = environment_config.getboolean('penalise_holes_increase')
    lock_delay = int(environment_config.get('lock_delay', '0'))
    step_reset = environment_config.getboolean('step_reset')
    observation = environment_config.get('observation', 'ram')
    extend_dims = environment_config.getboolean('extend_dims')

    agent = agent_config.get('agent')
    gamma = agent_config.getfloat('gamma')
    lam = agent_config.getfloat('lambda')
    clip_ratio = agent_config.getfloat('clip_ratio')
    entropy_beta = agent_config.getfloat('entropy_beta')
    critic_discount = agent_config.getfloat('critic_discount')
    delta = agent_config.getfloat('delta')
    hidden_sizes = json.loads(agent_config.get('hidden_sizes'))
    hidden_sizes_activation = agent_config.get('hidden_sizes_activation', 'relu')
    conv_layers = json.loads(agent_config.get('conv_layers'))
    conv_layers_activation = agent_config.get('conv_layers_activation', 'relu')
    flatten = agent_config.getboolean('flatten')

    if enable_logging:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = name + '/logs/test/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

    env = gym.make("SimpleTetris-v0",
                   obs_type=observation,
                   extend_dims=extend_dims,
                   reward_step=reward_step,
                   penalise_height=penalise_height,
                   penalise_height_increase=penalise_height_increase,
                   advanced_clears=advanced_clears,
                   high_scoring=high_scoring,
                   penalise_holes=penalise_holes,
                   penalise_holes_increase=penalise_holes_increase,
                   lock_delay=lock_delay,
                   step_reset=step_reset)

    observation_dimensions = env.observation_space.shape
    num_actions = env.action_space.n

    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    observation_input = Input(shape=observation_dimensions, dtype=tf.float32)
    logits = build_conv(observation_input, layers=conv_layers, flatten=flatten, activation=conv_layers_activation)
    logits = mlp(logits, hidden_sizes + [num_actions], hidden_sizes_activation, None)
    actor = Model(inputs=observation_input, outputs=logits)
    value = build_conv(observation_input, layers=conv_layers, flatten=flatten, activation=conv_layers_activation)
    value = tf.squeeze(
        mlp(value, hidden_sizes + [1], hidden_sizes_activation, None)
    )
    critic = Model(inputs=observation_input, outputs=value)

    start_epoch = 0
    end_epoch = 0

    if args.start > 0:
        start_epoch = args.start
    else:
        f = open(results_path, "w")
        f.write("Epoch,Time,Score,Lines Cleared\n")
        f.close()

    if args.end > 0:
        end_epoch = args.end + 1
    else:
        end_epoch = int(config['DEFAULT'].get('checkpoint', '0')) + 1

    with tqdm.trange(start_epoch, end_epoch, 50) as t:
        t.set_description(name)

        for i in t:
            t.set_postfix(epoch=i)
            if i > 0:
                actor.load_weights(checkpoint_path.format(model='actor', epoch=i))
                critic.load_weights(checkpoint_path.format(model='critic', epoch=i))

            num_episodes = 3
            max_steps = 4000
            time = 0
            score = 0
            lines_cleared = 0
            frames = []

            state_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                                          element_shape=observation_dimensions)
            action_buffer = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=())
            reward_buffer = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=())
            value_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=())
            logprob_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=())
            mask_buffer = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=())
            ts = 0
            done = 0

            for _ in range(num_episodes):
                state = env.reset()
                frames.append(env.render(mode="rgb_array"))
                info = None
                for _ in range(max_steps):
                    observation = tf.reshape(state, shape=(1, *observation_dimensions))
                    logits = actor(observation)
                    action = tf.random.categorical(logits, 1, dtype=tf.int32)[0, 0]

                    next_state, reward, done, info = env.step(action.numpy())
                    frames.append(env.render(mode="rgb_array"))

                    if enable_logging:
                        value = critic(observation)

                        logprob_all = tf.nn.log_softmax(logits)
                        logprob = tf.reduce_sum(tf.multiply(tf.one_hot(action, num_actions), logprob_all))

                        state_buffer = state_buffer.write(ts, state)
                        action_buffer = action_buffer.write(ts, action)
                        reward_buffer = reward_buffer.write(ts, reward)
                        value_buffer = value_buffer.write(ts, value)
                        logprob_buffer = logprob_buffer.write(ts, logprob)
                        mask_buffer = mask_buffer.write(ts, 1 - done)

                    state = next_state

                    ts = ts + 1

                    if done:
                        break

                time += info.get('time')
                score += info.get('score')
                lines_cleared += info.get('lines_cleared')

            time /= 3
            score /= 3
            lines_cleared /= 3

            f = open(results_path, "a")
            f.write(f"{i},{time},{score},{lines_cleared}\n")
            f.close()

            if args.gif > 0 and i % args.gif == 0:
                gif_file = gif_path.format(epoch=i)
                imageio.mimsave(gif_file, frames, format='gif', fps=10)

            if enable_logging:
                if done:
                    value_buffer = value_buffer.write(ts, tf.constant(0.0))
                else:
                    value_buffer = value_buffer.write(ts, critic(tf.reshape(state, shape=(1, *observation_dimensions))))

                state_buffer = state_buffer.stack()
                action_buffer = action_buffer.stack()
                reward_buffer = reward_buffer.stack()
                value_buffer = value_buffer.stack()
                logprob_buffer = logprob_buffer.stack()
                mask_buffer = mask_buffer.stack()

                log_values(actor,
                           critic,
                           state_buffer,
                           action_buffer,
                           reward_buffer,
                           value_buffer,
                           logprob_buffer,
                           mask_buffer,
                           tf.constant(num_actions),
                           tf.constant(i, dtype=tf.int64),
                           tf.constant(agent),
                           tf.constant(gamma),
                           tf.constant(lam),
                           tf.constant(clip_ratio),
                           tf.constant(entropy_beta),
                           tf.constant(critic_discount),
                           tf.constant(delta))
