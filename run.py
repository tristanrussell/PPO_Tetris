import random

import os
import imageio
import collections
import gym
import gym_simpletetris
import numpy as np
import statistics
import tqdm
import argparse
import configparser
import json

from EnvWrapper import EnvWrapper
from TetrisWrapper import TetrisWrapper
from PPO import PPO
from TRGPPO import TRGPPO

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

eps = np.finfo(np.float32).eps.item()


def run(name, load=0, gif=False, reset=False):
    export_prefix = name

    checkpoint_path = "training/" + export_prefix + "/weights/{model}-{epoch:09d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    gif_path = "training/" + export_prefix + "/gifs/{epoch:09d}.gif"
    gif_dir = os.path.dirname(gif_path)
    data_path = "training/" + export_prefix + "/data.csv"
    data_dir = os.path.dirname(data_path)
    config_path = "training/" + export_prefix + "/config.ini"
    config_dir = os.path.dirname(config_path)

    config = configparser.ConfigParser()
    config.read(config_path)

    checkpoint = int(config['DEFAULT'].get('checkpoint', '0'))
    environment_config = config['environment']
    agent_config = config['agent']
    criteria_config = config['criteria']

    # environment
    env_name = environment_config.get('env_name')
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
    render = environment_config.getboolean('render')

    # agent
    agent = agent_config.get('agent')
    steps_per_episode = agent_config.getint('steps_per_episode')
    batch_size = int(agent_config.get('batch_size', '0'))
    gamma = agent_config.getfloat('gamma')
    lam = agent_config.getfloat('lambda')
    clip_ratio = agent_config.getfloat('clip_ratio')
    entropy_beta = agent_config.getfloat('entropy_beta')
    critic_discount = agent_config.getfloat('critic_discount')
    delta = agent_config.getfloat('delta')
    policy_learning_rate = agent_config.getfloat('policy_learning_rate')
    value_function_learning_rate = agent_config.getfloat('value_function_learning_rate')
    train_policy_iterations = agent_config.getint('train_policy_iterations')
    train_value_iterations = agent_config.getint('train_value_iterations')
    target_kl = agent_config.getfloat('target_kl')
    hidden_sizes = json.loads(agent_config.get('hidden_sizes'))
    hidden_sizes_activation = agent_config.get('hidden_sizes_activation', 'relu')
    conv_layers = json.loads(agent_config.get('conv_layers'))
    conv_layers_activation = agent_config.get('conv_layers_activation', 'relu')
    flatten = agent_config.getboolean('flatten')

    # criteria
    max_episodes = criteria_config.getint('max_episodes')
    min_episodes_criterion = criteria_config.getint('min_episodes_criterion')
    reward_threshold = criteria_config.getint('reward_threshold')
    checkpoint_frequency = criteria_config.getint('checkpoint_frequency')
    gif_frequency = criteria_config.getint('gif_frequency')

    def run_episodes_and_create_video(actor, env, wrapped_env, max_steps, epoch):
        if gif:
            num_episodes = 3
            frames = []
            for _ in range(num_episodes):
                state = wrapped_env.tf_env_reset()
                frames.append(env.render(mode="rgb_array"))
                for _ in range(max_steps):
                    observation = tf.reshape(state, shape=(1, *observation_dimensions))
                    logits = actor(observation)
                    action = tf.random.categorical(logits, 1, dtype=tf.int32)[0, 0]

                    state, _, done, _ = wrapped_env.tf_env_step(action)
                    frames.append(env.render(mode="rgb_array"))

                    if done:
                        break
            gif_file = gif_path.format(epoch=epoch)
            imageio.mimsave(gif_file, frames, format='gif', fps=25)

    def run_episodes_and_create_video_random(env, wrapped_env, num_actions, max_steps, epoch):
        if gif:
            num_episodes = 3
            frames = []
            for _ in range(num_episodes):
                wrapped_env.tf_env_reset()
                frames.append(env.render(mode="rgb_array"))
                for _ in range(max_steps):
                    action = tf.constant(random.randint(0, num_actions - 1))

                    state, _, done, _ = wrapped_env.tf_env_step(action)
                    frames.append(env.render(mode="rgb_array"))

                    if done:
                        break
            gif_file = gif_path.format(epoch=epoch)
            imageio.mimsave(gif_file, frames, format='gif', fps=25)

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

    # Initialize the actor and the critic as keras models
    def build_net(observation_dimensions,
                  hidden_sizes,
                  conv_layers,
                  flatten,
                  num_actions,
                  hidden_sizes_activation='relu',
                  conv_layers_activation='relu'):
        observation_input = Input(shape=observation_dimensions, dtype=tf.float32)
        logits = build_conv(observation_input, layers=conv_layers, flatten=flatten, activation=conv_layers_activation)
        logits = mlp(logits, hidden_sizes + [num_actions], hidden_sizes_activation, None)
        actor = Model(inputs=observation_input, outputs=logits)
        value = build_conv(observation_input, layers=conv_layers, flatten=flatten, activation=conv_layers_activation)
        value = tf.squeeze(
            mlp(value, hidden_sizes + [1], hidden_sizes_activation, None)
        )
        critic = Model(inputs=observation_input, outputs=value)

        return actor, critic

    if env_name == 'breakout':
        env = gym.make("ALE/Breakout-v5", full_action_space=False)
        env = gym.wrappers.AtariPreprocessing(env, grayscale_newaxis=True, scale_obs=True, frame_skip=1)
    elif env_name == 'tetris':
        env = gym.make("ALE/Tetris-v5", full_action_space=False)
        env = gym.wrappers.AtariPreprocessing(env, grayscale_newaxis=True, scale_obs=True, frame_skip=1)
    elif env_name == 'simpletetris':
        if observation is not None:
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
        elif len(conv_layers) == 0:
            env = gym.make("SimpleTetris-v0",
                           reward_step=reward_step,
                           penalise_height=penalise_height,
                           penalise_height_increase=penalise_height_increase,
                           advanced_clears=advanced_clears,
                           high_scoring=high_scoring,
                           penalise_holes=penalise_holes,
                           penalise_holes_increase=penalise_holes_increase,
                           lock_delay=lock_delay,
                           step_reset=step_reset)
        else:
            env = gym.make("SimpleTetris-v0",
                           obs_type='grayscale',
                           extend_dims=True,
                           reward_step=reward_step,
                           penalise_height=penalise_height,
                           penalise_height_increase=penalise_height_increase,
                           advanced_clears=advanced_clears,
                           high_scoring=high_scoring,
                           penalise_holes=penalise_holes,
                           penalise_holes_increase=penalise_holes_increase,
                           lock_delay=lock_delay,
                           step_reset=step_reset)
    else:
        print("Unknown environment.\n")
        exit(1)

    if env_name == 'simpletetris':
        wrapped_env = TetrisWrapper(env)
    else:
        wrapped_env = EnvWrapper(env)

    # Initialize the environment and get the dimensionality of the
    # observation space and the number of possible actions
    observation_dimensions = env.observation_space.shape
    num_actions = env.action_space.n

    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    actor, critic = build_net(observation_dimensions,
                              hidden_sizes,
                              conv_layers,
                              flatten,
                              num_actions,
                              hidden_sizes_activation,
                              conv_layers_activation)

    current_epoch = 0
    if load > 0:
        current_epoch = load
        actor.load_weights(checkpoint_path.format(model='actor', epoch=current_epoch))
        critic.load_weights(checkpoint_path.format(model='critic', epoch=current_epoch))
        current_epoch += 1
    elif not reset:
        current_epoch = checkpoint
        if current_epoch != 0:
            actor.load_weights(checkpoint_path.format(model='actor', epoch=current_epoch))
            critic.load_weights(checkpoint_path.format(model='critic', epoch=current_epoch))
            current_epoch += 1

    if current_epoch == 0:
        run_episodes_and_create_video_random(env, wrapped_env, num_actions, steps_per_episode, 0)

    policy_optimizer = Adam(learning_rate=policy_learning_rate)
    value_optimizer = Adam(learning_rate=value_function_learning_rate)

    if agent == 'PPO':
        agent = PPO(num_actions,
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
                    batch_size)
    elif agent == 'TRGPPO':
        agent = TRGPPO(num_actions,
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
                       delta,
                       batch_size)
    else:
        print("Invalid agent name.")
        exit(1)

    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
    running_reward = 0

    with tqdm.trange(current_epoch, max_episodes + 1) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            sum_return, sum_length, sum_info, num_episodes = agent.run(wrapped_env, initial_state, actor, critic)
            sum_return, sum_length, sum_info, num_episodes = \
                int(sum_return), int(sum_length), int(sum_info), int(num_episodes)

            episode_reward = sum_return / num_episodes
            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            episode_info = sum_info / num_episodes

            t.set_description(f'Episode {i}')
            if env_name == 'simpletetris':
                t.set_postfix(clears=episode_info, ep_rew=episode_reward, run_rew=running_reward)
            else:
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            if i % checkpoint_frequency == 0:
                actor.save_weights(checkpoint_path.format(model='actor', epoch=i))
                critic.save_weights(checkpoint_path.format(model='critic', epoch=i))
                f = open(data_path, "a")
                if env_name == 'simpletetris':
                    f.write(f"{i},{episode_reward},{episode_info}\n")
                else:
                    f.write(f"{i},{episode_reward}\n")
                f.close()
                config['DEFAULT']['checkpoint'] = str(i)
                with open(config_path, 'w') as configfile:
                    config.write(configfile)

            if i > 0 and i % gif_frequency == 0:
                run_episodes_and_create_video(actor, env, wrapped_env, steps_per_episode, i)

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                run_episodes_and_create_video(actor, env, wrapped_env, steps_per_episode, i)
                actor.save_weights(checkpoint_path.format(model='actor', epoch=i))
                critic.save_weights(checkpoint_path.format(model='critic', epoch=i))
                f = open(data_path, "a")
                if env_name == 'simpletetris':
                    f.write(f"{i},{episode_reward},{episode_info}\n")
                else:
                    f.write(f"{i},{episode_reward}\n")
                f.close()
                config['DEFAULT']['checkpoint'] = str(i)
                with open(config_path, 'w') as configfile:
                    config.write(configfile)
                break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True,
                        help="The name of the test to run.")

    parser.add_argument('-l', '--load', type=int, default=0,
                        help="The checkpoint to load (default is the latest checkpoint).")

    parser.add_argument('-g', '--gif', action='store_true',
                        help="Enable gifs.")

    parser.add_argument('-r', '--reset', action='store_true',
                        help="Restart training.")

    args = parser.parse_args()

    run(args.name, args.load, args.gif, args.reset)
