import numpy as np

import tensorflow as tf
from typing import Tuple

eps = np.finfo(np.float32).eps.item()

def gaes(rewards: tf.Tensor,
         masks: tf.Tensor,
         values: tf.Tensor,
         next_values: tf.Tensor,
         gamma: float,
         lamda: float,
         normalize: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    n = tf.shape(rewards)[0]
    advantages = tf.TensorArray(dtype=tf.float32, size=n + 1, clear_after_read=False)
    advantages = advantages.write(n, tf.constant(0.0))

    rewards = tf.cast(rewards, dtype=tf.float32)
    values = tf.cast(values, dtype=tf.float32)
    next_values = tf.cast(next_values, dtype=tf.float32)
    masks = tf.cast(masks, dtype=tf.float32)

    for t in tf.range(n - 1, -1, -1):
        reward = rewards[t]
        mask = masks[t]
        value = values[t]
        next_value = next_values[t]
        next_adv = advantages.read(t + 1)

        delta = reward + (gamma * next_value * mask) - value
        advantages = advantages.write(t, delta + (gamma * lamda * next_adv * mask))

    advantages = advantages.stack()[:n]
    returns = advantages + values

    if normalize:
        advantage_mean, advantage_std = (
            tf.math.reduce_mean(advantages),
            tf.math.reduce_std(advantages),
        )

        advantages = (advantages - advantage_mean) / (advantage_std + eps)

    return advantages, returns
