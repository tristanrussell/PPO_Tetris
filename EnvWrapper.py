import numpy as np

import tensorflow as tf
from typing import List, Tuple


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(int(action))

        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32),
                np.array(0, np.int32))

    def env_reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32, tf.int32])

    def tf_env_reset(self) -> tf.Tensor:
        return tf.numpy_function(self.env_reset, [], tf.float32)
