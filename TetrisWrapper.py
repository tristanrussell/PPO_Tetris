import numpy as np

from typing import Tuple

from EnvWrapper import EnvWrapper


class TetrisWrapper(EnvWrapper):

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, info = self.env.step(int(action))

        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32),
                np.array(info.get('lines_cleared'), np.int32))
