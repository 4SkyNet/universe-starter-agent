from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import numpy as np
from collections import deque

log = logging.getLogger(__name__)


class ColorEnv(object):
    max_steps = 10

    def __init__(self, env='color', shape=(2, 2), action_size=10, history=4):
        self.env = env
        self.action_size = action_size
        self.timestep_limit = ColorEnv.max_steps

        assert len(shape) == 2, "You should provide shape as (x,y)"
        self._history = deque(maxlen=history)
        self.shape = shape

        self._scale = 1. / action_size
        self._episode_reward = 0
        self._step_count = 0
        self.reset()

    def step(self, action):
        reward, terminal = self._step(action)
        state = self._process_state()
        self._episode_reward += reward

        if terminal:
            info = {'episode_reward': self._episode_reward}
        else:
            info = {}
        return state, reward, terminal, info

    def reset(self):
        self._history.clear()
        state = self._process_state()

        self._step_count = 0
        self._episode_reward = 0
        return state

    def _step(self, action):
        avg_color = sum(self._history) / float(len(self._history))
        reward = 1.0 - abs(action - avg_color) * self._scale

        self._step_count += 1
        self._episode_reward += reward
        return reward, self._step_count >= self.timestep_limit

    def _process_state(self):
        color_num = np.random.randint(0, self.action_size)
        self._history.append(color_num)

        state = np.empty(self.shape, dtype=np.float32)
        state.fill(color_num * self._scale)

        return state
