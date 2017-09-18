from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from builtins import range
from builtins import object

import sys
import logging
import numpy as np
from PIL import Image

from relaax.environment.config import options
from relaax.common.rlx_message import RLXMessageImage

log = logging.getLogger(__name__)


class MazeEnv(object):
    moves = [np.array([-1, 0]),  # go up
             np.array([0, -1]),  # go left
             np.array([1, 0]),   # go down
             np.array([0, 1])]   # go right
    actions = ['^', '<', 'v', '>']
    step_cost = .01
    goal_cost = 1.0
    max_steps = 100

    def __init__(self, env='level_1'):
        self._level = self._read_level(env)
        self._no_op_max = options.get('environment/no_op_max', 0)

        self.shape = options.get('environment/shape', [7, 7])
        assert len(self.shape) == 2, "You should provide shape as [x, y]"
        self.range = [int((self.shape[0]-1)/2), int((self.shape[1]-1)/2)]

        self.action_size = len(MazeEnv.actions)
        self.timestep_limit = MazeEnv.max_steps

        self._maze, self._step_count = None, None
        self._goal_pos, self._player_pos = None, None
        self.reset()

    def step(self, action):
        reward, terminal = self._step(action)
        state = self._process_state()

        return state, reward, terminal, {}

    def reset(self):
        while True:
            self._init_maze()
            terminal = False

            if self._no_op_max != 0:
                no_op = np.random.randint(0, self._no_op_max)
                for _ in range(no_op):
                    reward, terminal = self._step(np.random.randint(0, self.action_size))

            if not terminal:
                state = self._process_state()
                break

        self._step_count = 0
        return state

    def _step(self, action):
        new_pos = self._player_pos + MazeEnv.moves[action]

        if new_pos[0] == self._goal_pos[0] and new_pos[1] == self._goal_pos[1]:
            return MazeEnv.goal_cost, True
        else:
            self._step_count += 1
            if self._maze[new_pos[0], new_pos[1], 0] != 1:
                self._maze[self._player_pos[0], self._player_pos[1], 2] = 0
                self._player_pos = new_pos
                self._maze[self._player_pos[0], self._player_pos[1], 2] = 1
            return -MazeEnv.step_cost, self._step_count >= MazeEnv.max_steps

    def _process_state(self):
        region = self._maze[self._player_pos[0]-self.range[0]:self._player_pos[0]+self.range[0],
                            self._player_pos[1]-self.range[1]:self._player_pos[1]+self.range[1],
                            ...]
        state = np.copy(region) * 255
        state = RLXMessageImage(Image.fromarray(state.astype(np.uint8)))
        return state

    @staticmethod
    def _read_level(level_name):
        lvl_read = []
        with open('maps/' + level_name + '.txt', 'r') as lvl_file:
            for i, line in enumerate(lvl_file):
                lvl_read.append([])
                for j in line[:-1]:
                    if j in ('0', '1'):
                        lvl_read[i].append(int(j))
                    else:
                        log.error("You map file should be defined with '0' and '1'!")
                        sys.exit(-1)
        return np.asarray(lvl_read)

    def _init_maze(self):
        # init goal position
        goal = np.zeros_like(self._level)
        while True:
            row_idx = np.random.randint(0, self._level.shape[0])
            col_idx = np.random.randint(0, self._level.shape[1])
            if self._level[row_idx, col_idx] == 0:
                goal[row_idx, col_idx] = 1
                self._goal_pos = np.array([row_idx, col_idx])
                break

        # init player position
        player = np.zeros_like(self._level)
        while True:
            row_idx = np.random.randint(0, self._level.shape[0])
            col_idx = np.random.randint(0, self._level.shape[1])
            if self._level[row_idx, col_idx] == 0 and goal[row_idx, col_idx] == 0:
                player[row_idx, col_idx] = 1
                self._player_pos = np.array([row_idx, col_idx])
                break

        # stack all together in depth (along third axis)
        self._maze = np.dstack((self._level, goal, player))
