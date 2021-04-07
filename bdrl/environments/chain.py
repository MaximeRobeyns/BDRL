# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Defines some simple additional environments.
"""

import gin
import gym
import pyglet
import numpy as np

from gym import Env, RewardWrapper, spaces
from gym.envs.classic_control import rendering

from argparse import Namespace

# TODO implement manim rendering for presentable outputs
# from manim import *
# from manim._config import *

# TODO put these methods in more approprate files
#
def register_envs():
    """Registers the BDRL environments defined below"""
    gym.register('ChainMDP-v0', entry_point=ChainMDP)

class ChainMDP(Env):
    """A super simple, resizeable, chain MDP"""

    # realtime uses pyglet and matplotlib
    # pretty uses manim and takes a lot longer to render
    metadata = {
        'render.modes': ['realtime', 'pretty'],
    }

    def __init__(self, length=3):
        """
        Length is the number of steps we can take. (i.e. length=1 is a 2 state
        MRP)
        """
        self.length = length
        self.state = 0.
        self.episodes = 0
        self._max_episode_steps = length
        self.states = []
        self.viewer = None
        # You can only move right
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, length, shape=(1,), dtype=np.float32)

    def step(self, action):
        self.state = self.state + 1
        done = self.state == self.length
        stochastic_reward = np.random.randint(0, 2) * 2 - 1 if done else 0
        return np.array([self.state]), stochastic_reward, done, {}

    def reset(self):
        self.state = 0.
        self.episodes = self.episodes + 1
        for s in range(len(self.states)):
            self.states[s].set_color(.8, .8, .8)
        return np.array([self.state])

    def render(self, mode='human'):
        screen_width = 1920
        screen_height = 1080
        padding = 100
        light = (.8, .8, .8)
        dark = (.2, .2, .2)
        state_spacing = (screen_width / 2 - padding) / (self.length+1)
        mid_x = screen_width / 2
        mid_y = screen_height / 2

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.labels = []
            for s in range(self.length + 1):
                tmp_state = rendering.make_circle(radius=40, res=50)
                tmp_state.set_color(*light)
                xloc = mid_x - (padding + s * state_spacing)
                xloc_next = mid_x - (padding + (s+1) * state_spacing)
                if s < self.length:
                    line = rendering.Line((xloc,mid_y), (xloc_next, mid_y))
                    self.viewer.add_geom(line)
                tmp_state.add_attr(rendering.Transform(translation=(xloc, mid_y)))
                self.viewer.add_geom(tmp_state)
                self.states.append(tmp_state)
            self.states.reverse()

        if int(self.state) == 0:
            self.states[0].set_color(*dark)
        else:
            self.states[int(self.state)-1].set_color(*light)
            self.states[int(self.state)].set_color(*dark)

        if self.state is None:
            return None

        # TODO extend gym viewer and overload render method to include
        # return distribution plot
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
