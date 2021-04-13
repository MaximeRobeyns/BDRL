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

import sys
import math

import gin
import gym
import pyglet
import numpy as np
import torch as t

from argparse import Namespace

from gym import Env, RewardWrapper, spaces
from gym.envs.classic_control import rendering

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# TODO implement manim rendering for presentable outputs
# from manim import *
# from manim._config import *

# TODO put these methods in more approprate files
def register_envs():
    """Registers the BDRL environments defined below"""
    gym.register('ChainMDP-v0', entry_point=ChainMDP)
    gym.register('CrossChainMDP-v0', entry_point=CrossChainMDP)

class GraphImg(rendering.Geom):
    def __init__(self, x_loc, y_loc, width, height):
        rendering.Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.flip = False

    def set_img(self, newimg):
        self.img = newimg

    def render1(self):
        self.img.blit(self.x_loc, self.y_loc,
                      width=self.width, height=self.height)

class HeavyLine(rendering.FilledPolygon):
    def __init__(self, x1, y1, x2, y2, diag=2):
        if diag is not None:
            eps = diag
        else:
            eps = 1
        v = [(x1,y1+eps),(x2,y2+eps), (x2, y2-eps), (x1,y1-eps)]
        super().__init__(v)
        self.set_color(.23, .31, .36)

@gin.configurable('ChainMDP')
class ChainMDP(Env):
    """A super simple, resizeable, chain MDP"""

    # realtime uses pyglet and matplotlib
    # pretty uses manim and takes a lot longer to render
    metadata = {
        'render.modes': ['realtime', 'pretty'],
    }

    def __init__(self, length=3, num_pts_render=100):
        """
        Length is the number of steps we can take. (i.e. length=1 is a 2 state
        MRP)
        """
        assert length >= 1
        self.length = length
        self.num_pts = num_pts_render
        self.state = 0.
        self.episodes = 0
        self._max_episode_steps = length

        self.states = []
        self.viewer = None
        self.graph_img = None

        # You can only move right
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, length, shape=(1,),
                                            dtype=np.float32)

    def step(self, action):
        assert action == 0
        self.state += 1
        done = int(self.state) == self.length
        stochastic_reward = np.random.randint(0, 2) * 2 - 1 if done else 0
        return_state = np.array([self.state], dtype=np.float32)
        return return_state, stochastic_reward, done, {}

    def reset(self):
        self.state = 0.
        self.episodes = self.episodes + 1
        for s in self.states:
            s.set_color(.8, .8, .8)
        return np.array([self.state], dtype=np.float32, copy=True)

    def render(self, dist=None, mode='human', rmin=-10, rmax=10):
        screen_width = 1920
        screen_height = 1080
        padding = 100
        light = (.8, .8, .8)
        dark = (.2, .2, .2)
        state_spacing = (screen_width / 2 - padding) / (self.length+1)
        mid_x = screen_width / 2
        left_x = mid_x - (2*padding + (self.length+1) * state_spacing)
        mid_y = screen_height / 2
        dpi_res = min(screen_width, screen_height) / 10

        graph_w = (screen_width) / 2
        graph_h = (screen_height-padding)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.labels = []
            for s in range(self.length+1):
                tmp_state = rendering.make_circle(radius=50 - 5*(self.length+1), res=50)
                tmp_state.set_color(*light)
                xloc = left_x + ((s+1) * state_spacing)
                xloc_next = left_x + ((s+2) * state_spacing)
                if s+1 < self.length+1:
                    line = HeavyLine(xloc, mid_y, xloc_next, mid_y)
                    self.viewer.add_geom(line)
                tmp_state.add_attr(rendering.Transform(translation=(xloc, mid_y)))
                self.viewer.add_geom(tmp_state)
                self.states.append(tmp_state)

        if self.graph_img is None:
            self.graph_img = GraphImg(screen_width/2, screen_height/2 - graph_h/2, graph_w, graph_h)
            self.viewer.add_geom(self.graph_img)

        if dist is not None:
            # assert len(dist) == self.action_space.n
            fig = Figure((graph_w/dpi_res, graph_h/dpi_res), dpi=dpi_res)
            ax = fig.add_subplot(111)

            with t.no_grad():
                ys = t.linspace(rmin, rmax, self.num_pts)
                ss = dist.log_prob(ys).exp().detach().cpu()
                ss_mean = ss.mean(0)
                ss_std  = ss.std(0)

            for d in range(dist.batch_shape[1]):
                label = f'Action: {d}'
                ax.plot(ys.detach().cpu(), ss_mean[d], label=label)
                ax.fill_between(ys.detach().cpu(), (ss_mean[d]+ss_std[d]), (ss_mean[d]-ss_std[d]), alpha=0.2)

            ax.legend()

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buffer = canvas.renderer.buffer_rgba()
            self.graph_img.set_img(
                pyglet.image.ImageData(int(graph_w), int(graph_h), 'RGBA', buffer.tobytes(), -4 * int(graph_w))
            )

        self.states[int(self.state)].set_color(*dark)
        if (self.state > 0):
            self.states[int(self.state)-1].set_color(*light)

        # TODO extend gym viewer and overload render method to include
        # return distribution plot
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

@gin.configurable('CrossChainMDP')
class CrossChainMDP(Env):
    """A less simple, resizeable, chain MDP"""

    def __init__(self, length=1, num_pts_render=100):
        """
        Args:
            length (Int): The number of steps we can take in this MDP (min 1)
        """
        assert length >= 1

        self.length = length
        self.num_pts = num_pts_render
        self.state = np.array([0., 0.5], dtype=np.float32)
        self.episodes = 0
        self._max_episode_steps = length

        # rendering
        self.initial_state = None
        self.states = []
        self.viewer = None
        self.graph_img = None

        # You can move up (1) or down (0) at every non terminal state.
        self.action_space = spaces.Discrete(2)
        # dimension 0 gives the position along the chain
        # dimension 1 indicates whether we are in the top or bottom branch;
        # initially 0.5, then in {0,1}
        self.observation_space = spaces.Box(0, length, (2,), dtype=np.float32)

    def step(self, action):
        assert action == 0 or action == 1
        self.state = np.array([
            self.state[0]+1,
            action
        ], dtype=np.float32)
        done = int(self.state[0]) == self.length
        reward = (-2) * ((self.state[0] + self.state[1])%2) + 1
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0., 0.5], dtype=np.float32)
        self.episodes += 1

        # Reset renders
        for s in range(len(self.states)):
            self.states[s].set_color(.8, .8, .8)

        return self.state

    def render(self, dist=None, mode='human', rmin=-10, rmax=10):
        screen_width = 1920
        screen_height = 1080
        padding = 100
        light = (.8, .8, .8)
        dark = (.2, .2, .2)
        opt_line_color = (1.,.2,.2)
        state_spacing = (screen_width / 2 - padding) / (self.length+1)
        mid_x = screen_width / 2
        left_x = mid_x - (2*padding + self.length * state_spacing)
        mid_y = screen_height / 2
        bot_y = mid_y + 100
        top_y = mid_y - 100
        dpi_res = min(screen_width, screen_height) / 10

        graph_w = (screen_width) / 2
        graph_h = (screen_height-padding)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Create the initial state
            self.initial_state = rendering.make_circle(radius=50 - 5*self.length, res=50)
            self.initial_state.set_color(*light)
            xloc = left_x
            xloc_next = left_x + state_spacing
            line_up = HeavyLine(xloc, mid_y, xloc_next, top_y, diag=2)
            # line_up.add_attr(rendering.LineWidth(20))
            line_down = HeavyLine(xloc, mid_y, xloc_next, bot_y, diag=2)
            line_down.set_color(*opt_line_color)
            self.viewer.add_geom(line_up)
            self.viewer.add_geom(line_down)
            self.initial_state.add_attr(rendering.Transform(translation=(xloc,mid_y)))
            self.viewer.add_geom(self.initial_state)

            self.labels = []
            # Create each of the subsequent states
            for s in range(self.length):
                # Create the top state:
                tmp_state = rendering.make_circle(radius=50 - 5*self.length, res=50)
                tmp_state.set_color(*light)
                xloc = left_x + ((s+1) * state_spacing)
                xloc_next = left_x + ((s+2) * state_spacing)
                yloc = top_y
                if s+1 < self.length:
                    line = HeavyLine(xloc, top_y, xloc_next, top_y, diag=2)
                    self.viewer.add_geom(line)
                    nline = HeavyLine(xloc, top_y, xloc_next, bot_y, diag=2)
                    if (s)%2 == 1:
                       nline.set_color(*opt_line_color)
                    self.viewer.add_geom(nline)
                tmp_state.add_attr(rendering.Transform(translation=(xloc, yloc)))
                self.viewer.add_geom(tmp_state)
                self.states.append(tmp_state)

                # Create the bottom state:
                tmp_state = rendering.make_circle(radius=50 - 5*self.length, res=50)
                tmp_state.set_color(*light)
                xloc = left_x + ((s+1) * state_spacing)
                xloc_next = left_x + ((s+2) * state_spacing)
                yloc = bot_y
                if s+1 < self.length:
                    line = HeavyLine(xloc, bot_y, xloc_next, bot_y, diag=2)
                    self.viewer.add_geom(line)
                    nline = HeavyLine(xloc, bot_y, xloc_next, top_y, diag=2)
                    if (s)%2 == 0:
                       nline.set_color(*opt_line_color)
                    self.viewer.add_geom(nline)
                tmp_state.add_attr(rendering.Transform(translation=(xloc, yloc)))
                self.viewer.add_geom(tmp_state)
                self.states.append(tmp_state)

        if self.graph_img is None:
            self.graph_img = GraphImg(screen_width/2, screen_height/2 - graph_h/2, graph_w, graph_h)
            self.viewer.add_geom(self.graph_img)

        # If the return distribution(s) are available, plot it.
        if dist is not None:
            # assert len(dist) == self.action_space.n

            fig = Figure((graph_w/dpi_res, graph_h/dpi_res), dpi=dpi_res)
            ax = fig.add_subplot(111)

            with t.no_grad():
                ys = t.linspace(rmin, rmax, self.num_pts)
                ss = dist.log_prob(ys).exp().detach().cpu()
                ss_mean = ss.mean(0)
                ss_std  = ss.std(0)

            for d in range(dist.batch_shape[1]):
                label = f'Action: {d}'
                ax.plot(ys.detach().cpu(), ss_mean[d], label=label)
                ax.fill_between(ys.detach().cpu(), (ss_mean[d]+ss_std[d]), (ss_mean[d]-ss_std[d]), alpha=0.2)

            ax.legend()

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buffer = canvas.renderer.buffer_rgba()
            self.graph_img.set_img(
                pyglet.image.ImageData(int(graph_w), int(graph_h), 'RGBA', buffer.tobytes(), -4 * int(graph_w))
            )

        if int(self.state[0]) == 0:
            self.initial_state.set_color(*dark)
        elif int(self.state[0]) == 1:
            self.initial_state.set_color(*light)
            self.states[2*int(self.state[0]-1)+int(self.state[1])].set_color(*dark)
        else:
            self.states[2*int(self.state[0]-2)].set_color(*light)
            self.states[2*int(self.state[0]-2)+1].set_color(*light)
            self.states[2*int(self.state[0]-1)+int(self.state[1])].set_color(*dark)

        # TODO extend gym viewer and overload render method to include
        # return distribution plot
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
