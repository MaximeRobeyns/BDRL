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
Defines a base agent class.
"""

import abc
import gym
import time
import torch as t

from bdrl.utils import buffer
from bdrl.environments import wrappers

class Agent(metaclass=abc.ABCMeta):
    """Agent is an abstract base agent class.
    Classes inheriting Agent must implement:
        - train_step
        - get_action
    """

    def __init__(self, env, seed, storage_device, device, dtype,
                 train_steps_per_transition, scale_rewards):
        """Initialise the base agent class
        Args:
            env           : The initialised environment
            seed          : The random seed
            storage_device: Device memory to use for replay buffer
            device        : Device to use for computation
            dtype         : PyTorch datatype to use
            train_steps_per_transition: Used in training loop.
            scale_rewards: Wraps environment with RewardScaler
        """
        super().__init__()

        self.seed = seed
        self.dtype = dtype
        self.storage_device = storage_device
        self.device = device
        self.train_steps_per_transition = train_steps_per_transition
        self.max_episode_steps = env._max_episode_steps

        # Setup environment
        self.env = env
        if scale_rewards:
            self.env = wrappers.RewardScaler(self.env)
        self.env.action_space.seed(seed)
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.n

        # Experience replay buffer
        self.buffer = buffer.Discrete(
            self.env.observation_space.shape,
            storage_device,
            device,
            self.dtype,
            t.int64
        )

    # # Enforce implementation of the following methods
    # @classmethod
    # def __subclasshook__(cls, subclass):
    #     return (
    #         hasattr(subclass, "train_step") and
    #         hasattr(subclass, "get_action") and
    #         callable(subclass.train_step) and
    #         callable(subclass.get_action) and
    #         NotImplemented
    #     )

    @abc.abstractmethod
    def train_step(self):
        """Run a training step"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, s, training=True):
        """Return the next action to take.
        Args:
            s: The current state (single feature vector i.e. not a batch)
            training: Whether we are training (True) or evaluating (False) the
                agent.

        Returns:
            A valid action to take in the environment at state s.
        """
        raise NotImplementedError

    def train_rollout(self, random_exploration=False, render=False):
        """Train until either done or max_episode_steps
        Args:
            random_exploration (Boolean): if True, collect initial data to fill
                replay buffer
            render (Boolean): whether to render the train rollout
        Returns:
            total_reward
        """
        s = self.env.reset()
        done = False
        t = 0
        total_reward = 0.

        while (not done) and (t < self.max_episode_steps):
            if random_exploration:
                a = self.env.action_space.sample()
            else:
                a = self.get_action(s)

            sp, r, done, _ = self.env.step(a)

            total_reward += r
            self.buffer.add_state(s, a, sp, r, done)

            if not random_exploration:
                for _ in range(self.train_steps_per_transition):
                    self.train_step()
            t += 1
            s = sp

        return total_reward

    def eval_rollout(self, render=False):
        """Evaluates a rollout
        Args:
            render (Boolean): whether to render this rollout
        """
        s = self.env.reset()
        done = False
        total_reward = 0.
        t = 0

        if render:
            self.env.render()
            # time.sleep(1)

        while not done:
            t += 1
            a = self.get_action(s)
            s, r, done, _ = self.env.step(a)
            if render:
                self.env.render()
                # time.sleep(1)
            total_reward += r
        return total_reward

class DistributionalAgent(Agent):

    @abc.abstractmethod
    def _Qdist(self, s, a, samples=100, sample_dict=None):
        """Return the value distribution.
        Args:
            s: A batch of states
            a: A batch of actions
            samples: The number of parameter samples to return
            sample_dict: Return the sampled weights
        Returns:
            Theta (f, alpha, beta) triple, the logpq term, and sample_dict.
        """
        raise NotImplementedError

    def eval_rollout(self, render=False):
        s = self.env.reset()
        done = False
        total_reward = 0.
        t = 0

        if render:
            a = self.get_action(s)
            theta, _, _ = self._Qdist(s, a)
            self.env.render_dist(theta)
            time.sleep(1)

        while not done:
            t += 1
            a = self.get_action(s)
            s, r, done, _ = self.env.step(a)
            total_reward += r

            # placing render here allows us to see the final state.
            if render:
                theta, _, _ = self._Qdist(s, a)
                self.env.render_dist(theta)
                time.sleep(1)

        return total_reward
