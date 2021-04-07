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
Defines a simple deep Q network agent.
"""

import gin
import copy
import math
import random
import torch as t
import torch.nn as nn

import bdrl.models as models
from bdrl.agents.base_agent import Agent

@gin.configurable('DQN_agent')
class DQN(Agent):
    """Deep Q Network agent"""

    def __init__(self, env, train_batch=1024, seed=42, dtype=t.float32,
                 device='cpu', storage_device='cpu', opt=t.optim.Adam, lr=0.05,
                 train_steps_per_transition=100, eps_min=0.05, eps_max=0.9,
                 eps_decay=200, gamma=0.99, target_update=10,
                 loss=nn.SmoothL1Loss, scale_rewards=False):
        """
        Args:
            env: The initialised environment
            train_batch: The size of each training batch
            seed: The random seed
            dtype: The datatype to work with
            device: The primary device to use for computation
            storage_device: The device on which to store transitions
            opt: The stochastic optimisation algorithm
            lr: The optimiser learning rate
            train_steps_per_transition: tin.
            eps_min: The minimum epsilon value
            eps_max: The maximum (i.e. initial) epsilon value
            eps_decay: Number of timesteps for epsilon to halve
            gamma: Discount factor
            target_update: frequency with which to update the target network
            loss: The loss function
            scale_rewards: Whether to wrap the environment in RewardScaler
        """
        # Initialises environment and replay buffer.
        Agent.__init__(self, env, seed, storage_device, device, dtype,
                       train_steps_per_transition, scale_rewards)

        self.network        = models.ANN(self.obs_shape[0], self.act_shape)
        self.target_network = models.ANN(self.obs_shape[0], self.act_shape)
        self.target_network.load_state_dict(self.network.state_dict())
        # self.target_network = copy.deepcopy(self.network)
        self.opt            = opt(self.network.parameters(), lr=lr)
        self.train_batch    = train_batch
        self.eps_min        = eps_min
        self.eps_max        = eps_max
        self.eps_decay      = eps_decay
        self.gamma          = gamma
        self.target_update  = target_update
        self.loss           = loss

        self.steps_done       = 0
        self.train_steps_done = 0

    def _sync_weights(self):
        """
        Synchronise the policy network and target network weights.
        """
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _Qs(self, s, network=None):
        """
        Computes Q(s_t, a) for all possible a, returning the highest for each
        state.
        Args:
            s: The value of s_t
            network: The network to use (e.g. policy network / target network)
        Returns:
            Argmax_a Q(s_t, a)
        """
        s = s.to(device=self.device, dtype=self.dtype)
        if network is None:
            Qs = self.network(s)
        else:
            Qs = network(s)
        return Qs.max(1)[0]

    def _Qsa(self, s, a):
        """
        Computes Q(s_t, a_t).
        """
        s = s.to(device=self.device, dtype=self.dtype)
        a = a.to(device=self.device, dtype=t.int64)
        Qs = self.network(s)
        return Qs.gather(1, a)

    def train_step(self):
        """
        Perform a single train step: sync target / policy network weights if
        necessary, sample buffer of transitions, calculate TD error, update
        policy net.
        """
        self.train_steps_done += 1
        if self.train_steps_done % self.target_update == 0:
            self._sync_weights()
        s, a, sp, r, done = self.buffer.random_batch(self.train_batch)
        Qsa = self._Qsa(s, a)
        Qsp = self._Qs(sp, self.target_network).detach().unsqueeze(1)
        EQsp = r + self.gamma * Qsp * (1-done)
        assert (Qsp.shape == EQsp.shape)
        obj = self.loss(Qsa, EQsp)
        self.opt.zero_grad()
        obj.backward()
        self.opt.step()

    def get_action(self, s, training=True):
        """
        Return the next action to take.
        Args:
            s: The current state (single feature vector i.e. not a batch)
            training: Whether we are training (True) or evaluating (False) the
                      agent.
        Returns:
            A valid action to take in the environment at state s.
        """
        random_sample = random.random()
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if True or random_sample > epsilon or not training:
            # Use the policy network to decide on the best action
            action_values = self.network(s)
            action = action_values.max(0)[1]
            return action.item()
        else:
            # epsilon greedy
            return t.tensor([[self.env.action_space.sample()]],
                            device=self.device).item()

