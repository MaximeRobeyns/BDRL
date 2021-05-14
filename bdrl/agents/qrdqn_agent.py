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
Implementation of QR-DQN
See https://arxiv.org/abs/1710.10044
"""

import sys
import gin
import copy
import math
import random
import torch as t
import torch.nn as nn

import bdrl.models as models
from bdrl.agents.base_agent import Agent

@gin.configurable('QRDQN_agent')
class QRDQN(Agent):
    def __init__(self, env, train_batch=1024, seed=42, dtype=t.float32,
                 device='cpu', storage_device='cpu', opt=t.optim.Adam, lr=0.05,
                 train_steps_per_transition=100, eps_min=0.05, eps_max=0.9,
                 eps_decay=200, gamma=0.99, target_update=10,
                 loss=nn.SmoothL1Loss, scale_rewards=False, N=200, kappa=1):
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
            N: The number of quantiles
            kappa: Point at which the quadratic loss becomes linear in Huber loss
        """
        # Initialises environment and replay buffer.
        Agent.__init__(self, env, seed, storage_device, device, dtype,
                       train_steps_per_transition, scale_rewards)

        self.network = models.QRNET(self.obs_shape[0], self.act_shape, N=N)
        self.target_network = models.QRNET(self.obs_shape[0], self.act_shape, N=N)
        self._sync_weights()
        self._disable_grads(self.target_network)

        # TODO remove Adam specific arguments.
        self.opt            = opt(self.network.parameters(), lr=lr, eps=1e-2/train_batch)

        self.train_batch    = train_batch
        self.eps_min        = eps_min
        self.eps_max        = eps_max
        self.eps_decay      = eps_decay
        self.gamma          = gamma
        self.target_update  = target_update
        self.loss           = loss

        self.steps_done       = 0
        self.train_steps_done = 0

        # Initialise quantile fractions:
        taus = t.arange(0, N+1, device=device, dtype=dtype)/N
        self.tau_hats = ((taus[1:] + taus[:-1])/2.0).view(1, N)
        self.N        = N
        self.kappa    = kappa

    def _disable_grads(self, net):
        for p in net.parameters():
            p.requires_grad = False

    def _sync_weights(self):
        """
        Synchronise the policy network and target network weights.
        """
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _Qs(self, s, network=None):
        """
        Computes Q(s_t, a) for all possible a, returning the quantiles for the
        highest one.
        Args:
            s: The value of s_t
            network: The network to use (e.g. policy network / target network)
        Returns:
            Quantiles for Argmax_a Q(s_t, a)
        """
        batch_size = s.shape[0]
        s = s.to(device=self.device, dtype=self.dtype)
        if network is None:
            network = self.network
        Qs = network.Q(s)
        next_a = t.argmax(Qs, dim=1, keepdim=True)
        assert next_a.shape == (batch_size, 1)

        quantiles_sa = self._Qsa(s, next_a).transpose(1, 2)
        assert quantiles_sa.shape == (batch_size, 1, self.N)
        return quantiles_sa

    def _Qsa(self, s, a):
        """
        Computes the quantiles for Q(s_t, a_t).
        """
        s = s.to(device=self.device, dtype=self.dtype)
        a = a.to(device=self.device, dtype=t.int64)
        quantiles_s = self.network(s)
        assert quantiles_s.shape[0] == a.shape[0]
        assert quantiles_s.shape[1] == self.N
        batch_size = quantiles_s.shape[0]

        a_idx = a[..., None].expand(batch_size, self.N, 1)
        Qs = quantiles_s.gather(2, a_idx)
        return Qs

    def _quantile_huber_loss(self, errors):
        batch_size, N, N_hat = errors.shape
        # Huber loss
        hl = t.where(errors.abs() <= self.kappa,
                      0.5 * errors.pow(2),
                      self.kappa * (errors.abs() - 0.5 * self.kappa))
        assert hl.shape == (batch_size, N, N_hat)

        # quantile Huber loss
        qhl = t.abs(
            self.tau_hats.unsqueeze(-1) - (errors.detach() < 0).to(dtype=self.dtype)
        ) * hl
        assert qhl.shape == (batch_size, N, N_hat)

        batch_loss = qhl.sum(dim=1).mean(dim=1, keepdim=True)
        assert batch_loss.shape == (batch_size, 1)

        return qhl.mean()

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

        # Calculate the quantiles for the current state and action
        Qsa_quantiles = self._Qsa(s, a)
        assert Qsa_quantiles.shape == (self.train_batch, self.N, 1)

        with t.no_grad():
            # Calculate the quantile values for the greedy actions at sp.
            Qsp_quantiles = self._Qs(sp, self.target_network).detach()
            assert Qsp_quantiles.shape == (self.train_batch, 1, self.N)
        target_quantiles = r.unsqueeze(-1) + self.gamma * Qsp_quantiles * (1.0-done.unsqueeze(-1))
        assert target_quantiles.shape == (self.train_batch, 1, self.N)

        td_errors = target_quantiles - Qsa_quantiles
        loss = self._quantile_huber_loss(td_errors)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_action(self, s, training=True):
        """
        Return the next action to take.
        Args:
            s (Tensor): The current state (single feature vector i.e. not a batch)
            training (Bool): Whether we are training (True) or evaluating
                             (False) the agent.
        Returns:
            A valid action to take in the environment at state s.
        """
        random_sample = random.random()
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if not t.is_tensor(s):
            s = t.from_numpy(s)

        if True or random_sample > epsilon or not training:
            if len(s.shape) == 1:
                # batch size = 1
                s = s.unsqueeze(0)
            Qs = self.network.Q(s)
            return t.argmax(Qs, dim=1, keepdim=True).item()
        else:
            # epsilon greedy
            return t.tensor([[self.env.action_space.sample()]],
                            device=self.device).item()

