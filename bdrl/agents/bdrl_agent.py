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
Defines the Bayesian distributional RL agent.
"""
import sys

import gym
import gin.torch
import bayesfunc as bf

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from bdrl.utils import one_hot
from bdrl.agents.base_agent import DistributionalAgent
from bdrl.models import BDR, BDR_dist

@gin.configurable('BDRL_agent')
class BDRL(DistributionalAgent):
    """Bayesian Distributional Reinforcement Learning agent"""

    def __init__(self, env, train_batch=32, seed=42, dtype=t.float32,
                 device='cpu', storage_device='cpu', opt=t.optim.Adam, lr=0.05,
                 train_steps_per_transition=100, gamma=0.99, S_eval=20,
                 S_train=10, N_train=10, S_explore=1, scale_rewards=True):
        """
        Args:
            env: The initialised environment
            train_batch: Batch size sampled from replay buffer
            seed: The random seed
            dtype: The datatype to work with
            device: The primary device to use for computation
            storage_device: The device on which to store transitions
            opt: The stochastic optimisation algorithm
            lr: Learning rate for above optimisation algorithm
            train_steps_per_transition: what it says on the tin...
            gamma: The discount factor
            S_eval: Number of posterior weight samples for evaluation
            S_train: Number of posterior weight samples for training
            N_train: The number of samples to draw from the target distribution
            S_explore: Number of posterior weight samples for exploration
            scale_rewards: Whether to re-scale rewards
        """
        # Initialises environment and replay buffer.
        super().__init__(env, seed, storage_device, device, dtype,
                       train_steps_per_transition, scale_rewards)

        self.lr = lr
        self.actions = self.env.action_space.n
        self.train_batch = train_batch
        self.gamma = gamma
        self.S_eval = S_eval
        self.S_train = S_train
        self.N_train = N_train
        self.S_explore = S_explore

        # Initialise network (args specified in config file)
        in_features = self.obs_shape[0] + self.act_shape
        self.net    = BDR(in_features)
        self.opt    = opt(self.net.parameters(), lr=lr)

    def _Qdist(self, s, a, samples=100, sample_dict=None):
        """Approximate the parameters of the return distribution at a state,
        action pair"""
        if not t.is_tensor(s):
            s = t.tensor(s, dtype=self.dtype).to(device=self.device).unsqueeze(0)
        if not t.is_tensor(a):
            a = t.tensor(a, dtype=self.dtype).to(device=self.device).reshape(s.shape)
        batch_sa = t.cat((s, one_hot(a, self.act_shape)), 1)
        batch_sa = batch_sa.expand(samples, -1, -1)
        assert batch_sa.shape[-1] == self.obs_shape[0] + self.act_shape
        return bf.propagate(self.net, batch_sa, sample_dict=sample_dict)

    def get_return_dists(self, s):
        """
        Safe to assume that s is a single state (numpy array)
        """
        s = t.tensor(s, dtype=self.dtype, device=self.device).unsqueeze(0)
        s = s.repeat_interleave(self.act_shape, 0)
        a_s = t.arange(self.act_shape, dtype=self.dtype, device=self.device).unsqueeze(1)
        batch_sa = t.cat((s, one_hot(a_s, self.act_shape)), 1)
        theta, _, _ = self._Qdist(s, a_s, self.S_train)
        return BDR_dist(*theta, fp=True)

    def get_action(self, s, samples=None):
        with t.no_grad():
            # convert s to tensor if not already
            if not t.is_tensor(s):
                s = t.tensor(s)
            s = s.to(dtype=self.dtype)
            assert s.shape[-1] == self.obs_shape[0]
            assert len(s.shape) <= 2
            item = False
            if len(s.shape) == 1:
                s = s.unsqueeze(0)
                item = True

            # cannot pass as default args because requries self
            if samples is None:
                samples = self.S_train

            # get all possible state action pairs
            s_a = s.repeat_interleave(self.act_shape, 0)
            a_s = t.arange(self.act_shape, dtype=self.dtype)
            a_s = a_s.repeat(s.shape[0]).unsqueeze(1)
            theta, _, _ = self._Qdist(s_a, a_s, samples)
            s_dist = BDR_dist(*theta, fp=True)

            # Select action with max expected value:
            means = s_dist.mean().reshape(-1, self.act_shape).argmax(-1, keepdim=True)
            if item:
                return means.item()
            return means

            # Or select action with max mode:
            # lst = []
            # for b in range(s.shape[0] * self.act_shape):
            #     # TODO fix modes and make more robust!
            #     tmp_mds = s_dist.mode_at(b, avg=True)
            #     if len(tmp_mds):
            #         lst.append(tmp_mds.max())
            #     else:
            #         lst.append(self.env.action_space.sample())
            # modes = t.tensor(lst).reshape(-1, self.act_shape).argmax(-1, keepdim=True)
            # if item:
            #     return modes.item()
            # return modes

            # TODO try using uncertainty estimates, UCB, Thompson etc..

    def train_step(self):
        # Training step for BDRL agent

        # Sample transitions from buffer
        s, a, sp, r, done = self.buffer.random_batch(self.train_batch)

        # Find the optimal next action:
        with t.no_grad():
            s = s.to(dtype=self.dtype, device=self.device)

            # Get all possible subsequent state-action pairs
            ap_h = t.arange(self.act_shape, dtype=self.dtype, device=self.device)
            ap_h = ap_h.repeat(sp.shape[0]).unsqueeze(1)
            sp   = sp.repeat_interleave(self.act_shape, 0)
            batch_spap_h = t.cat((sp, one_hot(ap_h, self.act_shape)), 1)
            batch_spap_h = batch_spap_h.expand(self.S_train, -1, -1)

            assert batch_spap_h.shape[-1] == self.obs_shape[0] + self.act_shape
            theta, _, _ = bf.propagate(self.net, batch_spap_h)
            sp_dist = BDR_dist(*theta, fp=True)

            # Selection the action with the max expected value
            # TODO try other exploration policies
            means = sp_dist.mean().reshape(-1, self.act_shape)
            ap = means.argmax(-1, keepdims=True)

            # Locate the parameters of the corresponding (sp, ap)
            # distribution(s) so that we can sample from these.
            f_tmp, a_tmp, b_tmp = theta
            idxs = (t.arange(ap.size(0))*self.act_shape)+ap.flatten()
            theta_spap = (f_tmp[:,idxs].squeeze(),
                          a_tmp[:,idxs].squeeze(),
                          b_tmp[:,idxs].squeeze())

            # Generate targets
            spap_dist = BDR_dist(*theta_spap, fp=True)
            samples = spap_dist.sample(self.N_train, avg=True)
            targets = r + (1-done) * self.gamma * samples

        batch_sa = t.cat((s, one_hot(a, self.act_shape)), 1)
        batch_sa = batch_sa.expand(self.S_train, -1, -1)
        theta_sa, logpq, _ = bf.propagate(self.net, batch_sa)
        ll = BDR_dist(*theta_sa).log_prob(targets).sum(-1).mean(-1)

        elbo = ll + logpq/(self.buffer.filled_buffer)
        self.opt.zero_grad()
        (-elbo.mean()).backward()
        self.opt.step()

