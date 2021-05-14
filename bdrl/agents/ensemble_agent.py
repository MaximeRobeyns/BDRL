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
A simple 'ensemble' agent; uses the 'Ensemble' network inspired by:

https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf

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
from bdrl.models import Ensemble, BDR_dist

@gin.configurable('Ensemble_agent')
class EnsembleAgent(DistributionalAgent):
    """'Ensemble' agent"""

    def __init__(self, env, train_batch=32, seed=42, dtype=t.float32,
                 device='cpu', storage_device='cpu',
                 train_steps_per_transition=100, gamma=0.99, N_train=10,
                 scale_rewards=True):
        """
        Args:
            env: The initialised environment
            train_batch: Batch size sampled from replay buffer (per network in
                the ensemble)
            seed: The random seed
            dtype: The datatype to work with
            device: The primary device to use for computation
            storage_device: The device on which to store transitions
            train_steps_per_transition: what it says on the tin...
            gamma: The discount factor
            N_train: The number of samples to draw from the target distribution
            scale_rewards: Whether to re-scale rewards
        """
        # Initialises environment and replay buffer.
        super().__init__(env, seed, storage_device, device, dtype,
                       train_steps_per_transition, scale_rewards)

        self.actions = self.env.action_space.n
        self.train_batch = train_batch
        self.gamma = gamma
        self.N_train = N_train

        # Initialise network (args specified in config file)
        in_features = self.obs_shape[0] + self.act_shape
        self.net = Ensemble(in_features)

    def _Qdist(self, s, a):
        """Approximate the parameters of the return distribution at a state,
        action pair"""
        if not t.is_tensor(s):
            s = t.tensor(s, dtype=self.dtype).to(device=self.device).unsqueeze(0)
        if not t.is_tensor(a):
            a = t.tensor(a, dtype=self.dtype).to(device=self.device).reshape(s.shape)
        batch_sa = t.cat((s, one_hot(a, self.act_shape)), 1)
        assert batch_sa.shape[-1] == self.obs_shape[0] + self.act_shape
        return self.net.f(batch_sa)

    def get_return_dists(self, s):
        """
        Safe to assume that s is a single state (numpy array)
        """
        s = t.tensor(s, dtype=self.dtype, device=self.device).unsqueeze(0)
        s = s.repeat_interleave(self.act_shape, 0)
        a_s = t.arange(self.act_shape, dtype=self.dtype, device=self.device).unsqueeze(1)
        batch_sa = t.cat((s, one_hot(a_s, self.act_shape)), 1)
        theta = self.net.f(batch_sa)
        return BDR_dist(*theta, fp=True)

    def get_action(self, s):
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

            # get all possible state action pairs
            s_a = s.repeat_interleave(self.act_shape, 0)
            a_s = t.arange(self.act_shape, dtype=self.dtype)
            a_s = a_s.repeat(s.shape[0]).unsqueeze(1)
            theta = self._Qdist(s_a, a_s)
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
        # Trainig Step for distributional agent

        # Sample some transitions from the buffer
        s, a, sp, r, done = self.buffer.random_batch(self.train_batch)

        # Find the optimal next actions, and the rreturn distributions for
        # (sp, ap)
        with t.no_grad():
            s = s.to(dtype=self.dtype, device=self.device)

            # Get all the possible state-action pairs
            ap_h = t.arange(self.act_shape, dtype=self.dtype, device=self.device)
            ap_h = ap_h.repeat(sp.shape[0]).unsqueeze(1)
            sp = sp.repeat_interleave(self.act_shape, 0)
            batch_spap_h = t.cat((sp, one_hot(ap_h, self.act_shape)), 1)
            assert batch_spap_h.shape[-1] == self.obs_shape[0] + self.act_shape
            theta = self.net.f(batch_spap_h)
            sp_dist = BDR_dist(*theta, fp=True)

            # Select the action with the max expected value
            means = sp_dist.mean().reshape(-1, self.act_shape)
            ap = means.argmax(-1, keepdims=True)

            # Locate the parameters of the corresponding (sp,ap) distributions
            # so that we can sample from these
            f_tmp, a_tmp, b_tmp = theta
            idxs = (t.arange(ap.size(0))*self.act_shape)+ap.flatten()
            theta_spap = (f_tmp[:,idxs].squeeze(),
                          a_tmp[:,idxs].squeeze(),
                          b_tmp[:,idxs].squeeze())

            # Sanity check
            assert s.shape == (self.train_batch, self.obs_shape[0])
            assert a.shape == (self.train_batch, 1)
            assert a.shape == ap.shape

            # Generate targets
            spap_dist = BDR_dist(*theta_spap, fp=True)
            samples = spap_dist.sample(self.N_train, avg=True)
            targets = r + (1-done) * self.gamma * samples

        # Evaluate the likelihood of the target points for each of the (s,a)
        # distributions in the original batch.
        batch_sa = t.cat((s, one_hot(a, self.act_shape)), 1)
        self.net.train(batch_sa, targets, epochs=2, batch_size=32)
