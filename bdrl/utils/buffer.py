# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
# Based on:
#
#   https://github.com/LaurenceA/rl_framework/blob/main/buffer.py
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
Defines a simple data buffer to store transitions.
"""

import abc
import torch as t
import numpy as np

class Buffer(metaclass=abc.ABCMeta):
    """Buffer interface, exposing the public methods add_state, random_batch
    and contiguous_batch.
    """

    def __init__(self, state_shape, storage_device='cpu', output_device='cpu',
                 output_dtype=t.float32, action_dtype=t.int64, init_buffer=32,
                 reward_range=None):
        """
        Args:
            state_shape: The shape of the state vector
            storage_device: The device memory in which to store transitions
            output_device: Device to copy memory to before returning samples
            output_dtype: The datatype to return values as
            action_dtype: The action datatype
            init_buffer: The initial size to make the buffer.
        """
        super().__init__()
        self.storage_device = storage_device
        self.output_device  = output_device
        self.output_dtype   = output_dtype
        self.action_dtype   = action_dtype
        self.filled_buffer  = 0
        self.ratio          = 4
        self.buffer_size    = init_buffer
        self.r  = t.zeros((init_buffer, 1),            device=storage_device)
        self.s  = t.zeros((init_buffer, *state_shape), device=storage_device)
        self.sp = t.zeros((init_buffer, *state_shape), device=storage_device)
        self.d  = t.zeros((init_buffer, 1),            device=storage_device)

    def _expand_buffer(self, old_buffer):
        assert old_buffer.shape[0] == self.filled_buffer
        new_buffer = t.zeros((self.buffer_size, *old_buffer.shape[1:]),
                             dtype  = old_buffer.dtype,
                             device = old_buffer.device)
        new_buffer[:self.filled_buffer] = old_buffer
        return new_buffer

    def _expand_all(self):
        """Expands all buffers (s, a, sp, r, d) by `ratio`"""
        self.buffer_size = self.ratio * self.buffer_size
        self.s  = self._expand_buffer(self.s)
        self.a  = self._expand_buffer(self.a)
        self.sp = self._expand_buffer(self.sp)
        self.r  = self._expand_buffer(self.r)
        self.d  = self._expand_buffer(self.d)

    def _random_idxs(self, T):
        """
        Return at most T elements selected uniformly at random from the buffer
        """
        return t.randperm(
            self.filled_buffer,
            device=self.storage_device
        )[:min(T, self.filled_buffer)]

    def _contiguous_idxs(self, T):
        """
        Returns T continuous elements, starting at a random index
        """
        if self.filled_buffer < T:
            return range(self.filled_buffer)
        else:
            start = np.random.randint(self.filled_buffer)
            if start + T < self.filled_buffer:
                return range(start, start+T)
            else:
                return list(
                    range(start, self.filled_buffer)
                ) + list(
                    range(T-(self.filled_buffer-start))
                )

    def _batch(self, idxs):
        """
        Return a batch from the given indexes.
        """
        kwargs = {'device': self.output_device, 'dtype': self.output_dtype}
        akwargs = {'device': self.output_device, 'dtype': self.action_dtype}
        return (self.s [idxs].to(**kwargs ),
                self.a [idxs].to(**akwargs),
                self.sp[idxs].to(**kwargs ),
                self.r [idxs].to(**kwargs ),
                self.d [idxs].to(**kwargs ))

    def add_state(self, s, a, sp, r, d):
        """Add a transition to the buffer
        Args:
            s : The current state (numpy array)
            a : The current action (usually integer)
            sp: The next state (s')
            r : The reward for this transition
        """
        # 'dynamically' resize buffer if required
        if self.filled_buffer == self.buffer_size:
            self._expand_all()

        i = self.filled_buffer
        self.s[i, :] = t.tensor(s, device=self.storage_device)

        # if discrete action
        if isinstance(a, int):
            self.a[i, 0] = a
        else:
            self.a[i, :] = t.tensor(a, device=self.storage_device)

        self.sp[i, :] = t.tensor(sp, device=self.storage_device)
        self.r[i, 0]  = r
        self.d[i, 0]  = d
        self.filled_buffer += 1

    # @classmethod
    def random_batch(self, T):
        """
        Returns a batch of T random transitions.
        """
        return self._batch(self._random_idxs(T))

    # @classmethod
    def contiguous_batch(self, T):
        """
        Returns a batch of T contiguous transitions with a random starting
        location.
        """
        return self._batch(self._contiguous_idxs(T))

class Continuous(Buffer):
    def __init__(
        self,
        state_shape,
        action_features,
        storage_device='cpu',
        output_device='cpu',
        output_dtype=t.float32,
        init_buffer=32
    ):
        super().__init__(state_shape, storage_device, output_device,
            output_dtype, init_buffer)
        super().__init__(state_shape, storage_device, output_device,
                         output_dtype, output_dtype, init_buffer)
        self.a = t.zeros(init_buffer, action_features, device=storage_device)

class Discrete(Buffer):
    def __init__(
        self,
        state_shape,
        storage_device='cpu',
        output_device='cpu',
        output_dtype=t.float32,
        action_dtype=t.int64,
        init_buffer=32
    ):
        """
        For use with discrete action environments.
        Args:
            state_shape: The shape of the state vector
            storage_device: The device memory in which to store transitions
            output_device: Device to copy memory to before returning samples
            output_dtype: The datatype to return values as
            action_dtype: The action datatype
            init_buffer: The initial size to make the buffer.
        """
        super().__init__(state_shape, storage_device, output_device,
                         output_dtype, action_dtype, init_buffer)
        self.a = t.zeros(init_buffer, 1, dtype=action_dtype, device=storage_device)

