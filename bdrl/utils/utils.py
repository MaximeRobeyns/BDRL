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
Utility functions
"""

import torch as t
import torch.nn as nn

def one_hot(a, actions, dtype=t.float32):
    """
    Converts an integer tensor a of shape [T, 1] to a one-hot representation of
    shape [T, actions]
    """
    assert 1 == a.shape[-1]
    return (a == t.arange(actions, device=a.device)).to(dtype=dtype)

class Squareplus(nn.Module):
    def __init__(self, a=2):
        super().__init__()
        self.a = a
    def forward(self, x):
        """The 'squareplus' activation function: has very similar properties to
        softplus, but is far cheaper computationally.
            - squareplus(0) = 1 (softplus(0) = ln 2)
            - gradient diminishes more slowly for negative inputs.
            - ReLU = (x + sqrt(x^2))/2
            - 'squareplus' becomes smoother with higher 'a'
        """
        return (x + t.sqrt(t.square(x)+self.a*self.a))/2

def squareplus_f(x, a=2):
    """The 'squareplus' activation function: has very similar properties to
    softplus, but is far cheaper computationally.
        - squareplus(0) = 1 (softplus(0) = ln 2)
        - gradient diminishes more slowly for negative inputs.
        - ReLU = (x + sqrt(x^2))/2
        - 'squareplus' becomes smoother with higher 'a'
    """
    return (x + t.sqrt(t.square(x)+a*a))/2

