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
Defines a simple feed-forward enural network, as used by DQN.
"""

import gin
import torch as t
import torch.nn as nn
import torch.nn.functional as F

@gin.configurable('ANN')
class ANN(nn.Module):
    """Artificial Neural Network"""
    def __init__(self, state_shape, num_actions, layer_sizes=(50, 25),
                 dtype=t.float32, device='cpu'):
        """
        Args:
            state_shape   : The shape of the state vector.
            num_actions   : The number of actions in the action set.
            layer_sizes   : A tuple with the size of the hidden layers. Note,
                            the number of hidden layers to use is implicitly
                            specified by the size of this tuple.
            dtype         : The datatype to use
            device        : The device memory to use
        """
        super().__init__()
        self.hidden         = len(layer_sizes)
        self.dtype          = dtype
        self.device         = device
        # must have at least 1 hidden layer...
        assert(len(layer_sizes) >= 1)

        layers = [nn.Linear(state_shape, layer_sizes[0]), nn.ReLU()]
        for i in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[i-1], layer_sizes[i]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-1], num_actions)]
        self.net_layers = nn.Sequential(*layers)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        if not t.is_tensor(x):
            x = t.from_numpy(x)
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.net_layers(x)
        return x
