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
Environment wrappers
"""

import gin
from gym import RewardWrapper

@gin.configurable
class RewardScaler(RewardWrapper):
    """Rescales the reward.

    Specify the min and max expected _return_, as well as the desired range for
    the rescaled return.
    """
    def __init__(self, environment, min=0, max=200, scale_min=-1, scale_max=1):
        super().__init__(environment)
        self.r_min = min
        self.rr = max - min
        self.sr = scale_max - scale_min
        self.s_min = scale_min

    def reward(self, r):
        tmp = (r - self.r_min) / self.rr
        return (tmp * self.sr) + self.s_min
