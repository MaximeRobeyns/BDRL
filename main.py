#!/usr/bin/env python3

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
Main entrypoint
"""

import os
import sys
import gin
import gym
import argparse
import torch as t

# Agents

from bdrl.agents import BDRL, DQN, EnsembleAgent

from bdrl.utils.gin import gin as gin
from bdrl import environments

@gin.configurable
def main(random_episodes=10, episodes=1000, agent=BDRL, env='CartPole-v0',
         render=True, render_training=False):
    """
    Main entrypoint, here we parse the configuration file, initialise the
    agent, and start training.
    """
    env = gym.make(env)
    agent = agent(env)

    # Optional random exploration step
    for i in range(random_episodes):
        tr = agent.train_rollout(random_exploration=True)
        er = agent.eval_rollout()
        print((f"random episode: {i}, train reward: {tr:.3f}, "
               f"eval reward: {er:.3f}"))

    for i in range(episodes):
        tr = agent.train_rollout(render=render_training)
        er = agent.eval_rollout(render=render)
        print((f"training episode: {i}, train reward: {tr:.3f}, "
               f"eval reward: {er:.3f}"))

    return os.EX_OK

if __name__ == "__main__":
    desc = """
Bayesian distributional reinforcement learning program.
    """
    parser = argparse.ArgumentParser("main.py", description=desc)

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="The configuration file to use.",
        required=True
    )

    args = parser.parse_args()

    try:
        parsed = gin.parse_config_file(args.config)
        print(f"INFO: successfully parsed configuration in {parsed.filename}")
    except:
        sys.exit(f"ERROR: could not read configuration file: {args.config}")

    environments.register_envs()

    exit(
        main()
    )
