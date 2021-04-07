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

import pathlib
from setuptools import setup
from setuptools import find_packages

wd = pathlib.Path(__file__).parent.resolve()
ld = (wd / 'README.org').read_text(encoding='utf-8')
install_requires = [
    'torch >= 1.5.1',
    'gin-config >= 0.4.0',
    'bayesfunc >= 0.1.0',
    'gym >= 0.18.0',
    'numpy >= 1.18.5',
    'argparse >= 1.4.0',
    ]

setup (
    name='bdrl',
    version='0.0.1',
    description='Experiments with Bayesian distributional reinforcement learning.',
    long_description=ld,
    author="Maxime Robeyns",
    author_email="maximerobeyns@gmail.com",
    url="https://github.com/MaximeRobeyns/bdrl",
    license='Apache 2.0',
    keywords='bdrl, reinforcement, machine, learning, research',
    include_package_data=True,
    packages=find_packages(exclude=['docs', 'configs', 'examples']),
    package_data={'configs': ['configs/*.gin']},
    install_requires=install_requires,
    project_urls={
        'Documentation': 'https://github.com/maximerobeyns/bdrl',
        'Bug Reports': 'https://github.com/maximerobeyns/bdrl/issues',
        'Source': 'https://github.com/maximerobeyns/bdrl',
    },
)
