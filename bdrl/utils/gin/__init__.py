"""
Init file for Pytorch-specific Gin-Config package.

Adapted from justanhduc/neuralnet-pytorch: Adapted from Gin-config
"""

try:
    from gin import *
    from bdrl.utils.gin import configurables
except ImportError:
    print('Please install gin-config via \'pip install gin-config\'')
    raise
