
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gin import config

# Prefix these with '@' in gin config files.

# optimizers
config.external_configurable(optim.SGD, 'sgd', module='t.optim')
config.external_configurable(optim.Adam, 'adam', module='t.optim')
config.external_configurable(optim.Adadelta, 'adadelta', module='t.optim')
config.external_configurable(optim.Adagrad, 'adagrad', module='t.optim')
config.external_configurable(optim.SparseAdam, 'sparse_adam', module='t.optim')
config.external_configurable(optim.Adamax, 'adamax', module='t.optim')
config.external_configurable(optim.AdamW, 'adamw', module='t.optim')
config.external_configurable(optim.ASGD, 'asgd', module='t.optim')
config.external_configurable(optim.LBFGS, 'lbfgs', module='t.optim')
config.external_configurable(optim.RMSprop, 'rmsprop', module='t.optim')
config.external_configurable(optim.Rprop, 'rprop', module='t.optim')

# lr scheduler
config.external_configurable(optim.lr_scheduler.LambdaLR, 'lambda_lr', module='t.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.StepLR, 'step_lr', module='t.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.MultiStepLR, 'multistep_lr', module='t.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.ExponentialLR, 'exp_lr', module='t.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.CosineAnnealingLR, 'cosine_lr', module='t.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.ReduceLROnPlateau, 'plateau_lr', module='t.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.CyclicLR, 'cyclic_lr', module='t.optim.lr_scheduler')

# losses
config.external_configurable(F.l1_loss, 'l1loss', module='t.nn')
config.external_configurable(F.mse_loss, 'mseloss', module='t.nn')
config.external_configurable(F.cross_entropy, 'celoss', module='t.nn')
config.external_configurable(F.binary_cross_entropy, 'bceloss', module='t.nn')
config.external_configurable(F.binary_cross_entropy_with_logits, 'bcelogit_loss', module='t.nn')
config.external_configurable(F.nll_loss, 'nllloss', module='t.nn')
config.external_configurable(F.kl_div, 'kldloss', module='t.nn')
config.external_configurable(F.smooth_l1_loss, 'huberloss', module='t.nn')
config.external_configurable(F.cosine_embedding_loss, 'cosineembed_loss', module='t.nn')

# activations
config.external_configurable(nn.ELU, 'elu', module='t.nn')
config.external_configurable(nn.Hardshrink, 'hardshrink', module='t.nn')
config.external_configurable(nn.Hardtanh, 'hardtanh', module='t.nn')
config.external_configurable(nn.LeakyReLU, 'lrelu', module='t.nn')
config.external_configurable(nn.LogSigmoid, 'logsig', module='t.nn')
config.external_configurable(nn.MultiheadAttention, 'multihead_att', module='t.nn')
config.external_configurable(nn.PReLU, 'prelu', module='t.nn')
config.external_configurable(nn.ReLU, 'relu', module='t.nn')
config.external_configurable(nn.RReLU, 'rrelu', module='t.nn')
config.external_configurable(nn.SELU, 'selu', module='t.nn')
config.external_configurable(nn.CELU, 'celu', module='t.nn')
config.external_configurable(nn.Sigmoid, 'sigmoid', module='t.nn')
config.external_configurable(nn.Softplus, 'softplus', module='t.nn')
config.external_configurable(nn.Softshrink, 'softshrink', module='t.nn')
config.external_configurable(nn.Softsign, 'softsign', module='t.nn')
config.external_configurable(nn.Tanh, 'tanh', module='t.nn')
config.external_configurable(nn.Tanhshrink, 'tanhshrink', module='t.nn')
config.external_configurable(nn.Threshold, 'threshold', module='t.nn')

# constants (prefix with '%' in gin config files)
config.constant('float16', t.float16)
config.constant('float32', t.float32)
config.constant('float64', t.float64)
config.constant('int8', t.int8)
config.constant('int16', t.int16)
config.constant('int32', t.int32)
config.constant('int64', t.int64)
config.constant('complex32', t.complex32)
config.constant('complex64', t.complex64)
config.constant('complex128', t.complex128)
config.constant('float', t.float)
config.constant('short', t.short)
config.constant('long', t.long)
config.constant('half', t.half)
config.constant('uint8', t.uint8)
config.constant('int', t.int)
