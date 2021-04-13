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
Defines the network to perform 'Bayesian density regression', and defines a
distribution which takes the resulting parameters, with some common methods
implemented (mean, mode(s), log_prob, log_norm, cdf, icdf, sample...).
"""

import gin
import math
import bayesfunc as bf
from bayesfunc.priors import ScalePrior, NealPrior
from numbers import Real, Number

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints, Gamma, Distribution
from torch.distributions.utils import broadcast_all

@gin.configurable('BDR')
class BDR(nn.Module):
    """Bayesian Density Regression (BDR) class"""

    def __init__(self, in_features, inducing_batch=40, N=10, layer_sizes=(50,),
                 inducing_data=None, dtype=t.float32, device='cpu',
                 f_postproc='sort'):
        """
        Args:
            in_features (Int): The number of input features.
            inducing_batch (Int): The number of global inducing points to use.
            N (Int): The number of components to use in the piecewise linear
                log-likelihood.
            layer_sizes (Tuple): A tuple with the size of the hidden layers.
                Note, the number of hidden layers to use is implicitly
                specified by the size of this tuple.
            inducing_data (Tensor): Provide some initial inducing point locations.
            dtype (torch.dtype): The datatype to use
            device (String): The device memory to use
            f_postproc (String): Either ``sort`` or ``sum``. Sort is more
                'stable', but sum can give better representations.

        Note:
            This is currently quite sensitive to the scale of the data. If
            possible, pre-process the data to have unit standard deviation.

        Example:

            >>> net = BDR(10, N=11, layer_sizes=(50,75,25), f_postproc='sum')
            >>> batch.shape
            torch.Size([75, 10])
            >>> batch = batch.expand(50, -1, -1)
            >>> (f, alpha, beta), _, _ = bf.propagate(net, batch)
            >>> f.shape
            torch.Size([50, 75, 11])
            >>> theta, _, _ = bf.propagate(net, batch)
            >>> dist = BDR_dist(*theta)
            >>> lp = dist.log_prob(batch_ys)
        """
        super().__init__()
        self.in_features    = (in_features,)
        self.hidden         = len(layer_sizes)
        self.inducing_batch = inducing_batch
        self.N              = N
        self.id             = inducing_data
        self.dtype          = dtype
        self.device         = device
        self.f_postproc     = f_postproc
        assert(len(layer_sizes) >= 1)

        if self.id is None:
            self.ia = bf.InducingAdd(inducing_batch=self.inducing_batch,
                                     inducing_shape=(inducing_batch, in_features))
        else:
            self.ia = bf.InducingAdd(inducing_batch=self.inducing_batch,
                                     inducing_data=self.id)
        self.ir = bf.InducingRemove(inducing_batch=self.inducing_batch)

        # TODO add options for other layer types (factorised linear, DGP etc.)
        kwargs = {'inducing_batch': self.inducing_batch}
        layers = [bf.GILinear(in_features, layer_sizes[0], **kwargs),
                  nn.ReLU()]
        for i in range(1, len(layer_sizes)):
            layers += [bf.GILinear(layer_sizes[i-1], layer_sizes[i], **kwargs),
                       nn.ReLU()]
        self.net_layers = nn.Sequential(*layers)
        kwargs['full_prec'] = True
        kwargs['bias']      = True
        # linspace inducing data
        # GP_block = nn.Sequential(
        #     bf.BiasFeature(),
        #     bf.ReluKernelFeatures(inducing_batch=inducing_batch),
        #     bf.GIGP(out_features=10, inducing_batch=inducing_batch)
        # )
        self.f      = bf.GILinear(layer_sizes[-1], N, **kwargs)
        self.alpha  = bf.GILinear(layer_sizes[-1], N, **kwargs)
        self.beta   = bf.GILinear(layer_sizes[-1], N, **kwargs)

    def forward(self, x):
        # in_features must have rank 1 (flatten features with rank > 1)
        assert x.shape[-1:] == self.in_features

        x = self.ia(x)
        x = self.net_layers(x)

        f      = self.ir(self.f(x))
        alpha  = self.ir(self.alpha(x))
        beta   = self.ir(self.beta(x))

        # Post processing f ----------------------------------------------------

        # Note: sorting gives density estimates with less variance. They also
        # tend to collapse around one mode (which tends to be the highest one).
        #
        # Offset-based methods model multi-modal data better (particularly with
        # modes of different scale) however they suffer from a 'smoothing'
        # effect. It is easy to see why: a change in one f location
        # (particularly a 'base' location to which many other fs are added)
        # will change the effect of all other fs (as well as alphas and betas)
        # that depend on it. These dependencies make learning the parameters
        # difficult...

        # Idea: try to enforce monotonicity of fs at every layer in the
        # network, so dependencies are not so strongly concentrated at the
        # output.

        # Option 1: simple sorting
        if self.f_postproc == 'sort':
            f = f.sort(-1)[0]

        # Option 2: sort f, and keep corresponding alpha and beta together:
        # (no observed effect on performance)
        # if self.f_postproc == 'sort':
        #     fab   = t.cat((f.unsqueeze(0), alpha.unsqueeze(0), beta.unsqueeze(0)), 0)
        #     idxs  = f.sort(-1).indices.expand(3, *list(f.shape))
        #     fab   = t.gather(fab, len(f.shape), idxs)
        #     f, alpha, beta = fab

        # Option 3: Offset from f0
        # (problem: tends to skew data to the right)
        # f0 = f[...,0].unsqueeze(-1)
        # f = t.cat((f0, (f0 + F.softplus(f[...,1:])).cumsum(-1)), -1)

        # Option 4: Midpoint offset
        if self.f_postproc == 'sum':
            m = math.floor(f.shape[-1]/2)
            fm = f[...,m].unsqueeze(-1)
            f = t.cat((
                fm - (F.softplus(f[...,:m]).flip(-1).cumsum(-1)),
                fm,
                fm + F.softplus(f[...,m+1:]).cumsum(-1)
            ), -1)

        # ---------------------------------------------------------------------

        alpha = t.cat((
            alpha[...,:-1],
            -alpha[...,:-1].sum(-1).unsqueeze(-1) +
            F.softplus(alpha[...,-1]).unsqueeze(-1)
        ), -1)

        beta = t.cat((
            -beta[...,1:].sum(-1).unsqueeze(-1) +
            F.softplus(beta[...,0]).unsqueeze(-1),
            beta[...,1:]
        ), -1)

        return f, alpha, beta

# Alternative Approaches ======================================================

class _DN_ANN(nn.Module):
    """ANN to output parameters for the piecewise-linear log likelihood"""

    def __init__(self, in_features, N=10, layer_sizes=(50,), f_postproc='sort'):
        super().__init__()
        self.in_features    = (in_features,)
        self.hidden         = len(layer_sizes)
        self.N              = N
        self.f_postproc     = f_postproc
        assert(len(layer_sizes) >= 1)

        layers = [nn.Linear(in_features, layer_sizes[0], bias=True),
                  nn.ReLU()]
        for i in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[i-1], layer_sizes[i], bias=True),
                       nn.ReLU()]
        self.net_layers = nn.Sequential(*layers)

        self.f     = nn.Linear(layer_sizes[-1], N, bias=True)
        self.alpha = nn.Linear(layer_sizes[-1], N, bias=True)
        self.beta  = nn.Linear(layer_sizes[-1], N, bias=True)

    def forward(self, x):
        # in_features must have rank 1 (flatten features with rank > 1)
        assert x.shape[-1:] == self.in_features

        x = self.net_layers(x)
        f     = self.f(x)
        alpha = self.alpha(x)
        beta  = self.beta(x)

        # Option 1: simple sorting
        if self.f_postproc == 'sort':
            f = f.sort(-1)[0]

        # Option 2: sort f, and keep corresponding alpha and beta together:
        # (no observed effect on performance)
        # if self.f_postproc == 'sort':
        #     fab   = t.cat((f.unsqueeze(0), alpha.unsqueeze(0), beta.unsqueeze(0)), 0)
        #     idxs  = f.sort(-1).indices.expand(3, *list(f.shape))
        #     fab   = t.gather(fab, len(f.shape), idxs)
        #     f, alpha, beta = fab

        # Option 3: Offset from f0
        # (problem: tends to skew data to the right)
        # f0 = f[...,0].unsqueeze(-1)
        # f = t.cat((f0, (f0 + F.softplus(f[...,1:])).cumsum(-1)), -1)

        # Option 4: Midpoint offset
        if self.f_postproc == 'sum':
            m = math.floor(f.shape[-1]/2)
            fm = f[...,m].unsqueeze(-1)
            f = t.cat((
                fm - (F.softplus(f[...,:m]).flip(-1).cumsum(-1)),
                fm,
                fm + F.softplus(f[...,m+1:]).cumsum(-1)
            ), -1)

        # ---------------------------------------------------------------------

        alpha = t.cat((
            alpha[...,:-1],
            -alpha[...,:-1].sum(-1).unsqueeze(-1) +
            F.softplus(alpha[...,-1]).unsqueeze(-1)
        ), -1)

        beta = t.cat((
            -beta[...,1:].sum(-1).unsqueeze(-1) +
            F.softplus(beta[...,0]).unsqueeze(-1),
            beta[...,1:]
        ), -1)

        return f, alpha, beta

@gin.configurable('Ensemble')
class Ensemble(nn.Module):
    """An ensemble of ANNs providing (non-Bayesian) uncertainty estimates."""

    def __init__(self, in_features, E=40, N=7, layer_sizes=(50,),
                 opt=t.optim.Adam, lr=0.05, dtype=t.float32, device='cpu',
                 f_postproc='sort'):
        """
        Args:
            in_features (Int): The number of input features.
            E (Int): The number of networks to use in the ensemble
            N (Int): The number of components to use in the piecewise linear
                log-likelihood.
            layer_sizes (Tuple): A tuple with the size of the hidden layers.
                Note, the number of hidden layers to use is implicitly
                specified by the size of this tuple.
            inducing_data (Tensor): Provide some initial inducing point locations.
            dtype (torch.dtype): The datatype to use
            device (String): The device memory to use
            f_postproc (String): Either ``sort`` or ``sum``. Sort is more
                'stable', but sum can give better representations.
        """
        super().__init__()
        self.in_features    = (in_features,)
        self.hidden         = len(layer_sizes)
        self.E              = E
        self.N              = N
        self.dtype          = dtype
        self.device         = device
        self.f_postproc     = f_postproc
        assert(len(layer_sizes) >= 1)

        self.models = []
        for _ in range(E):
            tmp_model = _DN_ANN(in_features, N, layer_sizes, f_postproc)
            tmp_model = tmp_model.to(dtype=self.dtype, device=self.device)
            tmp_opt = opt(tmp_model.parameters(), lr=lr)
            self.models.append({
                'm':   tmp_model,
                'opt': tmp_opt
            })

    def train(self, X, y, epochs=10, batch_size=64):
        data_len = X.shape[0]
        for m in self.models:
            for e in range(epochs):
                batch_idx = t.randperm(data_len)[:batch_size]
                (f, alpha, beta) = m['m'](X[batch_idx])
                # need to add in dim 1 because there are no posterior samples
                f     = f.unsqueeze(0)
                alpha = alpha.unsqueeze(0)
                beta  = beta.unsqueeze(0)
                ll = BDR_dist(f, alpha, beta).log_prob(y[batch_idx])
                ll = ll.sum(-1).mean(-1)
                m['opt'].zero_grad()
                (-ll.mean()).backward()
                m['opt'].step()

    def f(self, X):
        partial_fs = []
        partial_alphas = []
        partial_betas = []
        with t.no_grad():
            for m in self.models:
                (f, alpha, beta) = m['m'](X)
                partial_fs.append(f.unsqueeze(0))
                partial_alphas.append(alpha.unsqueeze(0))
                partial_betas.append(beta.unsqueeze(0))

        return t.cat(partial_fs, 0), t.cat(partial_alphas, 0), t.cat(partial_betas, 0)

# BDR distribution ------------------------------------------------------------

class BDR_dist(Distribution):
    """
    Creates a distribution parametrised by (f, alpha, beta) and
    implements standard methods.
    """
    arg_constraints = {'f': constraints.real,
                       'alpha': constraints.real,
                       'beta': constraints.real}
    support = constraints.real

    @property
    def support(self):
        return constraints.real_vector

    @property
    def num_samples(self):
        """Returns the number of samples for the distributions's parameters.
        This is the number of times your batch of values will be replicated
        along dimension 0.
        """
        return self._dist_samples

    def _calc_mean(self, f, a, b, z):
        """Calculates the mean using the provided parameters."""

        c_1 = (t.exp(a[...,0] * f[...,0] + b[...,0]) *
               (a[...,0] * f[...,0] - 1)
              ) / a[...,0]**2
        c_2 = ((t.exp(a[...,1:-1] * f[...,1:] + b[...,1:-1]) * (a[...,1:-1] * f[...,1:] - 1) -
               t.exp(a[...,1:-1] * f[...,:-1] + b[...,1:-1]) * (a[...,1:-1] * f[...,:-1] - 1)
               ) / a[...,1:-1]**2).sum(-1)
        c_3 = (t.exp(a[...,-1] * f[...,-1] + b[...,-1]) *
               (a[...,-1] * f[...,-1] - 1)
               ) / a[...,-1]**2

        return 1/z * (c_1 + c_2 - c_3)

    def _calc_a_b(self, f, alpha, beta, avg=False):
        """Pre-computes the a and b coefficient vectors.
        Args:
            f (Tensor): f parameters (batched)
            alpha (Tensor): alpha parameters (batched)
            beta (Tensor): beta parameters (batched)
            avg (Bool): whether the f, alpha and beta have been averaged over
                samples.
       """
        # Flip these matrices along last dimension by copying memory; O(n)
        b_flip = beta.flip(-1)
        f_flip = f.flip(-1)

        # Calculate partial sums; O(n) per dimension
        a_ps   = alpha.cumsum(-1)
        b_ps   = b_flip.cumsum(-1).flip(-1)
        af_ps  = (alpha*f).cumsum(-1)
        bf_ps  = (b_flip * f_flip).cumsum(-1).flip(-1)

        # Calculate terms for interval y \in (-\infty, f_1]
        a_0    =   b_ps[...,0]
        b_0    = - bf_ps[...,0]
        # Calculate main coefficient vectors
        a_vec  = b_ps[...,1:]   - a_ps[...,:-1]
        b_vec  = af_ps[...,:-1] - bf_ps[...,1:]
        # Calculate terms for interval y \in (f_N, \infty)
        a_N    = - a_ps[...,-1]
        b_N    =   af_ps[...,-1]

        # Concatenate for easier computation later.
        if avg:
            self.a_avg = t.cat((a_0.unsqueeze(-1), a_vec, a_N.unsqueeze(-1)), -1)
            self.b_avg = t.cat((b_0.unsqueeze(-1), b_vec, b_N.unsqueeze(-1)), -1)
        else:
            self.a = t.cat((a_0.unsqueeze(-1), a_vec, a_N.unsqueeze(-1)), -1)
            self.b = t.cat((b_0.unsqueeze(-1), b_vec, b_N.unsqueeze(-1)), -1)

    def _calc_norm(self, f, a, b, avg=False):
        """
        Calculates the normalising term, Z.
        """
        i_0 = 1/a[...,0] * t.exp(a[...,0] * f[...,0] + b[...,0])
        i_1 = 1/a[...,1:-1] * t.exp(b[...,1:-1]) * (
            t.exp(a[...,1:-1] * f[...,1:]) - t.exp(a[...,1:-1] * f[...,:-1])
        )
        i_2 = 1/a[...,-1] * t.exp(a[...,-1] * f[...,-1] + b[...,-1])

        if avg:
            self.i_0_avg = i_0
            self.i_1_avg = i_1
            self.i_2_avg = i_2
            self.Z_avg = i_0 + i_1.sum(-1) - i_2
        else:
            self.i_0 = i_0
            self.i_1 = i_1
            self.i_2 = i_2
            self.Z = i_0 + i_1.sum(-1) - i_2

    def _calc_modes(self, f):
        """
        Calculates the locations of the modes of the distribution.

        We cannot form a regularly shaped tensor with the mode(s) for each
        distribution in the batch because these may have different numbers of
        modes. Therefore we generate all the mode locations for all
        distributions in the batch concatenated into a 1D vector, along with an
        additional tensor of size batch_shape giving the number of modes per
        distribution in the batch. See the `mode_at` convenience method to use
        these vectors.
        """

        fd = self.log_prob(f).mean(0)

        # If f_i-1 <= f_i and f_i+1 <= f_i, then f_i is a mode
        f1 = fd[...,1:]
        f2 = fd[...,:-1]
        b1 = f1.le(f2)
        b2 = f2.le(f1)
        m1 = b1[...,1:].logical_and(b2[...,:-1])

        # if f_0 => f_1 then f_0 is a mode
        # if f_N => f_N-1 then f_n is a mode
        mask = t.cat((
            fd[...,0].ge(fd[...,1]).unsqueeze(-1),
            m1,
            fd[...,-1].ge(fd[...,-2]).unsqueeze(-1)
        ), -1)

        self._modes = self.f.masked_select(mask)
        self._modes_avg = self.f_avg.masked_select(mask)
        self._num_modes = mask.sum(-1)

    def _init_props(self):
        """
        Calculates the coefficient vectors a and b from the provided
        parameters. Also calculates the normalising constant.
        Note
            This method is intended to be called in the constructor.
        """
        # N is the number of components in the distribution
        self.N = self.f.shape[-1]
        self._dist_samples = self.f.shape[0]
        self._true_batch_shape = self.f.shape[1:-1]

        self._calc_a_b(self.f, self.alpha, self.beta)
        self._calc_norm(self.f, self.a, self.b)

    def _init_full_props(self):
        """
        Calculates further properties of the distribution which aren't required
        for basic likelihood operations.
        """
        self.fp_init = True
        self.f_inv   = self.cdf(self.f).squeeze(-1)
        self._mean   = self._calc_mean(self.f, self.a, self.b, self.Z)

        # Averaged parameters:
        self.f_avg     = self.f.mean(0).unsqueeze(0)
        self.alpha_avg = self.alpha.mean(0).unsqueeze(0)
        self.beta_avg  = self.beta.mean(0).unsqueeze(0)
        self._calc_a_b(self.f_avg, self.alpha_avg, self.beta_avg, avg=True)
        self._calc_norm(self.f_avg, self.a_avg, self.b_avg, avg=True)
        self._calc_modes(self.f_avg)
        # self._calc_modes(self.f_avg, avg=True)
        self.f_inv_avg = self.cdf(self.f_avg, avg=True).squeeze(-1)
        self._mean_avg = self._calc_mean(self.f_avg, self.a_avg, self.b_avg, self.Z_avg)

    def _validate_sample(self, value, avg=False):
        """Argument validation for distribution methods.

        We depart from PyTorch distribution conventions to stay in keeping with
        Bayesfunc conventions of having posterior samples at dimension 0.

        Valid samples are:

            a) Scalar values (will be replicated for all samples of all
            distributions)

            b) The same 1D batch of values (will be replicated for all samples
            of all distributions)

            c) A different 1D batch for each distribution. The value is
            therefore a 2D batch, with dimension 0 matching the distribution
            batch shape (will be replicated for all samples of the
            corresponding distribution).

            d) A different 1D batch for each sample of each distribution.

        Args:
            value (Tensor): the tensor of values to evaluate / compute
            avg (Boolean): if true, ensures sample shape is 1

        Raises
            ValueError: when the dimensions of value do not match any of the
                cases a--d above.
        """

        if (avg == True) and (self.fp_init == False):
            print(("\n\tWarning: initialise BDR_dist with full_props=True to "
                   "avoid computation spikes\n\twhen using avg=True in a method.\n"))
            self._init_full_props()

        if value.dtype != self.dtype:
            value = value.to(dtype=self.dtype)

        # Case a
        if not t.is_tensor(value) and type(value) == float:
            value = t.tensor([value], dtype=self.dtype).to(device=self.device)
        elif value.shape == t.Size([]):
            value = value.expand(1 if avg else self._dist_samples,
                                 self._true_batch_shape[0], 1, 1)
        # Case b
        elif len(value.shape) == 1 and value.shape[0] > 1:
            # to broadcast: add final 1, make dimension 2 and account for
            # distribution's parameter samples
            value = value.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        # Case c
        elif len(value.shape) == 2:
            # Just add final 1 and account for distribution parameter samples
            if value.shape[0] != self._true_batch_shape[0]:
                raise ValueError((
                    f"Sample shape must match batch shape. Got ({value.shape[0]}) "
                    f"and ({self._true_batch_shape[0]})"
                ))
            value = value.unsqueeze(-1).unsqueeze(0)
        # Case d
        elif len(value.shape) == 3:
            if avg:
                if value.shape[0] != 1:
                    raise ValueError((
                        f"Average over samples requested, yet sample shape in "
                        f"value is not 1 (got {value.shape[0]})"
                    ))
            else:
                if value.shape[0] != 1 and value.shape[0] != self._dist_samples:
                    raise ValueError((
                        f"Sample's first dimension ({value.shape[0]}) does not "
                        f"match number of parameter samples ({self._dist_samples}) "
                        f"for sample of dimension 3"
                    ))
            if value.shape[1] != self._true_batch_shape[0]:
                raise ValueError((
                    f"Sample's second dimension ({value.shape[1]}) does not "
                    f"match the batch shape ({self._true_batch_shape[0]}) for "
                    f"sample of dimension 3"
                ))
            value = value.unsqueeze(-1)
        elif len(value.shape) > 3:
            raise ValueError("Cannot sample batches with more than 3 dimensions")

        return value

    def __init__(self, f, alpha, beta, validate_args=True, fp=False,
                 full_props=False):
        """
        Args:
            f (Tensor): The f locations of the distribution
            alpha (Tensor): the alpha coefficients
            beta (Tensor): the beta coefficients
            validate_args (Boolean): Whether to validate arguments passed to
                this class' methods.
            fp (Boolean): Alias for ``full_props``
            full_props (Boolean): whether or not to initialise all the
                properties of the distribution when the constructor is called,
                or just the subset necessary to call log_prob (this is intended
                to reduce unnecessary computation during training).

        Hint:
            If you will only call this class' ``log_prob`` method, then set
            ``full_props=False`` (default), however if you intend to call
            ``sample``, ``cdf``, ``icdf``, ``modes``, ``mean`` (among others),
            then you can set ``full_props=True`` to compute required properties
            ahead of time.

        Example:

            >>> theta, logpq, sample_dict = bf.propagate(net, batch)
            >>> theta[0].shape
            torch.Size([25,10,5])
            >>> dist = BDR_dist(*theta, fp=True)
            >>> samples = dist.sample(100, avg=True)
            >>> samples.shape
            torch.Size([10, 100])

        """
        self.f = f
        self.alpha = alpha
        self.beta = beta
        assert f.shape == alpha.shape
        assert alpha.shape == beta.shape
        self.dtype = f.dtype
        self.device = self.f.device
        batch_shape = self.f.shape[:-1]
        event_shape = t.Size([1])
        super(BDR_dist, self).__init__(batch_shape, event_shape, validate_args)
        self._fp = (full_props or fp)
        self.fp_init = False
        self._init_props()
        if self._fp:
            self._init_full_props()

    def mean(self, avg=True):
        """Returns the mean of the distribution (or batch of distributions).

        Args:
            avg (Boolean): Whether to return the mean of the distribution found
                by averaging the sampled parameters (True) or the mean of each
                sampled distribution.
        """
        if not self.fp_init:
            if not avg:
                return self._calc_mean(self.f, self.a, self.b, self.Z)
            else:
                return self._calc_mean(self.f_avg, self.a_avg, self.b_avg,
                                       self.Z_avg)
        return self._mean if not avg else self._mean_avg

    def variance(self, avg=False):
        # TODO; this should be a fairly straightforward integral (similar to
        # the one for the mean)
        raise NotImplementedError

    def modes(self, avg=False):
        """Hard to use; consider using mode_at instead.
        See _calc_mode for more info"""
        if not self.fp_init:
            self._init_full_props
        if avg:
            return self._modes_avg, self._num_modes
        return self._modes, self._num_modes

    def mode_at(self, i, avg=False):
        """
        A convenience method for 1D distribution batches which returns a tensor
        of shape [samples x modes] if avg=False or [modes] if avg=True for
        distribution i.

        Args:
            i (Int): the index of the distribution in the batch for which to return
                the mode(s) (0 <= i < batch_shape)
            avg (Boolean): Whether to return the averaged mode location, or samples

        Returns:
            A 1D vector of mode locations if avg=True, else a tensor of shape
                [samples x N], for N the number of nodes.
        """
        if len(self._true_batch_shape) != 1:
            raise ValueError("mode_at only works for 1D batches of distributions")
        if i >= self._true_batch_shape[0]:
            raise ValueError((f"Requested distribution out of range ({i}) of "
                              f"max ({self._true_batch_shape})"))
        if not self.fp_init:
            self._init_full_props
        idxs = self._num_modes.cumsum(-1) - self._num_modes[0]
        if avg:
            return self._modes_avg[idxs[i]:idxs[i]+self._num_modes[i]]
        else:
            lst = []
            n = self._num_modes.sum()
            for j in range(self._dist_samples):
                lst.append(self._modes[idxs[i]+(j*n):
                                       idxs[i]+(j*n)+self._num_modes[i]
                                       ].unsqueeze(0))
            return t.cat(lst, 0)

    def sample(self, sample_shape=t.Size(), avg=True):
        """Generates a ``sample_shape`` shaped sample or sample_shape shaped
        batch of samples if the distribution parameters are batched.

        Args:
            sample_shape (t.Size): The shape of the sample
            avg (Boolean): Whether to sample from the distribution found by
                averaging the parameter samples (True) or not.

        Note:
            The uniformly sampled points (used in the inverse transforom
            sampling) are *the same* across distribution batches and parameter
            samples.
        """
        with t.no_grad():
            return self.rsample(sample_shape, avg)

    def rsample(self, sample_shape=t.Size(), avg=True):
        if type(sample_shape) == int:
            sample_shape = (sample_shape,)
        pts = t.rand(sample_shape, dtype=self.dtype, device=self.device).flatten()
        samples = self.icdf(pts, avg=avg).squeeze()
        base_shape = self._true_batch_shape if avg else self._batch_shape
        samples = samples.reshape(base_shape + sample_shape)
        return samples

    def log_prob(self, value, avg=False):
        """Evaluate the log probability of ``value``

        Args:
            value (Tensor or float): The location(s) at which to evaluate the
                log PDF.
            avg (Boolean): Whether to evluate from the average of the
                distribution's parameter samples.
        """
        value = self._validate_sample(value, avg)

        # Unsqueeze the parameters at location -2 to allow for an arbitrary
        # number of sample locations.
        tmp_f     = (self.f_avg if avg else self.f).unsqueeze(-2)
        tmp_alpha = (self.alpha_avg if avg else self.alpha).unsqueeze(-2)
        tmp_beta  = (self.beta_avg if avg else self.beta).unsqueeze(-2)
        tmp_Z     = (self.Z_avg if avg else self.Z).unsqueeze(-1)

        res = value - tmp_f
        assert res.shape[-1] == self.N
        ll_terms = -(res) * t.where(res > 0, tmp_alpha, -tmp_beta)
        lls = ll_terms.sum(-1) - t.log(tmp_Z)
        return lls

    def cdf(self, value, avg=False):
        """Cumulative distribution function

        Args:
            value (Tensor or float): The location(s) at which to evaluate the CDF.
            avg (Boolean): Whether to evluate from the average of the
                           distribution's parameter samples.
        """
        value = self._validate_sample(value, avg)

        tmp_f   = (self.f_avg   if avg else self.f  ).unsqueeze(-2)
        tmp_a   = (self.a_avg   if avg else self.a  ).unsqueeze(-2)
        tmp_b   = (self.b_avg   if avg else self.b  ).unsqueeze(-2)
        tmp_Z   = (self.Z_avg   if avg else self.Z  ).unsqueeze(-1).unsqueeze(-1)
        tmp_i_0 = (self.i_0_avg if avg else self.i_0).unsqueeze(-1).unsqueeze(-1)
        tmp_i_1 = (self.i_1_avg if avg else self.i_1).unsqueeze(-2)

        z = t.zeros((1), dtype=value.dtype).to(device=value.device)
        c_1 = t.where(tmp_f[...,1:] < value, tmp_i_1, z).sum(-1).unsqueeze(-1)

        idxs = tmp_f.lt(value).sum(-1).unsqueeze(-1)

        a = list(tmp_f.shape)
        b = list(tmp_a.shape)
        a[-2] = value.shape[-2]
        b[-2] = value.shape[-2]

        f_k = t.gather(tmp_f.expand(a), 3, (idxs-1).clamp(0))
        a_k = t.gather(tmp_a.expand(b), 3, idxs)
        b_k = t.gather(tmp_b.expand(b), 3, idxs)
        c_2 = 1/a_k * t.exp(b_k) * (t.exp(a_k * value) - t.exp(a_k * f_k))

        CDF = 1/tmp_Z * (tmp_i_0 + c_1 + c_2)
        return CDF

    def icdf(self, value, avg=False):
        """Evaluate the quantile function.

        Args:
            value (Tensor or float): the location(s) at which to evaluate the
                quantile function. Must be between 0 and 1; behaviour is
                undefined if this is not the case.
            avg (Boolean): whether to evaluate from the average of the
                distribution's parameter samples.
        """

        if not self.fp_init:
            self._init_full_props()
        value = self._validate_sample(value, True)

        if not constraints.unit_interval.check(value).all():
            raise ValueError(
                'The icdf value argument must be within the unit interval'
            )

        tmp_f     = (self.f_avg     if avg else self.f    ).unsqueeze(-2)
        tmp_f_inv = (self.f_inv_avg if avg else self.f_inv).unsqueeze(-2)

        tmp_a   = (self.a_avg   if avg else self.a  ).unsqueeze(-2)
        tmp_b   = (self.b_avg   if avg else self.b  ).unsqueeze(-2)
        tmp_Z   = (self.Z_avg   if avg else self.Z  ).unsqueeze(-1).unsqueeze(-1)
        tmp_i_0 = (self.i_0_avg if avg else self.i_0).unsqueeze(-1).unsqueeze(-1)
        tmp_i_1 = (self.i_1_avg if avg else self.i_1).unsqueeze(-2)

        a = list(tmp_f.shape)
        a[-2] = value.shape[-2]
        b = list(tmp_a.shape)
        b[-2] = value.shape[-2]
        z = t.zeros((1), dtype=value.dtype).to(device=value.device)

        idxs = tmp_f_inv.lt(value).sum(-1).unsqueeze(-1)
        f_l = t.gather(tmp_f.expand(a), 3, (idxs-1).clamp(0))
        a_l = t.gather(tmp_a.expand(b), 3, idxs)
        b_l = t.gather(tmp_b.expand(b), 3, idxs)

        c_1 = t.where(tmp_f_inv[...,1:] < value, tmp_i_1, z).sum(-1).unsqueeze(-1)
        c_2 = a_l * t.exp(-b_l)
        c_3 = t.exp(a_l * f_l)
        ICDF = t.log(((value * tmp_Z) - tmp_i_0 - c_1) * c_2 + c_3) / a_l
        return ICDF
