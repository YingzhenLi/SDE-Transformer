
"""
BNN with MFVI, for fully connected layer
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5  # define a small constant for numerical stability control


class MFVILinear(nn.Module):
    """Applies a linear transformation to the incoming data: y = xW^T + b, where
    the weight W and bias b are sampled from the q distribution.
    """

    def __init__(self, dim_in, dim_out, prior_weight_std=1.0, prior_bias_std=1.0, init_std=0.05,
                 sqrt_width_scaling=False, device=None, dtype=None, bias=True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MFVILinear, self).__init__()
        self.dim_in = dim_in  # dimension of network layer input
        self.dim_out = dim_out  # dimension of network layer output

        # define the trainable variational parameters for q distribtuion
        # first define and initialise the mean parameters
        self.weight_mean = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self._weight_std_param = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self.bias = bias
        if self.bias:
            self.bias_mean = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
            self._bias_std_param = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
        self.reset_parameters(init_std)

        # define the prior parameters (for prior p, assume the mean is 0)
        prior_mean = 0.0
        if sqrt_width_scaling:  # prior variance scales as 1/dim_in
            prior_weight_std /= self.dim_in ** 0.5
        # prior parameters are registered as constants
        self.register_buffer('prior_weight_mean', torch.full_like(self.weight_mean, prior_mean))
        self.register_buffer('prior_weight_std', torch.full_like(self._weight_std_param, prior_weight_std))
        if self.bias:
            self.register_buffer('prior_bias_mean', torch.full_like(self.bias_mean, prior_mean))
            self.register_buffer('prior_bias_std', torch.full_like(self._bias_std_param, prior_bias_std))

    def extra_repr(self):
        s = "dim_in={}, dim_in={}, bias={}".format(self.dim_in, self.dim_out, self.bias)
        weight_std = self.prior_weight_std.data.flatten()[0]
        if torch.allclose(weight_std, self.prior_weight_std):
            s += f", weight prior std={weight_std.item():.2f}"
        if self.bias:
            bias_std = self.prior_bias_std.flatten()[0]
            if torch.allclose(bias_std, self.prior_bias_std):
                s += f", bias prior std={bias_std.item():.2f}"
        return s

    def reset_parameters(self, init_std=0.05):
        _init_std_param = np.log(init_std)
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        self._weight_std_param.data = torch.full_like(self.weight_mean, _init_std_param)
        if self.bias: 
            bound = self.dim_in ** -0.5
            nn.init.uniform_(self.bias_mean, -bound, bound)
            self._bias_std_param.data = torch.full_like(self.bias_mean, _init_std_param)

    # define the q distribution standard deviations with property decorator
    @property
    def weight_std(self):
        return torch.clamp(torch.exp(self._weight_std_param), min=EPS)

    @property
    def bias_std(self):
        if self.bias:
            return torch.clamp(torch.exp(self._bias_std_param), min=EPS)
        else:
            return None

    # KL divergence KL[q||p] between two Gaussians
    def kl_divergence(self):
        q_weight = dist.Normal(self.weight_mean, self.weight_std)
        p_weight = dist.Normal(self.prior_weight_mean, self.prior_weight_std)
        kl = dist.kl_divergence(q_weight, p_weight).sum()
        if self.bias:
            q_bias = dist.Normal(self.bias_mean, self.bias_std)
            p_bias = dist.Normal(self.prior_bias_mean, self.prior_bias_std)
            kl += dist.kl_divergence(q_bias, p_bias).sum()
        return kl

    # forward pass with Monte Carlo (MC) sampling
    def forward(self, input):
        weight = self._normal_sample(self.weight_mean, self.weight_std)
        if self.bias:
            bias = self._normal_sample(self.bias_mean, self.bias_std)
        else:
            bias = None
        return F.linear(input, weight, bias)

    def _normal_sample(self, mean, std):
        return mean + torch.randn_like(mean) * std

