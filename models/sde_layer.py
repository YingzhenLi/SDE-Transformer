"""
Bayesian DL via SDEs

SDEs as building blocks for Bayesian neural networks

GPs with exponential kernel = linear SDEs

"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5  # define a small constant for numerical stability control
T_MAX = 10

# the SDE layer
class SDE_layer(nn.Module):
    """
    linear SDE layer, and we use another q SDE to approximate the posterior
    See Archambeau et al. "Variational Inference for Diffusion Processes"
    """
    def __init__(self, input_shape, linear_proj, device=None, dtype=None,  
                 inference_mode="marginal", r_min=0.01, r_max=0.2, t_max=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SDE_layer, self).__init__()
        self.input_shape = input_shape  # the shape of the states s_t, e.g., (dim,)
        assert inference_mode in ['marginal', 'joint']
        self.inference_mode = inference_mode    # should be "marginal" or "joint"
        if t_max is None:
            t_max = T_MAX
        self.t_max = t_max

        # prior SDE hyper-parameters
        # assume ds_t = -theta_p * s_t + sigma dW_t, t > 0, theta_p, sigma > 0
        # and theta_p is defined by prior (and fixed)
        self.log_sigma = nn.Parameter(torch.ones(size=input_shape, **factory_kwargs)*-5)

        # define q as another SDE with a different drift but the same diffusion
        # ds_t = -theta_q * s_t + sigma dW_t, t > 0, theta_q > 0
        self.param_theta_q = nn.Parameter(torch.randn(size=input_shape, **factory_kwargs)*0.1)
        # make sure that exp(-theta_q) is within [r_min, r_max] 
        self.max_theta_q, self.min_theta_q = -math.log(r_min), -math.log(r_max) 
        # as well as another initial distribution q(s_0) = N(s_0; q_mu, q_var)
        self.mu_q = nn.Parameter(torch.randn(size=input_shape, **factory_kwargs)*0.1)
        self.log_var_q = nn.Parameter(torch.randn(size=input_shape, **factory_kwargs)*0.1)

        # output mixing with linear projection
        if linear_proj is None:
            linear_proj = nn.Identity()
        self.linear_proj = linear_proj

    def extra_repr(self):
        s = "input_shape={}, inference_mode={}, r_min={}, r_max={}, t_max={}".format(\
             self.input_shape, self.inference_mode, 
             math.exp(-self.max_theta_q), math.exp(-self.min_theta_q), self.t_max)
        return s

    def set_inference_mode(mode):
        assert mode in ["marginal", "joint"]
        self.inference_mode = mode
        print("inference mode: {}".format(self.inference_mode))

    @property
    def var_q(self):
        return torch.exp(self.log_var_q)

    @property
    def log_var_p(self):
        # assume the initial prior distribution is the stationary distribution
        # i.e., p(s_0) = N(s_0; 0, 0.5 * sigma**2 / theta_p)
        return math.log(0.5) + 2 * self.log_sigma - torch.log(self.theta_p)

    @property
    def theta_q(self):
        # make sure theta_q is in (min_theta_q, max_theta_q)
        return self.min_theta_q + torch.sigmoid(self.param_theta_q) * (self.max_theta_q - self.min_theta_q)

    @property
    def theta_p(self):
        # we assume theta_p is 0.5 * (self.max_theta_q + self.min_theta_q)
        return torch.ones(size=(), device=self.log_sigma.device) * 0.5 * (self.min_theta_q + self.max_theta_q)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def var_scale(self):
        # the scaling constant associated with the variance
        return 0.5 * torch.exp(2 * self.log_sigma) / self.theta_q

    # KL divergence KL[q||p] between two SDEs
    # which is KL[q(s_0) || p(s_0)] + 0.5 * (theta_q - theta_p)**2 / sigma**2 * \int E_q(s_t)[s_t^2] d_t
    def kl_divergence(self):
        # KL for the initial distribution part, assume mu_p = 0
        kl_0 = kl_gaussian(self.mu_q, 0, self.log_var_q, self.log_var_p)
        # KL for the sde part, note that here we assume self.t_max = inf 
        tmp = (self.theta_q - self.theta_p).pow(2) / self.theta_q / 4
        kl_sde = tmp * (1 - 0.5 / self.theta_q + (self.mu_q.pow(2) + self.var_q) * torch.exp(-2*self.log_sigma))
        # but if not then kl_sde *= (1 - torch.exp(-2 * theta_q * self.t_max))
        if self.t_max is not None:
            kl_sde = kl_sde * (1 - torch.exp(-2 * theta_q * self.t_max))
        return kl_0.sum() + kl_sde.sum()

    def clamp_t(self, t):
        # clamp t from above if self.t_max is not None
        if self.t_max is not None:
            return torch.clamp(t, min=0, max=self.t_max)
        return t

    def mean_marginal(self, t):
        # need to assume the time-stamp for s_0 is t_0=0
        return torch.exp(-self.theta_q * t) * self.mu_q

    def var_marginal(self, t):
        # need to assume the time-stamp for s_0 is t_0=0
        return self.var_scale + torch.exp(-2 * self.theta_q * t) * (self.var_q - self.var_scale)
    
    def mean_and_var_marginal(self, t):
        theta_q_prod_t = -self.theta_q * t
        mean = torch.exp(theta_q_prod_t) * self.mu_q
        var = self.var_scale + torch.exp(2*theta_q_prod_t) * (self.var_q - self.var_scale)
        return mean, var

    def var_conditional(self, dt):
        # compute the variance of the conditional distribution p(s_t|s_t') with dt = |t - t'|
        return self.var_scale * (1 - torch.exp(-2 * self.theta_q * dt))

    def sample_marginal(self, t):
        # need to assume the time-stamp for s_0 is t_0=0
        mean, var = self.mean_and_var_marginal(t)
        return mean + var.sqrt() * torch.randn_like(mean)

    # cumsum method (fast and memory efficient)
    def sample_filtering_cumsum(self, t, sorted=False):
        # first sample s_0 and compute the non-cumsum part
        s_0 = self.mu_q + torch.exp(0.5 * self.log_var_q) * torch.randn_like(self.mu_q)
        # now compute the cumsum part: need to sort the input time indice in assending order
        if sorted:
            sorted_t = t
        else:
            sorted_t, indices = t.sort(1, descending=False)   # t has shape (N, T, ...)
        # compute the standard deviation of p(s_t | s_{t-1}) (after sorting)
        # compute dt = t_i - t_{i-1}, need to assume the time-stamp for s_0 is t_0=0
        dt = torch.cat([sorted_t[:, :1], sorted_t[:, 1:] - sorted_t[:, :-1]], dim=1)
        cond_std = (self.var_conditional(dt) + EPS).sqrt()
        # cumsum method to compute the zeta term
        sorted_exp_t = torch.exp(self.theta_q * sorted_t)
        zeta = torch.cumsum(cond_std * torch.randn_like(cond_std) * sorted_exp_t, dim=1)
        # need to reverse the sorting operation if applicable
        if not sorted:
            zeta = zeta.gather(1, indices.argsort(1))

        return (s_0 + zeta) * torch.exp(-self.theta_q * t)

    # forward pass with Monte Carlo (MC) sampling
    def forward(self, input, sample=True):
        # note: need to make sure input contain non-negative values
        if sample:
            if self.inference_mode == "marginal":
                out = self.sample_marginal(input)
            if self.inference_mode == "joint":
                out = self.sample_filtering_cumsum(input)
        else:   # just use the mean
            out = self.mean_marginal(input)
        return self.linear_proj(out)

def kl_gaussian(mu_q, mu_p, log_var_q, log_var_p):
    # compute KL[q||p] between two factorised Gaussians
    quadratic = (mu_q - mu_p).pow(2) * torch.exp(-log_var_p)
    logdet = log_var_q - log_var_p
    return 0.5 * (quadratic - 1 - logdet + torch.exp(logdet))

