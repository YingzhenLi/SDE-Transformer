import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sde_layer import kl_gaussian

EPS = 1e-5
T_MAX = 10

class MultiheadSDE_layer(nn.Module):

    def __init__(self, dim_input, dim_embed, num_heads, n_inducing, embed_func=None, device=None, dtype=None,
                 inference_mode='marginal', r_min=0.01, r_max=0.2, t_max=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim_input = dim_input
        if dim_embed is None:
            dim_embed = dim_input
        self.num_heads = num_heads
        self.vdim = int(dim_embed / num_heads)
        self.dim_embed = self.num_heads * self.vdim
        if embed_func is None:
            #self.embed_func = nn.Sequential(nn.Linear(dim_input, self.num_heads, **factory_kwargs),
            #                                nn.Softplus())
            self.embed_func = nn.Linear(dim_input, self.num_heads, **factory_kwargs)
        else:
            self.embed_func = embed_func    # need to ensure the output here is non-negative
        if t_max is None:
            t_max = T_MAX
        # division by n_inducing**2 seems to solve the gradient explosion issue with big n_inducing, but why?
        self.t_max = t_max / n_inducing**2
        self.t_min = -t_max / n_inducing**2
        
        # prior SDE hyper-parameters
        # assume ds_t = -theta * s_t + sigma dW_t, t > 0, theta, sigma > 0
        self.log_sigma = nn.Parameter(torch.randn(size=(self.num_heads, self.vdim), **factory_kwargs)*0.1)
        self.param_theta = nn.Parameter(torch.randn(size=(self.num_heads, self.vdim), **factory_kwargs)*0.1 - 1)
        self.max_theta, self.min_theta = -math.log(r_min), -math.log(r_max) 
        # combine the latent linear SDE together - linear weighting
        self.weight = nn.Parameter(torch.empty(size=(self.dim_embed, self.dim_embed), **factory_kwargs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.empty(size=(self.dim_embed,), **factory_kwargs))
        bound = self.dim_embed ** -0.5
        nn.init.uniform_(self.bias, -bound, bound)

        # q parameters
        self.n_inducing = n_inducing
        self.inference_mode = inference_mode
        self.Z = nn.Parameter(torch.randn(size=(n_inducing, dim_input), **factory_kwargs))
        self.mu_q_proj = nn.Linear(dim_input, self.dim_embed, **factory_kwargs)
        # make q distribution also having markov chain structure
        self.lbd_q = nn.Parameter(torch.randn(size=(n_inducing, self.num_heads, self.vdim), **factory_kwargs)*0.1)
        self.tau_q = nn.Parameter(torch.randn(size=(n_inducing, self.num_heads, self.vdim), **factory_kwargs)*0.1)
    
    def extra_repr(self):
        s = "dim_input={}, dim_embed={}, num_heads={}, n_inducing={}, inference_mode={}, ".format(\
             self.dim_input, self.dim_embed, self.num_heads, self.n_inducing, self.inference_mode)
        s += "r_min={}, r_max={}, t_max={}".format(math.exp(-self.max_theta), \
             math.exp(-self.min_theta), self.t_max)
        return s

    def set_inference_mode(self, mode):
        assert mode in ["marginal", "joint"]
        self.inference_mode = mode
        print("inference mode: {}".format(self.inference_mode))

    @property
    def theta_p(self):
        #return torch.exp(self.param_theta)
        # make sure theta is in (min_theta_q, max_theta_q)
        return (self.min_theta + torch.sigmoid(self.param_theta) * (self.max_theta - self.min_theta))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    @property
    def var_q(self):
        return torch.exp(self.log_var_q)
    
    @property
    def log_var_p(self):
        # assume the initial distribution is the stationary distribution N(s_0; 0, 0.5 * sigma**2 / theta_p)
        return math.log(0.5) + 2 * self.log_sigma - torch.log(self.theta_p)

    @property
    def t_q(self):
        return torch.clamp(self.embed_func(self.Z), min=self.t_min, max=self.t_max) # shape (n_inducing, num_head)

    @property
    def mu_q(self):
        return self.mu_q_proj(self.Z).reshape(-1, self.num_heads, self.vdim)   # shape (n_inducing, num_head, vdim)

    @property
    def var_scale(self):
        # the scaling constant associated with the variance
        return 0.5 * torch.exp(2 * self.log_sigma) / self.theta_p
    
    def make_q_cov(self, lbd, tau):
        # assume lbd = [lbd1, lbd2], tau = [tau1, tau2] (also extends to broadcasting)
        # assume a 2x2 cov matrix of form [[lbd1^2 + tau1^2, lbd1 * tau2], [lbd1 * tau2, lbd2^2 + tau2^2]]
        var = lbd.pow(2) + tau.pow(2)   # shape (M, D)
        cov = lbd[:-1] * tau[1:]    # shape (M-1, D)
        cov = torch.cat([cov, torch.zeros_like(var[:1])], dim=0)
        return var, cov

    def sort_inducing_points(self, add_endpoints=True, get_var=True):
        # self.time_q has shape (M, ...)
        t, indices = self.t_q.sort(dim=0, descending=False)
        indices = indices.unsqueeze(-1).tile((1, 1, self.vdim)).detach() # test this!
        mu_q = self.mu_q.gather(0, indices)
        if get_var:
            lbd_q = self.lbd_q.gather(0, indices)
            tau_q = self.tau_q.gather(0, indices)
            var_q, cov_q = self.make_q_cov(lbd_q, tau_q)
        else:
            var_q, cov_q = None, None
        if add_endpoints:
            t_max = torch.ones_like(t[-1:]) * self.t_max + EPS
            t_min = torch.ones_like(t[-1:]) * self.t_min - EPS
            t = torch.cat([t_min, t, t_max], dim=0)
            mu_q = torch.cat([torch.zeros_like(mu_q[:1]), mu_q, torch.zeros_like(mu_q[:1])], dim=0)
            if get_var:
                var_q = torch.cat([torch.zeros_like(var_q[:1]), var_q, torch.zeros_like(var_q[:1])], dim=0)
                cov_q = torch.cat([torch.zeros_like(cov_q[:1]), cov_q, torch.zeros_like(cov_q[:1])], dim=0)
        return t, mu_q, var_q, cov_q

    def get_interval(self, t, sorted_t=None, mu_q=None, var_q=None, get_var=True):
        # given t of shape (N, num_head), compute the interval t \in [t_l, t_r] for t_l, t_r \in self.time_q
        # first sort the inducing points, note that sorted_t has shape (M+2, num_head)
        # but mu_q and var_q, cov_q have shape (M+2, num_head, vdim)
        if sorted_t is None or mu_q is None or var_q is None:
            sorted_t, mu_q, var_q, cov_q = self.sort_inducing_points(add_endpoints=True)
        # now compute the interval
        ind_t_r = torch.searchsorted(sorted_t.permute(1, 0).contiguous(), t.permute(1, 0).contiguous(), right=False)
        ind_t_r = torch.clamp(ind_t_r, min=1, max=self.n_inducing+1).permute(1, 0).detach()
        ind_t_l = ind_t_r - 1
        dt_l = (t - sorted_t.gather(0, ind_t_l)).unsqueeze(-1)  # shape (N, num_head, 1)
        dt_r = (sorted_t.gather(0, ind_t_r) - t).unsqueeze(-1)

        theta_dt = -self.theta_p * (sorted_t[1:] - sorted_t[:-1]).unsqueeze(-1) # shape (M+1, num_head, vdim)
        var_dt = 1 - torch.exp(2*theta_dt) # 1 add, 1 mult, 1 exp        
        diff_mu_q = mu_q[1:] - torch.exp(theta_dt) * mu_q[:-1]
        ind_t_l = ind_t_l.unsqueeze(-1).tile((1, 1, self.vdim))
        diff_mu_q = diff_mu_q.gather(0, ind_t_l)    # shape (N, num_head, vdim)
        var_dt = var_dt.gather(0, ind_t_l)
        mu_q_l = mu_q.gather(0, ind_t_l)
        if get_var:
            var_q_l = var_q.gather(0, ind_t_l)
            ind_t_r = ind_t_r.unsqueeze(-1).tile((1, 1, self.vdim))
            var_q_r = var_q.gather(0, ind_t_r)
            cov_q_lr = cov_q.gather(0, ind_t_l) # share same indexing as var_q_l, see make_cov_matrix()
        else:
            var_q_l, var_q_r, cov_q_lr = None, None, None

        return dt_l, dt_r, mu_q_l, diff_mu_q, var_dt, var_q_l, var_q_r, cov_q_lr
    
    def log_prior(self, sorted_t=None, mu_q=None):
        if sorted_t is None or mu_q is None:
            sorted_t, mu_q, _, _ = self.sort_inducing_points(add_endpoints=False, get_var=False)
        if sorted_t.shape[0] > self.n_inducing:
            sorted_t = sorted_t[1:-1]
        if mu_q.shape[0] > self.n_inducing:
            mu_q = mu_q[1:-1]

        # log p(s_0 = mu_q[0]) for the first inducing variable
        # we assume initial dist p(s_0) has mean zero and variance self.var_scale
        # ignore the dim*log(2*pi) constant
        log_prior_0 = mu_q[0].pow(2) / self.var_scale + torch.log(self.var_scale)
        # p(s_t = mu_q[t] | s_{t-1} = mu_q[t-1]) for the rest of the inducing variables
        # mean and variance of p(s_r | s_l), time difference is dt = sorted_t[1:] - sorted_t[:-1]
        theta_dt = -self.theta_p * (sorted_t[1:] - sorted_t[:-1])
        mu_p_r = mu_q[:-1] * torch.exp(theta_dt)
        var_p_r = torch.clamp(self.var_scale * (1 - torch.exp(2*theta_dt)), min=EPS)
        # note the extra var_q_l / var_p_r term: taking expectation over q(s_l) with var_q_l
        log_prior_1 = (mu_q[1:] - mu_p_r).pow(2) / var_p_r + torch.log(var_p_r)
        return (log_prior_0.sum() + log_prior_1.sum()) * -0.5
    
    def kl_divergence(self, sorted_t=None, mu_q=None, var_q=None):
        if sorted_t is None or mu_q is None or var_q is None:
            sorted_t, mu_q, var_q, cov_q = self.sort_inducing_points(add_endpoints=False)
        if sorted_t.shape[0] > self.n_inducing:
            sorted_t = sorted_t[1:-1]
        if mu_q.shape[0] > self.n_inducing:
            mu_q = mu_q[1:-1]
        if var_q.shape[0] > self.n_inducing:
            var_q = var_q[1:-1]
        if cov_q.shape[0] > self.n_inducing:
            cov_q = cov_q[1:-1]

        # KL for the first inducing variable
        # mu_p = 0 because we assume initial dist p(s_0) has mean zero
        kl_0 = kl_gaussian(mu_q[0], 0, torch.log(torch.clamp(var_q[0], min=EPS)), self.log_var_p)
        # conditional KL for the rest of the inducing variables
        mu_q_r = mu_q[1:]
        # calculate the conditional variance of q(s_r | s_l)
        var_q_l, cov_q_lr = var_q[:-1], cov_q[:-1]
        tmp1 = cov_q_lr / torch.clamp(var_q_l, min=EPS)
        log_var_q_r = torch.log(torch.clamp(var_q[1:] - cov_q_lr * tmp1, min=EPS))
        # mean and variance of p(s_r | s_l), time difference is dt
        dt = (sorted_t[1:] - sorted_t[:-1]).unsqueeze(-1)
        tmp2 = torch.exp(-self.theta_p * dt)
        mu_p_r = mu_q[:-1] * tmp2
        var_p_r = torch.clamp(self.var_scale * (1 - tmp2.pow(2)), min=EPS)
        # note the extra var_q_l / var_p_r term: taking expectation over q(s_l) with var_q_l
        kl_1 = kl_gaussian(mu_q_r, mu_p_r, log_var_q_r, torch.log(var_p_r)) 
        kl_1 = kl_1 + var_q_l * (tmp1 - tmp2).pow(2)  / var_p_r
        
        return kl_0.sum() + kl_1.sum()

    def _comp_intermediate_terms(self, dt_l, dt_r, get_var_dt_r=True):
        theta_dt_l = -self.theta_p * dt_l
        theta_dt_r = -self.theta_p * dt_r
        var_dt_l = 1 - torch.exp(2 * theta_dt_l) # 1 add, 1 exp
        if get_var_dt_r:
            var_dt_r = 1 - torch.exp(2 * theta_dt_r) # 1 add, 1 exp
        else:
            var_dt_r = None
        return theta_dt_l, theta_dt_r, var_dt_l, var_dt_r

    def mean_marginal(self, t):
        dt_l, dt_r, mu_q_l, diff_mu_q, var_dt, _, _, _ = self.get_interval(t, get_var=False)
        theta_dt_l, theta_dt_r, var_dt_l, _ = self._comp_intermediate_terms(dt_l, dt_r, get_var_dt_r=False)
       
        scale_r = var_dt_l / var_dt
        mean = scale_r * torch.exp(theta_dt_r) * diff_mu_q + mu_q_l * torch.exp(theta_dt_l)

        # combine the sdes with linear projection
        return mean.flatten(start_dim=1) @ self.weight + self.bias

    def mean_and_var_marginal(self, t):
        # first check and reshape 
        dt_l, dt_r, mu_q_l, diff_mu_q, var_dt, var_q_l, var_q_r, cov_q_lr = self.get_interval(t, get_var=True)
        theta_dt_l, theta_dt_r, var_dt_l, var_dt_r = self._comp_intermediate_terms(dt_l, dt_r)
       
        scale_r = var_dt_l / var_dt
        gamma_r = scale_r * torch.exp(theta_dt_r)
        gamma_l = var_dt_r * torch.exp(theta_dt_l) / var_dt 
        
        mean = scale_r * torch.exp(theta_dt_r) * diff_mu_q + mu_q_l * torch.exp(theta_dt_l)
        var = gamma_l.pow(2) * var_q_l + gamma_r.pow(2) * var_q_r + 2 * gamma_l * gamma_r * cov_q_lr
        var = var + self.var_scale * var_dt_r * scale_r

        # combine the sdes with linear projection
        mean = mean.flatten(start_dim=1) @ self.weight + self.bias
        var = var.flatten(start_dim=1) @ self.weight.pow(2)

        return mean, torch.clamp(var, min=EPS)
    
    def sample_marginal(self, t):
        # sampling from the marginal distribution of q
        mean, var = self.mean_and_var_marginal(t)
        return mean + var.sqrt() * torch.randn_like(mean)
    
    # cumsum method (not implemented yet)
    def sample_smoothing_cumsum(self, t, sorted=False):
        raise NotImplementedError
    
    def forward(self, X, sample=True):
        assert X.shape[-1] == self.dim_input
        t = torch.clamp(self.embed_func(X.reshape(-1, self.dim_input)), min=self.t_min, max=self.t_max)
        if sample:
            if self.inference_mode == "marginal":
                out = self.sample_marginal(t)
            if self.inference_mode == "joint":
                out = self.sample_smoothing_cumsum(t)
        else:   # just use the mean
            out = self.mean_marginal(t)
        if len(X.shape) > 2:
            out = out.reshape(X.shape[:-1] + (self.dim_embed,))
        return out 

