"""
Bayesian DL via SDEs

ViT architectures with SDEs as building blocks for Bayesian neural networks

part of the code from https://github.com/chenw20/SGPA/blob/main/mlp.py

"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sde_layer import SDE_layer 
from models.sde_corr_sparse import SparseCorrSDE_layer
from models.multihead_sde import MultiheadSDE_layer
from models.mfvi import MFVILinear
from models.sgp_layer_vit import SGP_layer

EPS = 1e-5  # define a small constant for numerical stability control
BLOCK_TYPE = ['nn', 'mfvi', 'attention', 'sgpa', 'sde', 'sparsecorrsde', 'mhsde'] # supported blocks

# patch embedding needed for ViT
class Patch_embedding(nn.Module):
    def __init__(self, patch_size, in_channel, dim_emb, max_len, drop_rate=0, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Patch_embedding, self).__init__()
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.pos_emb = nn.Parameter(torch.randn(size=(max_len, dim_emb), **factory_kwargs)*0.1)  
        if drop_rate > 1 / dim_emb:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = nn.Identity()

        self.linear_proj = nn.Sequential(
            nn.Conv2d(in_channel, dim_emb, kernel_size=patch_size, stride=patch_size, **factory_kwargs)
            )
    
    def forward(self, x):  
        input_emb = self.linear_proj(x)
        b, e, h, w = input_emb.shape
        # from shape (b, e, h, w) to shape (b, h*w, e), note that max_len = h*w
        patch_emb = input_emb.reshape(b, e, -1).permute(0, 2, 1) + self.pos_emb
        return self.dropout(patch_emb)

# build a network with MLP or SDE blocks
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, drop_rate=0, device=None, dtype=None, norm=True, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.fn = fn
        if norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        if drop_rate > 1 / dim:
            self.dropout1 = nn.Dropout(drop_rate)
            self.dropout2 = nn.Dropout(drop_rate)
            self.dropout3 = nn.Dropout(drop_rate)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
            self.dropout3 = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(dim, dim, **factory_kwargs),
                                 self.dropout1, 
                                 nn.GELU(),
                                 nn.Linear(dim, dim, **factory_kwargs),
                                 self.dropout2)

    def forward(self, x):
        out = self.dropout3(self.fn(self.norm1(x))) + x
        out = self.mlp(self.norm2(out)) + x
        #out = self.dropout3(self.fn(self.norm1(x))) * x
        #out = self.mlp(self.norm2(out)) * out
        return out

class ClassificationHead(nn.Module):
    def __init__(self, dim_emb, num_class, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.fc = nn.Linear(dim_emb, num_class, **factory_kwargs)
        self.seqpool = nn.Linear(dim_emb, 1, **factory_kwargs)
        self.ln = nn.LayerNorm(dim_emb)

    def forward(self, x):  
        res = self.seqpool(x).permute(0,2,1) 
        res = torch.softmax(res, -1) 
        res = (res @ x).squeeze(1) 
        res = self.ln(res)
        res = self.fc(res) 
        return res

# SDE block
def SDE_block(n_channel, name='', device=None, dtype=None, **kwargs):
    factory_kwargs = {'device': device, 'dtype': dtype}
    net = nn.Sequential()
    sde = SDE_layer(input_shape=(n_channel,), device=device, dtype=dtype, 
                    linear_proj=nn.Linear(n_channel, n_channel, **factory_kwargs), **kwargs)
    net.add_module(f'SDE_Block{name}',
                   PreNormResidual(n_channel,
                                   nn.Sequential(nn.Linear(n_channel, n_channel, **factory_kwargs),
                                                 nn.Softplus(), 
                                                 sde),
                                   device=device, dtype=dtype, **kwargs))
    return net

# Sparse SDE block with correlated q
def SparseCorrSDE_block(n_channel, name='', device=None, dtype=None, **kwargs):
    factory_kwargs = {'device': device, 'dtype': dtype}
    net = nn.Sequential()
    sde = SparseCorrSDE_layer(dim_input=n_channel, dim_embed=n_channel, device=device, dtype=dtype, 
                              inducing_space='time', **kwargs)
    net.add_module(f'SparseCorrSDE_Block{name}',
                   PreNormResidual(n_channel,
                                   sde,
                                   device=device, dtype=dtype, **kwargs))
    return net

# multi-head SDE block
def MHSDE_block(n_channel, name='', device=None, dtype=None, **kwargs):
    net = nn.Sequential()
    factory_kwargs = {'device': device, 'dtype': dtype}
    num_heads = 4
    sde = MultiheadSDE_layer(dim_input=n_channel, dim_embed=n_channel, num_heads=num_heads, 
                             device=device, dtype=dtype, **kwargs)
    net.add_module(f'MultiheadSDE_Block{name}',
                   PreNormResidual(n_channel,
                                   sde,
                                   device=device, dtype=dtype, **kwargs))
    return net

# baseline: residual MLP_block
def MLP_block(n_channel, name='', device=None, dtype=None, **kwargs):
    factory_kwargs = {'device': device, 'dtype': dtype}
    net = nn.Sequential()
    net.add_module(f'MLP_Block{name}',
                   PreNormResidual(n_channel,
                                   nn.Sequential(nn.Linear(n_channel, n_channel, **factory_kwargs),
                                                 nn.Softplus(),
                                                 nn.Linear(n_channel, n_channel, **factory_kwargs)),
                                   device=device, dtype=dtype, **kwargs))
    return net

# baseline: residual BNN_block using MFVI
def MFVI_block(n_channel, name='', device=None, dtype=None, **kwargs):
    net = nn.Sequential()
    factory_kwargs = {'device': device, 'dtype': dtype}
    net.add_module(f'MFVI_Block{name}',
                   PreNormResidual(n_channel,
                                   nn.Sequential(MFVILinear(n_channel, n_channel, **factory_kwargs),
                                                 nn.Softplus(),
                                                 MFVILinear(n_channel, n_channel, **factory_kwargs)),
                                   device=device, dtype=dtype, **kwargs))
    return net

# baseline: residual multihead self-attention block
class MHA_layer(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True, device=dtype, dtype=dtype)

    def forward(self, x):
        return super().forward(x, x, x, need_weights=False)[0]

def MHA_block(n_channel, name='', device=None, dtype=None, **kwargs):
    net = nn.Sequential()
    factory_kwargs = {'device': device, 'dtype': dtype}
    num_heads = 4
    mha = MHA_layer(n_channel, num_heads, dropout=0.0, device=device, dtype=dtype)
    net.add_module(f'MultiheadSelfAttention_Block{name}',
                   PreNormResidual(n_channel,
                                   mha,
                                   device=device, dtype=dtype, **kwargs))
    return net

# baseline: residual SGPA block
def SGPA_block(n_channel, name='', device=None, dtype=None, **kwargs):
    net = nn.Sequential()
    factory_kwargs = {'device': device, 'dtype': dtype}
    num_heads = 4   # according to Chen and Li ICLR 2023
    max_len = 64    # only works for CIFAR images with patch size 4x4 
    sgpa = SGP_layer(num_heads, max_len, hdim=n_channel, kernel_type='exponential', sample_size=1, jitter=1e-5,
                     keys_len=8, device=device, dtype=dtype, **kwargs)
    net.add_module(f'SGPA_Block{name}',
                   PreNormResidual(n_channel,
                                   sgpa,
                                   **factory_kwargs))
    return net

def construct_net(input_shape, patch_size, dim_emb, n_block, dim_out, block_type='nn', **layer_kwargs):
    net = []
    # first do projection per batch, patch_size = (h_patch, w_patch)
    in_channel, h, w = input_shape
    n_patch = int(np.floor(h / patch_size[0]) * np.floor(w / patch_size[1]))
    # embedding have output (N, n_patches, dim_emb)
    net.append(Patch_embedding(patch_size, in_channel, dim_emb, n_patch, **layer_kwargs))
    # then add in the blocks
    assert block_type in BLOCK_TYPE
    for i in range(1, n_block + 1):
        if block_type == 'nn':
            net.append(*[MLP_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'mfvi':
            net.append(*[MFVI_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'attention':
            net.append(*[MHA_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'sgpa':
            net.append(*[SGPA_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'sde':
            net.append(*[SDE_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'sparsesde':
            net.append(*[SparseSDE_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'sparsecorrsde':
            net.append(*[SparseCorrSDE_block(dim_emb, name=str(i), **layer_kwargs)])
        if block_type == 'mhsde':
            net.append(*[MHSDE_block(dim_emb, name=str(i), **layer_kwargs)])
    # finally mix the patch features
    # mix along time and feature dims seperately 
    net.append(ClassificationHead(dim_emb, dim_out, **layer_kwargs))
    return nn.Sequential(*net)

