import math
import random
import os
import csv

import tensorflow as tf
import torch as t
import torch.nn as nn
import numpy as np
import tqdm.auto as tqdm
import matplotlib.pyplot as plt

from einops import reduce, rearrange, repeat
from torch import einsum

'''
Layer Normalization
'''
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        # calculate the standard deviation of each row
        residual = residual - reduce(residual, "b p d -> b p 1", "mean") # The 1 preserves the dimension?

        # divide by the square of each row + a small value, then find the square root
        scale = (reduce(residual.pow(2), "b p d -> b p 1", "mean") + self.cfg.ln_eps).sqrt()
        normalized = residual / scale

        return normalized


'''
Multi-Layer Perceptron Layer

Simple AF
'''
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)

        nn.init.normal_(self.fc1.weight, std=self.cfg.init_range)
        nn.init.normal_(self.fc2.weight, std=self.cfg.init_range)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


'''
Encoder Attention, Uncached
'''
class EncoderAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Attention Weights
        self.Qs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Ks = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Vs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.Qbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Kbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Ob = nn.Parameter(t.zeros(cfg.d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=self.cfg.init_range)
        nn.init.normal_(self.Ks, std=self.cfg.init_range)
        nn.init.normal_(self.Vs, std=self.cfg.init_range)
        nn.init.normal_(self.O, std=self.cfg.init_range)



    def forward(self, normalized_resid_pre):
        # Calculate query, key and value vectors
        q = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Ks) + self.Kbs
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Calculate attention scores, then scale, and apply softmax
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores = attn_scores / np.sqrt(self.cfg.d_head)
        attn_pattern = attn_scores.softmax(-1)

        z = einsum("b K h k, b h Q K -> b Q h k", v, attn_pattern)

        # Apply another transformation to convert back to the right dimensions
        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out


'''
Decoder Attention for Pairing with an Encoder, Uncached
'''
class DecoderAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_head: int, init_range: float):
        super().__init__()

        self.sqrt_d_head = np.sqrt(d_head)

        # Query Weights
        self.Qs = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Qbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Key Weights
        self.Ks = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Kbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Value Weights
        self.Vs = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Vbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Transformation Matrix
        self.O = nn.Parameter(t.empty((n_heads, d_head, d_model)))
        self.Ob = nn.Parameter(t.zeros(d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=init_range)
        nn.init.normal_(self.Ks, std=init_range)
        nn.init.normal_(self.Vs, std=init_range)
        nn.init.normal_(self.O, std=init_range)

        # This is for cuda management
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))


    def forward(self, normalized_resid_pre, encoder_output):
        # Calculate query, key matrices from the encoder output
        q = einsum("b p d, h d k -> b p h k", encoder_output, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", encoder_output, self.Ks) + self.Kbs

        # Calculate the value matrices from the normalized residual stream
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Form the visually appealing attention grid
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.sqrt_d_head)
        attn_scores = attn_scores_masked.softmax(-1)

        z = einsum("b K h k, b h Q K -> b Q h k", v, self.attn_cache)

        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


'''
Decoder Attention, Uncached
'''
class DecoderOnlyAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_head: int, init_range: float):
        super().__init__()

        self.sqrt_d_head = np.sqrt(d_head)

        # Query Weights
        self.Qs = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Qbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Key Weights
        self.Ks = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Kbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Value Weights
        self.Vs = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Vbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Transformation Matrix
        self.O = nn.Parameter(t.empty((n_heads, d_head, d_model)))
        self.Ob = nn.Parameter(t.zeros(d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=init_range)
        nn.init.normal_(self.Ks, std=init_range)
        nn.init.normal_(self.Vs, std=init_range)
        nn.init.normal_(self.O, std=init_range)

        # This is for cuda management
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))


    def forward(self, normalized_resid_pre):
        # Calculate query, key matrices from the encoder output
        q = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Ks) + self.Kbs

        # Calculate the value matrices from the normalized residual stream
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Form the visually appealing attention grid
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.sqrt_d_head)
        attn_pattern = attn_scores_masked.softmax(-1)

        z = einsum("b K h k, b h Q K -> b Q h k", v, attn_pattern)

        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


'''
Decoder Attention, Cached
'''
class DecoderOnlyAttentionCached(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_head: int, init_range: float):
        super().__init__()

        self.sqrt_d_head = np.sqrt(d_head)

        # Query Weights
        self.Qs = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Qbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Key Weights
        self.Ks = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Kbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Value Weights
        self.Vs = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.Vbs = nn.Parameter(t.zeros((n_heads, d_head)))

        # Transformation Matrix
        self.O = nn.Parameter(t.empty((n_heads, d_head, d_model)))
        self.Ob = nn.Parameter(t.zeros(d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=init_range)
        nn.init.normal_(self.Ks, std=init_range)
        nn.init.normal_(self.Vs, std=init_range)
        nn.init.normal_(self.O, std=init_range)

        # This is for cuda management
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

        # Cache for the attention grid
        self.attn_cache = None


    def forward(self, normalized_resid_pre):
        # Calculate query, key matrices from the encoder output
        q = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Ks) + self.Kbs

        # Calculate the value matrices from the normalized residual stream
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Form the visually appealing attention grid
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / np.sqrt(self.cfg.d_head))
        attn_scores = attn_scores_masked.softmax(-1)

        self.attn_cache = attn_scores

        z = einsum("b K h k, b h Q K -> b Q h k", v, self.attn_cache)

        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


    def get_attn_cache(self):
        return self.attn_cache
