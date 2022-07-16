#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 15 13:31:03 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     ASF.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
from torch import nn
import SP_Func as sf
from torch.nn import functional as F

# Masked Softmax Operation
def masked_softmax(X,valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tenser, `valid_lens`: 1D or 2D tenser
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sf.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])) # 1-Dimensional tensor
masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])) # 2-Dimensional tensor

# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, querices, keys, values, valid_lens):
        # `queries`: (`batch_size`, no.of queries, 1, `num_hidden`),
        # `keys`: (`batch_size`, 1, no. of key-value pairs, `num_hidden`)
        querices, keys = self.W_q(querices), self.W_k(keys)
        features = querices.unsqueeze(2) + keys.unsqueeze(1)
        tanh_features = torch.tanh(features) # apple tanh activation function
        scores = self.w_v(tanh_features).squeeze(-1) # # (`batch_size`, no. of queries, no. of key-value pairs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        self.qs, self.ks, self.vs, self.vls = queries.shape, keys.shape, values.shape, valid_lens.shape
        # `values`: (`batch_size`, no. of key-value pairs, value dimension)
        return torch.matmul(self.dropout(self.attention_weights), values)

    def detail(self):
        print('Attention Weights:', self.attention_weights.shape)
        print('Queries Shape:', self.qs, '\nKeys Shape:', self.ks)
        print('Values Shape:', self.vs, '\nValid_lens Shape:', self.vls)

queries, keys = torch.normal(0,1,(2,1,20)), torch.ones((2,10,2))
values = torch.arange(40,dtype=torch.float32).reshape((1,10,4)).repeat(2,1,1)
valid_lens = torch.tensor([2,6])
attention = AdditiveAttention(key_size=2,query_size=20,num_hiddens=8,dropout=0.1)
attention.eval()
'''
AdditiveAttention(
  (W_q): Linear(in_features=20, out_features=8, bias=False)
  (W_k): Linear(in_features=2, out_features=8, bias=False)
  (w_v): Linear(in_features=8, out_features=1, bias=False)
  (dropout): Dropout(p=0.1, inplace=False)
)
'''
attention(queries,keys,values,valid_lens)
attention.detail() # print out the shape
sf.heat_map(attention.attention_weights.detach().reshape((2,10)),xlabel='Keys', ylabel='Queries')

# Scaled Dot-Product Attention
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
    # Shape of `valid_lens`: (`batch_size`, ) or (`batch_size`, no. of queries)
    def forward(self,queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores,valid_lens)
        self.qs, self.ks, self.vs, self.vls = queries.shape, keys.shape,values.shape,valid_lens.shape
        return torch.matmul(self.dropout(self.attention_weights), values)

    def detail(self):
        print('Attention Weights:',self.attention_weights.shape)
        print('Queries Shape:', self.qs, '\nKeys Shape:', self.ks)
        print('Values Shape:', self.vs, '\nValid_lens Shape:', self.vls)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
'''
DotProductAttention(
  (dropout): Dropout(p=0.5, inplace=False)
)

'''
attention(torch.normal(0, 1, (2, 1, 2)), torch.ones((2,10,2)),
          torch.arange(40,dtype=torch.float32).reshape((1,10,4)).repeat(2,1,1), torch.tensor([2,6]))
attention.detail() # print out the shape
sf.heat_map(attention.attention_weights.detach().reshape(2,10),xlabel='Keys',ylabel='Queries')