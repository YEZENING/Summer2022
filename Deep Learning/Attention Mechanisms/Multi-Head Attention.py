#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 18 13:56:56 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Multi-Head Attention.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
from torch import nn
import SP_Func as sf

# Define Multihead Attention Model
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden,
                 num_head, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_head = num_head
        self.attention = sf.DotProductAttention(dropout)
        # define learnable parameters
        self.W_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.W_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.W_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.W_o = nn.Linear(num_hidden, num_hidden, bias=bias)

    def forward(self,queries, keys, values, valid_lens):
        '''
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hidden`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_head`, no. of queries or key-value pairs, `num_hidden` / `num_heads`)
        '''
        queries = transpose_qkv(self.W_q(queries), self.num_head)
        keys = transpose_qkv(self.W_k(keys), self.num_head)
        values = transpose_qkv(self.W_v(values), self.num_head)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats=self.num_head, dim=0)

        # output shape: (`batch_size` * `num_heads`, no. of queries, num_hidden` / `num_head`)
        output = self.attention(queries, keys, values, valid_lens)
        output_conc = transpose_output(output, self.num_head)
        return self.W_o(output_conc)

# Define transposition function for parallel computation
def transpose_qkv(X, num_head):
    """Transposition for parallel computation of multiple attention heads."""
    # X input shape: (`batch_size`, no. of queries or key-value pairs, `num_hidden`)
    # X output shape: (batch_size, no. of queries or key-value pairs, `num_head`, `num_hidden` / `num_head`)
    X = X.reshape(X.shape[0], X.shape[1], num_head, -1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_head):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_head, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# Implementation
num_hidden, num_head = 100, 5
attention = MultiHeadAttention(num_hidden,num_hidden,num_hidden,num_hidden,num_head,dropout=0.5)
attention.eval()
'''
MultiHeadAttention(
  (attention): DotProductAttention(
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (W_q): Linear(in_features=100, out_features=100, bias=False)
  (W_k): Linear(in_features=100, out_features=100, bias=False)
  (W_v): Linear(in_features=100, out_features=100, bias=False)
  (W_o): Linear(in_features=100, out_features=100, bias=False)
)
'''
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hidden))
Y = torch.ones((batch_size, num_kvpairs, num_hidden))
attention(X, Y, Y, valid_lens).shape # torch.Size([2, 4, 100])