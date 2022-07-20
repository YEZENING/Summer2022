#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 18 15:18:18 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     SA-PE.py
# @Software: PyCharm
"""
# Import Packages
import torch
import matplotlib.pyplot as plt
from torch import nn
import SP_Func as sf

'''Self-Attention and Positional Encoding'''
# Self-Attention
num_hidden, num_head = 100, 5
attention = sf.MultiHeadAttention(num_hidden,num_hidden,num_hidden,num_hidden,num_head,0.5)
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
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hidden))
attention(X, X, X, valid_lens).shape # torch.Size([2, 4, 100])

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, dropout, max_lens=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # create long enough P
        self.P = torch.zeros((1, max_lens, num_hidden))
        up = torch.arange(max_lens, dtype=torch.float32).reshape(-1,1)
        down = torch.pow(10000, torch.arange(0, num_hidden, 2, dtype=torch.float32) / num_hidden)
        X = up/down
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self,X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

## Setup Parameters
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]

## Plot
fig, ax = plt.subplots(1,figsize=(7,3))
ax.plot(torch.arange(num_steps), P[0, :, 6:10].T[0], '-')
ax.plot(torch.arange(num_steps), P[0, :, 6:10].T[1], '--')
ax.plot(torch.arange(num_steps), P[0, :, 6:10].T[2], 'r:')
ax.plot(torch.arange(num_steps), P[0, :, 6:10].T[3], 'm--')
ax.legend(['Col %d' % d for d in torch.arange(6, 10)])
ax.set_xlabel('Row (position)')
plt.grid(True)
fig.tight_layout()
plt.show()

# Absolute Position Information
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
'''
0 in binary is 000
1 in binary is 001
2 in binary is 010
3 in binary is 011
4 in binary is 100
5 in binary is 101
6 in binary is 110
7 in binary is 111
'''
P = P[0, :]
sf.heat_map(P,xlabel='Column (encoding dimension)',ylabel='Row (position)',cmap='Blues')