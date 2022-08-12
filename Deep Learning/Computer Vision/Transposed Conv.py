#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 10 14:40:30 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Transposed Conv.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf

# Basic Operation
def trans_conv(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
'''
tensor([[0., 1.],
        [2., 3.]])
tensor([[0., 1.],
        [2., 3.]])
'''
trans_conv(X,K)
'''
tensor([[ 0.,  0.,  1.],
        [ 0.,  4.,  6.],
        [ 4., 12.,  9.]])

'''
## Using API
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K # kernel
tconv(X)

# Padding, Strides, and Multiple Channels
## Padding
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,padding=1,bias=False) # padding = 1
tconv.weight.data = K # kernel
tconv(X)
'''
tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
'''

## Stride
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,bias=False) # stride = 2
tconv.weight.data = K # kernel
tconv(X)
'''
tensor([[[[0., 0., 0., 1.],
          [0., 0., 2., 3.],
          [0., 2., 0., 3.],
          [4., 6., 6., 9.]]]], grad_fn=<ConvolutionBackward0>)
'''

## Multiple Channels
X = torch.rand((1,10,16,16))
conv = nn.LazyConv2d(20,kernel_size=5,padding=2,stride=3)
tconv = nn.LazyConvTranspose2d(10,kernel_size=5,padding=2,stride=3)
tconv(conv(X)).shape == X.shape

# Matrix Transposition
X = torch.arange(9.0).reshape(3,3)
K = torch.tensor([[1.0,2.0],[3.0,4.0]])
Y = sf.corr_2d(X,K) # matirc multiplication
Y
'''
tensor([[27., 37.],
        [57., 67.]])
'''
def kernel2matirx(K):
    k, W = torch.zeros(5), torch.zeros((4,9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matirx(K)
W
'''
tensor([[1., 2., 0., 3., 4., 0., 0., 0., 0.],
        [0., 1., 2., 0., 3., 4., 0., 0., 0.],
        [0., 0., 0., 1., 2., 0., 3., 4., 0.],
        [0., 0., 0., 0., 1., 2., 0., 3., 4.]])
'''
Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2)

Z = trans_conv(Y,K)
'''
tensor([[ 27.,  91.,  74.],
        [138., 400., 282.],
        [171., 429., 268.]])
'''
Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)