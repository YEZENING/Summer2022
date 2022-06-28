#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 27 13:45:40 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Mutiple Input and Output.py
# @Software: PyCharm
"""
# Import Packages
import torch
from SP_Func import corr_2d

# Multiple input channels
def corr2d_multi_in(X, K):
    # X as input with multi dimention, K as kenel, return the sum of convolution
    return sum(corr_2d(x,k) for x, k in zip(X,K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]) # 3x3x2
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]) # 2x2x2

corr2d_multi_in(X,K)

# Multiple output channels
def corr2d_muti_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

K = torch.stack((K,K+1,K+2),0)
K.shape
corr2d_muti_in_out(X,K)

# 1x1 Convolutional layer
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully-connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X_1x1 = torch.normal(0,1,(3,3,3))
K_1x1 = torch.normal(0,1,(2,3,1,1))

Y1 = corr2d_multi_in_out_1x1(X_1x1,K_1x1)
Y2 = corr2d_muti_in_out(X_1x1,K_1x1)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6