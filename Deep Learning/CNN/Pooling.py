#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 27 15:24:24 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Pooling.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn

# Pooling
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode =='avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
'''
tensor([[0., 1., 2.],
        [3., 4., 5.],
        [6., 7., 8.]])
'''
pool2d(X,(2,2),mode='max')
'''
tensor([[4., 5.],
        [7., 8.]])
'''
pool2d(X,(2,2),mode='avg')
'''
tensor([[2., 3.],
        [5., 6.]])
'''

# Padding and stride
X = torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
X
'''
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
'''
pool2d = nn.MaxPool2d(3) # pool dim = 3x3
pool2d(X) # 10

pool2d = nn.MaxPool2d(3,padding=1,stride=2)
pool2d(X)
'''
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
'''

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1)) # stride height = 2, wideth = 3, padding height = 0, width = 1
pool2d(X)
'''
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
'''

# Multiple channels
X = torch.cat((X, X+1, X+2),1)
X
'''
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],
         [[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]],
         [[ 2.,  3.,  4.,  5.],
          [ 6.,  7.,  8.,  9.],
          [10., 11., 12., 13.],
          [14., 15., 16., 17.]]]])
'''
pool2d = nn.MaxPool2d(3,stride=2,padding=1)
pool2d(X)
'''
tensor([[[[ 5.,  7.],
          [13., 15.]],
         [[ 6.,  8.],
          [14., 16.]],
         [[ 7.,  9.],
          [15., 17.]]]])
'''