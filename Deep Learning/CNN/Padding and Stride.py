#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 24 16:02:06 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Padding and Stride.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn

# Padding
def comp_conv2d(conv2d,X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1,1,padding=1,kernel_size=3)
X = torch.rand(size=(8,8))
comp_conv2d(conv2d,X)
'''
tensor([[-0.1865, -0.3649, -0.3042, -0.2420, -0.4248, -0.2992, -0.4012, -0.1928],
        [-0.5375, -0.8100, -0.6750, -0.7128, -0.7231, -0.4193, -0.7189, -0.5300],
        [-0.3664, -0.5214, -0.4683, -0.6642, -0.3733, -0.3895, -0.4742, -0.2595],
        [-0.6505, -0.5401, -0.7547, -0.6992, -0.5451, -0.7282, -0.8709, -0.5771],
        [-0.5297, -0.6303, -0.7629, -0.4618, -0.5997, -0.3656, -0.5131, -0.4754],
        [-0.3077, -0.5019, -0.7508, -0.6377, -0.8487, -0.6249, -0.5536, -0.4076],
        [-0.6633, -0.6576, -0.4145, -0.4713, -0.4479, -0.6938, -0.3029, -0.5654],
        [-0.5694, -0.4434, -0.3918, -0.4436, -0.3699, -0.4010, -0.5340, -0.5151]],
       grad_fn=<ReshapeAliasBackward0>)
'''
comp_conv2d(conv2d,X).shape

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X)
'''
tensor([[ 0.0977, -0.4114, -0.3568, -0.2158, -0.3541, -0.1557, -0.3618,  0.0106],
        [ 0.0434, -0.3755, -0.3409, -0.3439, -0.4538, -0.0825, -0.2783,  0.0635],
        [ 0.2426, -0.5000, -0.4186, -0.2950, -0.7037, -0.2700, -0.5255,  0.1077],
        [-0.1773, -0.6819, -0.3096, -0.2702, -0.2487, -0.2932, -0.6991,  0.0367],
        [-0.2230, -0.7148, -0.5513, -0.4589, -0.3281, -0.2450, -0.7138, -0.1496],
        [-0.1075, -0.5628, -0.6069, -0.4421, -0.3432, -0.0968, -0.4274, -0.0905],
        [ 0.0687, -0.0776, -0.3145, -0.0110, -0.2471,  0.0534, -0.1922,  0.1905],
        [-0.0850, -0.0833, -0.0892, -0.0173, -0.0484, -0.1570,  0.1449, -0.0645]],
       grad_fn=<ReshapeAliasBackward0>)
'''
comp_conv2d(conv2d, X).shape

# Stride
conv2d_strid = nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
comp_conv2d(conv2d_strid,X)
'''
tensor([[ 0.0703,  0.0357,  0.2855,  0.1704],
        [ 0.2395,  0.0124, -0.1063,  0.1526],
        [ 0.2036,  0.3797,  0.3262,  0.0611],
        [ 0.2783, -0.1539,  0.1523, -0.2573]], grad_fn=<ReshapeAliasBackward0>)
'''
comp_conv2d(conv2d_strid,X).shape

## Complex setting
conv2d_strid2 = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d_strid2, X)
'''
tensor([[-0.5572, -0.6387],
        [-0.3015, -0.4446]], grad_fn=<ReshapeAliasBackward0>)
'''
comp_conv2d(conv2d_strid2, X).shape