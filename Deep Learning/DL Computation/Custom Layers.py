#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 23 16:13:11 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Custom Layers.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from torch.nn import functional as F

# Without parameter
class CenterLayer(nn.Module):
    def __init__(self):
        super(CenterLayer, self).__init__()

    def forward(self,X):
        return X - X.mean()

layer = CenterLayer()
layer(torch.FloatTensor([1,2,3,4,5]))

# Incoporate complex network
net = nn.Sequential(
    nn.Linear(8,128),
    CenterLayer()
)

Y = net(torch.rand((2,8)))
Y.mean()

# With parameter
class Linear_Layer(nn.Module):
    def __init__(self, input,unit):
        super(Linear_Layer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input,unit))
        self.bias = nn.Parameter(torch.randn(unit,))

    def forward(self,X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = Linear_Layer(5,3)
linear.weight

linear(torch.rand(2,5))
param_net = nn.Sequential(
    Linear_Layer(64,8),
    Linear_Layer(8,1)
)

param_net(torch.rand(2,64))