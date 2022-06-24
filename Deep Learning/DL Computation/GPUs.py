#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 23 22:02:41 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     GPUs.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn

'''
This section may not work with current device(No GPU available
'''

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
torch.cuda.device_count()

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()

# Tensor and GPU
x = torch.tensor([1,2,3])
x.device

# Store on GPU (cannot store on GPU since it is not support current device's GPU)
X = torch.ones(2,3,device=try_gpu())
X

Y = torch.rand(2, 3, device=try_gpu(1))
Y

# Coping (When GPU available)
Z = X.cuda(1)
print(Z)
print(X)
Y+Z
Z.cuda(1) is Z

# Neural Network and GPUS
net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())
net(X)
net[0].weight.data.device