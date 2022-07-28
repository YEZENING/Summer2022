#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 27 14:53:36 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     RMSProp.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
import numpy as np
import SP_Func as sf
import matplotlib.pyplot as plt

# Visualization of gamma
gammas = [0.95, 0.9, 0.8, 0.7]
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
for gamma in gammas:
    x = np.arange(40)
    ax.plot((1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
ax.legend()
ax.set_xlabel('Time')
plt.show()

# Implementation
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
sf.gd_trace_2d(f_2d,sf.train_2d(rmsprop_2d))

## Apply into Deep Learning Network
def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] = p[:] - hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

data_iter, feature_dim = sf.read_airfoil(batch_size=10)
sf.train_mini(rmsprop,init_rmsprop_states(feature_dim),{'lr':0.01, 'gamma':0.9},data_iter,feature_dim)

# Concise Implementation
optimizer = torch.optim.RMSprop
sf.train_ch11(optimizer,{'lr':0.01, 'alpha':0.9},data_iter,use_lazy=True) # Using LazyLinear
sf.train_ch11(optimizer,{'lr':0.01, 'alpha': 0.9},data_iter)