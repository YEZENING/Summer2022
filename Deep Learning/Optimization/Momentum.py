#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 26 14:41:11 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Momentum.py
# @Software: PyCharm
"""
# Import Packages
import torch
import numpy as np
import SP_Func as sf
import matplotlib.pyplot as plt

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def f_g_2d(x1, x2, s1, s2): # gradient decent x - x'*lr
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

sf.gd_trace_2d(f_2d,sf.train_2d(f_g_2d))

eta = 0.6
sf.gd_trace_2d(f_2d,sf.train_2d(f_g_2d))

# Leaky average
def momentum_2d(x1,x2,v1,v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
sf.gd_trace_2d(f_2d, sf.train_2d(momentum_2d))

eta, beta = 0.6, 0.25
sf.gd_trace_2d(f_2d, sf.train_2d(momentum_2d))

# Effective Sample Weight
betas = [0.95, 0.9, 0.6, 0.0]
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
for beta in betas:
    x = np.arange(40)
    ax.plot(beta ** x, label=f'beta = {beta:.2f}')
ax.legend()
ax.set_xlabel('Time')
plt.show()

# Implementation
def init_momentum_state(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w,v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] = p[:] - hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epoch=2):
    sf.train_mini(sgd_momentum,init_momentum_state(feature_dim),
                  {'lr':lr, 'momentum':momentum},data_iter,feature_dim,num_epoch)

data_iter, feature_dim = sf.read_airfoil(batch_size=10)
train_momentum(lr=0.02,momentum=0.5)
train_momentum(lr=0.01,momentum=0.9)
train_momentum(lr=0.005,momentum=0.9)

# Concise Implementation
optimizer = torch.optim.SGD
sf.train_ch11(optimizer,{'lr':0.005, 'momentum':0.9},data_iter)

# Scaler Functions
lambdas = [0.1, 1, 10, 19]
eta = 0.1
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
for lam in lambdas:
    t = np.arange(20)
    ax.plot((1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
ax.legend()
ax.set_xlabel('Time')
plt.show()