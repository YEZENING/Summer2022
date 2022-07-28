#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 27 20:07:21 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Adam.py
# @Software: PyCharm
"""
# Import Packages
import torch
import SP_Func as sf

# Implementation
def init_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v,s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1-beta1) * p.grad
            s[:] = beta2 * s + (1-beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] = p - hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] = hyperparams['t'] + 1

data_iter, feature_dim = sf.read_airfoil(batch_size=10)
sf.train_mini(adam, init_states(feature_dim), {'lr':0.01, 't':1}, data_iter, feature_dim)

# Concise Implementation
optimizer = torch.optim.Adam
sf.train_ch11(optimizer,{'lr':0.01},data_iter)
sf.train_ch11(optimizer,{'lr':0.01},data_iter,use_lazy=True)

# Yogi
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) *p.grad
            s[:] = s + (1 - beta2) * torch.sign(torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] = p[:] - hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps) # g'
            p.grad.data.zero_()
    hyperparams['t'] = hyperparams['t'] + 1

data_iter, feature_dim = sf.read_airfoil(batch_size=10)
sf.train_mini(yogi, init_states(feature_dim), {'lr':0.01, 't':1}, data_iter, feature_dim)