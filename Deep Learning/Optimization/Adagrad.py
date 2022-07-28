#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 27 12:29:24 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Adagrad.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
import SP_Func as sf

'''f(x) = 0.1x1**2 + 2x2**2'''
def adagrad_2d(x1,x2,s1,s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2 # gradient of x1, x2
    s1 = s1 + g1**2
    s2 = s2 + g2**2
    # SGD
    x1 = x1 - eta / math.sqrt(s1 + eps) * g1
    x2 = x2 - eta / math.sqrt(s2 + eps) * g2
    return x1, x2 ,s1, s2

def f_2d(x1,x2):
    return 0.1 * x1 ** 2 + 2 * x2 **2

eta = 0.4
sf.gd_trace_2d(f_2d, sf.train_2d(adagrad_2d),'lr=0.4')

eta = 2
sf.gd_trace_2d(f_2d, sf.train_2d(adagrad_2d),'lr=2')

# Implementation
def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

data_iter, feature_dim = sf.read_airfoil(batch_size=10)
sf.train_mini(adagrad,init_adagrad_states(feature_dim),{'lr':0.1},data_iter,feature_dim)

# Concise Implementation
optimizer = torch.optim.Adagrad
sf.train_ch11(optimizer,{'lr': 0.1}, data_iter)