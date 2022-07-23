#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 22 16:17:06 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     SGD.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
import SP_Func as sf

def f(x1,x2):
    return x1 ** 2 + 2 * x2 **2

def f_g(x1,x2):
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1,x2)
    # stimulate noise
    g1 = g1 + torch.normal(0.0, 1, (1,))
    g2 = g2 + torch.normal(0.0, 1, (1,))
    eta_t = eta * lr
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

eta = 0.1
lr = constant_lr()
sf.gd_trace_2d(f,sf.train_2d(sgd,step=50,f_grad=f_g))

# Dynamic learning rate
## Exponential learning rate decay
def exponential_lr():
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr()
sf.gd_trace_2d(f, sf.train_2d(sgd, step=50, f_grad=f_g))

## Polynomial learning rate decay
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t = t + 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr()
sf.gd_trace_2d(f, sf.train_2d(sgd, step=50, f_grad=f_g))