#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 21 22:04:39 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     GD.py
# @Software: PyCharm
"""
# Import Packages
import numpy as np
import torch
import matplotlib.pyplot as plt

# 1D Gradient Decents
def f(x):
    'Object function'
    return x**2

def f_g(x):
    'Gradient(derivative) of object function'
    return 2*x

def gd(lr, f_g, detail=False):
    x = 10.0 # initial
    res = [x]
    for i in range(10):
        x = x - lr * f_g(x)
        res.append(float(x))
        if detail:
            print(f'epoch {i+1}: {x:f}')
    print(f'epoch 10, x: {x:f}')
    return res

result = gd(0.2, f_g) # lr=0.2

# Define trace function
def gd_trace(res, f, title=None):
    n = max(abs(min(res)), abs(max(res)))
    line = torch.arange(-n, n, 0.01)
    fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
    ax.plot(line, [f(x) for x in line])
    ax.plot(res, [f(x) for x in res],'-o')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    if title:
        ax.set_title(title)
    ax.grid(True)
    plt.show()

gd_trace(result,f,title='lr = 0.2')

## Learning Rate
gd_trace(gd(0.05, f_g),f, 'lr = 0.05')
gd_trace(gd(1.1, f_g),f, 'lr = 1.1')

## Local minima
c = torch.tensor(0.15 * np.pi)

def f(x):
    return x * torch.cos(c*x)

def f_g(x):
    return torch.cos(c * x) - c * x * torch.sin(c * x)

gd_trace(gd(2,f_g), f)

# Multivariate Gradient Decent
def f_2d(x1, x2):
    ''' Object Fucntion'''
    return x1**2 + 2 * x2**2

def f_2d_g(x1, x2):
    '''Derivative(Partital) of object function'''
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1,x2)
    return (x1 - lr * g1, x2 - lr * g2, 0, 0)

def train_2d(trainer, step=20, f_grad=None):
    x1, x2, s1, s2 = -5, -2, 0, 0
    res = [(x1,x2)]
    for i in range(step):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        res.append((x1,x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return res

def gd_trace_2d(f, res, title=None):
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1), torch.arange(-3.0, 1.0, 0.1))
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=200)
    ax.plot(*zip(*res), '-o', color='#ff7f0e')
    ax.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if title:
        ax.set_title(title)
    plt.show()

lr = 0.1 # Setting Learning Rate
gd_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_g),title='lr = 0.1')

# Adative Method
## Newton's Method
'''
consider a hyperbolic consine function f(x) = cosh(cx) for some constant c
'''
c = torch.tensor(0.5)

def f(x):
    'Object function'
    return torch.cosh(c * x)

def f_g(x):
    'Gradient of f(x)'
    return c * torch.sinh(c * x)

def f_hess(x):
    'Hessian matrix of f(x)'
    return c**2 * torch.cosh(c * x)

def newton(lr=1,detail=False):
    x = 10.0
    res = [x]
    for i in range(10):
        x = x - lr * f_g(x)/f_hess(x)
        res.append(float(x))
        if detail:
            print(f'epoch{i+1}: {x:f}')
    print(f'epoch 10, x: {x:f}')
    return res

gd_trace(newton(),f,title='lr = 1')

'''
consider a nonconvex function f(x) = xcos(cx)
'''
c = torch.tensor(np.pi * 0.15)

def f(x): # object function
    return x * torch.cos(c * x)

def f_g(x): # Gradient of f(x)
    return torch.cos(c * x) - x * c * torch.sin(c * x)

def f_hess(x): # hessian matrix of f(x)
    return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

gd_trace(newton(), f,title='lr=1')
gd_trace(newton(lr=0.5), f, 'lr=0.5') # reduce lr
