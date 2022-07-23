#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 20 12:55:35 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Opt-DL.py
# @Software: PyCharm
"""
# Import Packages
import torch
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * torch.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

# Plot the functions define above
x = torch.arange(0.5, 1.5, 0.01)
fig, ax = plt.subplots(1,figsize=(7,4))
ax.plot(x, f(x))
ax.plot(x, g(x),'--')
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('Risk')
ax.annotate('min of risk',(1.1, -1.05), (0.95, -0.5), arrowprops=dict(arrowstyle='->'))
ax.annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1), arrowprops=dict(arrowstyle='->'))
plt.show()

# Local Minima
x = torch.arange(-1.0, 2.0, 0.1)
fig, ax = plt.subplots(1,figsize=(7,4))
ax.plot(x, f(x))
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.annotate('local min',(-0.3, -0.25), (-0.77, -1.0), arrowprops=dict(arrowstyle='->'))
ax.annotate('global min', (1.1, -0.95), (0.6, 0.8), arrowprops=dict(arrowstyle='->'))
plt.show()

# Saddle point
x = torch.arange(-2.0,2.0, 0.01)
fig, ax = plt.subplots(1,figsize=(7,4))
ax.plot(x, x**3)
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.annotate('saddle point',(0,0), (-0.52, -5.0), arrowprops=dict(arrowstyle='->'))
plt.show()

## Higher dimension
x, y = torch.meshgrid(torch.linspace(-1.0,1.0,101), torch.linspace(-1.0,1.0,101))
z = x**2 - y**2
fig, ax = plt.subplots(1,figsize=(3.5,3.5),subplot_kw={'projection':'3d'},dpi=200) # 3D projection
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])
ax.set_zticks([-1,0,1])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# Vanishing gradients
x = torch.arange(-2.0, 5, 0.01)
fig, ax = plt.subplots(1,figsize=(7,4))
ax.plot(x, torch.tanh(x))
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.annotate('Vanishing gradients',(4,1),(2,0.0),arrowprops=dict(arrowstyle='->'))
plt.show()