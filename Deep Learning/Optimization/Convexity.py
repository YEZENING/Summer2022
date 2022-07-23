#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 20 15:41:20 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Convexity.py
# @Software: PyCharm
"""
# Import Packages
import torch
import numpy as np
import matplotlib.pyplot as plt

# Convex Function
f = lambda x: 0.5 * x**2 # convex
g = lambda x: torch.cos(np.pi * x) # non-convex
h = lambda x: torch.exp(0.5*x) # convex

x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(9,3),dpi=200)
ax1.plot(x,f(x))
ax1.plot(segment,f(segment),'--')
ax2.plot(x,g(x))
ax2.plot(segment,g(segment),'--')
ax3.plot(x,h(x))
ax3.plot(segment,h(segment),'--')
for ax in (ax1,ax2,ax3):
    ax.grid(True)
plt.show()

# Convex Function Properties
## Local minima are global minima
f = lambda x: (x-1)**2
fig, ax = plt.subplots(1,figsize=(9,3),dpi=200)
ax.plot(x,f(x))
ax.plot(segment,f(segment),'--')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.grid(True)
plt.show()