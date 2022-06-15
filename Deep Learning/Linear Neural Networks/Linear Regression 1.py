#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 10 15:41:51 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Linear Regression 1.py
# @Software: PyCharm
"""
# Import Packages
import torch
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Vectorlization for speed
n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer:
    # Record multiple time
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time() # initiate the timer

    def stop(self):
        self.times.append(time.time() - self.tik) # stop timer and record as a list
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'

## faster way
timer.start()
d = a + b
f'{timer.stop():.5f} sec'

# Normal Distribution
def normal(x, mu, sigma):
    p= 1/np.sqrt(2*np.pi*sigma**2)
    return p*np.exp(-0.5*((x-mu)/sigma)**2)

x = np.arange(-7,7,0.01)
params = [(0, 1), (0, 2), (3, 1)]
fig, ax = plt.subplots(figsize=(5,3))
for i in range(len(params)):
    ax.plot(x, [normal(x, mu, sigma) for mu, sigma in params][i])
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
ax.legend([f'mean {mu}, std {sigma}' for mu, sigma in params])
plt.show()
