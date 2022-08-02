#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 29 14:05:26 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Asynchronous Computation.py
# @Software: PyCharm
"""
# Import Packages
import os
import torch
import subprocess
import numpy as np
from torch import nn
import SP_Func as sf

# compare numpy with pytorch
device = sf.try_gpu() # only cpu
a = torch.rand(size=(1000,1000),device=device)
b = torch.mm(a,a)

with sf.Benchmark('Numpy'):
    for _ in range(10):
        a = np.random.normal((1000,1000))
        b = np.dot(a,a)

with sf.Benchmark('PyTorch'):
    for _ in range(10):
        a = torch.rand(size=(1000,1000),device=device)
        b = torch.mm(a,a)

## The following code cannot be run since only CPUs are available in MacOS.
with sf.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)

