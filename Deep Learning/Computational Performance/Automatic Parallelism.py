#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 29 15:00:01 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Automatic Parallelism.py
# @Software: PyCharm
"""
# Import Packages
import torch
import SP_Func as sf

'''The following code need to run with 2 GPUs'''
# initiate GPUs
sf.num_gpus()
devices = sf.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])

# Parallel Computation on GPUs
## Single Running
run(x_gpu1)
run(x_gpu2)
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with sf.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with sf.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])

## Multiple running
with sf.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()

# Parallel Computation and Communication
## Copy data from GPU to CPU
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with sf.Benchmark('Run on GPU'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with sf.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()

with sf.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()