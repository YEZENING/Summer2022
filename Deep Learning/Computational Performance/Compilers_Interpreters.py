#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 29 12:28:09 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Compilers_Interpreters.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf

# Imperative Programming
def add(a,b):
    return a + b

def facny_fuc(a,b,c,d):
    e = add(a,b)
    f = add(c,d)
    g = add(e,f)
    return g

print(facny_fuc(1,2,3,4))

# Symbolic Programming
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)

# Hybrid Programming
def net():
    model = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(2)
    )
    return model

model = net()
x = torch.randn((1,512))
model(x)

model = torch.jit.script(model) # convert model into MLP
model(x)

# Acceleration by Hybridization
class Benchmark:
    def __init__(self, description='none'):
        self.description = description

    def __enter__(self):
        self.timer = sf.Timer()
        return self

    def __exit__(self,*args):
        print(f'{self.description}:{self.timer.stop():4f} sec')

model = net()
with Benchmark('Without torchscript'):
    for i in range(1000):
        model(x)
'''
Without torchscript:0.111906 sec
'''

model = torch.jit.script(model)
with Benchmark('With torchscript'):
    for i in range(1000):
        model(x)
'''
With torchscript:0.087516 sec
'''

# Serialization
model.save('Computational Performance/my_mlp')
'''
# Run the following code in terminal:
!ls -lh Computational\ Performance/my_mlp*
'''