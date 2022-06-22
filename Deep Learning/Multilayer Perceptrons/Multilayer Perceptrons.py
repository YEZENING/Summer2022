#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 15 22:13:24 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Multilayer Perceptrons.py
# @Software: PyCharm
"""
import torch
import matplotlib.pyplot as plt

# Activation function
## ReLU
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plt.plot(x.detach(), y.detach())
plt.xlabel('x'), plt.ylabel('ReLu(x)')
plt.grid(True)
plt.show()

## Derivative of ReLU
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad)
plt.xlabel('x'), plt.ylabel('grad of relu')
plt.grid(True)
plt.show()

## Sigmoid function
y = torch.sigmoid(x)
plt.plot(x.detach(), y.detach())
plt.xlabel('x'), plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()

## Derivative of sigmoid
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad)
plt.xlabel('x'), plt.ylabel('grad of sigmoid')
plt.grid(True)
plt.show()

## tanh function
y = torch.tanh(x)
plt.plot(x.detach(), y.detach())
plt.xlabel('x'), plt.ylabel('tanh(x)')
plt.grid(True)
plt.show()

## Derivative of tanh
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad)
plt.xlabel('x'), plt.ylabel('grad of tanh')
plt.grid(True)
plt.show()