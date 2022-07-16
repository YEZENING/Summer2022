#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 14 13:10:01 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Attention Pooling.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf
import seaborn as sns
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Generate dataset
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5) # training input

# f(x) = y function
def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0,0.5,(n_train,)) # training output
x_test = torch.arange(0,5,0.1) # test dataset
y_true = f(x_test)
n_test = len(x_test) # 50
y_hat = torch.repeat_interleave(y_train.mean(),n_test) # using average pooling

# plot the result
def plot_kernel(y_hat):
    fig, ax = plt.subplots(1, figsize=(6.5, 4))
    ax.plot(x_test, y_true, label='True')
    ax.plot(x_test, y_hat, '--', label='Pred')
    ax.plot(x_test, y_train, 'o')
    ax.set_xlim([0, 5])
    ax.legend()
    ax.grid(True)
    plt.show()

plot_kernel(y_hat)

# Non-Parametric Attention Pooling
'''
Each element of `y_hat` is weighted average of values, where weights are attention weights
'''
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train)) # Shape: (`n_test`, `n_train`)
attention_weights = F.softmax(-(X_repeat - x_train)**2 / 2, dim=1) # Shape: (`n_test`, `n_train`)
y_hat = torch.matmul(attention_weights, y_train)

## plot
plot_kernel(y_hat)

## Heatmap
sf.heat_map(attention_weights,xlabel='Sorted training input',ylabel='Sorted testing input')

# Paramerts Attention Pooling（Batch matrix multiplication)
X = torch.ones((2,1,4))
Y = torch.ones((2,4,6))
'''
tensor([[[1., 1., 1., 1.]],

        [[1., 1., 1., 1.]]])
        
tensor([[[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.]],
         
        [[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.]]])

'''
torch.bmm(X,Y).shape # torch.matmul(X,Y).shape same thing

weights = torch.ones((2,10)) * 0.1
values = torch.arange(20.0).reshape((2,10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

## Define model (Nadaraya-Watson Kernel Regression)
class NWKernelReg(nn.Module):
    def __init__(self, **kwargs):
        super(NWKernelReg, self).__init__(**kwargs)
        self.w = nn.Parameter(torch.rand(1,),requires_grad=True) # replace torch.rand(1,) with torch.rand(n_train,)

    def forward(self,queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1])) # repeat times: keys.shape[1]
        self.attention_weights = F.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

## Training
X_tile = x_train.repeat((n_train, 1)) # Shape: (n_train, n_train)
Y_tile = y_train.repeat((n_train, 1)) # Shape: (n_train, n_train)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) # ('n_train', 'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1) ) # ('n_train', 'n_train'-1)

model = NWKernelReg()
loss = nn.MSELoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
animator = sf.Animator(xlabel='Epcoh',ylabel='Loss',xlim=[1,5])

for epoch in range(5):
    optimizer.zero_grad()
    l = loss(model(x_train,keys,values), y_train)
    l.sum().backward()
    optimizer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
animator.display()

keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = model(x_test, keys, values).unsqueeze(1).detach()

## plot
plot_kernel(y_hat)

## Heatmap
sf.heat_map(model.attention_weights.detach(), xlabel='Sorted training input',ylabel='Sorted testing input')

# Comparison of three ways
y_hat1 = torch.repeat_interleave(y_train.mean(),n_test)
y_hat2 = torch.matmul(attention_weights, y_train)
y_hat3 = model(x_test, keys, values).unsqueeze(1).detach()

## Training plot
fig, tg = plt.subplots(figsize=(6.5,4), dpi=200)
tg.plot(x_test, y_true, label='True')
tg.plot(x_test, y_train, 'o', label='y_train')
tg.plot(x_test, y_hat1, '--', label='y_hat1')
tg.plot(x_test, y_hat2, '-.', label='y_hat2')
tg.plot(x_test, y_hat3, '--m', label='y_hat3')
tg.grid(True)
tg.legend()
tg.set_xlim([0,5])
plt.show()

## Heaatmap
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,5),sharex=True, sharey=True, dpi=200)
sns.heatmap(ax=ax1,data=attention_weights,cmap='Reds')
sns.heatmap(ax=ax2, data=model.attention_weights.detach(), cmap='Reds')
for ax in (ax1, ax2):
    ax.set_xlabel('Sorted training inputs')
    ax.set_ylabel('Sorted testing inputs')
fig.tight_layout()
plt.show()