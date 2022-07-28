#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 25 13:05:18 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     M-SGD.py
# @Software: PyCharm
"""
# Import Packages
import numpy as np
import torch
import time
from torch import nn
import SP_Func as sf
import matplotlib.pyplot as plt

# Vectorization
A = torch.zeros(256,256)
B = torch.randn(256,256)
C = torch.randn(256,256)

## Compute A = BC in Element-wise assignment
time_start = time.time()
for i in range(256):
    for j in range(256):
        A[i,j] = torch.dot(B[i,:],C[:,j])
time_stop = time.time()
print(f'Estimate Time: {time_stop - time_start}')
'''Estimate Time: 0.7650740146636963'''

## Compute A = BC in Column-wise assigment (Faster)
time_start = time.time()
for j in range(256):
    A[:,j] = torch.mv(B, C[:,j])
time_stop = time.time()
print(f'Estimate Time: {time_stop - time_start}')
'''Estimate Time: 0.006410121917724609'''

## Compute A = BC in one (Fastest)
time_start = time.time()
A = torch.matmul(B,C)
time_stop = time.time()
print(f'Estimate Time: {time_stop - time_start}')
'''Estimate Time: 0.0007228851318359375'''

# Minibatch
time_start = time.time()
for j in range(0, 256, 64):
    A[:,j:j+64] = torch.mm(B, C[:,j:j+64])
time_stop = time.time()
print(f'Estimate Time: {time_stop - time_start}')
'''Estimate Time: 0.003762960433959961'''


#####################################################################################
# Read data air foil selfnoise
def read_airfoil(batch_size=10, n=1500): # Sample size = 1500
    data = np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat',
                      unpack=False, dtype=np.float32)
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = sf.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_shuffle=True)
    return data_iter, data.shape[1]-1

## Define optimizer
def sgd(params, state, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()

## Trainig loop
def train_mini(optimizer, state, hyperparams, data_iter, feature_dim, num_epoch=2):
    # Initialization
    w = torch.normal(0.0, 0.01, size=(feature_dim,1),requires_grad=True)
    b = torch.zeros((1),requires_grad=True)
    model, loss = lambda X: (torch.matmul(X,w) + b), lambda y_hat, y: (0.5*(y_hat - y.reshape(y_hat.shape))**2)

    # Training
    animator = sf.Animator(xlabel='epoch', ylabel='loss',xlim=[0, num_epoch], ylim=[0.22, 0.35])
    n, timer = 0, sf.Timer()
    for _ in range(num_epoch):
        for X, y in data_iter:
            l = loss(model(X),y).mean() # loss
            l.backward() # BP
            optimizer([w,b], state, hyperparams)
            n = n + X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),(sf.evaluate_loss(model,data_iter,loss),))
                timer.start()
    animator.display()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_sgd(lr, batch_size, num_epoch=2):
    data_iter, feature_dim = read_airfoil(batch_size)
    return train_mini(sgd, None, {'lr':lr}, data_iter, feature_dim, num_epoch)

## Gradient Decent with batch_size = 1500
gd_res = train_sgd(1, 1500, 10)

## Stochastic Gradient Decent with batch_size = 1
sgd_res = train_sgd(0.005,1)

## Minibatch stochastic gradient descent with batch_size = 100 and 10
mini1_res = train_sgd(.4, 100)
mini2_res = train_sgd(.05, 10) # Execute more sample in one batch

## plot the result of three ways
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
ax.plot(*gd_res,label='gd')
ax.plot(*sgd_res, '--m,', label='sgd')
ax.plot(*mini1_res, '-.',label='batch_size = 100')
ax.plot(*mini2_res, '--',label='batch_size = 10')
ax.set_xlim([1e-2, 10])
ax.set_xlabel('time (sec)')
ax.set_ylabel('loss')
ax.set_xscale('log')
ax.grid('True')
ax.legend()
plt.show()

# Concise Implementation
def train_ch11(opt, hyperparams, data_iter, num_epoch=4):
    '''
    :param opt: Optimizer function
    :param hyperparams: NA
    :param data_iter: dataloader
    :param num_epoch: training epoch
    '''
    # Initialization
    model = nn.Sequential(nn.LazyLinear(1)) # using fully connected layer
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    model.apply(init_weights)
    optimizer = opt(model.parameters(),**hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = sf.Animator(xlabel='epoch', ylabel='loss',xlim=[0, num_epoch], ylim=[0.22, 0.35])
    n, timer = 0, sf.Timer()
    for _ in range(num_epoch):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = model(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n = n + X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n / X.shape[0] / len(data_iter), (sf.evaluate_loss(model,data_iter, loss) / 2,))
                timer.start()
    animator.display()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')

data_iter,_ = read_airfoil(10)
optimizer = torch.optim.SGD
train_ch11(optimizer,{'lr':0.01},data_iter)