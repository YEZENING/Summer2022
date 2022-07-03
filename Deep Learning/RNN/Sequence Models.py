#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 01 16:37:33 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Sequence Models.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import matplotlib.pyplot as plt
from SP_Func import load_array, evaluate_loss

# Generate data
T = 1000 # generate 1000 data point
time = torch.arange(1,T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0,0.2,(T,))
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
ax.plot(time,x)
ax.grid(True)
ax.set_xlim([0,1000])
ax.set_xlabel('time')
ax.set_ylabel('x')
plt.show()

# Transfer to feature and label
tau = 4
features = torch.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

# Define batch size and training time
batch_size, n_train = 16, 600
train_iter = load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_shuffle=True)

# Define model
## initialization
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

model = nn.Sequential(
    nn.Linear(4,10),
    nn.ReLU(),
    nn.Linear(10,1)
)
model.apply(init_weights)

# Define training loop
def train(model,train_data,lr,num_epochs):
    loss = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_data:
            optimizer.zero_grad()
            l = loss(model(X),y)
            l.sum().backward()
            optimizer.step()
        print(f'epoch {epoch + 1}', f'loss: {evaluate_loss(model, train_data, loss):f}')

train(model,train_data=train_iter,lr=0.01,num_epochs=5)

# One Step Prediction
pred1 = model(features)
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
ax.plot(time,x,label='origin')
ax.plot(time[tau:],pred1.detach().numpy(),label='pred1')
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.legend()
ax.grid(True)
ax.set_xlim([0,1000])
plt.show()

# Multi-step Prediction
multi_pred = torch.zeros(T)
multi_pred[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multi_pred[i] = model(multi_pred[i - tau:i].reshape((1, -1)))

## plot
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
ax.plot(time,x,label='origin')
ax.plot(time[tau:],pred1.detach().numpy(),label='pred1')
ax.plot(time[n_train + tau:], multi_pred[n_train + tau:].detach().numpy(),label='muti_pred')
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.legend()
ax.grid(True)
ax.set_xlim([0,1000])
plt.show()

# k steps prediction k = 1,4,16,64
max_step = 64
features = torch.zeros((T- max_step - tau + 1, tau + max_step))

# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_step + 1]

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau+max_step):
    features[:, i] = model(features[:, i - tau:i]).reshape(-1)

# steps = (1,4,16,64)
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
# step = 1
ax.plot(time[tau + 1 - 1: T - max_step + 1],
        features[:, (tau + 1 - 1)].detach().numpy(),label='pred_step=1')
# step = 4
ax.plot(time[tau + 4 - 1: T - max_step + 4],
        features[:, (tau + 4 - 1)].detach().numpy(),label='pred_step=4')
# step = 16
ax.plot(time[tau + 16 - 1: T - max_step + 16],
        features[:, (tau + 16 - 1)].detach().numpy(),label='pred_step=16')
# step = 64
ax.plot(time[tau + 64 - 1: T - max_step + 64],
        features[:, (tau + 64 - 1)].detach().numpy(),'--',label='pred_step=64')
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.legend()
ax.grid(True)
ax.set_xlim([0,1000])
plt.show()


# try to apply for loop into plt function, however it is not the same type of plots show below.
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
for i in (1,4,16,64):
    for j in ('-', 'm--', 'g-.', 'r:'):
        ax.plot(time[tau + i - 1: T - max_step + i],
                features[:, (tau + i - 1)].detach().numpy(),j)
ax.grid(True)
ax.set_xlabel('time')
ax.set_ylabel('x')
plt.show()

