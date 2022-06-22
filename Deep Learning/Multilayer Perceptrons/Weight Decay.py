#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 17 14:31:46 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Weight Decay.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch.utils import data
from torch import nn

# High dimensional linear regression
## Generate data
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

def load_array(data_arrays, batch_size, is_shuffle=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_shuffle)

def synthetic_data(w, b, num_examples):
    # Generate y = Xw + b + noise
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y = y + torch.normal(0, 0.01, y.shape) # add noise
    return X, y.reshape((-1, 1))

train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_shuffle=False)

## Initialize parameters1
def init_params():
    w = torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w,b]

## L2 regularization
def l2_regul(w):
    return torch.sum(w.pow(2)) /2

## Training loop
### linear regression
def lin_reg(X,w,b):
    return torch.matmul(X,w) + b

### Define loss function
def square_loss(y_hat, y):
    return 0.5*(y_hat - y.reshape(y_hat.shape))**2 # mean square

### optimizer
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

###
class Accumulator:
    # For accumulating sums over `n` variables.
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

### training
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: lin_reg(X, w, b), square_loss
    num_epochs, lr = 100, 0.003
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting
            # makes `l2_penalty(w)` a vector whose length is `batch_size`
            l = loss(net(X), y) + lambd * l2_regul(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
    print('L2 norm of w:', torch.norm(w).item())

## trainig without egularization
train(lambd=0)

## add regularization
train(lambd=3)


# Using API torch.nn
def train_API(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
    print('L2 norm of w:', net[0].weight.norm().item())

## Implement API
train_API(0) # without regularization
train_API(3) # apple regularization