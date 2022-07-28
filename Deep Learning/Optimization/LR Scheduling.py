#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 28 11:34:56 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     LR Scheduling.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
from torch import nn
from torch.optim import lr_scheduler
import SP_Func as sf
import matplotlib.pyplot as plt

# Create Model
model = nn.Sequential(
    # nn.Conv2d(1,6,kernel_size=5,padding=2),
    nn.LazyConv2d(6,kernel_size=5,padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    # nn.Conv2d(6,16,kernel_size=5),
    nn.LazyConv2d(16,kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    # nn.Linear(16*5*5,120),
    nn.LazyLinear(120),
    nn.ReLU(),
    # nn.Linear(120,84),
    nn.LazyLinear(84),
    nn.ReLU(),
    # nn.Linear(84,10)
    nn.LazyLinear(10)
)

loss = nn.CrossEntropyLoss()
device = sf.try_gpu()

batch_size = 256
train_iter, test_iter = sf.load_data_Fashion(batch_size)

# Define training progress
def train(model,train_iter,test_iter,num_epochs,loss,optimizer,device,scheduler=None):
    model.to(device)
    animator = sf.Animator(xlabel='epochs',xlim=[0,num_epochs],
                           legend=['train loss', 'train acc', 'test acc'],title=f'Learning Rate: {lr}')

    print('Processing.....')
    for epoch in range(num_epochs):
        metric =sf.Accumulator(3)
        for i, (X,y) in enumerate(train_iter):
            model.train() # start traininig
            optimizer.zero_grad() # clean optimizer
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward() # BP
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], sf.accuracy(y_hat,y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))
        test_acc = sf.evaluate_accuracy_gpu(model,test_iter,device)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    animator.display()
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

# Setup Parameters
lr, num_epochs = 0.3, 30
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
train(model,train_iter, test_iter, num_epochs, loss, optimizer, device)

# Schedulers
optimizer.param_groups[0]['lr'] = 0.1
print(f'learning rate is now {optimizer.param_groups[0]["lr"]:.2f}')

class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)

scheduler = SquareRootScheduler(lr=0.1)
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
ax.plot(list(range(num_epochs)),[scheduler(t) for t in range(num_epochs)])
ax.grid(True)
plt.show()

## Apply scheduler
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
train(model,train_iter, test_iter, num_epochs, loss, optimizer, device, scheduler)

# FactorScheduler
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9,stop_factor_lr=1e-2,base_lr=2.0)
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
ax.plot(list(range(50)),[scheduler(t) for t in range(50)])
ax.grid(True)
plt.show()

## Apply to the Network
train(model, train_iter, test_iter, num_epochs, loss, optimizer, device, scheduler)

# Multi-factor scheduler
optimizer = torch.optim.SGD(model.parameters(),lr=0.5)
scheduler = lr_scheduler.MultiStepLR(optimizer,gamma=.5, milestones=[15,30])

## Get learning rate from scheduler
def get_lr(optimizer, scheduler):
    lr = scheduler.get_lr()[0]
    optimizer.step()
    scheduler.step()
    return lr

fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
ax.plot(list(range(num_epochs)),[get_lr(optimizer,scheduler) for t in range(num_epochs)])
ax.grid(True)
plt.show()

## Apply
train(model, train_iter, test_iter, num_epochs, loss, optimizer, device, scheduler)

# Cosine Schduler
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20,base_lr=0.3,final_lr=0.1)
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
ax.plot(list(range(num_epochs)),[scheduler(t) for t in range(num_epochs)])
ax.grid(True)
plt.show()

## Apply
optimizer = torch.optim.SGD(model.parameters(),lr=0.3)
train(model, train_iter, test_iter, num_epochs, loss, optimizer, device, scheduler)

# Warm-Up
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
fig, ax = plt.subplots(1,figsize=(6,4),dpi=200)
ax.plot(list(range(num_epochs)),[scheduler(t) for t in range(num_epochs)])
ax.grid(True)
plt.show()

## Apply
optimizer = torch.optim.SGD(model.parameters(),lr=0.3)
train(model, train_iter, test_iter, num_epochs, loss, optimizer, device, scheduler)