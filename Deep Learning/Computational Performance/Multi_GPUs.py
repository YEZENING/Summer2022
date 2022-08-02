#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 01 14:42:01 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Multi_GPUs.py
# @Software: PyCharm
"""
# Import Packages
import torch
import SP_Func as sf
from torch import nn
from torch.nn import functional as F

'''The following code need to run on multiple GPUs'''
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Loss function
loss = nn.CrossEntropyLoss(reduction='none')

# Data Synchronization
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

## Get parameters
new_params = get_params(params, sf.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)

## Reduce data
'''adds up all vectors and broadcasts the result back to all GPUs'''
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

data = [torch.ones((1, 2), device=sf.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])

# Distributing data
data = torch.arange(20).reshape((4,5))
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)

## split batch (split X, y into different device)
def split_batch(X, y, devices):
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

# Training
## Training batch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X,y,devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        sf.sgd(param, lr, X.shape[0])

## Training loop
def train(num_gpu, batch_size, lr):
    train_iter, test_iter = sf.load_data_Fashion(batch_size)
    devices = [sf.try_gpu(i) for i in range(num_gpu)]
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = sf.Animator('epochs', 'test_acc', title=f'Learning Rate:{lr}', xlim=[1, num_epochs])
    timer = sf.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (sf.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    animator.display()
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')

## Training with 1 GPU
train(num_gpu=1,batch_size=256,lr=0.2)

## Training with 2 GPUs
train(num_gpu=2,batch_size=256,lr=0.2)

#########################################################################################
# Concise Implementation
## Define concise model
class Residual(nn.Module):
    def __init__(self, num_channel, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        # two convolutional layer with same paramters
        self.conv1 = nn.LazyConv2d(num_channel,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.LazyConv2d(num_channel,kernel_size=3,padding=1)
        # adding 1x1 convolutional layer
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channel,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        self.batch_norm1 = nn.BatchNorm2d(num_channel)
        self.batch_norm2 = nn.BatchNorm2d(num_channel)

    def forward(self,X):
        Y = F.relu(self.batch_norm1(self.conv1(X)))
        Y = self.batch_norm2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)

def restnet18(num_class):
    def resnet_block(out_channel, num_residual, first_block=False):
        block = []
        for i in range(num_residual):
            if i == 0 and not first_block:
                block.append(Residual(out_channel, use_1x1conv=True, strides=2))
            else:
                block.append(Residual(out_channel))
        return nn.Sequential(*block)

    model = nn.Sequential(
        nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    model.add_module("resnet_block1", resnet_block(64, 2, first_block=True))
    model.add_module("resnet_block2", resnet_block(128, 2))
    model.add_module("resnet_block3", resnet_block(256, 2))
    model.add_module("resnet_block4", resnet_block(512, 2))
    model.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    model.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_class)))
    return model

## Initialization
model = restnet18(10)
devices = sf.try_all_gpus()

## Training loop
def train(model, num_gpu, batch_size, lr):
    train_iter, test_iter = sf.load_data_Fashion(batch_size) # load data
    devices = [sf.try_gpu(i) for i in range(num_gpu)] # list gpus

    # initiate parameters
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    model.apply(init_weights)

    # implement model into multiple GPUs
    model = nn.DataParallel(model,devices)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = sf.Timer(), 10
    animator = sf.Animator('epochs','test_acc', title=f'Learning Rate:{lr}',xlim=[1,num_epochs])
    for epoch in range(num_epochs):
        model.train()
        timer.start()
        for X, y in train_iter:
            optimizer.zero_grad() # Clean gradient
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(model(X), y)
            l.backward() # BP
            optimizer.step()
        timer.stop()
        animator.add(epoch + 1, (sf.evaluate_accuracy_gpu(model, test_iter),)) # plot the trace
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')

## Trainig with 1 GPU
train(model, num_gpu=1, batch_size=256, lr=0.1)

## Trainig with 2 GPU
train(model, num_gpu=2, batch_size=256, lr=0.2)