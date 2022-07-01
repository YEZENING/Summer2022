#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 30 15:55:25 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     ResNet.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from torch.nn import functional as F
from SP_Func import try_gpu, train_GPU, load_data_Fashion

# Define Residual Block for Resnet
class Residual(nn.Module):
    def __init__(self, input_channel, num_channel, use_1x1conv=False,strides=1):
        super(Residual, self).__init__()
        # two convolutional layer with same paramters
        self.conv1 = nn.Conv2d(input_channel,num_channel,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channel,num_channel,kernel_size=3,padding=1)
        # adding 1x1 convolutional layer
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channel,num_channel,kernel_size=1,stride=strides)
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

# input and output shape
block1 = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = block1(X)
Y.shape

# apply 1x1 convolutional layer
block2 = Residual(3,6, use_1x1conv=True, strides=2)
block2(X).shape

# Define ResNet Model
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

## resnet block
def resnet_block(input_channel, num_channel, num_residual, first_block=False):
    block = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            block.append((Residual(input_channel, num_channel, use_1x1conv=True, strides=2)))
        else:
            block.append(Residual(num_channel,num_channel))
    return block

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

model = nn.Sequential(
    b1,b2,b3,b4,b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(512,10)
)

# Checking parameters
X = torch.rand(size=(1, 1, 224, 224))
for layer in model:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# training
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = load_data_Fashion(batch_size,resize=96)
train_GPU(model,train_iter,test_iter,num_epochs,lr,device=try_gpu())