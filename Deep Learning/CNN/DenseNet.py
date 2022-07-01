#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 01 12:41:12 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     DenseNet.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from SP_Func import try_gpu, train_GPU, load_data_Fashion

# Define dense block
def conv_block(input_channel,num_channel):
    block = nn.Sequential(
        nn.BatchNorm2d(input_channel),
        nn.ReLU(),
        nn.Conv2d(input_channel,num_channel,kernel_size=3,padding=1)
    )
    return block

class DenseBlock(nn.Module):
    def __init__(self,num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels,num_channels))
        self.layer = nn.Sequential(*layer)

    def forward(self,X):
        for blk in self.layer:
            Y = blk(X)
            X = torch.cat((X,Y),dim=1)
        return X

# Checking shape
block = DenseBlock(2,3,10)
X = torch.randn((4,3,8,8))
Y = block(X)
Y.shape # 3 + 10 x 2 = 23 output channel

# Transition layer
def transition_block(input_channels,num_channels):
    block = nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
    return block

tran_block = transition_block(23,10)
tran_block(Y).shape

# Define DenseNet Model
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

num_channels, grow_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
block = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    block.append(DenseBlock(num_convs,num_channels,grow_rate))
    # # This is the number of output channels in the previous dense block
    num_channels = num_channels + num_convs * grow_rate
    # A transition layer that halves the number of channels is added between the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        block.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

model = nn.Sequential(
    b1, *block,
    nn.BatchNorm2d(num_channels),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(num_channels,10)
)

# Training
lr, num_eopchs, batch_size = 0.1, 10, 256
train_iter, test_iter = load_data_Fashion(batch_size,resize=96)
train_GPU(model,train_iter,test_iter,num_eopchs,lr,device=try_gpu())