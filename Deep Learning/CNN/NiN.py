#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 29 15:38:46 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     NiN.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from SP_Func import train_GPU, try_gpu, load_data_Fashion

# Nin Block
def nin_block(in_channel, out_channel, kernel_size, strides, padding):
    block = nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size,strides,padding),
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,kernel_size=1),
        nn.ReLU()
    )
    return block

# NiN Model
model = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

X = torch.randn((1,1,224,224))
for layer in model:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
'''
Sequential output shape:	 torch.Size([1, 96, 54, 54])
MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
Sequential output shape:	 torch.Size([1, 256, 26, 26])
MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
Sequential output shape:	 torch.Size([1, 384, 12, 12])
MaxPool2d output shape:	 torch.Size([1, 384, 5, 5])
Dropout output shape:	 torch.Size([1, 384, 5, 5])
Sequential output shape:	 torch.Size([1, 10, 5, 5])
AdaptiveAvgPool2d output shape:	 torch.Size([1, 10, 1, 1])
Flatten output shape:	 torch.Size([1, 10])
'''

# Training
lr, num_epochs, batch_size = 0.01, 10, 128
train_iter, test_iter = load_data_Fashion(batch_size,resize=224)
train_GPU(model, train_iter, test_iter, num_epochs, lr, try_gpu())