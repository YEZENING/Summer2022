#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 29 19:51:15 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     GoogLeNet.py
# @Software: PyCharm
"""
# Import Pacakages
import torch
from torch import nn
from torch.nn import functional as F
from SP_Func import try_gpu, train_GPU, load_data_Fashion

# Define Inception block
class Inception(nn.Module):
    # c1 - c4 are the number of output channels
    def __init__(self,in_channel,c1,c2,c3,c4,**kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1: 1x1 Conv
        self.p1 = nn.Conv2d(in_channel,c1,kernel_size=1)
        # Path 2: 2.1 1x1 Conv; 2.2 3x3 Conv padd 1
        self.p2_1 = nn.Conv2d(in_channel,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        # Path 3: 3.1 1x1 Conv; 3.2 5x5 Conv padd 2
        self.p3_1 = nn.Conv2d(in_channel,c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        # Path 4: 4.1 3x3 Maxpooling padd 1; 4.2 1x1 Conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.p4_2 = nn.Conv2d(in_channel,c4,kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)

# Define Model
b1 = nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,padding=3,stride=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    nn.Conv2d(64,64,kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64,192,kernel_size=3,padding=1,stride=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
)

b2 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

model = nn.Sequential(
    b1,b2,b3,b4,
    nn.Linear(1024,10)
)

X = torch.randn((1,1,96,96))
for layer in model:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
'''
Sequential output shape:	 torch.Size([1, 192, 6, 6])
Sequential output shape:	 torch.Size([1, 480, 3, 3])
Sequential output shape:	 torch.Size([1, 832, 2, 2])
Sequential output shape:	 torch.Size([1, 1024])
Linear output shape:	 torch.Size([1, 10])
'''

# Training
lr, num_epochs, batch_size = 0.01, 10, 128
train_iter, test_iter = load_data_Fashion(batch_size,resize=96)
train_GPU(model,train_iter,test_iter,num_epochs,lr,device=try_gpu())