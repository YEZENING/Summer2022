#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 28 14:11:04 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     AlexNet.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from SP_Func import load_data_Fashion, train_GPU, try_gpu

# Define model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=11,padding=1,stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,kernel_size=5,padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,kernel_size=3,padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Linear(6400,4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self,x):
        return self.conv_seq(x)

model = AlexNet()

# Print model layer with parameters
X = torch.randn(1, 1, 224, 224)
for layer in model.conv_seq:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
'''
Conv2d output shape:	 torch.Size([1, 96, 54, 54])
ReLU output shape:	 torch.Size([1, 96, 54, 54])
MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
Conv2d output shape:	 torch.Size([1, 256, 26, 26])
ReLU output shape:	 torch.Size([1, 256, 26, 26])
MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
Conv2d output shape:	 torch.Size([1, 384, 12, 12])
ReLU output shape:	 torch.Size([1, 384, 12, 12])
Conv2d output shape:	 torch.Size([1, 384, 12, 12])
ReLU output shape:	 torch.Size([1, 384, 12, 12])
Conv2d output shape:	 torch.Size([1, 256, 12, 12])
ReLU output shape:	 torch.Size([1, 256, 12, 12])
MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
Flatten output shape:	 torch.Size([1, 6400])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 10])

'''

# Setting batch_size and load data
batch_size = 128
train_iter, test_iter = load_data_Fashion(batch_size,resize=224)

# Setup hyperparameters and train
lr, num_epochs = 0.0001, 35
train_GPU(model,train_iter,test_iter,num_epochs,lr,device=try_gpu())