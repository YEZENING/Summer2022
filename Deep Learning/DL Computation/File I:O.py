#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 23 21:32:19 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     File I/O.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from torch.nn import functional as F

# Loading and saving tensor
x = torch.arange(4)
torch.save(x,'x-file') # save into work direction

x2 = torch.load('x-file')
x2

## Save two variables
y = torch.zeros(4)
torch.save([x,y],'x-file')
x2,y2 = torch.load('x-file')

## Save as dictionary
dic = {'x':x,'y':y}
torch.save(dic,'mydict')
dic2 = torch.load('mydict')
dic2

# Loading and saving model parameters
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self,X):
        X = F.relu(self.hidden(X))
        return self.output(X)

net = MLP()
net
X = torch.randn((2,20))
Y = net(X)
torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
Y_clone = clone(X)
Y_clone == Y