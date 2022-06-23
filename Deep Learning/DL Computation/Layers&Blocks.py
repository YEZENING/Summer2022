#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 22 13:44:51 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Layers&Blocks.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
from torch.nn import functional as F
import tensorflow as tf

# Simple model
model = nn.Sequential(
    nn.Linear(20,256),
    nn.ReLU(),
    nn.Linear(256,10)
)

torch_X = torch.rand(2, 20)
model(torch_X)

# Custom Block
## Torch version
class MLP_torch(nn.Module):
    def __init__(self):
        super().__init__()
        # input = 20, output = 10
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self, X):
        x = self.hidden(X)
        x = F.relu(x)
        return self.out(x)

model_torch = MLP_torch()
model_torch(torch_X)

## TF version
class MLP_tf(tf.keras.Model):
    def __init__(self):
        super(MLP_tf,self).__init__()
        self.d1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.d2 = tf.keras.layers.Dense(units=10)

    def call(self,x):
        x = self.d1(x)
        return self.d2(x)

model_tf = MLP_tf()
tf_X = tf.random.uniform((2, 20))
model_tf(tf_X)

# Sequential Block
## Torch version
class torch_seq(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

model_torch_seq = torch_seq(
    nn.Linear(20,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
model_torch_seq(torch_X)

## TF version
class tf_seq(tf.keras.Model):
    def __init__(self, *args):
        super(tf_seq, self).__init__()
        self.denses = []
        for dense in args:
            self.denses.append(dense)

    def call(self,X):
        for layer in self.denses:
            X = layer(X)
        return X

model_tf_seq = tf_seq(
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(10)
)
model_tf_seq(tf_X)

# More flexible model -- Executing Code in the Forward Propagation Function
## Torch version
class torch_FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(torch_FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X,self.rand_weight)+1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X = X/2
        return X.sum()

model_torch_flex = torch_FixedHiddenMLP()
model_torch_flex(torch_X)

## TF version
class tf_FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super(tf_FixedHiddenMLP, self).__init__()
        self.rand_weight = tf.constant(tf.random.uniform((20,20)))
        self.flatten = tf.keras.layers.Flatten()
        self.d2 = tf.keras.layers.Dense(20, activation='relu')

    def call(self, inputs):
        X = self.flatten(inputs)
        X = tf.nn.relu(tf.matmul(X, self.rand_weight)+1)
        X = self.d2(X)
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X = X/2
        return tf.reduce_sum(X)

model_tf_flex = tf_FixedHiddenMLP()
model_tf_flex(tf_X)

# Mix with all
## Torch version
class torch_MixMLP(nn.Module):
    def __init__(self):
        super(torch_MixMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32,16)

    def forward(self,X):
        X = self.net(X)
        return self.linear(X)

model_torch_mix = nn.Sequential(
    torch_MixMLP(),
    nn.Linear(16,20),
    torch_FixedHiddenMLP()
)
model_torch_mix(torch_X)

## TF version
class tf_MixMLP(tf.keras.Model):
    def __init__(self):
        super(tf_MixMLP, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        self.d1 = tf.keras.layers.Dense(16,activation='relu')

    def call(self,inputs):
        X = self.net(inputs)
        return self.d1(X)

model_tf_mix = tf.keras.Sequential([
    tf_MixMLP(),
    tf.keras.layers.Dense(20),
    tf_FixedHiddenMLP()
])
model_tf_mix(tf_X)