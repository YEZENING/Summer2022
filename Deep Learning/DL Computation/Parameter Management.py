#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 23 13:16:53 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Parameter Management.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import tensorflow as tf

model_torch = nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
torch_X = torch.rand((2,4))
model_torch(torch_X)

model_tf = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(1)
])
tf_X = tf.random.uniform((2,4))
model_tf(tf_X)

# Parameter Access
# Only need to index the layer and print it out
print(model_torch[2].state_dict())
print(model_tf.layers[2].weights)

# Target Parameters
## Identify parameters
print(type(model_torch[2].bias)) #category
print(model_torch[2].bias) # bias data with gradiant status
print(model_torch[2].bias.data) # only bias parameter
model_torch[2].weight.grad == None # did not apply backpropagation.

print(type(model_tf.layers[2].weights[1]))
print(model_tf.layers[2].weights[1])
print(tf.convert_to_tensor(model_tf.layers[2].weights[1]))

# Access all parameter at once
print(*[(name,param.shape) for name,param in model_torch[0].named_parameters()])
print(*[(name,param.shape) for name,param in model_torch.named_parameters()])
model_torch.state_dict()['2.bias'].data

print(model_tf.layers[1].weights)
print(model_tf.get_weights())
model_tf.get_weights()[1]

# Collecting parameters from nest block
## Torch
def torch_block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )

def torch_block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', torch_block1())
    return net

torch_rgnet = nn.Sequential(torch_block2(), nn.Linear(4, 1))
torch_rgnet(torch_X)
print(torch_rgnet)
torch_rgnet[0][1][0].bias.data

## TF
def tf_block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4,activation='relu')
    ],
    name=name)

def tf_block2():
    net = tf.keras.Sequential()
    for i in range(4):
        net.add(tf_block1(name=f'block-{i}'))
    return net

tf_rgnet = tf.keras.Sequential([
    tf_block2(),
    tf.keras.layers.Dense(1)
])
tf_rgnet(tf_X)
print(tf_rgnet.summary())
tf_rgnet.layers[0].layers[1].layers[1].weights[1]

# Build in Initialization
## under normaldistribution
def init_normal(m):
    if type(m) == nn.Linear:
       nn.init.normal_(m.weight,mean=0,std=0.01)
       nn.init.zeros_(m.bias)
model_torch.apply(init_normal)
model_torch[0].weight.data[0], model_torch[0].bias.data[0]

## using constant as initialization
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
model_torch.apply(init_constant)
model_torch[0].weight.data[0], model_torch[0].bias.data[0]

## setup init in diffrent block
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

model_torch[0].apply(xavier)
model_torch[2].apply(init_42)
print(model_torch[0].weight.data[0])
print(model_torch[2].weight.data)

## TF
## normal distribution
tf_net = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation='relu',
                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                          bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)
])
tf_net(tf_X)
tf_net.weights[0], tf_net.weights[1]

## Constant
tf_net = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation='relu',
                          kernel_initializer=tf.keras.initializers.Constant(1),
                          bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)
])
tf_net(tf_X)
tf_net.weights[0], tf_net.weights[1]

## mixture
tf_net = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4,activation='relu',
                          kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Constant(42))
])
tf_net(tf_X)
print(tf_net.layers[1].weights[0])
print(tf_net.layers[2].weights[0])

# Custom Initialization
def cus_init(m):
    if type(m) == nn.Linear:
        print('Init', *[(name,param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data = m.weight.data * m.weight.data.abs() >= 5

model_torch.apply(cus_init)
model_torch[0].weight[:2]

# Tied parameters
## Torch
torch_shared = nn.Linear(8,8)
torch_net = nn.Sequential(
    nn.Linear(4,8),nn.ReLU(),
    torch_shared,nn.ReLU(),
    torch_shared,nn.ReLU(),
    nn.Linear(8,1)
)
torch_net(torch_X)
print(torch_net[2].weight.data[0] == torch_net[4].weight.data[0])
torch_net[2].weight.data[0, 0] = 100
print(torch_net[2].weight.data[0] == torch_net[4].weight.data[0])

## TF
tf_shared = tf.keras.layers.Dense(4,activation='relu')
tf_net_shared = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf_shared,
    tf_shared,
    tf.keras.layers.Dense(1)
])
tf_net_shared(tf_X)
print(len(tf_net_shared.layers)==3)