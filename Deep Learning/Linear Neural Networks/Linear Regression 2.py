#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 13 15:52:42 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Linear Regression 2.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch.utils import data
from torch import nn
import random
import matplotlib.pyplot as plt

# Generate data
def synthetic_data(w, b, num_examples):
    # Generate y = Xw + b + noise
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y = y + torch.normal(0, 0.01, y.shape) # add noise
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(f'features: {features[0]}',f'\nlabels: {labels[0]}') # print out feature and labels

fig, ax = plt.subplots(figsize=(10,4))
ax.scatter(features[:,(1)].detach(), labels.detach(),1)
plt.show()

# Loading data
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # randomly
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# Initiate parameter
w = torch.normal(0,1,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# Define model
def lin_reg(X,w,b):
    return torch.matmul(X,w) + b

# Define loss function
def square_loss(y_hat, y):
    return 0.5*(y_hat - y.reshape(y_hat.shape))**2 # mean square

# Define Optimize Algorithm
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# Training model
lr = 0.03
num_epochs = 3
net = lin_reg
loss = square_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # sgd update parameters
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'Error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'Error in estimating b: {true_b - b}')


# Concise Implement for Linear Regression (Easy version)
## Generate Dataset
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # Remember define the function

## Reading dataset
def load_array(data_arrays, batch_size, is_shuffle=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_shuffle)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

net = nn.Sequential(nn.Linear(2,1)) # Define Model
net[0].weight.data.normal_(0,0.01) # initiate parameters in first layer
net[0].bias.data.fill_(0)
loss = nn.MSELoss() # Mean square as loss function
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # optimizer

## Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)