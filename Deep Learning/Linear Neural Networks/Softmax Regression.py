#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 14 14:50:33 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Softmax Regression.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# Load data
def load_data(batch_size,num_worker):
    train = FashionMNIST(root='data',
                         transform=transforms.ToTensor,
                         train=True,
                         download=True)
    test = FashionMNIST(root='data',
                        transform=transforms.ToTensor,
                        train=False,
                        download=True)
    return DataLoader(train,batch_size,shuffle=True,num_workers=num_worker), \
           DataLoader(test,batch_size,shuffle=True,num_workers=num_worker)

batch_size = 256
num_worker= 4
mnist_train, mnist_test = load_data(batch_size,num_worker)

# Initialize model parameter
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# Initialize softmax operation
''' process for sum
X = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True)
'''

def softmax(X):
    X_exp = torch.exp(X) # find the exponential
    sum_exp = X_exp.sum(axis=1,keepdims=True) # sum all exponential
    return X_exp / sum_exp

X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)

# Define model
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# Define loss function
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)

# Classification accuracy
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat,y) / len(y)

def evaluate_accuracy(net, data_iter):  #@save
    # Compute the accuracy for a model on a dataset.
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    # For accumulating sums over `n` variables.
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

evaluate_accuracy(net, mnist_test)

# Training
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

'''
It will be too complicate to reproduce the further code since the textbook use d2l package instead of original packages
from Python. The following code will present an easy version which is using API of torch.
'''

# Concise Implementation of Softmax Regression
from torch import nn

## Deine parameter and data
batch_size = 256
train_iter, test_iter = load_data(batch_size)

## initialize paramter and build the model
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

## Loss function
loss = nn.CrossEntropyLoss(reduction='None')
## Optimizer
opt = torch.optim.SGD(net.parameters(), lr=0.1)

## Training network
num_epochs = 10

'''
For some reasons I am not be able to reproduce the entire code from 3.6 - 3.7. I did not install 'd2l' package
for my environment since it is useless. 

The important thing to know that we will use the pytorch framework to produce the neural network and they all have function
inside the packages. Same as tensorflow
'''