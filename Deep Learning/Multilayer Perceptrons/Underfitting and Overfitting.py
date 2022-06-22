#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 16 16:45:27 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Underfitting and Overfitting.py
# @Software: PyCharm
"""
# Import Packages
import numpy as np
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

# Polynomial Regression
## Generate dataset
'''
assume the polynomial is the following:
y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{where}\epsilon \sim \mathcal{N}(0, 0.1^2).
'''
max_df = 20 # maximum degreee of fr¸dom for polynomial
n_train, n_test = 100, 100 # size of dataset for train and test
true_w = np.zeros(max_df)
true_w[0:4] = np.array([5.0,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train+n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_df).reshape(1,-1))
for i in range(max_df):
    poly_features[:,i] = poly_features[:,i] / math.gamma(i+1) # gamma(n) = (n-1)!
labels = np.dot(poly_features, true_w)
fin_labels = labels + np.random.normal(0.1,size=labels.shape)

## conver numpy array to tensor
true_w, features, poly_features, fin_labels = [torch.tensor(x, dtype=torch.float32)
                                               for x in [true_w, features, poly_features, fin_labels]]
### Print out the tensor
features[:2], poly_features[:2, :], fin_labels[:2]

## Training and testing the model
from SP_Func import Accumulator, Animator, train_epoch

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def dataloader(data,batch_size,is_shuffle=True):
    df = TensorDataset(*data)
    return DataLoader(df,batch_size,shuffle=is_shuffle)

## Define training loop
def train(train_df, test_df, train_label, test_label, num_epochs):
    loss_func = nn.MSELoss(reduction='none')
    model = nn.Sequential(nn.Linear(train_df.shape[-1],1,bias=False))
    bs = min(10, train_label.shape[0])
    train_iter = dataloader((train_df,train_label.reshape(-1,1)),batch_size=bs)
    test_iter = dataloader((test_df,test_label.reshape(-1, 1)),batch_size=bs,is_shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    plot = Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                    yscale='log' ,legend=['Train', 'Test'])
    for epoch in range(num_epochs):
        train_epoch(model,train_iter,loss_func,optimizer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            plot.add(epoch+1,(evaluate_loss(model,train_iter, loss_func),
                              evaluate_loss(model,test_iter, loss_func)))
    print(f'Weight: {model[0].weight.data.numpy()}')
    plot.display()

train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      fin_labels[:n_train], fin_labels[n_train:], num_epochs=400)

# Linear Function Fitting (Underfitting)
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      fin_labels[:n_train], fin_labels[n_train:], num_epochs=1500)

# Higher-Order Polynomial Function Fitting (Overfitting)
train(poly_features[:n_train, :], poly_features[n_train:, :],
      fin_labels[:n_train], fin_labels[n_train:], num_epochs=1500)