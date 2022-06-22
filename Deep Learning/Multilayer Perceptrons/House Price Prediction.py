#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 21 15:12:47 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     House Price Prediction.py
# @Software: PyCharm
"""
# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading data
train = pd.read_csv('Multilayer Perceptrons/house_price/train.csv')
test = pd.read_csv('Multilayer Perceptrons/house_price/test.csv')
train.head(3)
test.head(3)
print(train.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# combine train and test
df = pd.concat((train.iloc[:,1:-1], test.iloc[:,1:]))
# df.info()

# Regularization and fill NA value
numeric_feature = df.dtypes[df.dtypes != 'object'].index
df_reg = df[numeric_feature].apply(lambda x: (x - x.mean()) / x.std()) # apply deal with more complex function
df_reg[numeric_feature] = df_reg[numeric_feature].fillna(0)
df_reg = pd.get_dummies(df_reg, dummy_na=True)
# df_reg.shape

# Training
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

## Transfer to tensor
num_train = train.shape[0]
train_df = torch.tensor(df_reg[:num_train].values, dtype=torch.float32)
test_df = torch.tensor(df_reg[num_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

## Define model
loss = nn.MSELoss()
model = nn.Sequential(
    nn.Linear(train_df.shape[1],64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(32,1)
)

## relatively error instead of absolute error
def log_rmse(model, features, labels):
    # To further stabilize the value when the logarithm is taken, set the value less than 1 as 1
    clipped_preds = torch.clamp(model(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def data_trans(data, batch_size, is_shuffle):
    df = TensorDataset(*data)
    return DataLoader(df,batch_size=batch_size,shuffle=is_shuffle)

# Define train loop
def train_loop(model, batch_size, train_df, train_labels,test_df, test_label,
          num_epochs, learning_rate, weight_decay):
    train_ls, test_ls = [], []
    # create tensor dataset
    train_iter = data_trans((train_df,train_labels),batch_size=batch_size,is_shuffle=True)
    # define optimizer with adam
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(model(X), y)
            l.backward() # back propagation
            optimizer.step()
        train_ls.append(log_rmse(model, train_df, train_labels))
        if test_label is not None:
            test_ls.append(log_rmse(model, test_df, test_label))
    return train_ls, test_ls

def train_and_pred(train_df, test_df, train_labels, test,
                   num_epochs, lr, weight_decay, batch_size):
    # get information
    train_ls, _ = train_loop(model=model, train_df=train_df, train_labels=train_labels,
                             test_df=None, test_label=None,num_epochs=num_epochs, learning_rate=lr,
                             weight_decay=weight_decay, batch_size=batch_size)
    # plot the loss
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(np.arange(1, num_epochs + 1), train_ls)
    ax.set(xlabel='epoch',ylabel='log rmse',xlim=[1, num_epochs])
    ax.grid(True)
    plt.show()
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = model(test_df).detach().numpy() # make prediction
    # Reformat it to export to Kaggle
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    output = pd.concat([test['Id'], test['SalePrice']], axis=1)
    output.to_csv('Multilayer Perceptrons/house_price/output.csv', index=False)

# Initiate
num_epochs, lr, weight_decay, batch_size = 100, 0.01, 0, 128
train_and_pred(train_df, test_df, train_labels, test, num_epochs, lr, weight_decay, batch_size)