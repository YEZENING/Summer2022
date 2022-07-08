#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 08 16:02:16 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     DRNN.py
# @Software: PyCharm
"""
# Import Packages
from torch import nn
import SP_Func as sf

# setup parameters and load data
batch_size, num_steps = 32, 35
train_iter, vocab = sf.load_data_time_machine(batch_size,num_steps)
vocab_size, num_hidden, num_layer = len(vocab), 256, 2
device = sf.try_gpu()
num_input = vocab_size
lstm_layer = nn.LSTM(num_input, num_hidden, num_layer)
model = sf.RNNModel(lstm_layer,vocab_size)
model.to(device)

# Training and predicting
num_epochs, lr = 500, 2
sf.train_rnn(model,train_iter,vocab,lr,num_epochs,device)