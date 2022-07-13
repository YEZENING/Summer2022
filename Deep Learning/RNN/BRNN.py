#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 11 13:50:12 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     BRNN.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf

'''
The following code illustrated a wrong application of bidirectional RNN 
'''

# Load data
batch_size, num_steps, device = 32, 35, sf.try_gpu()
train_iter, vocab = sf.load_data_time_machine(batch_size, num_steps)

# Define bidirectional LSTM layer
vocab_size, num_hidden, num_layer = len(vocab), 256, 2
num_input = vocab_size
lstm_layer = nn.LSTM(num_input, num_hidden, num_layer, bidirectional=True)
model = sf.RNNModel(lstm_layer,len(vocab))
model.to(device)

# Training
num_epochs, lr = 500, 1
sf.train_rnn(model,train_iter, vocab, lr, num_epochs, device)