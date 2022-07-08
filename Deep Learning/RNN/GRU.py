#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 08 12:29:16 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     GRU.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf

# Setting parameters
batch_size, num_steps = 32, 35
train_iter, vocab = sf.load_data_time_machine(batch_size,num_steps)

# Initializing model parameters
def get_params(vocab_size, num_hidden, device):
    num_input = num_output = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device) * 0.1

    def three():
        return (
            normal((num_input, num_hidden)),
            normal((num_hidden, num_hidden)),
            torch.zeros(num_hidden,device=device)
        )

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters

    # output layer parameters
    W_hq = normal((num_hidden, num_output))
    b_q = torch.zeros(num_output, device=device)

    # attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# Define model
def init_gru_state(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden),device=device), )

def gru(input, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    output = []
    for X in input:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        output.append(Y)
    return torch.cat(output, dim=0), (H,)

# Training and prediction
vocab_size, num_hidden, device = len(vocab), 256, sf.try_gpu()
num_epochs, lr = 500, 1
model = sf.RNNModelScrach(len(vocab), num_hidden, device, get_params, init_gru_state, gru)
sf.train_rnn(model,train_iter,vocab,lr,num_epochs,device)

# Concise Implementation
num_input = len(vocab)
gru_layer = nn.GRU(num_input, num_hidden)
model = sf.RNNModel(gru_layer, len(vocab))
model.to(device)
sf.train_rnn(model,train_iter,vocab,lr,num_epochs,device)