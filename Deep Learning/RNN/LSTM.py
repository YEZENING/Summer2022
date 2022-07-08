#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 08 14:17:04 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     LSTM.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf

# Setting parameters
batch_size, num_steps = 32, 35
train_iter, vocab = sf.load_data_time_machine(batch_size,num_steps)

# Initializae parameters
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

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters

    # output layer parameters
    W_hq = normal((num_hidden, num_output))
    b_q = torch.zeros(num_output, device=device)

    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# Define model
def init_lstm_state(batch_size, num_hidden, device):
    return (
        torch.zeros((batch_size, num_hidden), device=device),
        torch.zeros((batch_size, num_hidden), device=device)
    )

def lstm(input, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in input:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# Training and prediction
vocab_size, num_hidden, device = len(vocab), 256, sf.try_gpu()
num_epochs, lr = 500, 1
model = sf.RNNModelScrach(len(vocab), num_hidden, device, get_params, init_lstm_state, lstm)
sf.train_rnn(model,train_iter,vocab,lr,num_epochs,device)

# Concise Implementation
num_input = vocab_size
lstm_layer = nn.LSTM(num_input,num_hidden)
model = sf.RNNModel(lstm_layer, len(vocab))
model.to(device)
sf.train_rnn(model,train_iter,vocab,lr,num_epochs,device)