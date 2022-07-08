#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 06 14:03:20 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     RNNs.py
# @Software: PyCharm
"""
# Import Packages
import time
import math
import torch
import SP_Func as sf
from torch import nn
from torch.nn import functional as F

# RNN Basic
X, W_xh = torch.normal(0,1,(3,1)), torch.normal(0,1,(1,4))
H, W_hh = torch.normal(0,1,(3,4)), torch.normal(0,1,(4,4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
'''
torch.matmul(X, W_xh) 
tensor([[ 1.4975,  1.6034, -2.3826, -3.3558],
        [ 0.1781,  0.1907, -0.2834, -0.3991],
        [ 0.8912,  0.9542, -1.4179, -1.9970]])

torch.matmul(H, W_hh)
tensor([[ 0.6059,  1.1279, -0.1069, -1.8234],
        [-0.0305, -0.2899,  1.0020, -0.3866],
        [ 1.3294, -0.1625, -1.1080, -0.2070]])
        
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
tensor([[ 2.1035,  2.7313, -2.4895, -5.1792],
        [ 0.1476, -0.0992,  0.7187, -0.7857],
        [ 2.2206,  0.7917, -2.5259, -2.2040]])
'''
torch.matmul(torch.cat((X,H),1), torch.cat((W_xh, W_hh),0))
'''
torch.matmul(torch.cat((X,H),1), torch.cat((W_xh, W_hh),0))
tensor([[ 2.1035,  2.7313, -2.4895, -5.1792],
        [ 0.1476, -0.0992,  0.7187, -0.7857],
        [ 2.2206,  0.7917, -2.5259, -2.2040]])
'''

# Implementation
## Define batch size and read data
batch_size, num_steps = 32, 35
train_iter, vocab = sf.load_data_time_machine(batch_size, num_steps)

## One-hot encoding
X = torch.arange(10).reshape((2,5))
F.one_hot(X.T, 28).shape

## Initiate model parameter
def get_param(vocab_size, num_hiddens, device):
    num_input = num_output = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_input, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_output))
    b_q = torch.zeros(num_output, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

## RNN Model
def init_rnn(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device),)

def rnn(input, state, params):
    # Input shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of X: (batch_size, vocab_size)
    for X in input:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScrach:
    # RNN from scratch
    def __init__(self, vocab_size, num_hidden, device,
                 get_param, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hidden
        self.params = get_param(vocab_size, num_hidden, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

## Initiate network
num_hidden = 512
model = RNNModelScrach(len(vocab),num_hidden,sf.try_gpu(),get_param, init_rnn, rnn)
state = model.begin_state(X.shape[0], sf.try_gpu())
Y, new_state = model(X.to(sf.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

# Prediction
def predict_rnn(prefix, num_preds, model, vocab, device):
    """Generate new characters following the `prefix`."""
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_rnn('time traveller ', 10, model, vocab, sf.try_gpu())

## Gradient Clipping
def grad_clipping(model, theta):
    """Clip the gradient."""
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

## Training
def train_epoch_ch8(model, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state = None
    time_start = time.time()
    metric = sf.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = model.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(model, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = model(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(model, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(model, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
        time_stop = time.time()
    return math.exp(metric[0] / metric[1]), metric[1] / (time_stop-time_start)

def train_ch8(model, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = sf.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(model, nn.Module):
        updater = torch.optim.SGD(model.parameters(), lr)
    else:
        updater = lambda batch_size: sf.sgd(model.params, lr, batch_size)
    predict = lambda prefix: predict_rnn(prefix, 50, model, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(model, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    animator.display()

## Training model
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, sf.try_gpu())

model = RNNModelScrach(len(vocab), num_hidden, sf.try_gpu(), get_param, init_rnn, rnn)
train_ch8(model, train_iter, vocab, lr, num_epochs, sf.try_gpu(),use_random_iter=True)


#######################################################################################
# Concise Implementation
## Define parameter and layer
num_hidden = 256
batch_size, num_steps = 32, 35
train_iter, vocab = sf.load_data_time_machine(batch_size, num_steps)
rnn_layer = nn.RNN(len(vocab),num_hidden)

## Initialize tensor
'''with shape (number of hidden layers, batch size, number of hidden units)'''
state = torch.zeros((1, batch_size, num_hidden))
state.shape

X = torch.randn((num_steps, batch_size, len(vocab)))
Y, new_state = rnn_layer(X, state)
Y.shape, new_state.shape

## Define model
class RNNModel(nn.Module):
    def __init__(self,rnn_layer,vocab_size,**kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hidden = self.rnn.hidden_size
        # for bidirectional rnn: num_directions == 2, else 1
        if not self.rnn.bidirectional:
            self.num_direction = 1
            self.linear = nn.Linear(self.num_hidden, self.vocab_size)
        else:
            self.num_direction = 2
            self.linear = nn.Linear(self.num_hidden * 2, self.vocab_size)

    def forward(self,input,state):
        X = F.one_hot(input.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        #
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self,device,batch_size=-1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_direction * self.rnn.num_layers,
                                batch_size, self.num_hidden),device=device)
        else:
            return (
                torch.zeros(
                    (self.num_direction * self.rnn.num_layers, batch_size, self.num_hidden),
                    device=device
                ),
                torch.zeros(
                    (self.num_direction * self.rnn.num_layers, batch_size, self.num_hidden),
                    device=device
                )
            )

## Training and predicting
device = sf.try_gpu()
model = RNNModel(rnn_layer,vocab_size=len(vocab))
model.to(device)
sf.predict_rnn('time traveller', 10, model, vocab, device)

num_epochs, lr = 500, 1
sf.train_rnn(model,train_iter,vocab,lr,num_epochs,device)