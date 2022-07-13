#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 12 12:48:13 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Seq2Seq.py
# @Software: PyCharm
"""
# Import Packages
import math
import time
import torch
import collections
import SP_Func as sf
from torch import nn

# Conduct encoder
class Seq2SeqEncoder(sf.Encoder):
    def __init__(self, vocab_size, embed_size, num_hidden,
                 num_layer, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hidden, num_layer, dropout=dropout)

    def forward(self, X, *args):
        # output X shape: (batch_size, num_step, embed_size)
        X = self.embedding(X)
        # first axis represent time step in RNN
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # `output` shape: (`num_step`, `batch_size`, `num_hidden`)
        # `state` shape: (`num_layer`, `batch_size`, `num_hidden`)
        return output, state

encoder = Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hidden=16,num_layer=2)
encoder.eval()
X = torch.zeros((4,7),dtype=torch.long)
'''
tensor([[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
'''
output, state = encoder(X)
output.shape # torch.Size([7, 4, 16]) (`num_step`, `batch_size`, `num_hidden`)
state.shape # torch.Size([2, 4, 16]) (`num_layer`, `batch_size`, `num_hidden`)

# Decoder
class Seq2SeqDecoder(sf.Decoder):
    def __init__(self, vocab_size, embed_size, num_hidden,
                 num_layer, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hidden, num_hidden, num_layer, dropout=dropout)
        self.dense = nn.Linear(num_hidden, vocab_size)

    def init_state(self, enc_output, *args):
        return enc_output[1]

    def forward(self, X, state):
        # output X shape: (`num_step`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # broadcast 'context' and make it has same num_step in X
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_Context = torch.cat((X, context), 2)
        output, state = self.rnn(X_Context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

decoder = Seq2SeqDecoder(vocab_size=10,embed_size=8,num_hidden=16,num_layer=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape # torch.Size([4, 7, 10]) (`num_step`, `batch_size`, `embed_size`)
state.shape # torch.Size([2, 4, 16]) (`num_layer`, `batch_size`, `num_hidden`)

# Loss Function
## clean irrelavent entries by mask them into zero
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1,2,3],[4,5,6]])
sequence_mask(X, torch.tensor([1,2]))

X = torch.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)

# Define softmax cross entropy
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # `pred` shape: (`batch_size`, `num_step`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_step`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))

# Training
def train_seq2seq(model, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss = MaskedSoftmaxCELoss()
    model.train()
    animator = sf.Animator(xlabel='Epochs', ylabel='Loss',xlim=[10, num_epochs])
    print('Processing......')
    for epoch in range(num_epochs):
        time_start = time.time()
        metric = sf.Accumulator(2) # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                              device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward() # BP
            sf.grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        time_stop = time.time()
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / (time_stop-time_start):.1f} '
          f'tokens/sec on {str(device)}')
    animator.display()

## Define paramters
embed_size, num_hidden, num_layer, dropout = 32, 32, 2, 0.1
bacth_size, num_step = 64, 10
lr, num_epoch, device = 0.005, 300, sf.try_gpu()

## Load data
train_iter, src_vocab, tgt_vocab = sf.load_data_nmt(bacth_size,num_step)
encoder = Seq2SeqEncoder(len(src_vocab),embed_size,num_hidden,num_layer,dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab),embed_size,num_hidden,num_layer,dropout)
model = sf.EncoderDecoder(encoder, decoder)
'''
EncoderDecoder(
  (encoder): Seq2SeqEncoder(
    (embedding): Embedding(184, 32)
    (rnn): GRU(32, 32, num_layers=2, dropout=0.1)
  )
  (decoder): Seq2SeqDecoder(
    (embedding): Embedding(201, 32)
    (rnn): GRU(64, 32, num_layers=2, dropout=0.1)
    (dense): Linear(in_features=32, out_features=201, bias=True)
  )
)
'''
## Train the model
train_seq2seq(model,train_iter,lr,num_epoch,tgt_vocab,device)

# Prediction
def pred_seq2seq(model, src_sentence, src_vocab, tgt_vocab, num_step, device,
                 save_attention_weights=False):
    model.eval()
    src_token = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_token)], device=device)
    src_token = sf.truncate_pad(src_token, num_step, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(torch.tensor(src_token, dtype=torch.long, device=device), dim=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_step):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(model.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

# Evaluation
## Bilingual Evaluation Understudy
def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = pred_seq2seq(
        model, eng, src_vocab, tgt_vocab, num_step, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')