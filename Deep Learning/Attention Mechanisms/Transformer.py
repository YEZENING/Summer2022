#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 19 13:15:11 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Transformer.py
# @Software: PyCharm
"""
# Import Packages
import math
import torch
import pandas as pd
from torch import nn
import SP_Func as sf

# Positionwise Feed-Forward Networks
class PositionwiseFFN(nn.Module): # NEW API
    def __init__(self, ffn_num_hidden, ffn_num_output):
        super(PositionwiseFFN, self).__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_output)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))

# class PositionwiseFFN(nn.Module):
#     def __init__(self, ffn_num_input,ffn_num_hidden, ffn_num_output, **kwargs):
#         super(PositionwiseFFN, self).__init__(**kwargs)
#         self.dense1 = nn.Linear(ffn_num_input,ffn_num_hidden)
#         self.relu = nn.ReLU()
#         self.dense2 = nn.Linear(ffn_num_hidden,ffn_num_output)
#
#     def forward(self, X):
#         return self.dense2(self.relu(self.dense1(X)))

ffn = PositionwiseFFN(4,8)
ffn.eval()
ffn(torch.ones((2,3,4)))[0]
'''
tensor([[-0.5232,  0.2363,  0.1142, -0.1134, -0.1520,  0.5152, -0.1547,  0.5692],
        [-0.5232,  0.2363,  0.1142, -0.1134, -0.1520,  0.5152, -0.1547,  0.5692],
        [-0.5232,  0.2363,  0.1142, -0.1134, -0.1520,  0.5152, -0.1547,  0.5692]],
       grad_fn=<SelectBackward0>)
'''

# Residual Connection and Layer Normalization
ln = nn.LayerNorm(2)
bn = nn.LazyBatchNorm1d() # bn = nn.BatchNorm1d(2)
X = torch.tensor([[1,2],[2,3]],dtype=torch.float32)
print(f'Layer norm: {ln(X)}',f'\nBatch norm: {bn(X)}')

## Add norm
class AddNorm(nn.Module):
    'residual connection follow by layer normalization'
    def __init__(self,norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

addnorm = AddNorm(4, 0.5)
sf.check_shape(addnorm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))), (2,3,4))

## Define Transformer Encoder
class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hidden, ffn_num_hidden, num_head, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = sf.MultiHeadAttention(key_size,query_size,value_size,num_hidden,num_head,dropout,
                                               use_bias)
        self.addnorm1 = AddNorm(num_hidden, dropout)
        self.ffn = PositionwiseFFN(ffn_num_hidden, num_hidden)
        self.addnorm2 = AddNorm(num_hidden, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, 48, 8, 0.5)
encoder_blk.eval()
'''
EncoderBlock(
  (attention): MultiHeadAttention(
    (attention): DotProductAttention(
      (dropout): Dropout(p=0.5, inplace=False)
    )
    (W_q): Linear(in_features=24, out_features=24, bias=False)
    (W_k): Linear(in_features=24, out_features=24, bias=False)
    (W_v): Linear(in_features=24, out_features=24, bias=False)
    (W_o): Linear(in_features=24, out_features=24, bias=False)
  )
  (addnorm1): AddNorm(
    (dropout): Dropout(p=0.5, inplace=False)
    (ln): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
  )
  (ffn): PositionwiseFFN(
    (dense1): Linear(in_features=24, out_features=48, bias=True)
    (relu): ReLU()
    (dense2): Linear(in_features=48, out_features=24, bias=True)
  )
  (addnorm2): AddNorm(
    (dropout): Dropout(p=0.5, inplace=False)
    (ln): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
  )
)
'''
sf.check_shape(encoder_blk(X, valid_lens), X.shape)
encoder_blk(X, valid_lens).shape # torch.Size([2, 100, 24])

## Define TransformaerEncoder
class TransformerEncoder(sf.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hidden,
                 ffn_num_hidden, num_head, num_layer, dropout, use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hidden = num_hidden
        self.Embedding = nn.Embedding(vocab_size, num_hidden)
        self.PositionE = sf.PositionalEncoding(num_hidden,dropout)
        self.block = nn.Sequential()
        for i in range(num_layer):
            self.block.add_module('block' + str(i),
                                  TransformerEncoderBlock(key_size, query_size, value_size,
                                                          num_hidden, ffn_num_hidden, num_head, dropout,use_bias)
            )

    def forward(self, X, valid_lens):
        num_hidden = self.Embedding(X) * math.sqrt(self.num_hidden)
        X = self.PositionE(num_hidden)
        self.attention_weights = [None] * len(self.block)
        for i, blk in enumerate(self.block):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

encoder = TransformerEncoder(200,24,24,24,24,48,8,2,0.5)
encoder.eval()
sf.check_shape(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens), (2,100, 24))
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape # torch.Size([2, 100, 24])

## Define Decoder
class TransformerDecoderBlock(nn.Module):
    'The `i` blcok in Decoder'
    def __init__(self,key_size, query_size, value_size, num_hidden, ffn_num_hidden, num_head, dropout, i):
        super(TransformerDecoderBlock, self).__init__()
        self.i = i
        self.attention1 = sf.MultiHeadAttention(key_size,query_size,value_size,num_hidden,num_head,dropout)
        self.addnorm1 = AddNorm(num_hidden,dropout)
        self.attention2 = sf.MultiHeadAttention(key_size,query_size,value_size,num_hidden,num_head,dropout)
        self.addnorm2 = AddNorm(num_hidden, dropout)
        self.ffn = PositionwiseFFN(ffn_num_hidden,num_hidden)
        self.addnorm3 = AddNorm(num_hidden, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_value = X
        else:
            key_value = torch.cat((state[2][self.i], X),dim=1)
        state[2][self.i] = key_value
        if self.training:
            batch_size, num_steps, _ = X.shape
            de_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            de_valid_lens = None

        # Self attention
        X2 = self.attention1(X,key_value,key_value,de_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hidden`)
        Y2 = self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z = self.addnorm2(Y,Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

decoder_blk = TransformerDecoderBlock(24,24,24,24,48,8,0.5,0)
decoder_blk.eval()
X = torch.ones((2,100,24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape
sf.check_shape(decoder_blk(X,state)[0], X.shape)

## Conduct Entire transdormer decoder
class TransformerDecoder(sf.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hidden,
                 ffn_num_hidden, num_head, num_layer, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_hidden = num_hidden
        self.num_layer = num_layer
        self.embedding = nn.Embedding(vocab_size,num_hidden)
        self.positionE = sf.PositionalEncoding(num_hidden,dropout)
        self.block = nn.Sequential()
        for i in range(num_layer):
            self.block.add_module('block'+str(i),
                                  TransformerDecoderBlock(
                                      key_size,query_size,value_size,num_hidden,ffn_num_hidden,num_head,dropout,i
                                  ))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layer]

    def forward(self, X, state):
        num_hidden = self.embedding(X) * math.sqrt(self.num_hidden)
        X = self.positionE(num_hidden)
        self._attention_weights = [[None] * len(self.block) for _ in range(2)]
        for i, blk in enumerate(self.block):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-Decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# Training
## initialize parameters
num_hidden, num_layer, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, sf.try_gpu()
ffn_num_hidden, num_head = 64, 4
query_size, key_size, value_size = 32, 32, 32

## Load data
train_iter, src_vocab, tgt_vocab = sf.load_data_nmt(batch_size,num_steps)
encoder = TransformerEncoder(len(src_vocab),
                             key_size,query_size,value_size,num_hidden,
                             ffn_num_hidden,num_head,num_layer,dropout)
decoder = TransformerDecoder(len(tgt_vocab),
                             key_size,query_size,value_size,num_hidden,
                             ffn_num_hidden,num_head,num_layer,dropout)
model = sf.EncoderDecoder(encoder, decoder)
sf.train_seq2seq(model,train_iter,lr,num_epochs,tgt_vocab,device)

# Translation
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs,fras):
    translation, dec_attention_weight_seq = sf.pred_seq2seq(model,eng,src_vocab,tgt_vocab,num_steps,device,True)
    print(f'{eng} => {translation}, ',
          f'bleu {sf.bleu(translation, fra, k=2):.3f}')

# Visualization
enc_attention_weights = torch.cat(model.encoder.attention_weights, 0).reshape(
    (num_layer, num_head,-1, num_steps))
enc_attention_weights.shape

## Use seaborn plot one at each
for i in range(len(enc_attention_weights.shape)):
    for j in range(enc_attention_weights.shape[0]):
        sf.heat_map(enc_attention_weights.cpu().detach()[j][i], xlabel='Key positions',
                    ylabel='Query positions', title=f'Head {i+1}', xlim=[0,5],figsize=(3.5, 3.5))

## Multi-plots in once
sf.heat_map_multi(enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layer, num_head, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape

sf.heat_map_multi(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

sf.heat_map_multi(dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))