#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 15 21:51:21 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Bahdanau Attention.py
# @Software: PyCharm
"""
# Import Packages
import torch
from torch import nn
import SP_Func as sf

# Define Attention Decoder
class AttentionDecoder(sf.Decoder):
    def __init__(self,**kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

# Define Sequence to Sequence Decoder
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hidden, num_layer, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = sf.AdditiveAttention(num_hidden, num_hidden, num_hidden, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hidden, num_hidden, num_layer, dropout=dropout)
        self.dense = nn.Linear(num_hidden, vocab_size)

    def init_state(self, enc_output, enc_valid_lens, *args):
        '''
        output shape: (batch_size，num_step，num_hidden)
        hiddent_state shape: (num_layer，batch_size，num_hidden)
        '''
        output, hidden_state = enc_output
        return (output.permute(1,0,2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        '''
        # enc_output shape: (batch_size,num_step,num_hidden).
        # hidden_state shape: (num_layer,batch_size,num_hidden)
        '''
        enc_output, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1,0,2) # X shape: (num_step,batch_size,embed_size)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1) # Shape: (batch_size,1,num_hidden)
            context = self.attention(query, enc_output, enc_output,
                                     enc_valid_lens) # Shape: (batch_size,1,num_hidden)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)),dim=-1) # Shape: (batch_size,1,num_hidden)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # Apply fully connected layer
        outputs = self.dense(torch.cat(outputs, dim=0)) # shape: (num_step,batch_size,vocab_size)
        return outputs.permute(1, 0, 2), [enc_output, hidden_state,enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

# Generate encoder-decoder example
encoder = sf.Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hidden=16,num_layer=2)
encoder.eval()
'''
Seq2SeqEncoder(
  (embedding): Embedding(10, 8)
  (rnn): GRU(8, 16, num_layers=2)
)
'''
decoder = Seq2SeqAttentionDecoder(vocab_size=10,embed_size=8,num_hidden=16,num_layer=2)
decoder.eval()
'''
Seq2SeqAttentionDecoder(
  (attention): AdditiveAttention(
    (W_q): Linear(in_features=16, out_features=16, bias=False)
    (W_k): Linear(in_features=16, out_features=16, bias=False)
    (w_v): Linear(in_features=16, out_features=1, bias=False)
    (dropout): Dropout(p=0, inplace=False)
  )
  (embedding): Embedding(10, 8)
  (rnn): GRU(24, 16, num_layers=2)
  (dense): Linear(in_features=16, out_features=10, bias=True)
)
'''
X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape

# Training
## Define hpyerparameters
embed_size, num_hidden, num_layer, dropout = 32, 32, 2, 0.1
batch_size, num_step = 64, 10
lr, num_epochs, device = 0.005, 250, sf.try_gpu()

## Load dataset and prepare encoder-decoder
train_iter, src_vocab, tgt_vocab = sf.load_data_nmt(batch_size,num_step)
encoder = sf.Seq2SeqEncoder(len(src_vocab),embed_size,num_hidden,num_layer,dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab),embed_size,num_hidden,num_layer,dropout)
model = sf.EncoderDecoder(encoder, decoder)
sf.train_seq2seq(model,train_iter,lr,num_epochs,tgt_vocab,device)

## Compute BLEU
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = sf.pred_seq2seq(
        model, eng, src_vocab, tgt_vocab, num_step, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {sf.bleu(translation, fra, k=2):.3f}')

attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((-1, num_step))
attention_weights.shape

## Plot heatmap
sf.heat_map(attention_weights.detach(),xlabel='Key positions', ylabel='Query positions',xlim=[0,3])