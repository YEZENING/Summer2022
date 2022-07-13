#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 11 22:53:19 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Encoder-Decoder.py
# @Software: PyCharm
"""
# Import Packages
from torch import nn

# Encoder
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, **kwargs):
        raise NotImplementedError

# Decoder
class Decoder(nn.Module):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, eng_output, *args):
        raise NotImplementedError

    def forward(self, X, **kwargs):
        raise NotImplementedError

'''
The encoder architecture is: 
input --> Encoder --> state --> Decoder --> output  
'''
# All in one
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,enc_X, dec_X, *args):
        enc_output = self.encoder(enc_X,*args)
        dec_state = self.decoder.init_state(enc_output, *args)
        return self.decoder(dec_X, dec_state)