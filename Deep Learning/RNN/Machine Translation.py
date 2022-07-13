#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 11 14:04:19 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Machine Translation.py
# @Software: PyCharm
"""
# Import Packages
import torch
import SP_Func as sf
import matplotlib.pyplot as plt

# Read data
def read_data_nmt():
    with open('RNN/fra.txt','r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

# Preprocess data
def preprocess_nmt(text):
    # delete space
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

# tokenization
def tokenize_nmt(text, num_example=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_example and i > num_example:
            break
        part =line.split('\t')
        if len(part) == 2:
            source.append(part[0].split(' '))
            target.append(part[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]

# Plot a histogram of pair
fig, ax = plt.subplots(1,figsize=(6,4))
ax.hist([[len(l) for l in source], [len(l) for l in target]])
ax.set_xlim([0,30])
ax.set_xlabel('# tokens per fequence')
ax.set_ylabel('Count')
ax.legend(['source','target'])
plt.show()

# Create vocabulary for both sources
src_vocab = sf.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

# Apply truncate and padding on the src_vocab
def truncate_pad(line, num_step, pad_token):
    if len(line) > num_step:
        return line[:num_step]  # Truncate
    return line + [pad_token] * (num_step - len(line))  # Pad

truncate_pad(src_vocab[source[0]],10,src_vocab['<pad>'])

# Transform text sequences of machine translation into minibatch
def build_array_nmt(lines, vocab, num_step):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_step, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

build_array_nmt(source,src_vocab,num_step=600)

# All in one
def load_data_nmt(batch_size, num_step, num_example=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_example)
    src_vocab = sf.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>']) # source
    tgt_vocab = sf.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>']) # target
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_step)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_step)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = sf.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_step=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('valid lengths for Y:', Y_valid_len)
    break