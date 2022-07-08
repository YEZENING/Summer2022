#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 05 15:26:11 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Language Model.py
# @Software: PyCharm
"""
# Import Packages
import random
import torch
import matplotlib.pyplot as plt
from SP_Func import tokenize, read_txt, Vocab, load_time_machine

# Load dataset
html = 'https://www.gutenberg.org/files/35/35-0.txt'
tokens = tokenize(read_txt(html),token='word') # get toke
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus) # create vocabulary
vocab.tokens_freq[:10]

# plot the frequency (single token)
freqs = [freq for token, freq in vocab.tokens_freq]
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
ax.plot(freqs)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Token: x')
ax.set_ylabel('Frequency: n(x)')
ax.grid(True)
plt.show()

# mutil gram(word combination) bigram, trigram
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.tokens_freq[:10]

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.tokens_freq[:10]

# Plot multiple frequency
bigram_freqs = [freq for token, freq in bigram_vocab.tokens_freq]
trigram_freqs = [freq for token, freq in trigram_vocab.tokens_freq]
fig, ax = plt.subplots(1,figsize=(7,4),dpi=200)
ax.plot(freqs,label='unigram')
ax.plot(bigram_freqs,'--',label='bigram')
ax.plot(trigram_freqs,'g-.',label='trigram')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Token: x')
ax.set_ylabel('Frequency: n(x)')
ax.grid(True)
ax.legend()
plt.show()

# Generate sequence data randomly (random selection)
def random_sequence(corpus, batch_size, num_steps):
    # randomly select the copus
    corpus = corpus[random.randint(0, num_steps -1):] # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # shuffle the dataset
    random.shuffle(initial_indices)

    def data(enter):
        # Return a sequence of length `num_steps` starting from given word
        return corpus[enter: enter + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch] # featurs
        Y = [data(j + 1) for j in initial_indices_per_batch] # labels
        yield torch.tensor(X), torch.tensor(Y)

seq = list(range(35))
for X, Y in random_sequence(seq, 2, 5):
    print('X:', X, '\nY:',Y)

# Sequential Partition
def seq_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0,num_steps) # step number between two sequence
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens]) # features
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens]) # labels
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batch = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batch, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_sequential(seq, 2, 5):
    print('X:', X, '\nY:',Y)

# Combine togather
class SeqDataloader:
    def __init__(self, batch_size, num_steps, use_rand, max_token):
        if use_rand:
            self.data = random_sequence
        else:
            self.data = seq_sequential
        self.corpus, self.vocab = load_time_machine(max_token)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data(self.corpus, self.batch_size, self.num_steps)

# return data iter and vocabulary
def load_data_time_machine(batch_size, num_steps,
                           use_rand=False, max_token=10000):
    data = SeqDataloader(html,batch_size,num_steps,use_rand,max_token)
    return data, data.vocab