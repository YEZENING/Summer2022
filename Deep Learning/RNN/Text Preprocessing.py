#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 04 13:17:37 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Text Preprocessing.py
# @Software: PyCharm
"""
# Import Packages
import collections
import requests
import re

# Website
html = 'https://www.gutenberg.org/files/35/35-0.txt'

# Read document from website
def read_txt(html):
    response = requests.get(html)
    texts = response.text.lower().strip().split('\n')
    return [re.sub('[^A-Za-z]+', ' ', text) for text in texts]

text = read_txt()
print(f'# text lines: {len(text)}')
text[0]

# Tokenization
def tokenize(texts, token):
    if token == 'word':
        return [text.split() for text in texts]
    elif token == 'char':
        return [list(text) for text in texts]
    else:
        print('Error: Unrecognize type: ' + token)


tokens = tokenize(text,token='word')
for i in range(11):
    print(tokens[i])

# Define Vocabulary
class Vocab:
    def __init__(self,tokens=[],min_freq=0,reserved_tokens=[]):
        # change to a list if needed
        if tokens and isinstance(tokens[0],list):
            tokens = [token for line in tokens for token in line]
        # count the frequency of token
        counter = collections.Counter(tokens)
        self.tokens_freq = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        # list of unique token
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens +
                                            [token for token, freq in self.tokens_freq if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

# print the vocab
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

# All in one
def load_time_machine(html,max_token=-1):
    text = read_txt(html)
    tokens = tokenize(text,'char')
    vocab = Vocab(tokens)
    # return a single list for each token
    corpus = [vocab[tokens] for line in tokens for tokens in line]
    if max_token > 0:
        corpus = corpus[:max_token]
    return corpus, vocab

corpus, vocab = load_time_machine()
len(corpus), len(vocab)