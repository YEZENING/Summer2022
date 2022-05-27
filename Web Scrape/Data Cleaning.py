#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 26 14:01:59 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Data Cleaning.py
# @Software: PyCharm
"""
# Import Packages
import requests
import re
import string
from bs4 import BeautifulSoup

## Return list of 2-gram
def getngrams(content, n): # n is num of slice windows
    content = re.sub('\n|[[\d+\]]',' ',content) # using regex to extract the content
    content = bytes(content,'utf-8') # encode into utf-8
    content = content.decode('ascii','ignore')
    content = content.split(' ') # split empty space
    content = [word for word in content if word != '']
    output = []
    for i in range(len(content)-n+1):
        output.append(content[i:i+n])
    return output

html = requests.get('https://en.wikipedia.org/wiki/Python_(programming_language)')
bs = BeautifulSoup(html.text, 'html.parser')
content = bs.find('div',{'id':'mw-content-text'}).get_text()
ngrams = getngrams(content, 2)
print(ngrams)
print(f'2-grams count is: {str(len(ngrams))}')

## delete puntuation from previous code
def cleanSentence(sentence):
    sentence = sentence.split(' ')
    sentence = [word.strip(string.punctuation+string.whitespace) for word in sentence]
    sentence = [word for word in sentence if len(word) > 1 or (word.lower() == 'a' or word.lower() == 'i')]
    return sentence

def cleanInput(content):
    content = content.upper() # reduce number of unique 2-grams
    content = re.sub('\n|[[\d+\]]', ' ', content)
    content = bytes(content, "UTF-8")
    content = content.decode("ascii", "ignore")
    sentences = content.split('. ')
    return [cleanSentence(sentence) for sentence in sentences]

def getNgramsFromSentence(content, n): # num_shift n = 2
    output = []
    for i in range(len(content)-n+1):
        output.append(content[i:i+n])
    return output

def getNgrams(content, n):
    content = cleanInput(content)
    ngrams = []
    for sentence in content:
        ngrams.extend(getNgramsFromSentence(sentence, n))
    return(ngrams)

html = requests.get('https://en.wikipedia.org/wiki/Python_(programming_language)')
bs = BeautifulSoup(html.text, 'html.parser')
content = bs.find('div',{'id':'mw-content-text'}).get_text()
ngrams = getngrams(content, 2)
print(ngrams)
print(f'2-grams count is: {len(ngrams)}')
# print(string.punctuation)

## Data Normalization
from collections import Counter
def getNgrams(content, n):
    content = cleanInput(content)
    ngrams = Counter() # count the frequency of 2-grams appear in the text
    for sentence in content:
        newNgrams = [' '.join(ngram) for ngram in getNgramsFromSentence(sentence, 2)]
        ngrams.update(newNgrams)
    return (ngrams)

content = bs.find('div',{'id':'mw-content-text'}).get_text()
print(getNgrams(content, 2))