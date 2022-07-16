#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jul 13 14:56:37 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Attention Cue.py
# @Software: PyCharm
"""
# Import Packages
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Visualization of Attention
def heat_map(matrices, xlabel, ylabel, figsize=(6.5,4), title=None, cmap='Reds'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ax=ax, data=matrices, cmap=cmap) # data must be 2-dimentionas
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

attention_weights = torch.eye(10) # Shape: (10,10)
heat_map(attention_weights,xlabel='Keys',ylabel='Queries')