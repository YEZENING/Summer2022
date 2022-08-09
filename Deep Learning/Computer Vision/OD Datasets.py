#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 08 13:50:49 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     OD Datasets.py
# @Software: PyCharm
"""
# Import Packages
import io
import os
import torch
import requests
import torchvision
import pandas as pd
import SP_Func as sf
from zipfile import ZipFile

# Download data from server
response = requests.get('http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip')
zfile = ZipFile(io.BytesIO(response.content))
zfile.extractall('data')

# Read data
def banana_data(is_train):
    csv_file = os.path.join('data/banana-detection','bananas_train' if is_train else 'bananas_val','label.csv')
    csv_data = pd.read_csv(csv_file)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join('data/banana-detection', 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

# Create customize dataset
class BananaDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        self.features, self.labels = banana_data(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
                                                   is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

# Loading and create dataloader
def load_banana(batch_size):
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True),batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False),batch_size)
    return train_iter, test_iter

# SetUp parameters
batch_size, edge_size = 32, 256
train_iter, _ = load_banana(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape

# Demo
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = sf.show_img(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    sf.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])