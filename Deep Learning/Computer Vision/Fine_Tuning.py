#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 03 11:58:21 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Fine_Tuning.py
# @Software: PyCharm
"""
# Import Packages
import torch
import requests
import torchvision
from torch import nn
import SP_Func as sf
from zipfile import ZipFile

# Download dataset
'''
response = requests.get('http://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip')
zfile = ZipFile(io.BytesIO(response.content))
zfile.extractall('Computer Vision')
'''

# Read data and visualize
train_data = torchvision.datasets.ImageFolder('Computer Vision/hotdog/train') # dataset might too large
test_data = torchvision.datasets.ImageFolder('Computer Vision/hotdog/test')

# Augmentation and Normalization
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(244),
    torchvision.transforms.ToTensor(),
    normalize
])

# Define pre-train model
pretrain_model = torchvision.models.resnet18(pretrained=True)
pretrain_model.fc # output layer

# fine-tuning model
finetun_model = torchvision.models.resnet18(pretrained=True)
finetun_model.fc = nn.Linear(finetun_model.fc.in_features,2)
nn.init.xavier_uniform_(finetun_model.fc.weight)

# Fine-Tuning the Model
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fin_tuning(model,lr,batch_size=128,num_epcochs=5,param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        'data/hotdog/train',transform=train_augs),batch_size=batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder('data/hotdog/test',transform=test_augs),
                                            batch_size=batch_size)
    devices = sf.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_1x = [param for name, param in model.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                     {'params': model.fc.parameters(),'lr': lr * 10}],
                                    lr=lr, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=0.001)
    sf.train_cv(model,train_iter,test_iter,loss,optimizer,num_epcochs,devices)

# Training
train_fin_tuning(finetun_model,5e-5)

# Scrach mdoel
scrach_model = torchvision.models.resnet18()
scrach_model.fc = nn.Linear(scrach_model.fc.in_features,2)
train_fin_tuning(scrach_model,5e-4,param_group=False)