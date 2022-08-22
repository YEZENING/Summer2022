#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 15 14:11:01 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Image Classification.py
# @Software: PyCharm
"""
# Import Packages
import os
import io
import math
import torch
import shutil
import requests
import torchvision
import collections
import pandas as pd
from torch import nn
import SP_Func as sf
from zipfile import ZipFile

'''
This section I will use data from Kaggle--CIFAR-10 to develop a neural network
for image classification.
'''
# Download dataset (Already Download)
def cifar10_tiny():
    response = requests.get('http://d2l-data.s3-accelerate.amazonaws.com/kaggle_cifar10_tiny.zip', stream=True)
    zfile = ZipFile(io.BytesIO(response.content))
    zfile.extractall('Computer Vision/img')
    zfile.close()

# Organize dataset
data_dir = 'Computer Vision/img/kaggle_cifar10_tiny' # local direction, change into server direction
def read_csv_cif(fname):
    with open(fname,'r') as f:
        '''skip header'''
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name,label) for name, label in tokens))

labels = read_csv_cif(os.path.join(data_dir,'trainLabels.csv'))
print('# training examples:', len(labels))
print('# classes:', len(set(labels.values())))

def copyfile(filename, target_dir):
    '''copy file from original'''
    # if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename,target_dir) # shutil.copy not copyfile

def reorg_train_valid(data_dir,labels,valid_ratio):
    '''split validation set from the original training set'''
    # Number of example of the class
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # Numberof examples per class for validation set
    n_valid_per_lable = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir,'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir,'train',train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_lable:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))
    return n_valid_per_lable

def reorg_test(data_dir):
    '''organizing testing set for data loading in prediction'''
    for test_file in os.listdir(os.path.join(data_dir,'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10(data_dir, valid_ratio):
    labels = read_csv_cif(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

## Parameter pre-set
batch_size = 32 # If we use entire dataset use 128
valid_ratio = 0.1
reorg_cifar10(data_dir, valid_ratio)

# Image Augmentation
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Reading dataset
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder), transform=transform_train) for folder in ['train','test']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder), transform=transform_test) for folder in ['valid','test']]

## Dataloader
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset,batch_size, shuffle=True,drop_last=True) for dataset in (train_ds,train_valid_ds)]
valid_iter = torch.utils.data.DataLoader(valid_ds,batch_size,shuffle=False,drop_last=False)
test_iter = torch.utils.data.DataLoader(test_ds,batch_size,shuffle=False,drop_last=False)

# Define Model
def get_model():
    model = sf.restnet18(10)
    return model
loss = nn.CrossEntropyLoss(reduction='none')

def train(model, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    '''define training loop'''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # change SGD into Adam
    schedulor = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), sf.Timer()
    legend = ['train_loss','train_acc']
    if valid_iter is not None:
        legend.append('valid_loss')
    animator = sf.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    model = nn.DataParallel(model,device_ids=devices).to(devices[0])
    print('Processing Epoch...')
    for epoch in range(num_epochs):
        # print(f'Processing Epoch: {epoch}')
        model.train()
        metric = sf.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = sf.train_batch_cv(model,features,labels,loss,optimizer,devices)
            metric.add(l,acc,labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = valid_acc = sf.evaluate_accuracy_gpu(model,valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        schedulor.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    animator.display()
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(devices)}')

# Training and validing the model
devices, num_epochs, lr, wd = sf.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9
model = get_model()
model(next(iter(train_iter))[0])
train(model, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# Classifying testing set
model, preds = get_model(), []
model(next(iter(train_valid_iter))[0]) # raise Stop Iteration error
train(model, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

for X, _ in test_iter:
    y_hat = model(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1,len(test_ds) + 1))
sorted_ids.sort(key=lambda x:str(x))
df = pd.DataFrame({'id':sorted_ids, 'label':preds}) # Create csv
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)