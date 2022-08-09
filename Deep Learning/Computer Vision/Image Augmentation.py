#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 02 14:38:01 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Image Augmentation.py
# @Software: PyCharm
"""
# Import Packages
import torch
import torchvision
from torch import nn
import SP_Func as sf
import matplotlib.pyplot as plt
from PIL import Image

# Read image
img = Image.open('Computer Vision/img/cat1.jpg')
fig, ax = plt.subplots(1,dpi=200)
ax.imshow(img)
plt.show()

# Apply Augmentation into imgae
def show_img(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=200)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.numpy()
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()

def apply(imgs, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(imgs) for _ in range(num_rows * num_cols)]
    show_img(Y, num_rows, num_cols, scale=scale)

# Common Augmentation Methods
## Horizontal Flip
apply(img, torchvision.transforms.RandomHorizontalFlip())

## Vertical Flip
apply(img, torchvision.transforms.RandomVerticalFlip())

## Cropping randomly
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

## Change color
apply(imgs=img, aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0)) # brightness
apply(imgs=img, aug=torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5)) # hue

color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

## Combine multiple changes into the image
augs = torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(),color_aug, shape_aug])
apply(img,augs)

# Training
training_data = torchvision.datasets.CIFAR10(train=True, root='data',download=True)
show_img([training_data[i][0] for i in range(32)], 4, 8, scale=.5)

## Augmentation
train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

## Dataloader
def cifar10_load(is_train, augs, batch_size):
    data = torchvision.datasets.CIFAR10(root='data',train=is_train,transform=augs,download=False)
    dataloader = torch.utils.data.DataLoader(data,batch_size,shuffle=is_train,num_workers=4)
    return dataloader

## Train
def train_batch_cv(model, X, y, loss, optimizer, devices):
    # using multiple gpus
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])

    # training loop
    model.train()
    optimizer.zero_grad()
    pred = model(X) # get pred value
    l = loss(pred, y)
    l.sum().backward() # BP
    optimizer.step() # update parameters
    train_loss_sum = l.sum()
    train_acc_sum = sf.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_cv(model, train_iter, test_iter, loss, optimizer, num_epochs, devices=sf.try_all_gpus()):
    '''train a model with multiple GPUs'''
    timer, num_batch = sf.Timer(), len(train_iter)
    animator = sf.Animator(xlabel='epochs',xlim=[1,num_epochs], ylim=[0,1],
                           legend=['train_loss', 'train_acc', 'test_acc'])
    model = nn.DataParallel(model,device_ids=devices).to(devices[0])
    print('Processing....')
    for epoch in range(num_epochs):
        metric = sf.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_cv(model,features,labels,loss,optimizer,devices)
            metric.add(l,acc,labels.shape[0],labels.numel())
            timer.stop()
            if (i + 1) % (num_batch // 5) == 0 or i == num_batch - 1:
                animator.add(epoch + (i + 1) / num_batch, (metric[0] / metric[2], metric[1] / metric[3],None))
        test_acc = sf.evaluate_accuracy_gpu(model,test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

## Parameters initialization
def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
batch_size, devices, model = 256, sf.try_all_gpus(), sf.restnet18(10)
model.apply(init_cnn)

def train_with_augs(train_augs, test_augs, model, lr=0.001):
    train_iter = cifar10_load(True,train_augs,batch_size)
    test_iter = cifar10_load(False,test_augs,batch_size)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    model(next(iter(train_iter))[0])
    train_cv(model,train_iter,test_iter,loss,optimizer,10,devices)

## Train
train_with_augs(train_augs,test_augs,model)