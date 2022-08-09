#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 08 15:34:06 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     SSD.py
# @Software: PyCharm
"""
# Import Packages
import torch
import torchvision
import SP_Func as sf
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

'''SSD representative as Single Shoot Multibox Detection'''
# Model
## Class Prediction Layer
def cls_predictor(num_anchor, num_class):
    return nn.LazyConv2d(num_anchor * (num_class + 1), kernel_size=3, padding=1)

## Bouonding Box Prediction Layer
def bbox_predictor(num_anchor):
    return nn.LazyConv2d(num_anchor * 4,kernel_size=3,padding=1)

## Concatenating Predictions for Multiple Scales
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(torch.zeros((2,16,10,10)), cls_predictor(3,10))
Y1.shape, Y2.shape

def flatten_pred(pred):
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1)

def concat_pred(preds):
    # shape: (batch_size, height x width x number of channels)
    return torch.cat([flatten_pred(p) for p in preds],dim=1)

concat_pred([Y1,Y2]).shape

## Downsampling Blcok
def down_sample_blk(out_channel):
    block = []
    for _ in range(2):
        block.append(nn.LazyConv2d(out_channel,kernel_size=3,padding=1))
        block.append(nn.BatchNorm2d(out_channel))
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)

forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape

## Base Network Block
def base_net():
    block = []
    num_filters = [3,16,32,64]
    for i in range(len(num_filters) - 1):
        block.append(down_sample_blk(num_filters[i+1]))
    return nn.Sequential(*block)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape

## Complete Model
def get_block(i):
    if i == 0:
        block = base_net()
    elif i == 1:
        block = down_sample_blk(128)
    elif i == 4:
        block = nn.AdaptiveMaxPool2d((1, 1))
    else:
        block = down_sample_blk(128)
    return block

def block_forward(X, block, size, ratio, cls_predictor, bbox_predictor):
    Y = block(X)
    anchors = sf.multibox_prior(Y, sizes=size, ratio=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

## Define Parameters
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self,num_class,**kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_class = num_class
        for i in range(5):
            setattr(self, f'block_{i}', get_block(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors,num_class))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self,X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = block_forward(
                X, getattr(self, f'block_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors,dim=1)
        cls_preds = concat_pred(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_class + 1)
        bbox_preds = concat_pred(bbox_preds)
        return anchors, cls_preds, bbox_preds

model = TinySSD(num_class=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = model(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

# Training
## Read data
batch_size = 32
train_iter,_ = sf.load_banana(batch_size)

## Device and Optimizer
device, model = sf.try_gpu(), TinySSD(num_class=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, weight_decay=5e-4)

## Define Loss Function
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def cal_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
    batch_size,num_class = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_class), cls_labels.reshape((-1))).reshape(batch_size,-1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds,cls_labels):
    '''Since the class prediction result in the final dimension, `argmax` need to specify this dimension'''
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds,bbox_labels,bbox_mask):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_mask)).sum())

## Hyperparameter SetUp
num_epochs, timer = 20, sf.Timer()
animator = sf.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['class error', 'bbox mae'])
model = model.to(device)

## Training Loop
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = sf.Accumulator(4)
    model.train()
    for feature, label in train_iter:
        timer.start()
        optimizer.zero_grad()
        X, Y = feature.to(device), label.to(device)
        # Generate multiscale anchor boxes and predict their classes and offsets
        anchors, cls_preds, bbox_preds = model(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = sf.multibox_target(anchors, Y)
        l = cal_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks) # cal loss value
        l.mean().backward() # BP
        optimizer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
    animator.display()
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on 'f'{str(device)}')

# Prediction
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

## Define predict function
def predict(X):
    model.eval()
    anchors, cls_preds, bbox_preds = model(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = sf.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0,idx]
output = predict(X)

## Display function
def display(img, output, threshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        sf.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output,threshold=0.9)