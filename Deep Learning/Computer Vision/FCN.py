#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 11 12:41:23 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     FCN.py
# @Software: PyCharm
"""
# Import Packages
import io
import torch
import requests
import torchvision
import SP_Func as sf
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F

# Model
pretrain = torchvision.models.resnet18(pretrained=True) # /Users/zeningye/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
list(pretrain.children())[-3:] # print the outlet
'''
[Sequential(
   (0): BasicBlock(
     (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (relu): ReLU(inplace=True)
     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (downsample): Sequential(
       (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (1): BasicBlock(
     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (relu): ReLU(inplace=True)
     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   )
 ),
 AdaptiveAvgPool2d(output_size=(1, 1)),
 Linear(in_features=512, out_features=1000, bias=True)]
'''
model = nn.Sequential(*list(pretrain.children())[:-2])
X = torch.rand(size=(1,3,320,480))
model(X).shape # torch.Size([1, 512, 10, 15])

## Conduct 1x1 convolution layer to transform from number of output channel to number of classes
num_classes = 21
model.add_module('final_conv',nn.Conv2d(512,num_classes, kernel_size=1)) # cannot use LazyConv2d to conduct conv layer
model.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,
                                                     kernel_size=64,padding=16,stride=32)) # cannot use LazyConVTran
model

# Initializing transpose convolutional layer
def bilinear_kernel(in_channel, out_channel, kernal_size):
    factor = (kernal_size + 1) // 2
    if kernal_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernal_size).reshape(-1,1),
          torch.arange(kernal_size).reshape(1,-1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channel,out_channel,kernal_size,kernal_size))
    weight[range(in_channel),range(out_channel),:,:] = filt
    return weight

## Conduct transpose convolutional layer double the height and width
conv_trans = nn.ConvTranspose2d(3,3,kernel_size=4,padding=1,stride=2,bias=False) # nn.LazyConvTranspose2d(3,kernel_size=4,padding=1,stride=2)
conv_trans.weight.data.copy_(bilinear_kernel(3,3,4))

## Read image
response = requests.get('https://raw.githubusercontent.com/d2l-ai/d2l-en/master/img/catdog.jpg',stream=True)
img = torchvision.transforms.ToTensor()(Image.open(io.BytesIO(response.content))) # only can read their dataset
X = img.unsqueeze(0) # shape:　torch.Size([1, 3, 561, 728])
Y = conv_trans(X) # shape: torch.Size([1, 3, 1122, 1456])
out_img = Y[0].permute(1, 2, 0).detach()
print('input image shape:', img.permute(1, 2, 0).shape)
print('output image shape:', out_img.shape)
plt.imshow(img.permute(1,2,0))
plt.imshow(out_img)
plt.show()

W = bilinear_kernel(num_classes, num_classes, 64)
model.transpose_conv.weight.data.copy_(W)

# Read Dataset
batch_size, crop_size = 32, (320,480)
train_iter, test_iter = sf.load_voc(batch_size,crop_size)

# Training
def loss(inputs, targets):
    '''define loss function'''
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, sf.try_all_gpus()
optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
sf.train_cv(model,train_iter,test_iter,loss,optimizer,num_epochs,devices)

# Prediction
def predict(img):
    X = test_iter.dataset.normalize_img(img).unsqueeze(0)
    pred = model(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2img(pred):
    colormap = torch.tensor(sf.VOC_COLORMAP,device=devices[0])
    X = pred.long()
    return colormap[X,:]

test_imgs, test_labels = sf.read_voc_img(False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0,0,320,480)
    X = torchvision.transforms.functional.crop(test_imgs[i],*crop_rect)
    pred = label2img(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1,2,0)]
sf.show_img(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)

Image.open('https://scc-ondemand2.bu.edu/pun/sys/files/fs/usr4/ma675/znye/PyTorch%20Summer%202022/Computer%20Vision/data/VOC2012/SegmentationObject/2007_000032.png')