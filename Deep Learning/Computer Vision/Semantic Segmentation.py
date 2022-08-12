#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 09 21:16:14 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Semantic Segmentation.py
# @Software: PyCharm
"""
# Import Packages
import os
import io
import torch
import tarfile
import requests
import torchvision
import SP_Func as sf

# Pascal VOC2012 Semantic Segmentation
## Download data
'''file around 2GB, download into server instead in local'''
res = requests.get('http://d2l-data.s3-accelerate.amazonaws.com/VOCtrainval_11-May-2012.tar',stream=True)
z_file = tarfile.open(fileobj=io.BytesIO(res.content),mode='r:gz')
z_file.extractall('data')

## Read input images and labels
def read_voc_img(is_train=True):
    txt_name = os.path.join('data/VOC2012','ImageSets/Segmentation','train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_name,'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join('data/VOC2012','JPEGImages',f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join('data/VOC2012', 'SegmentationClass',
                                                             f'{fname}.png'),mode))
    return features, labels

train_features, train_labels = read_voc_img(True)

## Plot some images
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
sf.show_img(imgs,2,n)

## Enumerate RGB color values and class name
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    '''Build the mapping from RGB to class indices for VOC labels'''
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    '''Map any RGB values in VOC labels to their class indices'''
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]

## Data Preprocessing
def voc_rand_crop(feature, label, height, width):
    '''randomly crop feature and label in image'''
    rect = torchvision.transforms.RandomCrop.get_params(feature,(height,width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0],train_labels[0],200,300) # apply random crop function to imgs
imgs = [img.permute(1, 2, 0) for img in imgs]
sf.show_img(imgs[::2] + imgs[1::2], 2, n) # visualization after crop

## Customize dataset class
class VOCSegData(torch.utils.data.Dataset):
    def __init__(self,is_train,crop_size):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]) # Normalization
        self.crop_size = crop_size
        features, labels = read_voc_img(is_train=is_train)
        self.features = [self.normalize_img(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_img(self,img):
        return self.transform(img.float() / 255)

    def filter(self,imgs):
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and
                                        img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        features, labels = voc_rand_crop(self.features[idx],self.labels[idx],*self.crop_size)
        return (features, voc_label_indices(labels,self.colormap2label))

    def __len__(self):
        return len(self.features)

## Read dataset
crop_size = (320,480)
voc_train = VOCSegData(True,crop_size)
voc_test = VOCSegData(False,crop_size)

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train,batch_size,shuffle=True,drop_last=True,
                                         num_workers=4)
## All in One
def load_voc(batch_size,crop_size):
    train_iter = torch.utils.data.DataLoader(VOCSegData(True,crop_size), batch_size,
                                             shuffle=True, drop_last=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(VOCSegData(False,crop_size), batch_size,
                                             drop_last=True, num_workers=4)
    return train_iter, test_iter