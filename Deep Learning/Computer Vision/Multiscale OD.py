#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 08 12:11:05 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Multiscale OD.py
# @Software: PyCharm
"""
# Import Packages
import numpy as np
import torch
import SP_Func as sf
import matplotlib.pyplot as plt

# Multiscale Anchor Box
img = plt.imread('Computer Vision/img/dogcat.png')
h,w = img.shape[:2]
print(h,w)

def display_anchors(fmap_w, fmap_h, s):
    fmap = torch.zeros((1,4,fmap_h,fmap_w))
    anchors = sf.multibox_prior(data=fmap,sizes=s,ratio=[1,2,0.5])
    bbox_scale = torch.tensor((w,h,w,h))
    sf.show_bboxes(plt.imshow(img).axes,anchors[0] * bbox_scale)
    plt.show()

display_anchors(fmap_w=4, fmap_h=4, s=[0.15]) # col=row=4 with scale 0.15
display_anchors(fmap_w=2, fmap_h=2, s=[0.4]) # reduce by half with scale 0.4
display_anchors(fmap_w=1, fmap_h=1, s=[0.9]) # reduce by half with scale 0.9