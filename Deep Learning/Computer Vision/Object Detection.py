#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 03 14:22:04 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Object Detection.py
# @Software: PyCharm
"""
# Import Packages
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Plot data
img = Image.open('Computer Vision/img/dogcat.png')
fig, ax = plt.subplots(dpi=200)
ax.imshow(img)
plt.show()

# Define object box
def box_corner_to_center(boxes):
    '''convert from (upper-left, upper-right) to (center, center)'''
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2],boxes[:,3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx,cy,w,h),axis=-1)
    return boxes

def box_center_to_corner(boxes):
    '''convert from (center, center) to (upper-left, upper-right)'''
    cx,cy,w,h = boxes[:,0], boxes[:,1], boxes[:,2],boxes[:,3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# Implement into image
dog_bbox, cat_bbox = [38.0, 15.0, 184.0, 272.0], [186.0, 105.0, 321.0, 272.0]
boxes = torch.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes

def box_rect(bbox, color):
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
                         fill=False, edgecolor=color, linewidth=1)

fig = plt.imshow(img)
fig.axes.add_patch(box_rect(dog_bbox, 'blue'))
fig.axes.add_patch(box_rect(cat_bbox, 'red'))
plt.show()