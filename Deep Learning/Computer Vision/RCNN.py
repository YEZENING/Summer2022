#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 09 13:59:04 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     RCNN.py
# @Software: PyCharm
"""
# Import Packages
import torch
import torchvision

# Fast R-CNN
## Region of Interest Pooling Layer
X = torch.arange(16.0).reshape(1,1,4,4)
X
'''
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
'''
'''
input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements.
Each element contains ``C`` feature maps of dimensions ``H x W``.
boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
'''
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
torchvision.ops.roi_pool(X,rois,output_size=(2,2),spatial_scale=0.1)
'''
tensor([[[[ 5.,  6.],
          [ 9., 10.]]],
        [[[ 9., 11.],
          [13., 15.]]]])
'''