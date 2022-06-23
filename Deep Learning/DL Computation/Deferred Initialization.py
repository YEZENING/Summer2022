#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 23 15:47:21 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Deferred Initialization.py
# @Software: PyCharm
"""
'''
Tensorflow have better stable API than PyTorch for deferring initialization, so the following code will use tensorflow
'''
# Import Package
import tensorflow as tf

# Create a network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(10)
])
[model.layers[i].get_weights() for i in range(len(model.layers))]

# Create data then through the network
X = tf.random.uniform((2,20))
model(X)
[w.shape for w in model.get_weights()]
model.summary()