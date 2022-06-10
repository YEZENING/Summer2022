#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 05 14:34:08 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Preliminary.py
# @Software: PyCharm
"""
# Import Packages
import tensorflow as tf
import os
import pandas as pd

# Tensorflow basic operation
x = tf.range(12)
tf.size(x)

X = tf.reshape(x, (3,4))
X

tf.zeros((2,4,4))

# Cerate a csv file
os.makedirs('Data')
data_file = os.path.join('Data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # row name
    f.write('NA,Pave,127500\n')  # each line is a sample
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

## read csv via pandas
dta = pd.read_csv('Data/house_tiny.csv')
dta

inputs,outputs = dta.iloc[:,0:2], dta.iloc[:,2] # separate input and output to fiil the NA value
inputs = inputs.fillna(inputs.mean())
inputs = pd.get_dummies(inputs, dummy_na=True)

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y

## Matrix
A = tf.reshape(tf.range(25,dtype=tf.float64),(5,5))
A
tf.transpose(A)

## Symmetrix Metrix
B = tf.constant([[1,2,3],[2,0,4],[3,4,5]])
tf.transpose(B)
B == tf.transpose(B)

## Reduction
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)

'''
Matrix A:
<tf.Tensor: shape=(5, 5), dtype=int32, numpy=
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]], dtype=int32)>
'''

A.shape, tf.reduce_sum(A) # same as tf.reduce_sum(A, axis=[0,1])
A_sum_axis0 = tf.reduce_sum(A, axis=0) # reuction sum for each row
A_sum_axis0, A_sum_axis0.shape
A_sum_axis1 = tf.reduce_sum(A, axis=1) # reduction sum for each column
A_sum_axis1, A_sum_axis1.shape

tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()

## Non-Reduction Sum
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
A / sum_A

tf.cumsum(A) #cummulative sum of element

## Dot Product
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x,y,axes=1)
tf.reduce_sum(x * y)

## matrix-vector and matric multiplication
B = tf.reshape(tf.range(20,dtype=tf.float64),(5,4))
tf.matmul(A,B)

## Norm
u = tf.constant([3.0, -4.0])
tf.norm(u)
tf.norm(tf.ones((4,9)))
len(X)

# Automatic Differentiation
x = tf.range(4,dtype=tf.float32)
x

x = tf.Variable(x)
with tf.GradientTape() as t:
    y = 2*tf.tensordot(x,x,axes=1) # y = 2x1 + 2x2 + 2x3 + 2x4
y

x_grad = t.gradient(y,x) # take partial derivative for x1 - x4 of y
x_grad
x_grad == 4*x

with tf.GradientTape() as t:
    y = tf.reduce_sum(x) # y = x1 + x2 + x3 + x4
t.gradient(y, x)

## Bcakward for non-scalar variable
with tf.GradientTape() as t:
    y = x * x
t.gradient(y,x)

## Detaching Computation
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x
x_grad = t.gradient(z, x)
x_grad == u