#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 11 15:10:50 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Neural Style Transfer.py
# @Software: PyCharm
"""
# Import Packages
import torch
import torchvision
from torch import nn
import SP_Func as sf

# Read Content and Style image
'''also can read different image for content and style if you want'''
content_img = sf.get_img('rainier.jpg',True)
style_img = sf.get_img('autumn-oak.jpg',True)

# Preprocessing & Postprocessing
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, img_shape):
    '''transform image'''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean,std=rgb_std)
    ])
    return transform(img).unsqueeze(0)

def postprocess(img):
    '''restore image'''
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2,0,1))

# Extract features
pretrain = torchvision.models.vgg19(pretrained=True)
style_layers, content_layers = [0, 5, 10, 19, 28], [25] # last conv of the fourth conv blcok; the first conv of each
model = nn.Sequential(*[pretrain.features[i] for i in range(max(content_layers + style_layers) + 1)]) # New model

def extract_feature(X, content_layer, style_layer):
    '''extract feature from different layers'''
    contents, styles = [], []
    for i in range(len(model)):
        X = model[i](X)
        if i in style_layer:
            styles.append(X)
        if i in content_layer:
            contents.append(X)
    return contents, styles

def get_contents(img_shape,device):
    ''' extracts content features from the content image'''
    content_X = preprocess(content_img,img_shape).to(device)
    contents_Y, _ = extract_feature(content_X,content_layers,style_layers)
    return content_X, contents_Y

def get_styles(img_shape,device):
    '''extracts style features from the style image'''
    style_X = preprocess(style_img,img_shape).to(device)
    _,styles_Y = extract_feature(style_X,content_layers,style_layers)
    return style_X, styles_Y

# Define Loss Function
def content_loss(Y_hat, Y):
    '''mean square loss; Content loss'''
    return torch.square((Y_hat-Y.detach())).mean()

def gram(X):
    '''Gram matrix XX.t'''
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels,n))
    return torch.matmul(X,X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    '''mean square loss; Style loss'''
    return torch.square((gram(Y_hat) - gram_Y.detach())).mean()

def total_var_loss(Y_hat):
    '''total variation denoising, reduce noise for image'''
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    '''the weighted sum of content loss, style loss, and total variation loss'''
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = total_var_loss(X) * tv_weight
    # Add up all the losses
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# Initializing the Synthesized Image
class SynthesiedIMG(nn.Module):
    def __init__(self,img_shape,**kwargs):
        super(SynthesiedIMG, self).__init__(**kwargs)
        self.weights = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weights

def get_init(X,device,lr,styles_Y):
    '''creates a synthesized image model instance and initializes it into imageX'''
    gen_img = SynthesiedIMG(X.shape).to(device)
    gen_img.weights.data.copy_(X.data)
    optimizer = torch.optim.Adam(gen_img.parameters(),lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer

# Training
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, optimizer = get_init(X,device,lr,styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8) # default is 0.1
    animator = sf.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs],
                           legend=['content', 'style', 'TV'], ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_feature(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X,contents_Y_hat,styles_Y_hat,contents_Y,styles_Y_gram)
        l.backward() # BP
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

## Parameters setup
device, img_shape = sf.try_gpu(), (300, 450)
model = model.to(device)
content_X, content_Y = get_contents(img_shape,device)
_, styles_Y = get_styles(img_shape, device)

output = train(content_X, content_Y, styles_Y, device, 0.3, 500, 50)