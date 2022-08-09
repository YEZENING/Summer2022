#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Aug 03 16:23:58 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Deep Learning
# @File:     Anchor Box.py
# @Software: PyCharm
"""
# Import Packages
import io

import PIL.Image
import requests
import torch
import SP_Func as sf
import matplotlib.pyplot as plt

# SetUp
torch.set_printoptions(2)

# Generating Multiple Anchor Boxes
def multibox_prior(data, sizes, ratio):
    '''Generate anchor boxes with different shapes centered on each pixel'''
    in_height, in_width = data.shape[-2:]
    device, num_size, num_ratio = data.device, len(sizes), len(ratio)
    boxes_per_pixel = (num_size + num_ratio - 1) # number of anchor boxes centered on the same pixel: n + m - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratio, device=device)
    # Offsets are required to move the anchor to the center of a pixel. height and width = 1, choose offset = 0.5
    offset_h, offset_w = 0.5, 0.5
    step_h = 1.0 / in_height # scale step in y
    step_w = 1.0 / in_width # scale step in x

    # Generate all center point for the anchor box
    center_h = (torch.arange(in_height, device=device) + offset_h) * step_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * step_w
    shift_y, shift_x = torch.meshgrid((center_h, center_w))
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

# Read image
img = plt.imread('Computer Vision/img/dogcat.png')
h, w = img.shape[:2] # get height and width
print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratio=[1, 2, 0.5])
Y.shape

boxes = Y.reshape(h, w, 5, 4)
boxes[150, 150, 0, :]
# boxes[250, 250, 0, :]

def show_bboxes(axes, bboxes, labels=None, colors=None):
    '''show bounding box'''

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = sf.box_rect(bbox.detach(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# plot the anchor box
bbox_scale = torch.tensor((w,h,w,h))
img = plt.imread('Computer Vision/img/dogcat.png') # read image as array
fig = plt.imshow(img)
show_bboxes(fig.axes, boxes[122, 110, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.745, r=1', 's=0.75, r=1.25',
             's=0.75, r=0.5'])
plt.show()

# Interaction over Union (IoU)
def box_iou(box1, box2):
    '''Compute pairwise IoU across two lists of anchor or bounding boxes'''
    box_area = lambda boxes: ((boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]))
    # Shape of `box1`, `box2`, `area1`, `area2`:
    # (no. of boxes1, 4), (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    area1 = box_area(box1)
    area2 = box_area(box2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`:
    # (no. of box1, no. of box2, 2)
    inter_upperlefts = torch.max(box1[:, None, :2], box2[:, :2])
    inter_lowerrights = torch.min(box1[:, None, 2:], box2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of inter-area, union-area: (no. of box1, no. of box2)
    inter_area = inters[:,:,0] * inters[:,:,1]
    union_area = area1[:,None] + area2 - inter_area
    return inter_area / union_area # IoU

# Label Anchor box in training set
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_gt, num_anchor = ground_truth.shape[0], anchors.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    # initialize the tensor to hold the assigned ground-truth bounding box for each anchor
    anchor_bbox_map = torch.full((num_anchor,),-1,dtype=torch.long,device=device)
    # assign ground_truth box according to the threshold
    max_iou, indice = torch.max(jaccard,dim=1)
    anc_i = torch.nonzero(max_iou >= iou_threshold).reshape(-1)
    box_i = indice[max_iou >= iou_threshold]
    anchor_bbox_map[anc_i] = box_i
    col_discard = torch.full((num_anchor,),-1)
    row_discard = torch.full((num_gt,),-1)
    for _ in range(num_gt):
        max_idx = torch.argmax(jaccard) # find the largest IoU
        box_idx = (max_idx % num_gt).long()
        anc_idx = (max_idx / num_gt).long()
        anchor_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchor_bbox_map

# Label classes and offset
def offset_box(anchor,assigned_bbox, eps=1e-6):
    '''transform for anchor box offset'''
    central_anc = sf.box_corner_to_center(anchor)
    central_ass_bb = sf.box_corner_to_center(assigned_bbox)
    offset_xy = 10 * (central_ass_bb[:, :2] - central_anc[:, :2]) / central_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + central_ass_bb[:, 2:] / central_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1,4)
        # initialize class labels and assign bounding box corrdinate with zero
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)

        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        idx_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[idx_true]
        class_labels[idx_true] = label[bb_idx, 0].long() + 1
        assigned_bb[idx_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_box(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

## example
ground_truth = torch.tensor([[0, 0.1, 0.04, 0.515, 0.97], [1, 0.525, 0.37, 0.9, 0.97]]) # percentage of image
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.52, 0.05, 0.78, 0.98], [0.55, 0.45, 0.7, 0.8],
                    [0.515, 0.3, 0.92, 0.9]])

fig = plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
labels[0]
'''
tensor([[-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00,  1.30e+00,  1.02e+01,
          2.53e+00,  7.68e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00,
         -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -9.29e-01,  1.17e+00,
          3.45e-01,  5.36e-06]])

'''
labels[1]
'''
tensor([[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
         1., 1.]])
'''
labels[2]
'''
tensor([[0, 1, 0, 0, 2]])
'''

# Predicting bounding boxes with non-maximun supression
def offset_inverse(anchors, offset_pred):
    '''Predict bounding boxes based on anchor boxes with predicted offsets'''
    anc = sf.box_corner_to_center(anchors)
    pred_offset_xy = (offset_pred[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_offset_wh = torch.exp(offset_pred[:,2:] / 5) * anc[:,2:]
    pred_bbox = torch.cat((pred_offset_xy,pred_offset_wh),axis=1)
    fin_pred_bbox = sf.box_center_to_corner(pred_bbox)
    return fin_pred_bbox

## Non-maximum suppression
def nms(boxes,scores,iou_threshold):
    '''Sort confidence score in predicted bounding box in descending order and return a list'''
    B = torch.argsort(scores,dim=-1,descending=True)
    keep = [] # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep,device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    '''Predict bounding boxes using non-maximum suppression'''
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    output = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1,4)
        conf, class_id = torch.max(cls_prob[1:],0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors,dtype=torch.long,device=device)
        combine = torch.cat((keep,all_idx))
        unique, count = combine.unique(return_counts=True)
        non_keep = unique[count == 1]
        all_id_sort = torch.cat((keep,non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sort]
        conf, predicted_bb = conf[all_id_sort], predicted_bb[all_id_sort]
        # pos_threshold` is a threshold for positive (non-background) predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),conf.unsqueeze(1),predicted_bb), dim=1)
        output.append(pred_info)
    return output

## Implementation
anchors = torch.tensor([[0.1, 0.05, 0.52, 0.97], [0.08, 0.2, 0.56, 0.96],
                      [0.15, 0.3, 0.62, 0.95], [0.52, 0.38, 0.9, 0.96]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # Predicted background likelihood
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood

## plot the predict anchors
fig = plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
plt.show()

## Result
output = multibox_detection(cls_probs.unsqueeze(dim=0),offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),nms_threshold=0.5)
output
'''
'-1' indicate background or removal in suppresion
'0':'dog', '1':'cat'
[tensor([[ 0.00,  0.90,  0.10,  0.05,  0.52,  0.97],
         [ 1.00,  0.90,  0.52,  0.38,  0.90,  0.96],
         [-1.00,  0.80,  0.08,  0.20,  0.56,  0.96],
         [-1.00,  0.70,  0.15,  0.30,  0.62,  0.95]])]
'''

# Final Output after implement
fig = plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1: continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
plt.show()