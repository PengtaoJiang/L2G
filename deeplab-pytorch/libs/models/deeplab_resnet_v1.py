#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   19 February 2019

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation)

class FOV(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, num_classes):
        super(FOV, self).__init__()
        self.num_classes = num_classes
        self.conv6 = conv3x3(in_planes=2048, out_planes=1024, padding=12, dilation=12)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(0.5)

        self.conv7 = conv1x1(in_planes=1024, out_planes=1024, padding=0)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(0.5)

        self.conv8 = conv1x1(in_planes=1024, out_planes=self.num_classes, padding=0)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.conv8(x)
        
        return x

# class FOV(nn.Module):
#     """
#     Atrous spatial pyramid pooling (ASPP)
#     """

#     def __init__(self, num_classes):
#         super(FOV, self).__init__()
#         self.num_classes = num_classes
#         self.conv6 = conv3x3(in_planes=2048, out_planes=num_classes, padding=12, dilation=12)

#         for m in self.children():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv6(x)
#         return x

class DeepLabV1_largeFOV(nn.Sequential):
    """
    DeepLab v1: Dilated ResNet + 1x1 Conv
    Note that this is just a container for loading the pretrained COCO model and not mentioned as "v1" in papers.
    """

    def __init__(self, n_classes, n_blocks):
        super(DeepLabV1_largeFOV, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("fov", FOV(n_classes))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

if __name__== "__main__":
    model = DeepLabV1_largeFOV(n_classes=21, n_blocks=[3, 4, 23, 3])
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
