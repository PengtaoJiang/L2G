#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

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

def conv3x3_relu(inplanes, planes, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                    stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, num_classes):
        super(_ASPP, self).__init__()
        self.num_classes = num_classes
        self.conv6 = conv3x3(in_planes=512, out_planes=1024, padding=12, dilation=12)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(0.5)

        self.conv7 = conv1x1(in_planes=1024, out_planes=1024, padding=0)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(0.5)

        self.conv8 = conv1x1(in_planes=1024, out_planes=num_classes, padding=0)

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


class DeepLabV1_vgg16(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, atrous_rates):
        super(DeepLabV1_vgg16, self).__init__()
        features = nn.Sequential(conv3x3_relu(3, 64),
                                      conv3x3_relu(64, 64),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(64, 128),
                                      conv3x3_relu(128, 128),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(128, 256),
                                      conv3x3_relu(256, 256),
                                      conv3x3_relu(256, 256),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(256, 512),
                                      conv3x3_relu(512, 512),
                                      conv3x3_relu(512, 512),
                                      nn.MaxPool2d(3, stride=1, padding=1))
        features2 = nn.Sequential(conv3x3_relu(512, 512, rate=2),
                                       conv3x3_relu(512, 512, rate=2),
                                       conv3x3_relu(512, 512, rate=2),
                                       nn.MaxPool2d(3, stride=1, padding=1))
        self.add_module("layer1", features)
        self.add_module("layer2", features2)
        self.add_module("aspp", _ASPP(n_classes))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


if __name__ == "__main__":
    model = DeepLabV1_vgg16(
        n_classes=21, n_blocks=[3, 4, 23, 3]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
