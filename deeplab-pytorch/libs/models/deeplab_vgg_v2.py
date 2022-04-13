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


def conv3x3_relu(inplanes, planes, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                    stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Sequential(
                    nn.Conv2d(512, 1024, 3, 1, padding=rate, dilation=rate, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Conv2d(1024, 1024, 1, 1, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Conv2d(1024, out_ch, 1, 1, bias=True),
                )
            )

        for head in self.children():
            for m in head:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2_vgg16(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, atrous_rates):
        super(DeepLabV2_vgg16, self).__init__()
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
        self.add_module("aspp", _ASPP(n_classes, atrous_rates))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
