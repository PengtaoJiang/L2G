import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.fc_extra_3 = nn.Conv2d(512, 64, 1, bias=False)
        self.fc_extra_4 = nn.Conv2d(512, 64, 1, bias=False)
        self.extra_att_module = nn.Conv2d(131, 128, 1, bias=False)
        self.extra_last_conv = nn.Conv2d(512, num_classes + 1, 1, bias=False)
        self._initialize_weights()

        torch.nn.init.xavier_uniform_(self.extra_last_conv.weight)
        torch.nn.init.kaiming_normal_(self.fc_extra_4.weight)
        torch.nn.init.kaiming_normal_(self.fc_extra_3.weight)
        torch.nn.init.xavier_uniform_(self.extra_att_module.weight, gain=4)

    def forward(self, image):
        out_layer = [23]
        out_ans = []
        x = image.clone()
        for i in range(len(self.features)):
            x = self.features[i](x)
            if(i in out_layer):
                out_ans.append(x)
        _, _, h, w = x.size()
        for o in out_ans:
            o = F.interpolate(o, (h, w), mode='bilinear', align_corners=True)
        image = F.interpolate(image, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([image, self.fc_extra_3(out_ans[0].detach()), self.fc_extra_4(x.detach())], dim=1)
        x = self.extra_convs(x)
        x_att = self.PCM(x, f)
        cam_att = self.extra_last_conv(x_att)
        cam = self.extra_last_conv(x)
        # loss = torch.mean(torch.abs(cam_att[:, 1:, :, :] - cam[:, 1:, :, :])) * 0.02


        self.featmap = cam_att + cam
        logits = F.avg_pool2d(self.featmap[:, :-1], kernel_size=(cam_att.size(2), cam_att.size(3)), padding=0)
        logits = logits.view(-1, 20)

        return self.featmap, logits

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.extra_att_module(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv

    def get_heatmaps(self):
        return self.featmap.clone().detach()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model
